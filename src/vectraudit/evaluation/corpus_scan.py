"""Scan a corpus for text that tries to instruct a model, before retrieval ever serves it.

A retrieval-augmented system pastes retrieved documents into a prompt. A document that
contains instructions, role markers or hidden text can steer the model that reads it: indirect
prompt injection, and the document side of RAG poisoning. This looks for the *traces* such
text leaves, with deterministic detectors that need no model and no network. The detectors
follow the attack families in promptfoo's RAG red-team plugins (``rag-poisoning``,
``indirect-prompt-injection``, ``rag-document-exfiltration``; MIT) and its encoding
strategies, reimplemented here:

- ``instruction_override``: "ignore (all) previous instructions" and its variants.
- ``role_reassignment``: "you are now ...".
- ``prompt_reference``: "system prompt", "developer message".
- ``chat_template_token``: ``<|im_start|>``, ``[INST]``, ``<<SYS>>``, ``### System:``, a line
  starting ``assistant:``.
- ``zero_width_character`` and ``bidi_control``: characters that hide or reorder text.
- ``html_comment`` and ``hidden_style``: text a reader never sees (comments, ``display:none``,
  ``font-size:0``).
- ``exfiltration_beacon``: a Markdown image or link whose URL carries a template placeholder,
  the way injected text leaks data to a server.
- ``encoded_instruction``: a long base64 or hex run that decodes to one of the phrases above.
- ``mixed_script_word``: a word mixing Latin with Cyrillic or Greek letters, a homoglyph
  candidate.

A finding is a **hypothesis**: ordinary documents can say "you are now logged in", and an
attack can avoid every pattern. It marks a document for a person to read, never proves one
safe or unsafe. The report holds document IDs, detector IDs, character offsets and counts,
never document text; ``--show-excerpts`` prints a short escaped excerpt to the terminal only.
A patterns file adds detectors of your own (regular expressions run locally on your own
corpus; avoid patterns with nested repetition).
"""

from __future__ import annotations

import base64
import binascii
import itertools
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCAN_FORMAT = "vectraudit.corpus_scan"
SCAN_FORMAT_VERSION = 1
SEVERITIES = ("low", "medium", "high")
MAX_MATCHES_PER_DETECTOR = 5
MAX_FINDINGS_PER_DOCUMENT = 20
MAX_PATTERNS = 50
MAX_PATTERN_LENGTH = 300
EXCERPT_CHARS = 120
MAX_LINE_BYTES = 8 * 2**20


class CorpusScanError(ValueError):
    """The corpus or patterns file cannot be used; the message says why."""


@dataclass(frozen=True)
class Detector:
    id: str
    version: int
    severity: str
    explanation: str


_OVERRIDE = (
    r"\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any|the|your|every|of)\s+)*"
    r"(?:previous|prior|above|earlier|preceding|former)\s+"
    r"(?:instructions?|prompts?|context|messages?|rules?|directions?)\b"
)
_ROLE = r"\byou\s+are\s+now\b"
_PROMPT_REF = r"\b(?:system\s+prompt|developer\s+message|developer\s+mode)\b"
_CHAT = (
    r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>"
    r"|\[/?INST\]|<</?SYS>>|^\#{1,6}\s*system\s*:|^\s*assistant\s*:"
)
_ZERO_WIDTH = "[\u200b-\u200f\u2060\ufeff]"
_BIDI = "[\u202a-\u202e\u2066-\u2069]"
_COMMENT = r"<!--.*?-->"
_HIDDEN = r"display\s*:\s*none\b|font-size\s*:\s*0+(?:\.0+)?(?:px|pt|em|rem|%)?(?![\d.])"
_BEACON = (
    r"!?\[[^\]\n]{0,200}\]\(\s*[^)\s]*\?[^)\s]*"
    r"(?:\{\{[^}\s]{1,60}\}\}|\$\{[^}\s]{1,60}\}|\{[A-Za-z_][\w.-]{0,40}\}|<[A-Za-z_][\w.-]{0,40}>"
    r"|%7B%7B[^)\s]{1,60}%7D%7D)[^)\s]*\s*\)"
)
_FLAGS = re.IGNORECASE | re.MULTILINE
_BASE64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}){20,}")
_WORD = re.compile(r"\w+")

DETECTORS: dict[str, Detector] = {
    d.id: d
    for d in (
        Detector(
            "instruction_override",
            1,
            "high",
            "Text that tells a model to ignore or override its earlier instructions.",
        ),
        Detector(
            "role_reassignment",
            1,
            "low",
            'Text such as "you are now ...", which can reassign the model\'s role. Ordinary '
            'prose says it too ("you are now logged in"), so it is a weak signal on its own.',
        ),
        Detector(
            "prompt_reference",
            1,
            "medium",
            "Text that refers to the system prompt or a developer message, which documents "
            "rarely do.",
        ),
        Detector(
            "chat_template_token",
            1,
            "high",
            "A chat-template or role marker (such as <|im_start|>, [INST], <<SYS>> or a line "
            "starting 'assistant:') that can make retrieved text read as a different speaker.",
        ),
        Detector(
            "zero_width_character",
            1,
            "medium",
            "A zero-width character, which can hide text from a reader or split a phrase "
            "past a filter.",
        ),
        Detector(
            "bidi_control",
            1,
            "high",
            "A bidirectional control character, which can reorder text so it reads differently "
            "from how a program sees it.",
        ),
        Detector(
            "html_comment",
            1,
            "medium",
            "An HTML comment: text a reader of the rendered page never sees.",
        ),
        Detector(
            "hidden_style",
            1,
            "high",
            "A style that hides text (display:none or font-size:0).",
        ),
        Detector(
            "exfiltration_beacon",
            1,
            "high",
            "A Markdown image or link whose URL has a template placeholder in its query string, "
            "the pattern injected text uses to send data to an outside server.",
        ),
        Detector(
            "encoded_instruction",
            1,
            "high",
            "A long base64 or hex run that decodes to an instruction or role marker.",
        ),
        Detector(
            "mixed_script_word",
            1,
            "low",
            "A word mixing Latin letters with Cyrillic or Greek ones, which can be a homoglyph "
            "used to slip past a filter.",
        ),
    )
}
_PATTERNS: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(_OVERRIDE, _FLAGS),
    "role_reassignment": re.compile(_ROLE, _FLAGS),
    "prompt_reference": re.compile(_PROMPT_REF, _FLAGS),
    "chat_template_token": re.compile(_CHAT, _FLAGS),
    "html_comment": re.compile(_COMMENT, re.DOTALL),
    "hidden_style": re.compile(_HIDDEN, re.IGNORECASE),
    "exfiltration_beacon": re.compile(_BEACON, re.IGNORECASE),
}
_NON_ASCII_PATTERNS = {
    "zero_width_character": re.compile(_ZERO_WIDTH),
    "bidi_control": re.compile(_BIDI),
}
_DECODED = [_PATTERNS[k] for k in ("instruction_override", "role_reassignment", "prompt_reference")]
_DECODED.append(_PATTERNS["chat_template_token"])
_SCRIPTS = ("LATIN", "CYRILLIC", "GREEK")
_UNIT_SYMBOLS = frozenset("\u03a9\u03bc\u03c0\u0394\u03a3\u03b8\u03bb")
"""Greek letters that are units or maths (Ohm, mu, pi, Delta, Sigma, theta, lambda): ordinary in
technical text and not look-alikes of Latin letters, so a word like ``10k`` + Ohm is not a
homoglyph candidate."""
_MIN_HOMOGLYPH_LETTERS = 3


@dataclass(frozen=True)
class CustomPattern:
    """A detector supplied by a patterns file."""

    detector: Detector
    regex: re.Pattern[str]


def load_patterns(path: str | Path) -> list[CustomPattern]:
    """Detectors from a JSON file: ``[{"id", "pattern", "explanation", "severity"?}]``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise CorpusScanError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise CorpusScanError(f"{path} is not JSON.") from exc
    if not isinstance(data, list) or not data or len(data) > MAX_PATTERNS:
        raise CorpusScanError(f"{path} must be a list of 1-{MAX_PATTERNS} pattern objects.")
    out: list[CustomPattern] = []
    seen: set[str] = set()
    for number, item in enumerate(data, 1):
        where = f"{path} pattern {number}"
        if not isinstance(item, dict):
            raise CorpusScanError(f"{where}: expected an object")
        name, pattern = item.get("id"), item.get("pattern")
        explanation, severity = item.get("explanation"), item.get("severity", "medium")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_.-]{1,40}", name):
            raise CorpusScanError(f"{where}: id must be 1-40 characters of a-z, 0-9, _ . -")
        if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 300:
            raise CorpusScanError(f"{where}: explanation must say, in plain words, what it finds")
        if severity not in SEVERITIES:
            raise CorpusScanError(f"{where}: severity must be one of {', '.join(SEVERITIES)}")
        if not isinstance(pattern, str) or not 1 <= len(pattern) <= MAX_PATTERN_LENGTH:
            raise CorpusScanError(f"{where}: pattern must be 1-{MAX_PATTERN_LENGTH} characters")
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            raise CorpusScanError(f"{where}: not a valid regular expression ({exc})") from exc
        if regex.search(""):
            raise CorpusScanError(f"{where}: pattern must not match the empty string")
        key = f"custom:{name}"
        if key in seen:
            raise CorpusScanError(f"{where}: id {name!r} repeats")
        seen.add(key)
        out.append(CustomPattern(Detector(key, 1, severity, explanation.strip()), regex))
    return out


def read_corpus(path: str | Path) -> Iterator[tuple[str, str]]:
    """``(doc_id, text)`` per line of a JSONL corpus, streamed; ``id`` works for ``doc_id``.

    A repeated ID is allowed (chunks of one source document share its ``doc_id``): this scans
    text, it does not load a corpus, and the report records each flagged row's position.
    """
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if len(line) > MAX_LINE_BYTES:
                    raise CorpusScanError(f"{path} line {number} is larger than 8 MiB.")
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise CorpusScanError(f"{path} line {number}: not JSON") from exc
                doc = row.get("doc_id", row.get("id")) if isinstance(row, dict) else None
                text = row.get("text") if isinstance(row, dict) else None
                if isinstance(doc, int) and not isinstance(doc, bool):
                    doc = str(doc)
                if not isinstance(doc, str) or not doc or not isinstance(text, str):
                    raise CorpusScanError(f"{path} line {number}: needs a doc_id (or id) and text")
                yield doc, text
    except OSError as exc:
        raise CorpusScanError(f"Cannot read {path}: {exc.strerror or exc}") from exc


def _decoded_instruction(text: str) -> Iterator[tuple[int, int]]:
    """Spans of base64 or hex runs that decode to an instruction or role marker."""
    for pattern, decode in ((_BASE64, _from_base64), (_HEX, _from_hex)):
        for match in pattern.finditer(text):
            decoded = decode(match.group())
            if decoded and any(p.search(decoded) for p in _DECODED):
                yield match.start(), match.end()


def _from_base64(run: str) -> str | None:
    try:
        raw = base64.b64decode(run + "=" * (-len(run) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    return _printable(raw)


def _from_hex(run: str) -> str | None:
    try:
        return _printable(bytes.fromhex(run))
    except ValueError:
        return None


def _printable(raw: bytes) -> str | None:
    """The text if ``raw`` is mostly printable ASCII; otherwise it is not an encoded phrase."""
    if not raw or sum(32 <= b < 127 or b in (9, 10, 13) for b in raw) < 0.9 * len(raw):
        return None
    return raw.decode("ascii", errors="replace")


def _mixed_script(text: str) -> Iterator[tuple[int, int]]:
    for word in _WORD.finditer(text):
        token = word.group()
        if token.isascii() or sum(c.isalpha() for c in token) < _MIN_HOMOGLYPH_LETTERS:
            continue
        scripts: set[str] = set()
        for char in token:
            if char.isalpha() and char not in _UNIT_SYMBOLS:
                name = unicodedata.name(char, "")
                scripts.update(s for s in _SCRIPTS if name.startswith(s))
        if "LATIN" in scripts and len(scripts) > 1:
            yield word.start(), word.end()


def _first(regex: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    """At most ``MAX_MATCHES_PER_DETECTOR`` matches, so one hostile document stays cheap."""
    return itertools.islice(regex.finditer(text), MAX_MATCHES_PER_DETECTOR)


def scan_text(text: str, custom: Sequence[CustomPattern] = ()) -> list[tuple[str, int, int]]:
    """``(detector id, start, end)`` of the matches in one document, by position."""
    found: list[tuple[str, int, int]] = []
    for name, regex in _PATTERNS.items():
        found += [(name, m.start(), m.end()) for m in _first(regex, text)]
    if not text.isascii():
        for name, regex in _NON_ASCII_PATTERNS.items():
            found += [(name, m.start(), m.end()) for m in _first(regex, text)]
        mixed = itertools.islice(_mixed_script(text), MAX_MATCHES_PER_DETECTOR)
        found += [("mixed_script_word", s, e) for s, e in mixed]
    if len(text) >= 40:
        encoded = itertools.islice(_decoded_instruction(text), MAX_MATCHES_PER_DETECTOR)
        found += [("encoded_instruction", s, e) for s, e in encoded]
    for pattern in custom:
        found += [(pattern.detector.id, m.start(), m.end()) for m in _first(pattern.regex, text)]
    return sorted(found, key=lambda f: (f[1], f[2], f[0]))


def safe_excerpt(text: str, start: int, end: int, limit: int = EXCERPT_CHARS) -> str:
    """A short excerpt safe to print: control and formatting characters are shown escaped.

    Documents are untrusted, and a terminal acts on escape sequences, so nothing is printed
    raw. At most ``limit`` characters of the original are shown, around the match.
    """
    span = end - start
    pad = max(0, (limit - span) // 2)
    lo, hi = max(0, start - pad), min(len(text), max(end, start + limit) + pad)
    shown = text[lo : min(hi, lo + limit)]
    return "".join(c if c.isprintable() else f"\\u{ord(c):04x}" for c in shown)


def scan_corpus(
    documents: Iterable[tuple[str, str]],
    custom: Sequence[CustomPattern] = (),
    *,
    on_excerpt: Callable[[str, str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """The scan report for a corpus; ``on_excerpt`` receives terminal-safe excerpts only.

    ``on_excerpt(doc_id, detector, start, end, excerpt)`` is never stored in the report.
    """
    detectors = {**DETECTORS, **{c.detector.id: c.detector for c in custom}}
    scanned = 0
    flagged: list[dict[str, Any]] = []
    by_detector: Counter[str] = Counter()
    documents_by_detector: Counter[str] = Counter()
    for doc_id, text in documents:
        scanned += 1
        row = scanned
        matches = scan_text(text, custom)
        if not matches:
            continue
        shown = matches[:MAX_FINDINGS_PER_DOCUMENT]
        flagged.append(
            {
                "doc_id": doc_id,
                "row": row,
                "findings": [
                    {
                        "detector": name,
                        "start": start,
                        "end": end,
                        "severity": detectors[name].severity,
                        "evidence": "hypothesis",
                    }
                    for name, start, end in shown
                ],
                **(
                    {"more_findings": len(matches) - len(shown)}
                    if len(matches) > len(shown)
                    else {}
                ),
            }
        )
        by_detector.update(name for name, _, _ in matches)
        documents_by_detector.update({name for name, _, _ in matches})
        if on_excerpt is not None:
            for name, start, end in shown:
                on_excerpt(doc_id, name, start, end, safe_excerpt(text, start, end))
    rank = {s: i for i, s in enumerate(SEVERITIES)}

    def worst(doc: dict[str, Any]) -> int:
        return max(rank[f["severity"]] for f in doc["findings"])

    review = sorted(flagged, key=lambda d: (-worst(d), -len(d["findings"]), d["doc_id"], d["row"]))
    return {
        "format": SCAN_FORMAT,
        "format_version": SCAN_FORMAT_VERSION,
        "evidence": "hypothesis",
        "note": (
            "Findings are hypotheses that mark a document for a person to read. They do not "
            "show a document is safe or unsafe, and no document text is recorded."
        ),
        "summary": {
            "documents": scanned,
            "flagged": len(flagged),
            "findings": sum(by_detector.values()),
            "by_detector": dict(sorted(by_detector.items())),
            "documents_by_detector": dict(sorted(documents_by_detector.items())),
            "by_severity": dict(
                sorted(
                    Counter(
                        detectors[n].severity for n, c in by_detector.items() for _ in range(c)
                    ).items()
                )
            ),
        },
        "detectors": [
            {
                "id": d.id,
                "version": d.version,
                "severity": d.severity,
                "explanation": d.explanation,
            }
            for d in detectors.values()
        ],
        "review": [
            {
                "doc_id": d["doc_id"],
                "row": d["row"],
                "severity": SEVERITIES[worst(d)],
                "findings": len(d["findings"]),
            }
            for d in review
        ],
        "documents": sorted(flagged, key=lambda d: (d["doc_id"], d["row"])),
    }


def render(report: dict[str, Any]) -> str:
    """A short terminal summary of a scan report."""
    s = report["summary"]
    lines = [
        f"Scanned {s['documents']} documents: {s['flagged']} flagged, {s['findings']} findings "
        f"(hypotheses for a person to read, not proof)."
    ]
    explain = {d["id"]: d["explanation"] for d in report["detectors"]}
    for name, n in sorted(s["documents_by_detector"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {n} documents: {name.replace('_', ' ')}. {explain[name]}")
    for item in report["review"][:10]:
        lines.append(
            f"  read first: {item['doc_id']} (row {item['row']}, {item['severity']}, "
            f"{item['findings']} findings)"
        )
    return "\n".join(lines)
