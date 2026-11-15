"""Attach answer judgments from existing tools to a run, and diagnose each question.

Vectraudit does not grade answers. Teams already do, with human review or tools such as
Ragas, DeepEval, Langfuse scores or Phoenix evaluations. This module reads those judgments
from a JSONL file, one judgment per line::

    {"query_id": "q1", "evaluator": "ragas.answer_correctness", "evaluator_version": "0.2",
     "kind": "model", "score": 0.42}
    {"trace_id": "4bf9…", "evaluator": "support-review", "rubric": "v3", "kind": "human",
     "verdict": "incorrect"}

The file may also be a Langfuse scores export or LangSmith feedback export, read through
:mod:`vectraudit.evaluation.judgment_formats` once the caller names which scores judge
answers. A judgment names its question by ``query_id`` or, for runs imported from traces, by
``trace_id``. ``verdict`` is ``correct``, ``partial`` or ``incorrect``; a judgment with only a
``score`` gets a verdict from a threshold the caller declares for that evaluator, never from
a default. ``outcome`` (``answered``, ``abstained`` or ``handoff``) separates an answer from a
generated abstention or an appropriate human handoff. Explanations and answer text are not
read, so nothing private is copied into the run.

A judgment may also carry the answer's ``citations`` (document IDs the answer cited) and
``citation_support`` (``{"doc-id": "supported" | "unsupported"}`` from review or a tool).
Each cited document is placed against what the run recorded: in the model context, absent
from an observed context, delivered with context unknown, or never delivered at all. Support
verdicts follow the same rule as answer verdicts (humans over models, disagreement is
``disputed``). Citations never change the answer verdict or the diagnosis.

A judgment may instead (or also) carry ``claims`` and ``claim_support``: counts of the
reference answer's and the response's claims that the delivered documents entail, from a
claim-level checker such as RAGChecker (see :mod:`vectraudit.evaluation.ragchecker_import`).
They are attached to the question as ``claim_evidence`` and never change a verdict or the
diagnosis; a line with only claims needs no verdict or score. Counts and document IDs only:
no claim text. :func:`claim_findings` reads them beside the diagnosis (``claim_findings`` on
the answer): which delivered documents the answer was or was not built from, and which ones
misled it. Findings are model-judged, so their evidence level is ``supported``, never
``verified``, and they are marked ``disputed`` when human review disagrees with the model.

Human judgments decide a question's verdict when present; otherwise model judgments do.
Disagreement among the deciding judgments makes the verdict ``disputed`` instead of picking
one. Each model evaluator's agreement with human review is reported where both judged.

The diagnosis states the first observable loss on the path from retrieval to answer, with
the facts behind it; it is not a root cause. It needs ``required_evidence`` in the
benchmark: without it, which documents suffice to answer is not known.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.evaluation import judgment_formats
from vectraudit.evaluation.answer_estimates import estimate_rates
from vectraudit.evaluation.review_sample import ReviewSample, ReviewSampleError, read_sample

ANSWERS_FORMAT_VERSION = 1
VERDICTS = ("correct", "partial", "incorrect")
OUTCOMES = ("answered", "abstained", "handoff")
KINDS = ("human", "model")
SAFETY_KIND = "safety"
"""A record of what an answer did (for example echoed an injected canary), kept apart from
answer verdicts: following an injected instruction is a safety failure that never changes
whether the answer was correct."""
MAX_SAFETY_ITEMS = 100
SUPPORT = ("supported", "unsupported")
MAX_JUDGMENTS = 200_000
MAX_CITATIONS = 100
MAX_CLAIM_DOCUMENTS = 500
MAX_CLAIM_COUNT = 100_000
CLAIM_REQUIRED = (
    "reference",
    "response",
    "reference_in_context",
    "reference_in_response",
    "unsupported_incorrect",
    "unsupported_correct",
    "from_irrelevant_incorrect",
    "from_relevant_incorrect",
)
CLAIM_OPTIONAL = (
    "response_incorrect",
    "reference_used",
    "context_documents",
    "useful_documents",
)
SUPPORT_REQUIRED = ("reference", "response", "contradicts_reference")
SUPPORT_OPTIONAL = ("response_incorrect",)
MAX_REFERENCE_CLAIMS = 1_000
_TEXT_LIMIT = 200
DIAGNOSES = {
    "answered_correctly": "Answered correctly.",
    "correct_without_required_evidence": (
        "Answered correctly although no required evidence group reached the model; the model "
        "may have used other knowledge, or the required evidence is incomplete."
    ),
    "answered_wrong_despite_evidence": (
        "Answered wrong although a complete required evidence group reached the model."
    ),
    "lost_in_context": (
        "A complete required evidence group was retrieved and delivered, but did not reach "
        "the model intact."
    ),
    "lost_downstream": (
        "Required evidence appeared in a recorded stage but was dropped before delivery."
    ),
    "not_observed": (
        "Required evidence was not in any recorded stage. Missing instrumentation or an "
        "absent or stale source is still possible."
    ),
    "context_unknown": (
        "Retrieval delivered a complete required group, but whether it reached the model is "
        "unknown."
    ),
    "abstained_despite_evidence": (
        "Abstained although a complete required evidence group reached the model."
    ),
    "abstained": "Abstained; the required evidence did not demonstrably reach the model.",
    "handoff": "Handed off to a person.",
    "disputed": "Judgments disagree; resolve them before diagnosing.",
    "unjudged": "No usable verdict (a score without a declared threshold).",
    "no_required_evidence": "The benchmark declares no required evidence for this question.",
}


class AnswerJudgmentError(ValueError):
    """The judgments file or settings cannot be used; the message says why."""


@dataclass(frozen=True)
class Judgment:
    """One evaluator's judgment of one question's answer."""

    evaluator: str
    kind: str
    verdict: str | None
    score: float | None
    outcome: str
    evaluator_version: str | None
    rubric: str | None
    citations: tuple[str, ...] | None = None
    support: tuple[tuple[str, str], ...] = ()
    claims: tuple[tuple[str, int], ...] | None = None
    claim_support: tuple[tuple[str, dict[str, Any]], ...] = ()
    safety: dict[str, Any] | None = None
    redteam: dict[str, Any] | None = None

    def record(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "kind": self.kind,
            "verdict": self.verdict,
            "score": self.score,
            "outcome": self.outcome,
            **({"evaluator_version": self.evaluator_version} if self.evaluator_version else {}),
            **({"rubric": self.rubric} if self.rubric else {}),
            **({"citations": list(self.citations)} if self.citations is not None else {}),
            **({"citation_support": dict(self.support)} if self.support else {}),
        }


def _short(value: Any, what: str, line: int, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > _TEXT_LIMIT:
        raise AnswerJudgmentError(
            f"line {line}: {what} must be a non-empty string of at most {_TEXT_LIMIT} characters"
        )
    return value


def _citations(
    obj: Mapping[str, Any], line: int
) -> tuple[tuple[str, ...] | None, tuple[tuple[str, str], ...]]:
    raw, support = obj.get("citations"), obj.get("citation_support")
    cited: tuple[str, ...] | None = None
    if raw is not None:
        if not isinstance(raw, list) or len(raw) > MAX_CITATIONS:
            raise AnswerJudgmentError(
                f"line {line}: citations must be a list of at most {MAX_CITATIONS} document IDs"
            )
        cited = tuple(sorted({_short(c, "a citation", line, required=True) or "" for c in raw}))
    pairs: tuple[tuple[str, str], ...] = ()
    if support is not None:
        if not isinstance(support, dict) or not support:
            raise AnswerJudgmentError(f"line {line}: citation_support must be an object")
        if cited is None:
            raise AnswerJudgmentError(f"line {line}: citation_support needs citations")
        for doc, verdict in support.items():
            if doc not in cited:
                raise AnswerJudgmentError(f"line {line}: citation_support names uncited {doc!r}")
            if verdict not in SUPPORT:
                raise AnswerJudgmentError(
                    f"line {line}: citation support must be {' or '.join(SUPPORT)}"
                )
        pairs = tuple(sorted(support.items()))
    return cited, pairs


def _counts(
    value: Any, required: Sequence[str], optional: Sequence[str], what: str, line: int
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, dict):
        raise AnswerJudgmentError(f"line {line}: {what} must be an object of counts")
    unknown = sorted(set(value) - {*required, *optional})
    missing = [k for k in required if k not in value]
    if unknown or missing:
        raise AnswerJudgmentError(
            f"line {line}: {what} "
            + (f"has unknown {', '.join(unknown)}" if unknown else f"lacks {', '.join(missing)}")
        )
    for key, count in value.items():
        if type(count) is not int or not 0 <= count <= MAX_CLAIM_COUNT:
            raise AnswerJudgmentError(
                f"line {line}: {what} {key} must be a whole number from 0 to {MAX_CLAIM_COUNT}"
            )
    return tuple(sorted(value.items()))


def _claim_positions(value: Any, reference: int, doc: str, line: int) -> tuple[int, ...]:
    """Positions of the reference claims a document entails: distinct, below ``reference``."""
    if (
        not isinstance(value, list)
        or len(value) > MAX_REFERENCE_CLAIMS
        or any(type(i) is not int or not 0 <= i < reference for i in value)
        or len(set(value)) != len(value)
    ):
        raise AnswerJudgmentError(
            f"line {line}: claim_support {doc!r} reference_claims must be distinct positions "
            f"below the {reference} reference claims"
        )
    return tuple(sorted(value))


def _claims(
    obj: Mapping[str, Any], line: int
) -> tuple[tuple[tuple[str, int], ...] | None, tuple[tuple[str, dict[str, Any]], ...]]:
    """Claim counts and per-document support; counts must be consistent with each other."""
    raw, support = obj.get("claims"), obj.get("claim_support")
    if raw is None:
        if support is not None:
            raise AnswerJudgmentError(f"line {line}: claim_support needs claims")
        return None, ()
    claims = _counts(raw, CLAIM_REQUIRED, CLAIM_OPTIONAL, "claims", line)
    c = dict(claims)
    incorrect = c.get("response_incorrect", c["response"])
    problems = [
        (c["reference_in_context"] > c["reference"], "reference_in_context exceeds reference"),
        (c["reference_in_response"] > c["reference"], "reference_in_response exceeds reference"),
        (c.get("reference_used", 0) > c["reference"], "reference_used exceeds reference"),
        (incorrect > c["response"], "response_incorrect exceeds response"),
        (
            c["unsupported_incorrect"] + c["unsupported_correct"] > c["response"],
            "unsupported claims exceed response",
        ),
        (
            c["unsupported_incorrect"]
            + c["from_relevant_incorrect"]
            + c["from_irrelevant_incorrect"]
            > incorrect,
            "incorrect claims are counted more than once",
        ),
        (
            c.get("useful_documents", 0) > c.get("context_documents", MAX_CLAIM_COUNT),
            "useful_documents exceeds context_documents",
        ),
    ]
    for bad, message in problems:
        if bad:
            raise AnswerJudgmentError(f"line {line}: claims inconsistent: {message}")
    pairs: tuple[tuple[str, dict[str, Any]], ...] = ()
    if support is not None:
        if not isinstance(support, dict) or len(support) > MAX_CLAIM_DOCUMENTS:
            raise AnswerJudgmentError(
                f"line {line}: claim_support must map at most {MAX_CLAIM_DOCUMENTS} document "
                "IDs to counts"
            )
        rows: list[tuple[str, dict[str, Any]]] = []
        for doc, counts in support.items():
            _short(doc, "a claim_support document ID", line, required=True)
            positions: tuple[int, ...] | None = None
            if isinstance(counts, dict) and "reference_claims" in counts:
                positions = _claim_positions(counts["reference_claims"], c["reference"], doc, line)
                counts = {k: v for k, v in counts.items() if k != "reference_claims"}
            d = dict(
                _counts(counts, SUPPORT_REQUIRED, SUPPORT_OPTIONAL, f"claim_support {doc!r}", line)
            )
            if positions is not None and len(positions) != d["reference"]:
                raise AnswerJudgmentError(
                    f"line {line}: claim_support {doc!r} lists {len(positions)} reference "
                    f"claims but counts {d['reference']}"
                )
            if (
                d["reference"] > c["reference"]
                or d["contradicts_reference"] > c["reference"]
                or d["response"] > c["response"]
                or d.get("response_incorrect", 0) > d["response"]
            ):
                raise AnswerJudgmentError(
                    f"line {line}: claim_support {doc!r} counts more claims than were extracted"
                )
            rows.append((doc, d if positions is None else {**d, "reference_claims": positions}))
        pairs = tuple(sorted(rows, key=lambda r: r[0]))
    return claims, pairs


def _safety(
    obj: Mapping[str, Any], kind: str, line: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The ``safety`` or ``redteam`` object of a safety record; ``(None, None)`` otherwise.

    A safety record has kind ``safety`` and exactly one of the two objects; no other line may
    have either. It carries no verdict or score: it is kept apart from answer judgments.
    """
    safety, redteam = obj.get("safety"), obj.get("redteam")
    if kind != SAFETY_KIND:
        if safety is not None or redteam is not None:
            raise AnswerJudgmentError(
                f'line {line}: a safety record has kind "safety" and a safety or redteam '
                "object, and nothing else has either"
            )
        return None, None
    if (safety is None) == (redteam is None):
        raise AnswerJudgmentError(
            f'line {line}: a safety record has kind "safety" and exactly one of a safety '
            "object (canary check) or a redteam object (red-team result)"
        )
    if obj.get("verdict") is not None or obj.get("score") is not None:
        raise AnswerJudgmentError(
            f"line {line}: a safety record carries no verdict or score; it is kept apart from "
            "answer judgments"
        )
    if redteam is not None:
        return None, _redteam(redteam, line)
    if not isinstance(safety, dict) or set(safety) != {
        "followed_injection",
        "attacks",
        "documents",
    }:
        raise AnswerJudgmentError(
            f"line {line}: safety needs followed_injection, attacks and documents"
        )
    followed, attacks, documents = (
        safety["followed_injection"],
        safety["attacks"],
        safety["documents"],
    )
    if not isinstance(followed, bool):
        raise AnswerJudgmentError(f"line {line}: followed_injection must be true or false")
    for name, items in (("attacks", attacks), ("documents", documents)):
        if (
            not isinstance(items, list)
            or len(items) > MAX_SAFETY_ITEMS
            or any(not isinstance(i, str) or not i.strip() or len(i) > _TEXT_LIMIT for i in items)
            or len(set(items)) != len(items)
        ):
            raise AnswerJudgmentError(
                f"line {line}: safety {name} must be at most {MAX_SAFETY_ITEMS} distinct names"
            )
    if followed != bool(attacks) or (not followed and documents):
        raise AnswerJudgmentError(
            f"line {line}: safety is inconsistent: an injection was followed exactly when "
            "attacks are named"
        )
    return (
        {
            "followed_injection": followed,
            "attacks": sorted(attacks),
            "documents": sorted(documents),
        },
        None,
    )


def _redteam(raw: Any, line: int) -> dict[str, Any]:
    """One red-team test result: the plugin, the strategy if any, and whether it was resisted."""
    if (
        not isinstance(raw, dict)
        or set(raw) - {"plugin", "strategy", "passed"}
        or not ({"plugin", "passed"} <= set(raw))
    ):
        raise AnswerJudgmentError(f"line {line}: redteam needs plugin and passed (and a strategy)")
    plugin, strategy, passed = raw["plugin"], raw.get("strategy"), raw["passed"]
    if not isinstance(plugin, str) or not re.fullmatch(r"[a-z0-9:_.-]{1,80}", plugin):
        raise AnswerJudgmentError(f"line {line}: redteam plugin must be a plugin ID")
    if strategy is not None and (
        not isinstance(strategy, str) or not re.fullmatch(r"[a-z0-9:_.-]{1,80}", strategy)
    ):
        raise AnswerJudgmentError(f"line {line}: redteam strategy must be a strategy ID")
    if not isinstance(passed, bool):
        raise AnswerJudgmentError(f"line {line}: redteam passed must be true or false")
    return {"plugin": plugin, "strategy": strategy, "passed": passed}


def _records(text: str) -> tuple[list[tuple[int, Any]], bool]:
    """``(position, object)`` pairs and whether the text was JSON lines.

    Reads a JSON array, an API page (``{"data": [...]}``) or JSON lines. A line that is not
    JSON is kept as its raw text, so the line-by-line parser reports it with its line number.
    """
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        try:
            whole = json.loads(stripped)
        except ValueError:
            whole = None  # JSON lines, or broken JSON reported line by line
        if isinstance(whole, dict) and isinstance(whole.get("data"), list):
            whole = whole["data"]
        if isinstance(whole, list):
            return list(enumerate(whole, 1)), False
    out: list[tuple[int, Any]] = []
    for line, raw in enumerate(text.splitlines(), 1):
        if raw.strip():
            try:
                out.append((line, json.loads(raw)))
            except ValueError:
                out.append((line, _BadLine(raw)))
    return out, True


@dataclass(frozen=True)
class _BadLine:
    raw: str


@dataclass(frozen=True)
class JudgmentFile:
    """The judgments read from a file, its format, and what was skipped and why."""

    rows: list[tuple[str, str, Judgment]]
    source_format: str
    skipped: dict[str, dict[str, int]] = field(default_factory=dict)


def read_judgment_file(
    path: str | Path,
    thresholds: Mapping[str, float] | None = None,
    *,
    evaluators: Mapping[str, str | None] | None = None,
    labels: Mapping[str, str] | None = None,
) -> JudgmentFile:
    """Judgments from a judgments file, a Langfuse scores export or LangSmith feedback.

    ``evaluators`` names the vendor scores that judge answers, each with the kind (``human``
    or ``model``) to use where the export does not say who judged, or ``None``. ``labels``
    maps categorical values to verdicts. Both apply to vendor exports only. Vendor records
    become judgment records and go through the same validation as a judgments file.
    """
    for name, kind in (evaluators or {}).items():
        if kind is not None and kind not in KINDS:
            raise AnswerJudgmentError(f"evaluator {name}: kind must be human or model")
    for label, verdict in (labels or {}).items():
        if verdict not in VERDICTS:
            raise AnswerJudgmentError(
                f"label {label!r}: verdict must be one of {', '.join(VERDICTS)}"
            )
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise AnswerJudgmentError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    records, lines = _records(text)
    shapes = {
        judgment_formats.shape_of(obj) or "vectraudit"
        for _, obj in records
        if not isinstance(obj, _BadLine)
    }
    if len(shapes) > 1:
        raise AnswerJudgmentError(
            f"{path} mixes formats ({', '.join(sorted(shapes))}); use one per file"
        )
    shape = shapes.pop() if shapes else "vectraudit"
    if shape == "vectraudit":
        if evaluators or labels:
            raise AnswerJudgmentError(
                "--evaluator and --label apply to Langfuse and LangSmith exports; a judgments "
                "file names its evaluators and verdicts itself"
            )
        if not lines:  # a JSON array of judgment records: one per line, in order
            text = "\n".join(json.dumps(obj) for _, obj in records)
        return JudgmentFile(_parse_judgments(text, thresholds, path), shape)
    if len(records) > MAX_JUDGMENTS:
        raise AnswerJudgmentError(f"more than {MAX_JUDGMENTS} judgments; split the file")
    if not evaluators:
        names = Counter(
            judgment_formats.evaluator_of(obj, shape) or "(unnamed)"
            for _, obj in records
            if isinstance(obj, Mapping)
        )
        listed = ", ".join(f"{n} ({c})" for n, c in sorted(names.items())[:30])
        raise AnswerJudgmentError(
            f"{path} is a {shape} export with scores named {listed}. Pass --evaluator NAME "
            "for each score that judges answers (NAME=human or NAME=model where the export "
            "does not say who judged); nothing is read by default."
        )
    selection = judgment_formats.Selection(evaluators, labels or {})
    skipped: dict[str, Counter[str]] = defaultdict(Counter)
    # Converted records keep their position (line or record number) for error messages;
    # skipped ones leave a blank line, which the parser ignores.
    converted_lines = [""] * (max((n for n, _ in records), default=0))
    for n, obj in records:
        if isinstance(obj, _BadLine):
            converted_lines[n - 1] = obj.raw
            continue
        converted = judgment_formats.convert(obj, shape, selection)
        if isinstance(converted, str):
            skipped[converted][judgment_formats.evaluator_of(obj, shape) or "(unnamed)"] += 1
        else:
            converted_lines[n - 1] = json.dumps(converted)
    summary = {reason: dict(sorted(names.items())) for reason, names in sorted(skipped.items())}
    if not any(converted_lines):
        why = "; ".join(
            f"{judgment_formats.SKIPPED[r]}: {sum(n.values())}" for r, n in summary.items()
        )
        raise AnswerJudgmentError(f"{path} contains no judgments" + (f" ({why})" if why else ""))
    rows = _parse_judgments("\n".join(converted_lines), thresholds, path)
    return JudgmentFile(rows=rows, source_format=shape, skipped=summary)


def read_judgments(
    path: str | Path, thresholds: Mapping[str, float] | None = None
) -> list[tuple[str, str, Judgment]]:
    """``(key kind, key, judgment)`` rows; key kind is ``query_id`` or ``trace_id``."""
    return read_judgment_file(path, thresholds).rows


def _parse_judgments(
    text: str, thresholds: Mapping[str, float] | None, path: str | Path
) -> list[tuple[str, str, Judgment]]:
    """Validate judgment records given as JSON lines."""
    thresholds = dict(thresholds or {})
    for name, value in thresholds.items():
        if not math.isfinite(value):
            raise AnswerJudgmentError(f"threshold for {name} must be finite")
    rows = []
    for line, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as exc:
            raise AnswerJudgmentError(f"line {line}: not JSON") from exc
        if not isinstance(obj, dict):
            raise AnswerJudgmentError(f"line {line}: expected a JSON object")
        keys = [k for k in ("query_id", "trace_id") if obj.get(k) is not None]
        if len(keys) != 1:
            raise AnswerJudgmentError(f"line {line}: give exactly one of query_id or trace_id")
        key = _short(obj[keys[0]], keys[0], line, required=True)
        evaluator = _short(obj.get("evaluator"), "evaluator", line, required=True)
        assert key is not None
        assert evaluator is not None
        kind = obj.get("kind", "model")
        if kind not in (*KINDS, SAFETY_KIND):
            raise AnswerJudgmentError(f"line {line}: kind must be human, model or safety")
        safety, redteam = _safety(obj, kind, line)
        outcome = obj.get("outcome", "answered")
        if outcome not in OUTCOMES:
            raise AnswerJudgmentError(f"line {line}: outcome must be one of {', '.join(OUTCOMES)}")
        score = obj.get("score")
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
            raise AnswerJudgmentError(f"line {line}: score must be a finite number")
        verdict = obj.get("verdict")
        if verdict is not None and verdict not in VERDICTS:
            raise AnswerJudgmentError(f"line {line}: verdict must be one of {', '.join(VERDICTS)}")
        if verdict is None and score is not None and evaluator in thresholds:
            verdict = "correct" if score >= thresholds[evaluator] else "incorrect"
        claims, claim_support = _claims(obj, line)
        if (
            verdict is None
            and score is None
            and outcome == "answered"
            and claims is None
            and safety is None
            and redteam is None
        ):
            raise AnswerJudgmentError(
                f"line {line}: a judgment needs a verdict, a score, claims or a safety or "
                "red-team record"
            )
        cited, support = _citations(obj, line)
        rows.append(
            (
                keys[0],
                key,
                Judgment(
                    evaluator=evaluator,
                    kind=kind,
                    verdict=verdict,
                    score=None if score is None else float(score),
                    outcome=outcome,
                    evaluator_version=_short(
                        obj.get("evaluator_version"), "evaluator_version", line
                    ),
                    rubric=_short(obj.get("rubric"), "rubric", line),
                    citations=cited,
                    support=support,
                    claims=claims,
                    claim_support=claim_support,
                    safety=safety,
                    redteam=redteam,
                ),
            )
        )
        if len(rows) > MAX_JUDGMENTS:
            raise AnswerJudgmentError(f"more than {MAX_JUDGMENTS} judgments; split the file")
    if not rows:
        raise AnswerJudgmentError(f"{path} contains no judgments")
    return rows


def _decide(judgments: Sequence[Judgment]) -> tuple[str | None, str | None, str]:
    """``(verdict, decided_by, outcome)``. Humans decide when present; ties are disputes."""
    outcomes = {j.outcome for j in judgments}
    outcome = (
        "handoff"
        if "handoff" in outcomes
        else "abstained"
        if "abstained" in outcomes
        else ("answered")
    )
    for kind in KINDS:
        verdicts = {j.verdict for j in judgments if j.kind == kind and j.verdict is not None}
        if len(verdicts) > 1:
            return "disputed", kind, outcome
        if verdicts:
            return verdicts.pop(), kind, outcome
    return None, None, outcome


def diagnose(query: Mapping[str, Any], verdict: str | None, outcome: str) -> dict[str, Any]:
    """The first observable loss for one question, with the facts that support it."""
    facts: list[str] = []
    sufficiency = query.get("sufficiency")
    evidence = query.get("evidence") or {}
    context = evidence.get("context") if isinstance(evidence, Mapping) else None
    if verdict is not None:
        facts.append(f"answer judged {verdict}")
    if outcome != "answered":
        facts.append(f"outcome: {outcome}")
    if isinstance(context, Mapping):
        for loss in context.get("losses") or []:
            facts.append(f"{loss['doc_id']} {loss['kind']} in the model context")
    if outcome == "handoff":
        return {"class": "handoff", "facts": facts}
    if verdict == "disputed":
        return {"class": "disputed", "facts": facts}
    if verdict is None and outcome == "answered":
        return {"class": "unjudged", "facts": facts}
    if not isinstance(sufficiency, Mapping):
        return {"class": "no_required_evidence", "facts": facts}
    delivered, in_context = sufficiency["delivered"], sufficiency["context"]
    facts.append(f"required evidence delivered: {delivered}; in model context: {in_context}")
    if outcome == "abstained":
        name = "abstained_despite_evidence" if in_context == "sufficient" else "abstained"
        return {"class": name, "facts": facts}
    if verdict == "correct":
        name = (
            "correct_without_required_evidence"
            if in_context == "insufficient"
            else ("answered_correctly")
        )
        return {"class": name, "facts": facts}
    if in_context == "sufficient":
        return {"class": "answered_wrong_despite_evidence", "facts": facts}
    if delivered == "sufficient":
        name = "lost_in_context" if in_context == "insufficient" else "context_unknown"
        return {"class": name, "facts": facts}
    missing = set(sufficiency.get("missing_from_delivered") or [])
    located = [
        f
        for f in (evidence.get("failures") if isinstance(evidence, Mapping) else None) or []
        if f.get("doc_id") in missing and f.get("stage")
    ]
    if located:
        facts += [f"{f['doc_id']} dropped at {f['stage']}" for f in located]
        return {"class": "lost_downstream", "facts": facts}
    facts += [f"{doc} not in any recorded stage" for doc in sorted(missing)]
    return {"class": "not_observed", "facts": facts}


CLAIM_FINDINGS = {
    "reference_not_in_context": (
        "Reference claims no delivered document entails: a retrieval gap, unless the label or "
        "the required evidence is incomplete or stale."
    ),
    "evidence_unused": (
        "Claims the delivered documents support were left out of the response: generation or "
        "context packing."
    ),
    "misled_by_irrelevant_context": (
        "Incorrect response claims entailed only by documents that support no reference "
        "claim: noise, hubs, poisoned or stale documents."
    ),
    "misled_by_relevant_context": (
        "Incorrect response claims entailed by documents that do support reference claims: "
        "conflicting content inside relevant documents."
    ),
    "unsupported_incorrect": "Incorrect response claims no delivered document supports: invented.",
    "unsupported_correct": (
        "Correct response claims no delivered document supports: the model answered from its "
        "own knowledge."
    ),
    "contradicting_document": (
        "A delivered document contradicts the reference answer, which often means it is stale: "
        "check it with `vectraudit workbench freshness`."
    ),
}
MAX_FINDING_DOCUMENTS = 20


def _documents(
    support: Mapping[str, Mapping[str, int]], key: str, keep: Any
) -> tuple[list[str], int]:
    """Documents whose ``key`` count is positive and that ``keep`` accepts, most first."""
    found = sorted(
        (doc for doc, s in support.items() if s.get(key, 0) > 0 and keep(s)),
        key=lambda doc: (-support[doc][key], doc),
    )
    return found[:MAX_FINDING_DOCUMENTS], max(0, len(found) - MAX_FINDING_DOCUMENTS)


def claim_findings(
    query: Mapping[str, Any], evidence: Mapping[str, Any], *, disputed: bool
) -> list[dict[str, Any]]:
    """What the claim-level evidence says about one question, beside (never in) its diagnosis.

    Only documents the run delivered or put in the model context are named; any other in
    ``claim_support`` is listed as unmatched on the evidence. Where the counts per document
    lack ``response_incorrect`` the finding keeps its count but names no documents.
    """
    claims = evidence["claims"]
    run_docs = set(query.get("ids") or [])
    context = query.get("model_context")
    if isinstance(context, Mapping):
        run_docs |= {i.get("id") for i in context.get("items") or [] if isinstance(i, Mapping)}
    support = {d: s for d, s in evidence["claim_support"].items() if d in run_docs}
    findings: list[dict[str, Any]] = []

    def add(
        name: str, count: int, docs: tuple[list[str], int] | None = None, note: str = ""
    ) -> None:
        if count <= 0:
            return
        finding: dict[str, Any] = {
            "finding": name,
            "count": count,
            "points_to": CLAIM_FINDINGS[name],
            "evidence": "supported",
            "disputed": disputed,
        }
        if docs is not None:
            finding["documents"] = docs[0]
            if docs[1]:
                finding["more_documents"] = docs[1]
        if note:
            finding["note"] = note
        findings.append(finding)

    missing = claims["reference"] - claims["reference_in_context"]
    sufficiency = query.get("sufficiency")
    note = ""
    if isinstance(sufficiency, Mapping) and sufficiency.get("delivered") == "sufficient":
        note = (
            "A complete required evidence group was delivered, so the label or the required "
            "evidence may be incomplete or stale."
        )
    add("reference_not_in_context", missing, None, note)
    used = claims.get("reference_used")
    if used is not None:  # absent from older exports: unknown, not zero
        add("evidence_unused", claims["reference_in_context"] - used)
    named = any("response_incorrect" in s for s in support.values())
    irrelevant = _documents(support, "response_incorrect", lambda s: s["reference"] == 0)
    relevant = _documents(support, "response_incorrect", lambda s: s["reference"] > 0)
    add(
        "misled_by_irrelevant_context",
        claims["from_irrelevant_incorrect"],
        irrelevant if named else None,
    )
    add(
        "misled_by_relevant_context",
        claims["from_relevant_incorrect"],
        relevant if named else None,
    )
    add("unsupported_incorrect", claims["unsupported_incorrect"])
    add("unsupported_correct", claims["unsupported_correct"])
    contradicting = _documents(support, "contradicts_reference", lambda s: True)
    add("contradicting_document", len(contradicting[0]) + contradicting[1], contradicting)
    return findings


CITATION_CLASSES = {
    "in_context": "in the model context",
    "not_in_context": "delivered, but absent from the observed model context",
    "delivered_context_unknown": "delivered; whether it reached the model is unknown",
    "not_delivered": "never delivered by retrieval (memory, another source, or an invalid ID)",
    "not_in_corpus": (
        "absent from the corpus inventory: a fabricated citation, found by lookup, not judged"
    ),
}


def analyze_citations(
    query: Mapping[str, Any],
    judgments: Sequence[Judgment],
    inventory: AbstractSet[str] | None = None,
) -> dict[str, Any] | None:
    """Place each cited document against the run's recorded evidence; ``None`` if none given.

    With a corpus ``inventory``, a cited ID the inventory does not hold is ``not_in_corpus``:
    a fabricated citation, found deterministically (no model), whatever retrieval delivered.
    """
    reported = {j.citations for j in judgments if j.citations is not None}
    if not reported:
        return None
    if len(reported) > 1:
        return {"status": "conflicting", "cited": sorted({c for r in reported for c in r or ()})}
    cited = next(iter(reported)) or ()
    delivered = set(query.get("ids") or [])
    context = query.get("model_context")
    observed = isinstance(context, Mapping) and context.get("coverage") == "observed"
    in_context = {i.get("id") for i in (context or {}).get("items", [])} if context else set()
    classes = {}
    for doc in cited:
        if inventory is not None and doc not in inventory:
            classes[doc] = "not_in_corpus"
        elif observed:
            classes[doc] = (
                "in_context"
                if doc in in_context
                else ("not_in_context" if doc in delivered else "not_delivered")
            )
        else:
            classes[doc] = "delivered_context_unknown" if doc in delivered else "not_delivered"
    support: dict[str, dict[str, Any]] = {}
    for doc in cited:
        votes = [(j.kind, v) for j in judgments for d, v in j.support if d == doc]
        for kind in KINDS:
            verdicts = {v for k, v in votes if k == kind}
            if len(verdicts) > 1:
                support[doc] = {"verdict": "disputed", "decided_by": kind}
                break
            if verdicts:
                support[doc] = {"verdict": verdicts.pop(), "decided_by": kind}
                break
    return {"status": "consistent", "cited": list(cited), "classes": classes, "support": support}


def _meta_evaluation(answers: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """How often human reviewers judged answers incorrect, for each claim finding.

    Over questions that have claim evidence: those with a given finding, those with none, and
    all of them. Only questions a human decided count as judged (a ``disputed`` verdict does
    not). It shows whether a finding goes with answers people find wrong for this extractor
    and checker, before anyone relies on it.
    """
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for answer in answers.values():
        if "claim_findings" not in answer:
            continue
        human = answer["decided_by"] == "human" and answer["verdict"] in VERDICTS
        names = {f["finding"] for f in answer["claim_findings"]} or {"no_findings"}
        for name in names | {"all_questions"}:
            groups[name]["questions"] += 1
            if human:
                groups[name]["human_judged"] += 1
                groups[name][f"human_{answer['verdict']}"] += 1
    out: dict[str, dict[str, Any]] = {}
    for name, counts in sorted(groups.items()):
        row: dict[str, Any] = {
            k: counts[k]
            for k in (
                "questions",
                "human_judged",
                "human_incorrect",
                "human_partial",
                "human_correct",
            )
        }
        row["incorrect_rate"] = (
            counts["human_incorrect"] / counts["human_judged"] if counts["human_judged"] else None
        )
        out[name] = row
    return out


def _safety_record(qid: str, judgments: Sequence[Judgment]) -> dict[str, Any] | None:
    """The question's safety record (what its answer did), or ``None`` when none was given."""
    given = [j for j in judgments if j.safety is not None]
    if not given:
        return None
    if len(given) > 1:
        raise AnswerJudgmentError(
            f"question {qid!r} has {len(given)} safety records; use one canary check per file"
        )
    j = given[0]
    return {
        **(j.safety or {}),
        "evaluator": j.evaluator,
        **({"evaluator_version": j.evaluator_version} if j.evaluator_version else {}),
    }


def _disagrees(judgments: Sequence[Judgment]) -> bool:
    """Whether a human verdict contradicts a model verdict on this question."""
    human = {j.verdict for j in judgments if j.kind == "human" and j.verdict}
    model = {j.verdict for j in judgments if j.kind == "model" and j.verdict}
    return bool(human) and bool(model - human)


def _claim_evidence(qid: str, judgments: Sequence[Judgment]) -> dict[str, Any] | None:
    """The question's claim counts and per-document support, or ``None`` when none were given."""
    given = [j for j in judgments if j.claims is not None]
    if not given:
        return None
    if len(given) > 1:
        raise AnswerJudgmentError(
            f"question {qid!r} has {len(given)} claim-evidence judgments; import one "
            "extractor and checker per file, so each question has one set of claim facts"
        )
    j = given[0]
    return {
        "evaluator": j.evaluator,
        "kind": j.kind,
        **({"evaluator_version": j.evaluator_version} if j.evaluator_version else {}),
        "claims": dict(j.claims or ()),
        "claim_support": {
            doc: {k: list(v) if isinstance(v, tuple) else v for k, v in counts.items()}
            for doc, counts in j.claim_support
        },
    }


def attach_answers(
    run: Mapping[str, Any],
    rows: Sequence[tuple[str, str, Judgment]],
    *,
    source_sha256: str | None = None,
    inventory: AbstractSet[str] | None = None,
    sample: ReviewSample | None = None,
    source_format: str = "vectraudit",
    skipped: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """A copy of the run with ``answer`` on each judged question and an ``answers`` summary.

    ``inventory`` is the corpus's document IDs; with it, a cited ID outside it is a fabricated
    citation (``not_in_corpus``). ``sample`` is the random review sample (see
    :mod:`vectraudit.evaluation.review_sample`): with it, ``answers.estimates`` gives a
    judge-corrected correct rate per model evaluator, or says why it cannot.

    Judgments for questions the run does not contain are counted, never attached. Failed
    questions keep their judgments but get no diagnosis: there is no evidence to read.
    """
    out: dict[str, Any] = json.loads(json.dumps(run))
    by_query = {q["query_id"]: q for q in out["queries"]}
    by_trace = {q["trace_id"]: q for q in out["queries"] if isinstance(q.get("trace_id"), str)}
    grouped: dict[str, list[Judgment]] = defaultdict(list)
    unknown = 0
    for key_kind, key, judgment in rows:
        q = by_query.get(key) if key_kind == "query_id" else by_trace.get(key)
        if q is None:
            unknown += 1
            continue
        grouped[q["query_id"]].append(judgment)
    verdicts: Counter[str] = Counter()
    diagnoses: Counter[str] = Counter()
    by_prompt: dict[str, Counter[str]] = defaultdict(Counter)
    citation_flags: Counter[str] = Counter()
    cited_questions = 0
    claim_questions = 0
    safety_checked = safety_followed = 0
    redteam_counts: dict[str, Counter[str]] = defaultdict(Counter)
    safety_attacks: Counter[str] = Counter()
    finding_counts: Counter[str] = Counter()
    pairs: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for qid, judgments in grouped.items():
        q = by_query[qid]
        verdict, decided_by, outcome = _decide(judgments)
        answer: dict[str, Any] = {
            "verdict": verdict,
            "decided_by": decided_by,
            "outcome": outcome,
            "judgments": [j.record() for j in judgments],
        }
        if q.get("error") is None:
            answer["diagnosis"] = diagnose(q, verdict, outcome)
            diagnoses[answer["diagnosis"]["class"]] += 1
            for prompt in q.get("prompts") or []:
                key = f"{prompt['name']}@{prompt['version']}"
                by_prompt[key][answer["diagnosis"]["class"]] += 1
        safety = _safety_record(qid, judgments)
        if safety is not None:
            answer["safety"] = safety
            safety_checked += 1
            safety_followed += safety["followed_injection"]
            safety_attacks.update(safety["attacks"])
        red = sorted(
            (
                {
                    **j.redteam,
                    "evaluator": j.evaluator,
                    **({"evaluator_version": j.evaluator_version} if j.evaluator_version else {}),
                }
                for j in judgments
                if j.redteam is not None
            ),
            key=lambda r: (r["plugin"], r["strategy"] or "", r["passed"]),
        )
        if red:
            answer["redteam"] = red
            for result in red:
                redteam_counts[result["plugin"]]["tests"] += 1
                redteam_counts[result["plugin"]]["failed"] += not result["passed"]
        claim_evidence = _claim_evidence(qid, judgments)
        if claim_evidence is not None:
            answer["claim_evidence"] = claim_evidence
            claim_questions += 1
            known = set(q.get("ids") or []) | {
                i.get("id")
                for i in (q.get("model_context") or {}).get("items") or []
                if isinstance(i, Mapping)
            }
            unmatched = sorted(set(claim_evidence["claim_support"]) - known)
            if unmatched:
                claim_evidence["unmatched_documents"] = unmatched[:MAX_FINDING_DOCUMENTS]
            if q.get("error") is None:
                disagree = _disagrees(judgments)
                answer["claim_findings"] = claim_findings(
                    q, claim_evidence, disputed=verdict == "disputed" or disagree
                )
                for finding in answer["claim_findings"]:
                    finding_counts[finding["finding"]] += 1
        if q.get("error") is None:
            citations = analyze_citations(q, judgments, inventory)
            if citations is not None:
                answer["citations"] = citations
                cited_questions += 1
                classes = set(citations.get("classes", {}).values())
                for name in ("not_delivered", "not_in_context", "not_in_corpus"):
                    citation_flags[name] += name in classes
                citation_flags["unsupported"] += any(
                    s["verdict"] == "unsupported" for s in citations.get("support", {}).values()
                )
                citation_flags["conflicting"] += citations["status"] == "conflicting"
        q["answer"] = answer
        verdicts[verdict or "unjudged"] += 1
        human = {j.verdict for j in judgments if j.kind == "human" and j.verdict}
        if len(human) == 1:
            reference = next(iter(human))
            for j in judgments:
                if j.kind == "model" and j.verdict is not None:
                    pairs[j.evaluator].append((qid, j.verdict == reference))
    evaluators = sorted(
        {(j.evaluator, j.kind, j.evaluator_version, j.rubric) for _, _, j in rows},
        key=lambda e: tuple(x or "" for x in e),
    )
    out["answers"] = {
        "format_version": ANSWERS_FORMAT_VERSION,
        "source_sha256": source_sha256,
        "source_format": source_format,
        "skipped_judgments": {r: dict(n) for r, n in (skipped or {}).items()},
        "judged_questions": len(grouped),
        "unmatched_judgments": unknown,
        "evaluators": [
            {"evaluator": e, "kind": k, "evaluator_version": v, "rubric": r}
            for e, k, v, r in evaluators
        ],
        "verdicts": dict(sorted(verdicts.items())),
        "diagnoses": dict(sorted(diagnoses.items())),
        "diagnoses_by_prompt": {
            key: dict(sorted(counts.items())) for key, counts in sorted(by_prompt.items())
        },
        "citations": {
            "questions_with_citations": cited_questions,
            "questions_citing_undelivered": citation_flags["not_delivered"],
            "questions_citing_outside_context": citation_flags["not_in_context"],
            "questions_with_unsupported_citation": citation_flags["unsupported"],
            "questions_with_conflicting_citations": citation_flags["conflicting"],
            **(
                {
                    "questions_citing_outside_corpus": citation_flags["not_in_corpus"],
                    "inventory_documents": len(inventory),
                }
                if inventory is not None
                else {}
            ),
        },
        "redteam": {
            plugin: {"tests": c["tests"], "failed": c["failed"]}
            for plugin, c in sorted(redteam_counts.items())
        },
        "safety": {
            "questions_checked": safety_checked,
            "followed_injection": safety_followed,
            "by_attack": dict(sorted(safety_attacks.items())),
        },
        "claim_evidence": {
            "questions": claim_questions,
            "findings": dict(sorted(finding_counts.items())),
            "meta_evaluation": _meta_evaluation({qid: by_query[qid]["answer"] for qid in grouped}),
        },
        "estimates": estimate_rates(
            out,
            {qid: [(j.evaluator, j.kind, j.verdict) for j in js] for qid, js in grouped.items()},
            sample,
        ),
        "agreement_with_human": {
            name: {
                "questions": len(results),
                "agreed": sum(ok for _, ok in results),
                "disagreed_on": sorted(qid for qid, ok in results if not ok)[:50],
            }
            for name, results in sorted(pairs.items())
        },
    }
    return out


def attach_file(
    run: Mapping[str, Any],
    path: str | Path,
    thresholds: Mapping[str, float] | None = None,
    inventory: AbstractSet[str] | None = None,
    sample_path: str | Path | None = None,
    *,
    evaluators: Mapping[str, str | None] | None = None,
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """:func:`attach_answers` from a judgments file or vendor export, recording its digest."""
    read = read_judgment_file(path, thresholds, evaluators=evaluators, labels=labels)
    rows = read.rows
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    try:
        sample = None if sample_path is None else read_sample(sample_path)
    except ReviewSampleError as exc:
        raise AnswerJudgmentError(f"{exc} {exc.hint}".strip()) from exc
    return attach_answers(
        run,
        rows,
        source_sha256=digest,
        inventory=inventory,
        sample=sample,
        source_format=read.source_format,
        skipped=read.skipped,
    )


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _estimate_lines(block: Mapping[str, Any] | None) -> list[str]:
    """Terminal lines for the judge-corrected rates (judge only, human only, corrected)."""
    lines = []
    for name, row in ((block or {}).get("by_evaluator") or {}).items():
        if row["status"] != "ok":
            lines.append(f"{name}: no corrected estimate. {row['reason']}")
            continue
        human, corrected = row["human_only"], row["corrected"]
        lines.append(
            f"{name} correct rate: judge only {_pct(row['judge_only']['estimate'])} (biased by the "
            f"judge's errors), people only {_pct(human['estimate'])} "
            f"({_pct(human['ci_low'])}-{_pct(human['ci_high'])}), corrected "
            f"{_pct(corrected['estimate'])} ({_pct(corrected['ci_low'])}-"
            f"{_pct(corrected['ci_high'])}, 95%); {row['sample_families']} of "
            f"{row['frame_families']} families reviewed, the judge agreed with people on "
            f"{_pct(row['agreement_on_sample'])} of them."
        )
    return lines


def render_summary(run: Mapping[str, Any]) -> str:
    """The attached judgments as plain text for the terminal."""
    answers = run["answers"]
    lines = [
        f"Judged {answers['judged_questions']} of {len(run['queries'])} questions"
        + (
            f"; {answers['unmatched_judgments']} judgments matched no question in this run"
            if answers["unmatched_judgments"]
            else ""
        )
        + ".",
        "Verdicts: " + (", ".join(f"{n} {v}" for v, n in answers["verdicts"].items()) or "none"),
    ]
    if answers["diagnoses"]:
        lines += ["", "Diagnosis (first observable loss, not a root cause):"]
        lines += [
            f"  {n}: {name.replace('_', ' ')}. {DIAGNOSES[name]}"
            for name, n in sorted(answers["diagnoses"].items(), key=lambda kv: -kv[1])
        ]
    by_prompt = answers.get("diagnoses_by_prompt") or {}
    if len(by_prompt) > 1:  # one version explains nothing on its own
        lines += ["", "Diagnosis by prompt version (questions whose traces recorded one):"]
        lines += [
            f"  {key}: " + ", ".join(f"{n} {c.replace('_', ' ')}" for c, n in counts.items())
            for key, counts in by_prompt.items()
        ]
    lines += _estimate_lines(answers.get("estimates"))
    for name, stats in answers["agreement_with_human"].items():
        lines.append(
            f"{name} (model judgments) agrees with human review on {stats['agreed']} of "
            f"{stats['questions']} questions judged by both."
        )
    hints = {
        "kind_undeclared": "pass --evaluator NAME=human or NAME=model if these are reviews; "
        "end-user ratings are not",
        "label_unmapped": "pass --label LABEL=correct|partial|incorrect",
    }
    for reason, names in (answers.get("skipped_judgments") or {}).items():
        lines.append(
            f"Skipped {sum(names.values())}: {judgment_formats.SKIPPED[reason]} "
            f"({', '.join(sorted(names)[:10])})"
            + (f"; {hints[reason]}" if reason in hints else "")
            + "."
        )
    cites = answers.get("citations") or {}
    if cites.get("questions_with_citations"):
        lines.append(
            f"Citations on {cites['questions_with_citations']} questions: "
            f"{cites['questions_citing_undelivered']} cite a document retrieval never delivered, "
            f"{cites['questions_citing_outside_context']} cite one absent from the model "
            f"context, {cites['questions_with_unsupported_citation']} have a citation judged "
            f"unsupported, {cites['questions_with_conflicting_citations']} have conflicting "
            "citation lists."
        )
        if "questions_citing_outside_corpus" in cites:
            lines.append(
                f"{cites['questions_citing_outside_corpus']} questions cite an ID absent from the "
                f"{cites['inventory_documents']}-document corpus inventory: fabricated citations."
            )
    for plugin, row in (answers.get("redteam") or {}).items():
        lines.append(
            f"Red-team {plugin}: {row['failed']} of {row['tests']} tests matched to questions "
            "failed (model-graded; recorded apart from answer verdicts)."
        )
    safety = answers.get("safety") or {}
    if safety.get("questions_checked"):
        lines.append(
            f"Canary check: {safety['followed_injection']} of {safety['questions_checked']} "
            "answers followed an injected instruction (recorded as a safety failure, apart from "
            "answer verdicts)."
        )
    claims = answers.get("claim_evidence") or {}
    if claims.get("questions"):
        lines.append(f"Claim-level evidence attached to {claims['questions']} questions.")
        lines += [
            f"  {n}: {name.replace('_', ' ')}. {CLAIM_FINDINGS[name]}"
            for name, n in sorted(claims.get("findings", {}).items(), key=lambda kv: -kv[1])
        ]
        meta = claims.get("meta_evaluation") or {}
        judged = {k: v for k, v in meta.items() if v["human_judged"]}
        if judged:
            lines.append("Human review of the answers, by claim finding (incorrect / judged):")
            for name, row in sorted(judged.items()):
                label = {"all_questions": "all questions", "no_findings": "no finding"}.get(
                    name, name.replace("_", " ")
                )
                lines.append(
                    f"  {label}: {row['human_incorrect']} of {row['human_judged']} "
                    f"judged incorrect ({row['questions']} questions)"
                )
    unjudged = answers["verdicts"].get("unjudged")
    if unjudged:
        lines.append(
            f"{unjudged} questions have no verdict. For scores, pass --threshold "
            "EVALUATOR=VALUE for each scored evaluator; claim evidence alone gives none."
        )
    return "\n".join(lines)
