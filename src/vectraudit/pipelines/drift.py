"""Query drift: is the benchmark still representative of production questions?

A release check is only as good as its benchmark. When production starts asking different
questions (longer ones, another language, a new tenant, words the benchmark has never
seen), a passing check says less about production than it did. This module compares the
benchmark's questions (reference) with production questions read from traces (current),
in the manner of Evidently's data drift reports:

- each question gets **descriptors**: word count, script (Latin, Cyrillic, CJK, ...),
  whether it contains digits, whether it is phrased as a question, whether it uses a term
  the rest of the questions never use, and any declared **context fields** (tenant,
  language, product);
- each descriptor's distribution is compared with the **population stability index**
  (PSI), computed as Evidently does: numeric descriptors with more than 20 distinct
  reference values are binned with the Sturges rule over both samples, other descriptors
  compare category shares, and empty bins get a small floor. PSI of 0.1 or more is drift
  (Evidently's default threshold);
- **new values** (a script, tenant or category production uses and the benchmark lacks) and
  the most frequent **unseen terms** are listed, because a drift score says that something
  moved but not what.

The unseen-term descriptor is computed leave-one-out on the benchmark (a term no *other*
benchmark question uses), so both sides measure the same thing. Everything stays local:
the report holds production terms and context values and is never uploaded.

Reimplemented with numpy; Evidently (Apache-2.0) is the reference for the method, not a
dependency.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

PSI_THRESHOLD = 0.1
SMALL_SAMPLE = 30
QUESTION_MARKS = ("?", "\uff1f")  # ASCII, and the fullwidth mark CJK questions use
NUMERIC_UNIQUE = 20  # Evidently bins numbers only above this many distinct reference values
_WORD = re.compile(r"\w+", re.UNICODE)
_SCRIPTS = (
    ("CJK", "cjk"), ("HIRAGANA", "japanese"), ("KATAKANA", "japanese"), ("HANGUL", "korean"),
    ("LATIN", "latin"), ("CYRILLIC", "cyrillic"), ("GREEK", "greek"), ("ARABIC", "arabic"),
    ("HEBREW", "hebrew"), ("DEVANAGARI", "devanagari"), ("THAI", "thai"),
)  # fmt: skip


def psi(reference: Sequence[Any], current: Sequence[Any], *, numeric: bool) -> float:
    """Population stability index of ``current`` against ``reference`` (Evidently's method)."""
    if not reference or not current:
        raise ValueError("PSI needs at least one reference and one current value")
    if numeric and len(set(reference)) > NUMERIC_UNIQUE:
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        edges = np.histogram_bin_edges(np.concatenate([ref, cur]), bins="sturges")
        ref_share = np.histogram(ref, edges)[0] / len(ref)
        cur_share = np.histogram(cur, edges)[0] / len(cur)
    else:
        keys = sorted({str(v) for v in [*reference, *current]})
        ref_counts, cur_counts = Counter(map(str, reference)), Counter(map(str, current))
        ref_share = np.array([ref_counts[k] / len(reference) for k in keys])
        cur_share = np.array([cur_counts[k] / len(current) for k in keys])
    ref_share, cur_share = _floor(ref_share), _floor(cur_share)
    return float(np.sum((ref_share - cur_share) * np.log(ref_share / cur_share)))


def _floor(shares: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Empty bins get 0.0001, or a millionth of the smallest share when that is tiny."""
    positive = shares[shares > 0]
    smallest = float(positive.min()) if positive.size else 1.0
    out = shares.astype(float).copy()
    out[out == 0] = smallest / 10**6 if smallest <= 0.0001 else 0.0001
    return out


def terms(text: str) -> list[str]:
    """Lower-cased words of two or more characters."""
    return [w for w in (m.group().lower() for m in _WORD.finditer(text)) if len(w) >= 2]


def script(text: str) -> str:
    """The script most of the letters are written in, or ``none`` without letters."""
    counts: Counter[str] = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        counts[next((label for prefix, label in _SCRIPTS if name.startswith(prefix)), "other")] += 1
    return counts.most_common(1)[0][0] if counts else "none"


def describe(text: str, rare: bool) -> dict[str, Any]:
    """The descriptors of one question."""
    stripped = text.strip()
    return {
        "words": len(_WORD.findall(stripped)),
        "script": script(stripped),
        "has_digits": "yes" if any(c.isdigit() for c in stripped) else "no",
        "question_mark": "yes" if stripped.endswith(QUESTION_MARKS) else "no",
        "unseen_term": "yes" if rare else "no",
    }


def query_drift(
    reference: Sequence[tuple[str, Mapping[str, Any]]],
    current: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    context_fields: Sequence[str] = (),
    top: int = 20,
) -> dict[str, Any]:
    """Compare production questions (``current``) with the benchmark's (``reference``).

    Each item is ``(question text, context)``. Questions with no text are not counted.
    """
    reference = [(t, c) for t, c in reference if isinstance(t, str) and t.strip()]
    current = [(t, c) for t, c in current if isinstance(t, str) and t.strip()]
    if not reference:
        raise ValueError("The benchmark has no question text to compare against.")
    if not current:
        raise ValueError("No production question text was found in the traces.")

    ref_terms = [set(terms(t)) for t, _ in reference]
    usage: Counter[str] = Counter(term for found in ref_terms for term in found)
    vocabulary = set(usage)
    ref_rows = [
        {**describe(t, any(usage[w] == 1 for w in found)), **_context(c, context_fields)}
        for (t, c), found in zip(reference, ref_terms, strict=True)
    ]  # leave-one-out: a term only this benchmark question uses
    unseen: Counter[str] = Counter()
    unseen_questions: Counter[str] = Counter()
    cur_rows = []
    for text, context in current:
        found = set(terms(text))
        new = found - vocabulary
        unseen.update(new)
        unseen_questions.update(new)
        cur_rows.append({**describe(text, bool(new)), **_context(context, context_fields)})

    names = ["words", "script", "has_digits", "question_mark", "unseen_term"]
    names += [f"context.{f}" for f in context_fields]
    descriptors = []
    for name in names:
        ref_values = [row[name] for row in ref_rows]
        cur_values = [row[name] for row in cur_rows]
        numeric = name == "words"
        value = psi(ref_values, cur_values, numeric=numeric)
        entry: dict[str, Any] = {
            "name": name,
            "kind": "numeric" if numeric else "categorical",
            "psi": round(value, 4),
            "drift": value >= PSI_THRESHOLD,
        }
        if numeric:
            entry["reference"] = _numeric_summary(ref_values)
            entry["current"] = _numeric_summary(cur_values)
        else:
            entry["reference"] = _shares(ref_values, top)
            entry["current"] = _shares(cur_values, top)
            new_values = Counter(v for v in cur_values if v not in set(ref_values))
            entry["new_values"] = [
                {"value": v, "questions": n} for v, n in new_values.most_common(top)
            ]
        descriptors.append(entry)

    ref_texts = {_norm(t) for t, _ in reference}
    covered = sum(_norm(t) in ref_texts for t, _ in current)
    warnings = []
    if len(current) < SMALL_SAMPLE or len(reference) < SMALL_SAMPLE:
        warnings.append(
            f"Small samples ({len(reference)} benchmark, {len(current)} production questions): "
            "PSI is unstable below about 30 values per side; read shifts as indicative."
        )
    drifted = [d["name"] for d in descriptors if d["drift"]]
    return {
        "format": "vectraudit.query_drift",
        "format_version": 1,
        "method": {"statistic": "psi", "threshold": PSI_THRESHOLD, "binning": "evidently"},
        "reference_questions": len(reference),
        "current_questions": len(current),
        "status": "drift" if drifted else "no_drift",
        "drifted": drifted,
        "descriptors": descriptors,
        "benchmark_coverage": {
            "questions_in_benchmark": covered,
            "share": round(covered / len(current), 4),
        },
        "unseen_terms": [
            {"term": term, "questions": unseen_questions[term]}
            for term, _ in unseen.most_common(top)
        ],
        "warnings": warnings,
    }


def _context(context: Mapping[str, Any], fields: Sequence[str]) -> dict[str, str]:
    out = {}
    for name in fields:
        value = (context or {}).get(name)
        out[f"context.{name}"] = (
            "(missing)" if value is None or isinstance(value, (list, dict)) else str(value)
        )
    return out


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 2),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _shares(values: Sequence[Any], top: int) -> dict[str, float]:
    counts = Counter(values)
    return {str(v): round(n / len(values), 4) for v, n in counts.most_common(top)}


def _norm(text: str) -> str:
    return " ".join(terms(text))


def production_questions(
    traces: str | Path, settings: Any, context_fields: Sequence[str] = ()
) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    """``(question text, context)`` per trace, and how many traces had no readable question."""
    from vectraudit.pipelines.intake import _context as trace_context
    from vectraudit.pipelines.traces import read_spans, view_trace

    spans, _ = read_spans(traces, settings.stage_kinds)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for span in spans:
        grouped[span.trace_id].append(span)
    out, unreadable = [], 0
    for group in grouped.values():
        text = next(iter(view_trace(group, settings).query_texts), None)
        if not text:
            unreadable += 1
            continue
        out.append((text, trace_context(group, context_fields)))
    return out, unreadable


def render(report: Mapping[str, Any]) -> str:
    """The drift report as plain text for the terminal."""
    status = "DRIFT" if report["status"] == "drift" else "no drift"
    lines = [
        f"Query drift: {status} ({report['current_questions']} production vs "
        f"{report['reference_questions']} benchmark questions; PSI >= "
        f"{report['method']['threshold']} is drift).",
        "",
    ]
    for d in report["descriptors"]:
        mark = "DRIFT" if d["drift"] else "ok"
        line = f"  {d['name']:<22} PSI {d['psi']:>7.4f}  {mark}"
        if d["kind"] == "numeric":
            line += f"  (mean {d['reference']['mean']} → {d['current']['mean']})"
        elif d.get("new_values"):
            shown = ", ".join(f"{v['value']} ({v['questions']})" for v in d["new_values"][:5])
            line += f"  new: {shown}"
        lines.append(line)
    coverage = report["benchmark_coverage"]
    lines += [
        "",
        f"{coverage['questions_in_benchmark']} production questions ({coverage['share']:.0%}) "
        "are benchmark questions word for word.",
    ]
    if report["unseen_terms"]:
        lines.append(
            "Most frequent terms the benchmark never uses: "
            + ", ".join(f"{t['term']} ({t['questions']})" for t in report["unseen_terms"][:10])
        )
    lines += [f"Warning: {w}" for w in report["warnings"]]
    lines.append(
        "Drift means the benchmark may no longer represent production; add or relabel "
        "questions (intake's review queue) before trusting a pass."
        if report["status"] == "drift"
        else "No descriptor moved past the threshold."
    )
    return "\n".join(lines)
