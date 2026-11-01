"""Query and retrieval-score drift, with Evidently's default drift tests.

A release check is only as good as its benchmark. When production starts asking different
questions (longer ones, another language, a new tenant, words the benchmark has never
seen), a passing check says less about production than it did. This module compares the
benchmark's questions (reference) with production questions read from traces (current),
and compares a retriever's scores between two windows of traces:

- each question gets **descriptors**: word count, script (Latin, Cyrillic, CJK, ...),
  whether it contains digits, whether it is phrased as a question, whether it uses a term
  the rest of the questions never use, and any declared **context fields** (tenant,
  language, product);
- each descriptor is compared with the test **Evidently selects by default**: with at most
  1,000 reference values, the two-sample Kolmogorov-Smirnov test for numbers with more than
  five distinct values, and otherwise chi-square (or a two-proportion Z-test for two
  values), drift at p < 0.05; with more, the normed Wasserstein distance for numbers and the
  Jensen-Shannon distance otherwise, drift at 0.1. The population stability index is
  available as an explicit choice; at the usual 0.1 threshold it raises false alarms on
  samples of a few hundred (measured in the tests), so it is not the default;
- **new values** (a script, tenant or category production uses and the benchmark lacks) and
  the most frequent **unseen terms** are listed, because a drift test says that something
  moved but not what.

The unseen-term descriptor is computed leave-one-out on the benchmark (a term no *other*
benchmark question uses), so both sides measure the same thing. Everything stays local:
the report holds production terms and context values and is never uploaded.

Reimplemented with numpy (p-values from the asymptotic Kolmogorov distribution and the
regularized incomplete gamma function); Evidently (Apache-2.0) and SciPy are the
references for the methods, not dependencies.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

PSI_THRESHOLD = 0.1
P_VALUE = 0.05  # Kolmogorov-Smirnov, chi-square and Z-test: drift below this p-value
DISTANCE_THRESHOLD = 0.1  # normed Wasserstein and Jensen-Shannon: drift at or above this
LARGE_REFERENCE = 1000  # Evidently switches from p-value tests to distances above this
FEW_VALUES = 5  # numbers with at most this many distinct values are tested as categories
METHODS = ("auto", "psi")
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
    ref_share, cur_share = _binned(reference, current, numeric=numeric)
    ref_share, cur_share = _floor(ref_share), _floor(cur_share)
    return float(np.sum((ref_share - cur_share) * np.log(ref_share / cur_share)))


def drift_test(
    reference: Sequence[Any], current: Sequence[Any], *, numeric: bool, method: str = "auto"
) -> dict[str, Any]:
    """Compare two samples with Evidently's default test (or PSI); the test and its result."""
    if not reference or not current:
        raise ValueError("A drift test needs at least one reference and one current value")
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    if method == "psi":
        value = psi(reference, current, numeric=numeric)
        return _result("psi", value, PSI_THRESHOLD, value >= PSI_THRESHOLD)
    distinct = len({str(v) for v in [*reference, *current]})
    numbers = numeric and distinct > FEW_VALUES
    if len(reference) <= LARGE_REFERENCE:
        if numbers:
            p = ks_p_value(reference, current)
            return _result("ks", p, P_VALUE, p <= P_VALUE)
        if distinct > 2:
            p = chi_square_p_value(reference, current)
            return _result("chi_square", p, P_VALUE, p < P_VALUE)
        p = z_test_p_value(reference, current)
        return _result("z_test", p, P_VALUE, p < P_VALUE)
    if numbers:
        value = wasserstein_normed(reference, current)
        return _result("wasserstein", value, DISTANCE_THRESHOLD, value >= DISTANCE_THRESHOLD)
    value = jensen_shannon(reference, current, numeric=numeric)
    return _result("jensen_shannon", value, DISTANCE_THRESHOLD, value >= DISTANCE_THRESHOLD)


def _result(test: str, value: float, threshold: float, drift: bool) -> dict[str, Any]:
    kind = "p_value" if test in ("ks", "chi_square", "z_test") else "score"
    return {
        "test": test,
        kind: round(float(value), 6),
        "threshold": threshold,
        "drift": bool(drift),
    }


def ks_p_value(reference: Sequence[float], current: Sequence[float]) -> float:
    """Two-sided two-sample Kolmogorov-Smirnov p-value (asymptotic, with Stephens' correction)."""
    a = np.sort(np.asarray(reference, dtype=float))
    b = np.sort(np.asarray(current, dtype=float))
    grid = np.concatenate([a, b])
    d = float(
        np.max(
            np.abs(
                np.searchsorted(a, grid, side="right") / len(a)
                - np.searchsorted(b, grid, side="right") / len(b)
            )
        )
    )
    if d == 0:
        return 1.0
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d
    total = sum(2 * (-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(min(max(total, 0.0), 1.0))


def chi_square_p_value(reference: Sequence[Any], current: Sequence[Any]) -> float:
    """Pearson chi-square of current counts against reference shares scaled to its size.

    As in Evidently: a value the reference never has gets an expected count of zero, so any
    occurrence of it is drift (p = 0).
    """
    keys = sorted({str(v) for v in [*reference, *current]})
    ref, cur = Counter(map(str, reference)), Counter(map(str, current))
    scale = len(current) / len(reference)
    statistic = 0.0
    for key in keys:
        expected = ref[key] * scale
        if expected == 0:
            if cur[key]:
                return 0.0
            continue
        statistic += (cur[key] - expected) ** 2 / expected
    dof = len(keys) - 1
    return _upper_gamma(dof / 2, statistic / 2) if dof > 0 else 1.0


def z_test_p_value(reference: Sequence[Any], current: Sequence[Any]) -> float:
    """Two-sided pooled two-proportion Z-test on the first of (at most) two values."""
    keys = sorted({str(v) for v in [*reference, *current]})
    if len(keys) < 2:
        return 1.0
    p1 = sum(str(v) == keys[0] for v in reference) / len(reference)
    p2 = sum(str(v) == keys[0] for v in current) / len(current)
    pooled = (p1 * len(reference) + p2 * len(current)) / (len(reference) + len(current))
    spread = math.sqrt(pooled * (1 - pooled) * (1 / len(reference) + 1 / len(current)))
    if spread == 0:
        return 1.0
    return math.erfc(abs(p1 - p2) / spread / math.sqrt(2))


def wasserstein_normed(reference: Sequence[float], current: Sequence[float]) -> float:
    """First Wasserstein distance divided by the reference's standard deviation (min 0.001)."""
    a = np.asarray(reference, dtype=float)
    b = np.asarray(current, dtype=float)
    grid = np.sort(np.concatenate([a, b]))
    widths = np.diff(grid)
    fa = np.searchsorted(np.sort(a), grid[:-1], side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid[:-1], side="right") / len(b)
    distance = float(np.sum(np.abs(fa - fb) * widths))
    return distance / max(float(np.std(a)), 0.001)


def jensen_shannon(reference: Sequence[Any], current: Sequence[Any], *, numeric: bool) -> float:
    """Jensen-Shannon distance (natural log) of the binned shares, as Evidently bins them."""
    ref_share, cur_share = _binned(reference, current, numeric=numeric)
    mid = (ref_share + cur_share) / 2

    def kl(p: npt.NDArray[np.float64]) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / mid[mask])))

    return math.sqrt(max((kl(ref_share) + kl(cur_share)) / 2, 0.0))


def _binned(
    reference: Sequence[Any], current: Sequence[Any], *, numeric: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if numeric and len(set(reference)) > NUMERIC_UNIQUE:
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        edges = np.histogram_bin_edges(np.concatenate([ref, cur]), bins="sturges")
        return (
            np.histogram(ref, edges)[0] / len(ref),
            np.histogram(cur, edges)[0] / len(cur),
        )
    keys = sorted({str(v) for v in [*reference, *current]})
    ref_counts, cur_counts = Counter(map(str, reference)), Counter(map(str, current))
    return (
        np.array([ref_counts[k] / len(reference) for k in keys]),
        np.array([cur_counts[k] / len(current) for k in keys]),
    )


def _upper_gamma(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x): the chi-square survival function."""
    if x <= 0:
        return 1.0
    log_front = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:  # series for P(a, x)
        term = total = 1.0 / a
        n = a
        for _ in range(1000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, min(1.0, 1.0 - total * math.exp(log_front)))
    b = x + 1 - a  # continued fraction for Q(a, x) (modified Lentz)
    c = 1 / 1e-300
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return max(0.0, min(1.0, math.exp(log_front) * h))


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
    method: str = "auto",
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
        entry: dict[str, Any] = {
            "name": name,
            "kind": "numeric" if numeric else "categorical",
            **drift_test(ref_values, cur_values, numeric=numeric, method=method),
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
            "drift tests have little power below about 30 values per side; read results as "
            "indicative."
        )
    drifted = [d["name"] for d in descriptors if d["drift"]]
    return {
        "format": "vectraudit.query_drift",
        "format_version": 2,
        "method": _method(method),
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


_TEST_LABELS = {
    "ks": "K-S p",
    "chi_square": "chi² p",
    "z_test": "Z p",
    "wasserstein": "Wasserstein",
    "jensen_shannon": "J-S",
    "psi": "PSI",
}


def _method(method: str) -> dict[str, Any]:
    if method == "psi":
        return {
            "name": "psi",
            "summary": f"PSI >= {PSI_THRESHOLD} is drift",
        }
    return {
        "name": "auto",
        "summary": (
            f"Evidently's default tests: p < {P_VALUE} (up to {LARGE_REFERENCE} reference "
            f"values) or distance >= {DISTANCE_THRESHOLD} is drift"
        ),
    }


def _shown(d: Mapping[str, Any]) -> str:
    value = d["p_value"] if "p_value" in d else d["score"]
    return f"{_TEST_LABELS[d['test']]:>11} {value:>8.4f}"


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
        f"{report['reference_questions']} benchmark questions; {report['method']['summary']}).",
        "",
    ]
    for d in report["descriptors"]:
        mark = "DRIFT" if d["drift"] else "ok"
        line = f"  {d['name']:<22} {_shown(d)}  {mark}"
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


SCORE_DESCRIPTORS = ("top_score", "mean_score", "score_spread", "returned")


def retrieval_samples(
    traces: str | Path, settings: Any, stage: str | None = None
) -> tuple[list[dict[str, float]], list[tuple[str, dict[str, Any]]], Counter[str]]:
    """Per trace: retrieval score descriptors and the question; and why traces were skipped.

    The stage is the first retriever span, or the span named ``stage``. Descriptors come
    from the scores in the order the retriever returned documents: the first document's
    score, the mean, the spread from first to last, and how many documents came back.
    """
    from vectraudit.pipelines.traces import read_spans, view_trace

    spans, _ = read_spans(traces, settings.stage_kinds)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for span in spans:
        grouped[span.trace_id].append(span)
    samples: list[dict[str, float]] = []
    questions: list[tuple[str, dict[str, Any]]] = []
    skipped: Counter[str] = Counter()
    for group in grouped.values():
        view = view_trace(group, settings)
        text = next(iter(view.query_texts), None)
        if text:
            questions.append((text, {}))
        candidates = [
            s
            for s in view.stages
            if not s.derived
            and (s.span.name == stage if stage is not None else s.span.kind == "RETRIEVER")
        ]
        if not candidates:
            skipped["no_stage"] += 1
            continue
        chosen = candidates[0]
        if chosen.documents is None:
            skipped["documents_unrecorded"] += 1
            continue
        scores = [s for s in chosen.scores if s is not None]
        if not scores or len(scores) != len(chosen.scores):
            skipped["scores_unrecorded"] += 1
            continue
        samples.append(
            {
                "top_score": scores[0],
                "mean_score": float(np.mean(scores)),
                "score_spread": scores[0] - scores[-1],
                "returned": float(len(chosen.documents)),
            }
        )
    return samples, questions, skipped


def score_drift(
    reference: str | Path,
    current: str | Path,
    settings: Any,
    *,
    stage: str | None = None,
    top: int = 20,
    method: str = "auto",
) -> dict[str, Any]:
    """Did the retriever's scores move between two windows of production traces?

    Questions in the two windows are compared too (``query_drift``), so a score shift can
    be read against whether the questions moved: scores that drift while the questions do
    not point at the retriever (an embedding model, index or similarity change).
    """
    ref_samples, ref_questions, ref_skipped = retrieval_samples(reference, settings, stage)
    cur_samples, cur_questions, cur_skipped = retrieval_samples(current, settings, stage)
    if not ref_samples or not cur_samples:
        side = "earlier" if not ref_samples else "later"
        why = dict(ref_skipped if not ref_samples else cur_skipped)
        raise ValueError(
            f"No retrieval scores in the {side} traces ({why or 'no traces'}). Scores come from "
            "retrieval.documents.N.document.score; name the stage with --stage if needed."
        )
    descriptors = []
    for name in SCORE_DESCRIPTORS:
        ref_values = [s[name] for s in ref_samples]
        cur_values = [s[name] for s in cur_samples]
        descriptors.append(
            {
                "name": name,
                **drift_test(ref_values, cur_values, numeric=True, method=method),
                "reference": _numeric_summary(ref_values),
                "current": _numeric_summary(cur_values),
            }
        )
    questions = (
        query_drift(ref_questions, cur_questions, top=top, method=method)
        if ref_questions and cur_questions
        else None
    )
    scores_moved = any(d["drift"] for d in descriptors)
    questions_moved = bool(questions and questions["status"] == "drift")
    if not scores_moved:
        status, reading = "no_drift", "Retrieval scores did not move past the threshold."
    elif questions_moved:
        status = "drift_with_question_drift"
        reading = (
            "Scores moved, and so did the questions; the questions may explain the shift. "
            "Compare on a fixed benchmark (a recorded run) to isolate the retriever."
        )
    else:
        status = "retriever_shift_likely"
        reading = (
            "Scores moved while the questions did not: the retriever likely changed (embedding "
            "model, index, similarity or score scale). Check before trusting older baselines."
        )
    warnings = []
    if min(len(ref_samples), len(cur_samples)) < SMALL_SAMPLE:
        warnings.append(
            f"Small samples ({len(ref_samples)} earlier, {len(cur_samples)} later traces with "
            "scores): drift tests have little power below about 30 values per side."
        )
    if questions is None:
        warnings.append("No question text in one window: question drift was not checked.")
    return {
        "format": "vectraudit.score_drift",
        "format_version": 1,
        "method": _method(method),
        "stage": stage or "first retriever span",
        "reference_traces": len(ref_samples),
        "current_traces": len(cur_samples),
        "skipped": {"reference": dict(ref_skipped), "current": dict(cur_skipped)},
        "status": status,
        "reading": reading,
        "descriptors": descriptors,
        "questions": (
            {"status": questions["status"], "drifted": questions["drifted"]} if questions else None
        ),
        "warnings": warnings,
    }


def render_scores(report: Mapping[str, Any]) -> str:
    """The score drift report as plain text."""
    lines = [
        f"Retrieval score drift: {report['status'].replace('_', ' ')} "
        f"({report['reference_traces']} earlier vs {report['current_traces']} later traces; "
        f"stage: {report['stage']}).",
        "",
    ]
    for d in report["descriptors"]:
        lines.append(
            f"  {d['name']:<14} {_shown(d)}  {'DRIFT' if d['drift'] else 'ok'}"
            f"  (mean {d['reference']['mean']} → {d['current']['mean']})"
        )
    q = report["questions"]
    if q is not None:
        moved = ", ".join(q["drifted"]) or "none"
        lines += ["", f"Questions: {q['status'].replace('_', ' ')} (drifted: {moved})."]
    for side in ("reference", "current"):
        if report["skipped"][side]:
            label = "earlier" if side == "reference" else "later"
            shown = ", ".join(
                f"{n} {k.replace('_', ' ')}" for k, n in sorted(report["skipped"][side].items())
            )
            lines.append(f"Not counted in the {label} window: {shown}.")
    lines += [f"Warning: {w}" for w in report["warnings"]]
    lines += ["", report["reading"]]
    return "\n".join(lines)
