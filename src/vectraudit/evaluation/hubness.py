"""Hubs: documents a retriever returns for many unrelated questions.

High-dimensional embeddings produce *hubs*, a few documents that are near almost every
query (Radovanović et al., "Hubs in Space", JMLR 2010). A hub crowds relevant documents out
of the top k for questions it does not answer, and a new embedding model or index can
create one without moving the average quality much. This report measures it on a recorded
run, adapting Evidently's popularity-bias metrics from recommendations to retrieval:

- **k-occurrence:** for each document, the number of questions whose delivered top k holds
  it. Questions that failed are left out.
- **Gini index** of the k-occurrences of the documents retrieved at least once, with
  Evidently's formula: 0 when every retrieved document appears equally often, near 1 when
  a few dominate.
- **Coverage:** the share of the corpus retrieved for any question, when the run records
  the corpus size (``backend.count``).
- **Skewness** of the k-occurrence distribution, the standard hubness measure. Documents
  never retrieved count as zero when the corpus size is known; otherwise only retrieved
  documents are counted, and the report says so.
- **Hubs:** documents in the top k of at least ``hub_share`` of the questions. With the
  benchmark, each hub's **relevance** is reported: for how many of the questions that
  retrieved it the benchmark labels it relevant. A hub that is rarely relevant is crowding
  out answers.

A baseline run turns this into a comparison: hubs new in the candidate are listed. The
report never changes a release verdict.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

DEFAULT_HUB_SHARE = 0.1
MIN_HUB_QUESTIONS = 3


def _rankings(run: Mapping[str, Any]) -> dict[str, list[str]]:
    k = int(run["search"]["k"])
    out = {}
    for q in run.get("queries") or []:
        if q.get("error") is None and isinstance(q.get("ids"), list):
            # A document repeated in one ranking occurs once for that question.
            out[str(q["query_id"])] = list(dict.fromkeys(str(d) for d in q["ids"][:k]))
    return out


def gini(counts: list[int]) -> float:
    """Evidently's Gini index of positive counts (0 even, near 1 concentrated)."""
    values = sorted(c for c in counts if c > 0)
    n = len(values)
    if n < 2:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * np.asarray(values)) / ((n - 1) * sum(values)))


def skewness(values: list[int]) -> float | None:
    """Sample skewness (Fisher-Pearson, biased), or ``None`` without spread."""
    if len(values) < 3:
        return None
    x = np.asarray(values, dtype=float)
    spread = float(x.std())
    if spread == 0:
        return None
    return float(np.mean((x - x.mean()) ** 3) / spread**3)


def hubness(
    run: Mapping[str, Any],
    *,
    labels: Mapping[str, Mapping[str, float]] | None = None,
    hub_share: float = DEFAULT_HUB_SHARE,
    top: int = 20,
) -> dict[str, Any]:
    """Hubness of one run; ``labels`` maps question IDs to their relevance labels."""
    if not 0 < hub_share <= 1:
        raise ValueError("hub_share must be in (0, 1]")
    rankings = _rankings(run)
    if not rankings:
        raise ValueError("The run has no successful questions with a delivered ranking.")
    occurrence: Counter[str] = Counter(d for ids in rankings.values() for d in ids)
    corpus = (run.get("backend") or {}).get("count")
    corpus = corpus if isinstance(corpus, int) and corpus > 0 else None
    counts = list(occurrence.values())
    if corpus is not None and corpus >= len(occurrence):
        with_zeros = counts + [0] * (corpus - len(occurrence))
    else:
        with_zeros = None
    threshold = max(MIN_HUB_QUESTIONS, math.ceil(hub_share * len(rankings)))
    hubs = []
    for doc, n in occurrence.most_common():
        if n < threshold:
            break
        hub: dict[str, Any] = {"doc_id": doc, "questions": n, "share": round(n / len(rankings), 4)}
        if labels is not None:
            judged = [qid for qid, ids in rankings.items() if doc in ids and qid in labels]
            relevant = sum((labels[qid].get(doc) or 0) > 0 for qid in judged)
            hub["relevant_for"] = relevant
            hub["judged_questions"] = len(judged)
        hubs.append(hub)
        if len(hubs) >= top:
            break
    return {
        "format": "vectraudit.hubness",
        "format_version": 1,
        "k": int(run["search"]["k"]),
        "questions": len(rankings),
        "documents_retrieved": len(occurrence),
        "corpus_size": corpus,
        "coverage": round(len(occurrence) / corpus, 4) if corpus else None,
        "gini": round(gini(counts), 4),
        "skewness": _rounded(skewness(with_zeros if with_zeros is not None else counts)),
        "skewness_includes_unretrieved": with_zeros is not None,
        "hub_threshold": {"share": hub_share, "questions": threshold},
        "hubs": hubs,
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def compare(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """What changed between two hubness reports: the measures, and hubs new to the candidate."""
    before = {h["doc_id"]: h for h in baseline["hubs"]}
    return {
        "gini": {"baseline": baseline["gini"], "candidate": candidate["gini"]},
        "skewness": {"baseline": baseline["skewness"], "candidate": candidate["skewness"]},
        "coverage": {"baseline": baseline["coverage"], "candidate": candidate["coverage"]},
        "new_hubs": [h for h in candidate["hubs"] if h["doc_id"] not in before],
        "gone_hubs": [
            h
            for h in baseline["hubs"]
            if h["doc_id"] not in {c["doc_id"] for c in candidate["hubs"]}
        ],
    }


def render(report: Mapping[str, Any], comparison: Mapping[str, Any] | None = None) -> str:
    """The hubness report as plain text."""
    coverage = (
        f"{report['coverage']:.1%} of {report['corpus_size']} documents"
        if report["coverage"] is not None
        else "unknown (the run does not record the corpus size)"
    )
    skew = "n/a" if report["skewness"] is None else f"{report['skewness']:.2f}"
    lines = [
        f"Hubness at k={report['k']} over {report['questions']} questions: "
        f"{report['documents_retrieved']} distinct documents retrieved.",
        f"  Coverage: {coverage}",
        f"  Gini: {report['gini']:.3f}  (0 even, near 1 a few documents dominate)",
        f"  k-occurrence skewness: {skew}"
        + ("" if report["skewness_includes_unretrieved"] else " (retrieved documents only)"),
        "",
    ]
    threshold = report["hub_threshold"]
    if report["hubs"]:
        lines.append(
            f"Hubs (in the top {report['k']} of at least {threshold['questions']} questions):"
        )
        for hub in report["hubs"]:
            line = f"  {hub['doc_id']}: {hub['questions']} questions ({hub['share']:.0%})"
            if "relevant_for" in hub:
                line += (
                    f", labelled relevant for {hub['relevant_for']} of "
                    f"{hub['judged_questions']} judged"
                )
            lines.append(line)
    else:
        lines.append(f"No hubs (no document in the top k of {threshold['questions']}+ questions).")
    if comparison is not None:
        lines += ["", "Against the baseline:"]
        for name in ("gini", "skewness", "coverage"):
            pair = comparison[name]
            lines.append(f"  {name}: {pair['baseline']} → {pair['candidate']}")
        new = comparison["new_hubs"]
        lines.append("  New hubs: " + (", ".join(h["doc_id"] for h in new) if new else "none"))
    rarely = [
        h for h in report["hubs"]
        if h.get("judged_questions") and h["relevant_for"] / h["judged_questions"] < 0.5
    ]  # fmt: skip
    if rarely:
        lines += [
            "",
            "Hubs labelled relevant for fewer than half the questions that retrieve them crowd "
            "out answers: " + ", ".join(h["doc_id"] for h in rarely) + ".",
        ]
    return "\n".join(lines)
