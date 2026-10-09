"""Per-query metrics and the versioned contract that defines them.

Two kinds of quality are kept apart and never substituted for each other:

- **Semantic** metrics (recall, precision, MRR, nDCG@k) compare results with labelled
  relevant documents. They need labels.
- **ANN recall@k** is the overlap with an exact nearest-neighbour search over the same
  corpus, vectors and distance. It measures approximation loss, not whether the documents
  answer the query, and needs an exact reference.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vectraudit.core.relevance import relevance_metrics

METRIC_CONTRACT_VERSION = 2
SEMANTIC_METRICS = ("recall", "precision", "mrr", "ndcg")
QUALITY_METRICS = (*SEMANTIC_METRICS, "ann_recall")


def metric_contract(k: int) -> dict[str, Any]:
    """Everything that must match for two runs' metrics to be comparable."""
    return {
        "version": METRIC_CONTRACT_VERSION,
        "repeat_aggregation": "mean_per_query",
        "k": k,
        "relevant": "grade > 0",
        "gain": "linear: grade / max grade of the query",
        "unjudged": "non-relevant",
        "duplicates": "no extra credit",
        "ties": "backend order",
        "precision": "relevant hits / k",
        "ann_recall": "|top-k ∩ exact top-k| / |exact top-k|",
    }


def repeat_quality(
    rankings: Sequence[Sequence[str]],
    labels: Mapping[str, float],
    k: int,
    exact: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score each execution, then average within the query (never extra sample units)."""
    if not rankings:
        raise ValueError("Quality requires at least one recorded ranking")
    metrics = [semantic_metrics(ids, labels, k) if labels else None for ids in rankings]
    recalls = [ann_recall(ids, exact) if exact is not None else None for ids in rankings]
    return {
        "repeat_metrics": metrics,
        "metrics": {
            name: statistics.fmean(m[name] for m in metrics if m is not None)
            for name in SEMANTIC_METRICS
        }
        if labels
        else None,
        "repeat_ann_recall": recalls,
        "ann_recall": statistics.fmean(r for r in recalls if r is not None)
        if all(r is not None for r in recalls)
        else None,
    }


def semantic_metrics(found: Sequence[str], labels: Mapping[str, float], k: int) -> dict[str, float]:
    """Recall, precision, MRR and nDCG at ``k`` for one labelled query."""
    m = relevance_metrics(found, labels, k)
    relevant = {doc for doc, grade in labels.items() if grade > 0}
    hits = len(set(found[:k]) & relevant)
    return {"recall": m.recall, "precision": hits / k, "mrr": m.mrr, "ndcg": m.ndcg}


def ann_recall(found: Sequence[str], exact: Sequence[str]) -> float | None:
    """Share of the exact top-k found; ``None`` when the exact search returned nothing."""
    if not exact:
        return None
    return len(set(found) & set(exact)) / len(exact)


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float] | None:
    """p50 / p95 / mean / max of per-query latencies, or ``None`` with no measurements."""
    if not latencies_ms:
        return None
    x = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "mean": float(x.mean()),
        "max": float(x.max()),
    }
