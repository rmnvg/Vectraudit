"""Metric names in the notation other IR tools use, shown beside Vectraudit's own.

Vectraudit's metric keys (``ndcg``, ``precision``, ...) are what policies and run files use and
they never change. Reports and exports also show the notation of ir-measures, which pytrec_eval,
ranx and most retrieval research follow, so a number can be matched to the same measure in
another tool: ``nDCG@10``, ``P@10``, ``R@10``, ``RR@10`` (``MRR`` is its alias), ``AP@10``
(``MAP``) and ``Success@10`` (hit rate). A label of 1 or more counts as relevant, which is
ir-measures' default ``rel=1`` for integer grades; a measure at another threshold is written
``P(rel=2)@5``.

Credit: the notation is ir-measures' (Macdonald and MacAvaney, ECIR 2022). Display only; see
docs/credits.md.
"""

from __future__ import annotations

from collections.abc import Mapping

IR_MEASURES = {
    "recall": "R",
    "precision": "P",
    "mrr": "RR",
    "ndcg": "nDCG",
    "map": "AP",
    "hit_rate": "Success",
}


def notation(metric: str, k: int, *, rel: int | None = None) -> str | None:
    """``nDCG@10`` for ``("ndcg", 10)``; ``None`` for a metric with no ir-measures name.

    ``rel`` writes a relevance threshold other than the default, as ``P(rel=2)@5``.
    """
    base = IR_MEASURES.get(metric)
    if base is None:
        return None
    return f"{base}(rel={rel})@{k}" if rel is not None else f"{base}@{k}"


def label(metric: str, k: int, fallback: str | None = None) -> str:
    """The notation with Vectraudit's key beside it (``nDCG@10 (ndcg)``), or the fallback."""
    name = notation(metric, k)
    return f"{name} ({metric})" if name else (fallback or metric)


def notations(metrics: Mapping[str, object], k: int) -> dict[str, str]:
    """``{key: notation}`` for the metrics in ``metrics`` that have one."""
    return {m: n for m in metrics if (n := notation(m, k)) is not None}
