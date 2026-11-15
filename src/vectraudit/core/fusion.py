"""Rank fusion: combine several rankings of the same query into one.

The methods and normalisations follow ranx (Bassani, MIT), reimplemented here and checked
against ranx 0.3.21's outputs (``tests/fixtures/ranx/fusion.json``); no ranx code is used.

- **rrf**: reciprocal rank fusion, ``w / (rrf_k + rank)`` summed over stages (Cormack,
  Clarke and Büttcher, SIGIR 2009).
- **isr**: inverse square rank, ``w / rank²`` summed, then multiplied by the number of
  stages that returned the document (Mourão, Martins and Magalhães, 2015).
- **bordafuse**: Borda count (Aslam and Montague, SIGIR 2001). Over the union of candidates
  (n), rank r earns ``n - r + 1`` points and a document a stage did not return earns
  ``(n - len + 1) / 2``.
- **combsum**, **combmnz**, **combmax** (Fox and Shaw, TREC 1994): the sum of normalised
  scores, that sum times the number of stages returning the document, and the maximum.
- **wsum**: ``sum(w * normalised score)``; a document a stage did not return scores 0 there.

Rank methods use input order and ignore ``norm``. Score methods normalise each stage first
and need every hit to carry a score. ``weights`` multiply each stage's contribution (all
1 by default); a stage weighted 0 contributes nothing, not even candidates, which is what
the hybrid baseline has always done. Results are ordered by ``(-score, id)``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Literal, get_args

Ranking = Sequence[tuple[str, float | None]]
"""Ordered ``(doc id, score)`` pairs, best first. Scores are optional for rank methods."""

NormMethod = Literal["none", "min_max", "max", "sum", "zmuv", "rank", "borda"]
FusionMethod = Literal["rrf", "wsum", "combsum", "combmnz", "combmax", "isr", "bordafuse"]
NORM_METHODS: tuple[str, ...] = get_args(NormMethod)
FUSION_METHODS: tuple[str, ...] = get_args(FusionMethod)
RANK_METHODS = ("rrf", "isr", "bordafuse")
SCORE_METHODS = ("wsum", "combsum", "combmnz", "combmax")

_FLOOR = 1e-9


def normalize(
    ranking: Ranking, method: NormMethod, *, candidates: Sequence[str] | None = None
) -> list[tuple[str, float]]:
    """Normalise one ranking's scores, keeping its order.

    ``rank`` and ``borda`` use positions only. ``borda`` also scores the ``candidates`` the
    ranking did not return (the union over all fused rankings; defaults to the ranking's
    own ids). Denominators are floored at 1e-9, so a constant ranking normalises to 0
    (``min_max``, ``sum``, ``zmuv``).
    """
    if method not in NORM_METHODS:
        raise ValueError(f"Unknown normalisation {method!r}; choose from {NORM_METHODS}")
    items = _dedupe(ranking)
    n = len(items)
    if method == "rank":
        return [(doc, 1 - i / n) for i, (doc, _) in enumerate(items)]
    if method == "borda":
        seen = {doc for doc, _ in items}
        pool = list(dict.fromkeys(candidates)) if candidates is not None else list(seen)
        if not seen <= set(pool):
            raise ValueError("borda candidates must include every ranked document")
        if not pool:
            return []
        total = len(pool)
        ranked = [(doc, 1 - i / total) for i, (doc, _) in enumerate(items)]
        missing = 0.5 - (n - 1) / (2 * total)
        return ranked + [(doc, missing) for doc in pool if doc not in seen]
    scores = _scores(items, method)
    if method == "none" or not scores:
        return list(zip([d for d, _ in items], scores, strict=True))
    if method == "min_max":
        low = min(scores)
        span = max(max(scores) - low, _FLOOR)
        values = [(s - low) / span for s in scores]
    elif method == "max":
        top = max(max(scores), _FLOOR)
        values = [s / top for s in scores]
    elif method == "sum":
        low = min(scores)
        span = max(sum(scores) - low * n, _FLOOR)
        values = [(s - low) / span for s in scores]
    else:  # zmuv
        mean = statistics.fmean(scores)
        spread = max(statistics.pstdev(scores), _FLOOR)
        values = [(s - mean) / spread for s in scores]
    return list(zip([d for d, _ in items], values, strict=True))


def fuse(
    rankings: Mapping[str, Ranking],
    method: FusionMethod,
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    norm: NormMethod = "min_max",
    depth: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse named stage rankings into one ``(doc id, score)`` ranking.

    ``depth`` keeps only each input's top ``depth`` hits before fusing. A document listed
    twice in one ranking keeps its first (best) position. Raises ``ValueError`` for an
    unknown method or normalisation, invalid weights or ``rrf_k``, or a score method given
    a hit without a finite score.
    """
    if method not in FUSION_METHODS:
        raise ValueError(f"Unknown fusion method {method!r}; choose from {FUSION_METHODS}")
    if norm not in NORM_METHODS:
        raise ValueError(f"Unknown normalisation {norm!r}; choose from {NORM_METHODS}")
    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError("rrf_k must be a non-negative integer")
    if depth is not None and (type(depth) is not int or depth < 1):
        raise ValueError("depth must be a positive integer")
    if not rankings:
        raise ValueError("fuse needs at least one ranking")
    w = _weights(rankings, weights)
    stages = {name: _dedupe(ranking)[:depth] for name, ranking in rankings.items() if w[name] != 0}
    fused: dict[str, float] = {}
    if method in RANK_METHODS:
        pool = list(dict.fromkeys(doc for items in stages.values() for doc, _ in items))
        counts: dict[str, int] = {}
        for name, items in stages.items():
            for rank, (doc, _) in enumerate(items, 1):
                if method == "rrf":
                    part = w[name] / (rrf_k + rank)
                elif method == "isr":
                    part = w[name] / rank**2
                else:
                    part = w[name] * (len(pool) - rank + 1)
                fused[doc] = fused.get(doc, 0) + part
                counts[doc] = counts.get(doc, 0) + 1
            if method == "bordafuse":
                listed = {doc for doc, _ in items}
                unranked = w[name] * (len(pool) - len(items) + 1) / 2
                for doc in pool:
                    if doc not in listed:
                        fused[doc] = fused.get(doc, 0) + unranked
        if method == "isr":
            fused = {doc: score * counts[doc] for doc, score in fused.items()}
        return _ordered(fused)

    for items in stages.values():
        _scores(items, method)  # refuse missing scores before any normalisation
    pool = list(dict.fromkeys(doc for items in stages.values() for doc, _ in items))
    normed = {
        name: normalize(items, norm, candidates=pool if norm == "borda" else None)
        for name, items in stages.items()
    }
    if method == "combmax":
        for name, scored in normed.items():
            for doc, score in scored:
                part = w[name] * score
                fused[doc] = max(fused[doc], part) if doc in fused else part
        return _ordered(fused)
    counts = {}
    for name, scored in normed.items():
        for doc, score in scored:
            fused[doc] = fused.get(doc, 0) + w[name] * score
            # As in ranx, Borda normalisation gives every candidate a score in every stage,
            # so CombMNZ then counts every stage.
            counts[doc] = counts.get(doc, 0) + 1
    if method == "combmnz":
        fused = {doc: score * counts.get(doc, 0) for doc, score in fused.items()}
    return _ordered(fused)


def _dedupe(ranking: Ranking) -> list[tuple[str, float | None]]:
    seen: set[str] = set()
    out = []
    for doc, score in ranking:
        if doc not in seen:
            seen.add(doc)
            out.append((doc, score))
    return out


def _scores(items: Sequence[tuple[str, float | None]], method: str) -> list[float]:
    values = []
    for doc, score in items:
        if score is None or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError(
                f"{method!r} needs a finite score for every hit, and {doc!r} has none; "
                "use 'rrf' or 'bordafuse', which use ranks only"
            )
        values.append(float(score))
    return values


def _weights(
    rankings: Mapping[str, Ranking], weights: Mapping[str, float] | None
) -> dict[str, float]:
    if weights is None:
        return dict.fromkeys(rankings, 1.0)
    if set(weights) != set(rankings):
        raise ValueError("weights must name exactly the fused stages")
    out = {}
    for name in rankings:
        value = weights[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("weights must be numbers")
        if not math.isfinite(value) or value < 0:
            raise ValueError("weights must be finite and non-negative")
        out[name] = value
    if not any(out.values()):
        raise ValueError("at least one weight must be positive")
    return out


def _ordered(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))
