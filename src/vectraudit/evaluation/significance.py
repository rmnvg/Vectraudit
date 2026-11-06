"""Significance tests for comparing retrieval systems on shared questions.

Everything here works on **family means** (see :mod:`vectraudit.evaluation.families`), the
independent sampling unit of a comparison, and is seeded, so a report reproduces exactly.
The tests are information for a reader: they never change a release verdict.

- :func:`paired_randomization`: a sign-flip randomization test of the mean paired difference
  between two systems. Under the null hypothesis that the systems are interchangeable, each
  family's difference is as likely to be positive as negative. Two-sided, with
  ``p = (b + 1) / (n + 1)`` for ``b`` of ``n`` random sign assignments at least as extreme as
  the observed mean, so it is never 0. ranx's Fisher randomization test (Bassani, MIT)
  returns ``b / n``, which can be 0 and so overstates certainty; this differs on purpose.
- :func:`randomized_tukey_hsd`: Carterette, "Multiple testing in statistical analysis of
  systems-based information retrieval experiments", ACM TOIS 2012. For three or more
  systems, system labels are permuted within each family and the largest difference between
  any two system means is recorded; a pair's adjusted p-value is how often that maximum
  reaches the pair's observed difference. Comparing every pair to one null maximum controls
  the chance of any false winner across all pairs. It is paired by design: ranx's Tukey test
  calls SciPy's ``tukey_hsd``, which assumes independent groups and ignores that every
  system answered the same questions.
- :func:`holm_adjust`: Holm (1979) step-down adjustment, which controls the family-wise
  error at least as powerfully as Bonferroni.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vectraudit.evaluation.families import family_id, family_means

DEFAULT_RESAMPLES = 10_000
ALPHA = 0.05
MAX_CELLS = 2_000_000
"""Cap on the size of one batch of random draws, so memory stays bounded."""
_TOLERANCE = 1e-12
"""Float slack when asking whether a permuted statistic is at least the observed one."""


@dataclass(frozen=True)
class PairedTest:
    """Result of :func:`paired_randomization`."""

    p_value: float
    mean_delta: float
    n_families: int
    resamples: int


def paired_randomization(
    family_deltas: Sequence[float], n: int = DEFAULT_RESAMPLES, seed: int = 0
) -> PairedTest:
    """Two-sided sign-flip test that the mean of ``family_deltas`` is zero."""
    deltas = _finite(family_deltas, "family_deltas")
    if len(deltas) == 0:
        raise ValueError("family_deltas needs at least one family")
    _check_resamples(n)
    observed = abs(float(deltas.mean()))
    if not np.any(deltas):
        return PairedTest(1.0, 0.0, len(deltas), n)
    rng = np.random.default_rng(seed)
    batch = max(1, MAX_CELLS // len(deltas))
    extreme = 0
    for start in range(0, n, batch):
        size = min(batch, n - start)
        signs = rng.integers(0, 2, size=(size, len(deltas))) * 2 - 1
        means = np.abs((signs * deltas).mean(axis=1))
        extreme += int(np.count_nonzero(means >= observed - _TOLERANCE))
    return PairedTest((extreme + 1) / (n + 1), float(deltas.mean()), len(deltas), n)


def randomized_tukey_hsd(
    scores_by_system: Mapping[str, Sequence[float]],
    n: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Randomized Tukey HSD over systems scored on the same families.

    ``scores_by_system`` maps each system to its per-family mean scores, in the same family
    order for every system. Returns each system's mean, every pair's difference and adjusted
    p-value, and ``groups``: maximal sets of systems that cannot be told apart at ``alpha``,
    best system first, each system's ``letters`` (``"ab"`` when it belongs to groups a and b,
    as in a compact letter display), ``leader`` (the best mean) and ``indistinguishable``
    (those sharing a group with it).
    """
    if len(scores_by_system) < 3:
        raise ValueError("randomized Tukey HSD needs three or more systems; use a paired test")
    _check_resamples(n)
    names = list(scores_by_system)
    columns = [_finite(scores_by_system[name], name) for name in names]
    families = len(columns[0])
    if families == 0 or any(len(c) != families for c in columns):
        raise ValueError("every system needs one score per family, in the same families")
    scores = np.stack(columns, axis=1)  # (families, systems)
    means = scores.mean(axis=0)
    rng = np.random.default_rng(seed)
    spreads = np.empty(n)
    batch = max(1, MAX_CELLS // scores.size)
    for start in range(0, n, batch):
        size = min(batch, n - start)
        order = rng.random((size, *scores.shape)).argsort(axis=2)
        permuted = np.take_along_axis(np.broadcast_to(scores, (size, *scores.shape)), order, 2)
        column_means = permuted.mean(axis=1)
        spreads[start : start + size] = column_means.max(axis=1) - column_means.min(axis=1)
    pairs = []
    adjusted: dict[tuple[int, int], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = abs(float(means[i] - means[j]))
            p = (int(np.count_nonzero(spreads >= diff - _TOLERANCE)) + 1) / (n + 1)
            adjusted[(i, j)] = adjusted[(j, i)] = p
            pairs.append({"a": names[i], "b": names[j], "diff": diff, "p_adjusted": p})
    ranked = sorted(range(len(names)), key=lambda i: (-means[i], names[i]))
    groups = _groups(ranked, adjusted, alpha)
    leader = ranked[0]
    together = {i for g in groups if leader in g for i in g}
    letters = {
        names[i]: "".join(chr(ord("a") + k) for k, g in enumerate(groups) if i in g) for i in ranked
    }
    return {
        "method": "randomized_tukey_hsd",
        "alpha": alpha,
        "resamples": n,
        "n_families": families,
        "systems": {names[i]: float(means[i]) for i in ranked},
        "pairs": pairs,
        "groups": [[names[i] for i in g] for g in groups],
        "letters": letters,
        "leader": names[leader],
        "indistinguishable": [names[i] for i in ranked if i in together and i != leader],
        "p_vs_leader": {names[i]: (None if i == leader else adjusted[(leader, i)]) for i in ranked},
    }


def _groups(
    ranked: Sequence[int], adjusted: Mapping[tuple[int, int], float], alpha: float
) -> list[list[int]]:
    """Maximal runs of systems, best first, whose ends are not significantly different.

    With systems sorted by mean, a pair's difference (and so its p-value ordering) grows with
    the distance between them, so a run is valid exactly when its two ends are.
    """
    runs: list[tuple[int, int]] = []
    for start in range(len(ranked)):
        end = start
        while end + 1 < len(ranked) and adjusted[(ranked[start], ranked[end + 1])] >= alpha:
            end += 1
        runs.append((start, end))
    kept = [
        (s, e)
        for s, e in runs
        if not any((s2 <= s and e <= e2 and (s2, e2) != (s, e)) for s2, e2 in runs)
    ]
    return [list(ranked[s : e + 1]) for s, e in dict.fromkeys(kept)]


def hsd_for_runs(
    runs: Mapping[str, Mapping[str, Any]],
    metric: str,
    *,
    n: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Randomized Tukey HSD of ``metric`` across run records, on the questions all share.

    Each system's score is the mean of its questions within a family, then the families are
    the units, as in a release comparison. ``None`` when fewer than three runs or no question
    is scored by all of them.
    """
    if len(runs) < 3:
        return None
    scored = {
        name: {
            q["query_id"]: (family_id(q), float(q["metrics"][metric]))
            for q in run["queries"]
            if q.get("metrics") and q["metrics"].get(metric) is not None
        }
        for name, run in runs.items()
    }
    shared = sorted(set.intersection(*(set(s) for s in scored.values())))
    if not shared:
        return None
    first = scored[next(iter(scored))]
    scores: dict[str, list[float]] = {}
    for name, per_query in scored.items():
        scores[name] = family_means((first[q][0], per_query[q][1]) for q in shared)
    result = randomized_tukey_hsd(scores, n=n, seed=seed)
    result["metric"] = metric
    result["n_queries"] = len(shared)
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the order given.

    Sort ascending, multiply the i-th smallest of m by ``m - i`` (counting from 0), make the
    sequence non-decreasing and cap at 1. A hypothesis is rejected at level ``alpha`` when its
    adjusted p-value is below ``alpha``.
    """
    values = list(p_values)
    if any(not 0 <= p <= 1 or math.isnan(p) for p in values):
        raise ValueError("p-values must be between 0 and 1")
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running = 0.0
    for position, i in enumerate(order):
        running = max(running, min(1.0, (len(values) - position) * values[i]))
        adjusted[i] = running
    return adjusted


def _finite(values: Sequence[float], what: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{what} must be a one-dimensional list of finite numbers")
    return array


def _check_resamples(n: int) -> None:
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
