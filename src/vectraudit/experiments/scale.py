"""Project recall@k measured on a corpus sample to the full corpus, with an interval.

Recall on a sample is inflated: fewer documents compete with each relevant one. The sample
always holds every labelled document and a known share ``g`` of the others, drawn uniformly
at random. For each pair of a question and one of its relevant documents:

- ``a``: labelled documents scoring above the relevant one (present at every size);
- ``u``: unlabelled documents in the sample scoring above it, a thinned draw of the pair's
  full-corpus count ``U`` (each unlabelled document is in the sample with chance ``g``).

The relevant document is in the full corpus's top ``k`` when ``a + U < k``. ``U`` is modelled
as Poisson with a rate drawn from a mixing distribution over pairs, estimated without a
parametric form (the nonparametric maximum likelihood estimate on a grid, by EM; Kiefer and
Wolfowitz 1956, Koenker and Mizera 2014). Given ``u``, the unsampled count ``U - u`` is
Poisson with rate ``(1 - g) * lambda`` under the pair's posterior, which gives each pair's
chance of staying in the top ``k``. The projection is their weighted mean. The interval
refits the mixing distribution on resampled families (each refit starting from a uniform
prior) and draws every pair's outcome, so it covers the full-corpus recall of these questions.

Validated in the Phase 48.0 spike (``scripts/corpus_projection_spike.py``,
``docs/DECISIONS.md``): from samples of 10% of the corpus the 95% interval held the measured
full-corpus recall@10 in 18 to 20 of 20 sample draws, on two synthetic corpora and one real
one, with two embedders each.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

Ints = npt.NDArray[np.int64]
Floats = npt.NDArray[np.float64]

GRID = 160
EM_STEPS = 400
DRAWS = 2000
OUTCOMES_PER_REFIT = 10
ALPHA = 0.05


@dataclass(frozen=True)
class Thinned:
    """Counts for every (question, relevant document) pair at one sample size."""

    a: Ints
    """Labelled documents scoring above the relevant one."""
    u: Ints
    """Unlabelled documents in the sample scoring above it."""
    g: float
    """Chance that an unlabelled document of the full corpus is in the sample."""
    unit: Ints
    """Resampling unit (query family) of each pair, numbered from 0."""
    weight: Floats
    """Each pair's weight in the mean: equal per family, then per question, then per pair."""

    def __post_init__(self) -> None:
        n = len(self.a)
        if not (len(self.u) == len(self.unit) == len(self.weight) == n) or n == 0:
            raise ValueError("a, u, unit and weight need one entry per pair, at least one")
        if not 0 < self.g <= 1:
            raise ValueError("g must be in (0, 1]")
        if np.any(self.a < 0) or np.any(self.u < 0):
            raise ValueError("counts must be non-negative")


def rate_grid(unlabelled: int, size: int = GRID) -> Floats:
    """Poisson rates from 0 to the number of unlabelled documents, log-spaced."""
    top = max(10.0, float(unlabelled))
    return np.concatenate([[0.0], np.geomspace(0.01, top, size - 1)])


def _likelihood(u: Ints, g: float, grid: Floats) -> Floats:
    lam = np.maximum(g * grid, 1e-300)
    lgam = np.array([math.lgamma(int(v) + 1) for v in u])[:, None]
    log = u[:, None] * np.log(lam)[None, :] - lam[None, :] - lgam
    return np.asarray(np.exp(log))


def fit_mixing(u: Ints, g: float, grid: Floats, steps: int = EM_STEPS) -> Floats:
    """Weights of the mixing distribution on ``grid`` (NPMLE by EM, from a uniform start)."""
    like = _likelihood(u, g, grid)
    w = np.full(len(grid), 1.0 / len(grid))
    for _ in range(steps):
        post = like * w
        post /= post.sum(axis=1, keepdims=True) + 1e-300
        w = post.mean(axis=0)
    return np.asarray(w)


def _below(room: Ints, lam: Floats) -> Floats:
    """P(V < room) for V ~ Poisson(lam): one row per pair, one column per rate."""
    out = np.zeros((len(room), len(lam)))
    term = np.repeat(np.exp(-lam)[None, :], len(room), axis=0)
    for v in range(max(0, int(room.max()))):
        out += np.where((v < room)[:, None], term, 0.0)
        term = term * lam[None, :] / (v + 1)
    return out


def stay_chances(t: Thinned, w: Floats, grid: Floats, k: int) -> Floats:
    """Each pair's posterior chance that its relevant document is in the full top ``k``."""
    post = _likelihood(t.u, t.g, grid) * w
    post /= post.sum(axis=1, keepdims=True) + 1e-300
    room = (k - t.a - t.u).astype(np.int64)
    return np.asarray((post * _below(room, (1 - t.g) * grid)).sum(axis=1))


def measured(t: Thinned, k: int) -> float:
    """Recall@k on the sample itself."""
    return float(np.sum(t.weight * ((t.a + t.u) < k)))


def project(
    thinned: Sequence[Thinned],
    k: int,
    unlabelled: int,
    *,
    seed: int = 0,
    draws: int = DRAWS,
) -> list[dict[str, Any]]:
    """Project several candidates' full-corpus recall@k, with paired draws.

    Every candidate must have the same pairs, units and weights (the same labelled
    questions). Draws share the resampled units and the random numbers deciding each pair's
    outcome, so they compare candidates as well as bound each one. Returns per candidate the
    point projection, the 95% interval and the draws themselves.
    """
    if not thinned:
        raise ValueError("nothing to project")
    first = thinned[0]
    for t in thinned[1:]:
        if not (np.array_equal(t.unit, first.unit) and np.array_equal(t.weight, first.weight)):
            raise ValueError("every candidate must be projected on the same pairs")
    grid = rate_grid(unlabelled)
    rng = np.random.default_rng(seed)
    units = int(first.unit.max()) + 1
    members = [np.flatnonzero(first.unit == j) for j in range(units)]
    refits = max(1, draws // OUTCOMES_PER_REFIT)
    picks = [rng.integers(0, units, units) for _ in range(refits)]
    uniforms = rng.random((refits, OUTCOMES_PER_REFIT, len(first.a)))
    out = []
    for t in thinned:
        chances = stay_chances(t, fit_mixing(t.u, t.g, grid), grid, k)
        values = []
        for pick, uniform in zip(picks, uniforms, strict=True):
            rows = np.concatenate([members[j] for j in pick])
            refit = fit_mixing(t.u[rows], t.g, grid)
            p = stay_chances(t, refit, grid, k)
            values.append((uniform < p) @ t.weight)
        sample = np.concatenate(values)
        out.append(
            {
                "recall": float(chances @ t.weight),
                "interval": [
                    float(np.quantile(sample, ALPHA / 2)),
                    float(np.quantile(sample, 1 - ALPHA / 2)),
                ],
                "draws": sample,
            }
        )
    return out
