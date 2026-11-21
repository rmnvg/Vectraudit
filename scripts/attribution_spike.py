"""Spike (not shipped): which of several changes caused the regression? Phase 47.0.

A synthetic pipeline has settings that a release changes together: an embedding model (the
planted culprit: weaker on a domain that is 40% of the questions), a chunk size and a reranker
(each a small gain alone, a loss together: the planted interaction), and a prompt (no effect on
retrieval). Every combination of settings is run on the same questions; each question's recall@10
depends on the settings and on a little per-run noise.

Each setting's contribution to the difference between baseline and candidate is its Shapley
value over the measured combinations (per question, then averaged, so the contributions add up
to the measured difference exactly), with a 95% paired bootstrap interval over questions. A pair
is reported as interacting when its interaction (the effect of changing both minus the effects
of changing each, averaged over the other settings) is shown to be larger than ``MIN_EFFECT``:
a sign-flip test of the interaction moved ``MIN_EFFECT`` toward zero, Holm-adjusted across the
pairs tested. Two weaker rules were tried first and report extra pairs (``--test bootstrap``
keeps one: a Bonferroni bootstrap interval that excludes zero, with a point estimate of at
least ``MIN_EFFECT``): tiny real interactions, overestimated by chance, pass them. Two designs:

- ``full``: all 2^n combinations (16 runs for four settings);
- ``screen``: the baseline, the candidate, each setting alone from the baseline (one at a time)
  and each setting reverted from the candidate (leave one out): 2n + 2 runs. The average of a
  setting's two effects is its Shapley value exactly when no three settings interact. A setting
  whose two effects differ by more than ``MIN_EFFECT`` (the same test, across settings)
  interacts with something; each pair of flagged settings then gets one more run (both from the
  baseline) to find the pair.

    uv run python scripts/attribution_spike.py [--seeds 1 2 ... ] [--questions 200]

Bar: in every seed and both designs the culprit is ranked first (the most negative
contribution), the planted pair is reported as interacting and no other pair is, and the
culprit's interval contains its true contribution (from 20,000 questions) in at least 90% of
seeds. A confounded case (the embedding model and the prompt can only change together) must
come out as one group, never as two shares.
"""

from __future__ import annotations

import argparse
import itertools
import math
import zlib
from collections.abc import Sequence

import numpy as np

from vectraudit.evaluation.significance import holm_adjust, paired_randomization

POOL = 40
RELEVANT = 4
K = 10
SETTINGS = ("embedding", "chunk_size", "reranker", "prompt")
CULPRIT = "embedding"
PAIR = ("chunk_size", "reranker")
MIN_EFFECT = 0.01
BOOT = 2000
CONFIG = {"test": "randomization"}


def recall(rng_seed: int, n: int, on: frozenset[str], noise_key: int) -> np.ndarray:
    """Per-question recall@K of the pipeline with ``on`` settings changed to the candidate's.

    The questions, documents and base scores come from ``rng_seed`` (the same for every
    combination: paired); ``noise_key`` adds a small per-run jitter.
    """
    rng = np.random.default_rng(rng_seed)
    truth = np.zeros((n, POOL), dtype=bool)
    for q in range(n):
        truth[q, rng.choice(POOL, RELEVANT, replace=False)] = True
    common = rng.normal(size=(n, POOL))
    own = rng.normal(size=(n, POOL))
    domain = rng.random(n) < 0.4
    signal = np.full(n, 1.5)
    if "embedding" in on:
        signal = signal - 0.7 * domain
    if "chunk_size" in on:
        signal += 0.12
    if "reranker" in on:
        signal += 0.12
    if "chunk_size" in on and "reranker" in on:
        signal -= 0.55
    jitter = np.random.default_rng([rng_seed, noise_key]).normal(size=(n, POOL))
    score = truth * signal[:, None] + 0.8 * common + 0.6 * own + 0.1 * jitter
    top = np.argsort(-score, axis=1)[:, :K]
    return truth[np.arange(n)[:, None], top].sum(1) / RELEVANT


def measure(
    seed: int, n: int, combos: Sequence[frozenset[str]]
) -> dict[frozenset[str], np.ndarray]:
    return {c: recall(seed, n, c, zlib.crc32(",".join(sorted(c)).encode())) for c in combos}


def shapley_full(
    v: dict[frozenset[str], np.ndarray], players: Sequence[str]
) -> dict[str, np.ndarray]:
    """Exact Shapley per question; a player may be a group of settings joined by ``+``."""
    n = len(players)
    out = {}
    for p in players:
        others = [x for x in players if x != p]
        total = 0.0
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for s in itertools.combinations(others, r):
                total = total + w * (v[_set(s, p)] - v[_set(s)])
        out[p] = np.asarray(total)
    return out


def _set(players: Sequence[str], extra: str | None = None) -> frozenset[str]:
    names = [*players, *([extra] if extra else [])]
    return frozenset(x for p in names for x in p.split("+"))


def interactions_full(
    v: dict[frozenset[str], np.ndarray], players: Sequence[str]
) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for a, b in itertools.combinations(players, 2):
        others = [x for x in players if x not in (a, b)]
        terms = [
            v[_set(s, a) | _set((), b)] - v[_set(s, a)] - v[_set(s, b)] + v[_set(s)]
            for r in range(len(others) + 1)
            for s in itertools.combinations(others, r)
        ]
        out[(a, b)] = np.mean(terms, axis=0)
    return out


def interval(x: np.ndarray, rng: np.random.Generator, alpha: float = 0.05) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def reported(effects: dict, seed: int) -> set:
    """Keys whose effect is shown larger than ``MIN_EFFECT``: sign-flip tests, Holm across keys."""
    keys = list(effects)
    if CONFIG["test"] == "bootstrap":
        rng = np.random.default_rng(seed)
        out = set()
        for key in keys:
            lo, hi = interval(effects[key], rng, 0.05 / len(keys))
            if (lo > 0 or hi < 0) and abs(float(effects[key].mean())) >= MIN_EFFECT:
                out.add(key)
        return out
    # Material: the effect is shown to be larger than MIN_EFFECT, not merely nonzero, by
    # testing the effect moved MIN_EFFECT toward zero.
    ps = [
        paired_randomization(
            (effects[key] - np.sign(effects[key].mean()) * MIN_EFFECT).tolist(), seed=seed
        ).p_value
        for key in keys
    ]
    adjusted = holm_adjust(ps)
    return {
        key
        for key, p in zip(keys, adjusted, strict=True)
        if p < 0.05 and abs(float(effects[key].mean())) >= MIN_EFFECT
    }


def attribute_full(seed: int, n: int, players: Sequence[str]) -> tuple[dict, dict, int]:
    combos = [_set(s) for r in range(len(players) + 1) for s in itertools.combinations(players, r)]
    v = measure(seed, n, combos)
    return shapley_full(v, players), interactions_full(v, players), len(combos)


def attribute_screen(seed: int, n: int, players: Sequence[str]) -> tuple[dict, dict, int]:
    all_on = _set(players)
    combos = {frozenset(), all_on}
    combos |= {_set((p,)) for p in players} | {all_on - _set((p,)) for p in players}
    v = measure(seed, n, sorted(combos, key=sorted))
    contrib, gaps = {}, {}
    for p in players:
        alone = v[_set((p,))] - v[frozenset()]
        last = v[all_on] - v[all_on - _set((p,))]
        contrib[p] = (alone + last) / 2
        gaps[p] = last - alone
    flagged = [p for p in players if p in reported(gaps, seed)]
    pairs = [frozenset(_set((a, b))) for a, b in itertools.combinations(flagged, 2)]
    v.update(measure(seed, n, pairs))
    inter = {
        (a, b): v[_set((a, b))] - v[_set((a,))] - v[_set((b,))] + v[frozenset()]
        for a, b in itertools.combinations(flagged, 2)
    }
    return contrib, inter, len(combos) + len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--test", choices=("randomization", "bootstrap"), default=CONFIG["test"])
    args = parser.parse_args()
    CONFIG["test"] = args.test
    truth, _, _ = attribute_full(999, 20_000, SETTINGS)
    true_value = {p: float(x.mean()) for p, x in truth.items()}
    print("true contributions:", {p: round(x, 4) for p, x in true_value.items()})
    for design, run in (("full", attribute_full), ("screen", attribute_screen)):
        first = pair_found = clean = covered = 0
        runs = 0
        for seed in args.seeds:
            rng = np.random.default_rng(seed + 7)
            contrib, inter, runs = run(seed, args.questions, SETTINGS)
            means = {p: float(x.mean()) for p, x in contrib.items()}
            first += min(means, key=lambda p: means[p]) == CULPRIT
            pairs = reported(inter, seed)
            pair_found += PAIR in pairs
            clean += pairs <= {PAIR}
            lo, hi = interval(contrib[CULPRIT], rng)
            covered += lo <= true_value[CULPRIT] <= hi
            if seed == args.seeds[0]:
                total = sum(means.values())
                print(
                    f"\n{design} ({runs} runs), seed {seed}: difference {total:+.4f}; "
                    + ", ".join(f"{p} {m:+.4f}" for p, m in means.items())
                    + f"; interacting: {sorted(pairs) or 'none'}"
                )
        s = len(args.seeds)
        ok = first == s and pair_found == s and clean == s and covered >= 0.9 * s
        print(
            f"{design}: culprit first {first}/{s}, pair reported {pair_found}/{s}, "
            f"no other pair {clean}/{s}, culprit interval covers truth {covered}/{s}: "
            f"{'holds' if ok else 'missed'}"
        )
    contrib, inter, runs = attribute_full(1, args.questions, ("embedding+prompt", *PAIR))
    print(
        f"\nconfounded (embedding and prompt change together, {runs} runs): "
        + ", ".join(f"{p} {float(x.mean()):+.4f}" for p, x in contrib.items())
    )


if __name__ == "__main__":
    main()
