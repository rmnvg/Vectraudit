"""Spike (not shipped): which judgments settle a release decision first? Phase 46.0.

Synthetic baseline and candidate runs over the same questions. A reviewer judges the
unjudged pooled documents (the union of both runs' top k) one at a time and learns the truth;
after each judgment the verdict is recomputed on the labels judged so far (unjudged documents
count as not relevant, as every metric counts them). The verdict **stabilises** at the
judgment after which it never changes again. Three orders are compared:

- ``random``: the pool in random order;
- ``pool``: the current judgment queue, documents only one run returned first (pool
  disagreement), then the rest, random within each group;
- ``swing``: documents ranked by how far the difference between the runs could move if the
  document were relevant, recomputed for a question after each of its judgments. Documents
  both runs return at the same rank move no difference and come last;
- ``expected``: swing times the share of already judged documents at the document's best
  rank that are relevant (estimated from the labels given, never the truth), after minimal
  test collections (Carterette, Allan and Sitaraman, SIGIR 2006).

    uv run python scripts/decision_queue_spike.py [--seeds 1 2 3] [--scenarios 63]

Bar: the verdict stabilises with at most half the judgments random order needs (the median
ratio over scenarios whose verdict was not already settled before any judgment, for the
better of ``swing`` and ``expected``).

The rule has the default policy's shape: fail when mean nDCG@10 or recall@10 drops by more
than 0.01 or MRR by more than 0.02. Two starting points: labels pooled from the baseline only
(the candidate's new documents unjudged), and one known relevant document per question.
Metrics come from ``vectraudit.evaluation.metrics``, the product's own definitions.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vectraudit.evaluation.metrics import semantic_metrics

QUESTIONS = 80
POOL = 40
RELEVANT = 4
K = 10
RULES = {"ndcg": 0.01, "recall": 0.01, "mrr": 0.02}
EFFECTS = (-0.3, -0.2, -0.12, -0.06, 0.0, 0.06, 0.12)


@dataclass
class Scenario:
    grades: np.ndarray  # (Q, POOL) true grade 0-3
    base: list[list[str]]
    cand: list[list[str]]


def make(rng: np.random.Generator, effect: float) -> Scenario:
    grades = np.zeros((QUESTIONS, POOL), dtype=int)
    for q in range(QUESTIONS):
        grades[q, rng.choice(POOL, RELEVANT, replace=False)] = rng.integers(1, 4, RELEVANT)
    common = rng.normal(size=(QUESTIONS, POOL))
    rel = grades > 0
    b = rel * 1.4 + 0.8 * common + 0.7 * rng.normal(size=(QUESTIONS, POOL))
    c = rel * (1.4 + effect) + 0.8 * common + 0.7 * rng.normal(size=(QUESTIONS, POOL))

    def top(x: np.ndarray) -> list[list[str]]:
        return [[f"d{j}" for j in np.argsort(-row)[:K]] for row in x]

    return Scenario(grades, top(b), top(c))


def deltas(labels: dict[str, float], base: list[str], cand: list[str]) -> dict[str, float] | None:
    if not any(g > 0 for g in labels.values()):
        return None
    mb, mc = semantic_metrics(base, labels, K), semantic_metrics(cand, labels, K)
    return {m: mc[m] - mb[m] for m in RULES}


def verdict(per_q: list[dict[str, float] | None]) -> bool:
    measured = [d for d in per_q if d is not None]
    return all(
        statistics.fmean(d[m] for d in measured) >= -drop - 1e-9 for m, drop in RULES.items()
    )


Order = Callable[
    [np.random.Generator, Scenario, list[dict[str, float]], list[tuple[int, str]]],
    list[tuple[int, str]],
]


def swing(labels: dict[str, float], base: list[str], cand: list[str], doc: str) -> float:
    """How far this question's difference could move if ``doc`` were relevant (grade 3)."""
    now = deltas(labels, base, cand)
    then = deltas({**labels, doc: 3.0}, base, cand)
    assert then is not None
    if now is None:
        return max(abs(v) for v in then.values())
    return max(abs(then[m] - now[m]) / drop for m, drop in RULES.items())


def best_rank(s: Scenario, q: int, doc: str) -> int:
    ranks = [r.index(doc) for r in (s.base[q], s.cand[q]) if doc in r]
    return min(ranks)


def rank_prior(labels: list[dict[str, float]], s: Scenario) -> dict[int, float]:
    """Share of judged documents at each rank (best of both runs) that are relevant.

    Estimated from the labels already given, never from the truth; Laplace-smoothed toward
    the overall share so a rank with few judgments is not certain either way.
    """
    hits = np.zeros(K)
    seen = np.zeros(K)
    for q, known in enumerate(labels):
        for doc, grade in known.items():
            if doc in s.base[q] or doc in s.cand[q]:
                r = best_rank(s, q, doc)
                seen[r] += 1
                hits[r] += grade > 0
    overall = (hits.sum() + 1) / (seen.sum() + 2)
    return {r: float((hits[r] + 2 * overall) / (seen[r] + 2)) for r in range(K)}


def simulate(
    rng: np.random.Generator, s: Scenario, start: str, order: str
) -> tuple[int, int, bool]:
    """(judgments until the verdict stabilises, judgments available, settled before any)."""
    labels: list[dict[str, float]] = []
    pool: list[tuple[int, str]] = []
    for q in range(QUESTIONS):
        truth = {f"d{j}": float(g) for j, g in enumerate(s.grades[q])}
        docs = sorted(set(s.base[q]) | set(s.cand[q]), key=lambda d: int(d[1:]))
        if start == "baseline":
            known = {d: truth[d] for d in s.base[q]}
            known.update({d: g for d, g in truth.items() if g > 0 and d not in s.cand[q]})
        else:
            first = next(d for d, g in truth.items() if g > 0)
            known = {first: truth[first]}
        labels.append(known)
        pool += [(q, d) for d in docs if d not in known]
    per_q = [deltas(labels[q], s.base[q], s.cand[q]) for q in range(QUESTIONS)]
    history = [verdict(per_q)]
    remaining = list(pool)
    rng.shuffle(remaining)
    if order == "pool":
        remaining.sort(key=lambda x: (x[1] in s.base[x[0]]) and (x[1] in s.cand[x[0]]))
    prior = rank_prior(labels, s) if order in ("swing", "expected") else {}

    def score(x: tuple[int, str]) -> float:
        q, doc = x
        value = swing(labels[q], s.base[q], s.cand[q], doc)
        return value * prior[best_rank(s, q, doc)] if order == "expected" else value

    scores = {x: score(x) for x in remaining} if prior else {}
    while remaining:
        if scores:
            pick = max(range(len(remaining)), key=lambda i: scores[remaining[i]])
            q, doc = remaining.pop(pick)
        else:
            q, doc = remaining.pop(0)
        labels[q][doc] = float(s.grades[q, int(doc[1:])])
        per_q[q] = deltas(labels[q], s.base[q], s.cand[q])
        history.append(verdict(per_q))
        if scores:
            for x in remaining:
                if x[0] == q:
                    scores[x] = score(x)
    final = history[-1]
    stable = max((i for i, v in enumerate(history) if v != final), default=-1) + 1
    return stable, len(pool), stable == 0


def ratio(counts: dict[str, list[int]], order: str) -> float:
    """Median over scenarios of this order's judgments to stability over random order's."""
    pairs = zip(counts[order], counts["random"], strict=True)
    return statistics.median(n / max(1, r) for n, r in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--scenarios", type=int, default=63)
    args = parser.parse_args()
    print(
        "start      seed  unsettled  pool  random  pool  swing  expected  swing/r  expected/r  bar"
    )
    for start in ("baseline", "one-known"):
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            counts: dict[str, list[int]] = {o: [] for o in ("random", "pool", "swing", "expected")}
            sizes = []
            for i in range(args.scenarios):
                s = make(rng, EFFECTS[i % len(EFFECTS)])
                gen = seed * 1000 + i
                runs = {o: simulate(np.random.default_rng(gen), s, start, o) for o in counts}
                if runs["random"][2]:
                    continue  # already decided before any judgment: nothing to compare
                sizes.append(runs["random"][1])
                for o, (n, _, _) in runs.items():
                    counts[o].append(n)
            if not sizes:
                continue

            med = {o: statistics.median(v) for o, v in counts.items()}
            by_swing, by_expected = ratio(counts, "swing"), ratio(counts, "expected")
            best = min(by_swing, by_expected)
            print(
                f"{start:<10} {seed:>4}  {len(sizes):>9}  {statistics.median(sizes):>4.0f}  "
                f"{med['random']:>6.0f}  {med['pool']:>4.0f}  {med['swing']:>5.0f}  "
                f"{med['expected']:>8.0f}  {by_swing:>7.2f}  {by_expected:>10.2f}  "
                f"{'holds' if best <= 0.5 else 'missed'}"
            )


if __name__ == "__main__":
    main()
