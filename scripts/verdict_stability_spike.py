"""Spike (not shipped): can we predict whether a verdict survives label errors? Phase 45.0.

Synthetic baseline and candidate runs are scored against clean labels (the truth) and against
observed labels with planted errors at a known rate. The **true flip rate** is how often the
verdict on the observed labels differs from the verdict on the clean labels. The **predicted
flip probability** is what a reader could compute from the observed labels and a declared
error rate alone. Two predictors are tried:

- ``perturb`` (the plan's method): flip the observed labels at the declared rates many times
  and count how often the verdict changes. Rates are given per observed class (a relevant
  label is wrong; a judged not-relevant label is wrong; an unjudged retrieved document is
  relevant), as reviewing a sample of the file would measure them.
- ``simex`` (simulation-extrapolation, for measurement error): add noise at the declared rate
  to the observed labels again, take the change in the measured difference as the bias the
  noise caused, and extrapolate back: ``2 * delta(observed) - delta(noised again)``.

    uv run python scripts/verdict_stability_spike.py [--seeds 1 2 3] [--draws 150]

Bar: across noise rates of 2, 5 and 10%, the predicted flip probability is within 10 points
of the truth: per scenario (mean prediction against the true rate over many draws of observed
labels), and by calibration bins (draws grouped by their prediction, against how often their
verdict really flipped; bins with at least 30 draws).

The decision rule has the default policy's shape: fail when mean recall@10 drops by more than
0.02. Scenarios set the candidate's true change from well below to above that margin. Errors
are independent of retrieval here, friendlier than in real labels, so a method that misses on
these data will not do better on real ones.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

QUESTIONS = 100
POOL = 40  # documents a question could retrieve
RELEVANT = 4  # truly relevant documents per question
K = 10
MAX_DROP = 0.02
NOISE_RATES = (0.02, 0.05, 0.10)
SAMPLES = 100  # perturbations per prediction
EFFECTS = (-0.9, -0.6, -0.45, -0.35, -0.25, -0.15, -0.05, 0.1, 0.3)
MIN_BIN = 30
BAR = 0.10
ROWS = np.arange(QUESTIONS)[:, None]


@dataclass
class Scenario:
    truth: np.ndarray  # (Q, POOL) bool: truly relevant
    judged: np.ndarray  # (Q, POOL) bool: in the labelling pool
    base_top: np.ndarray  # (Q, K) document indices
    cand_top: np.ndarray  # (Q, K)
    retrieved: np.ndarray  # (Q, POOL) bool: in either top K


def make_scenario(rng: np.random.Generator, effect: float, pool_from: str) -> Scenario:
    truth = np.zeros((QUESTIONS, POOL), dtype=bool)
    for q in range(QUESTIONS):
        truth[q, rng.choice(POOL, RELEVANT, replace=False)] = True
    common = rng.normal(size=(QUESTIONS, POOL))
    base = truth * 1.6 + 0.8 * common + 0.6 * rng.normal(size=(QUESTIONS, POOL))
    cand = truth * (1.6 + effect) + 0.8 * common + 0.6 * rng.normal(size=(QUESTIONS, POOL))
    base_top = np.argsort(-base, axis=1)[:, :K]
    cand_top = np.argsort(-cand, axis=1)[:, :K]
    in_base = np.zeros_like(truth)
    in_base[ROWS, base_top] = True
    in_cand = np.zeros_like(truth)
    in_cand[ROWS, cand_top] = True
    # "both": every relevant document was judged. "baseline": labels pooled from the baseline,
    # so the candidate's new documents are unjudged (relevant documents neither run retrieved
    # were judged, found some other way).
    judged = in_base | in_cand | truth if pool_from == "both" else in_base | (truth & ~in_cand)
    return Scenario(truth, judged, base_top, cand_top, in_base | in_cand)


def delta(labels: np.ndarray, s: Scenario) -> np.ndarray:
    """Mean recall@K change for label matrices (..., Q, POOL); unlabelled questions skipped."""
    total = labels.sum(-1)
    ok = total > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        change = (labels[..., ROWS, s.cand_top].sum(-1) - labels[..., ROWS, s.base_top].sum(-1)) / (
            total
        )
    return np.where(ok, change, 0.0).sum(-1) / ok.sum(-1)


def passes(d: np.ndarray | float) -> np.ndarray:
    return np.asarray(d) >= -MAX_DROP - 1e-9


def observe(rng: np.random.Generator, s: Scenario, rate: float) -> np.ndarray:
    """Observed labels: each judged label is wrong with probability ``rate``; unjudged = 0."""
    flips = s.judged & (rng.random(s.truth.shape) < rate)
    return s.judged & (s.truth ^ flips)


def class_rates(observed: np.ndarray, s: Scenario) -> tuple[float, float, float]:
    """Error rates given what the file says, as a review of a sample of it would measure."""

    def share(mask: np.ndarray, want: bool) -> float:
        return float((s.truth[mask] == want).mean()) if mask.any() else 0.0

    return (
        share(s.judged & observed, False),
        share(s.judged & ~observed, True),
        share(~s.judged & s.retrieved, True),
    )


Predictor = Callable[[np.random.Generator, np.ndarray, Scenario, float, float], float]


def perturb(
    rng: np.random.Generator, observed: np.ndarray, s: Scenario, rate: float, scale: float
) -> float:
    rel, notrel, unj = (min(1.0, x * scale) for x in class_rates(observed, s))
    u = rng.random((SAMPLES, *observed.shape))
    flipped = (
        (observed & ~(s.judged & (u < rel)))
        | (s.judged & ~observed & (u < notrel))
        | (~s.judged & s.retrieved & (u < unj))
    )
    return float(np.mean(passes(delta(flipped, s)) != passes(delta(observed, s))))


def simex(
    rng: np.random.Generator, observed: np.ndarray, s: Scenario, rate: float, scale: float
) -> float:
    here = delta(observed, s)
    again = s.judged & (rng.random((SAMPLES, *observed.shape)) < rate * scale)
    corrected = 2 * here - delta(observed ^ again, s)
    return float(np.mean(passes(corrected) != passes(here)))


def evaluate(
    predictor: Predictor, seed: int, draws: int, pool_from: str, scale: float
) -> dict[float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    results: dict[float, dict[str, float]] = {}
    for rate in NOISE_RATES:
        scenario_gaps: list[float] = []
        preds: list[float] = []
        flipped: list[bool] = []
        for effect in EFFECTS:
            s = make_scenario(rng, effect, pool_from)
            clean = passes(delta(s.truth, s))
            p_here, f_here = [], []
            for _ in range(draws):
                observed = observe(rng, s, rate)
                p_here.append(predictor(rng, observed, s, rate, scale))
                f_here.append(bool(passes(delta(observed, s)) != clean))
            scenario_gaps.append(abs(float(np.mean(p_here)) - float(np.mean(f_here))))
            preds += p_here
            flipped += f_here
        p, f = np.asarray(preds), np.asarray(flipped)
        bins = np.minimum((p * 10).astype(int), 9)
        gaps = [
            abs(p[bins == b].mean() - f[bins == b].mean())
            for b in range(10)
            if (bins == b).sum() >= MIN_BIN
        ]
        results[rate] = {
            "scenario_max_gap": max(scenario_gaps),
            "scenario_mean_gap": float(np.mean(scenario_gaps)),
            "bin_max_gap": max(gaps),
            "true_flip_rate": float(f.mean()),
            "mean_prediction": float(p.mean()),
        }
    return results


VARIANTS = (
    ("both", 1.0, "wrong labels, declared rates as measured"),
    ("both", 0.5, "wrong labels, declared rates half the true ones"),
    ("both", 2.0, "wrong labels, declared rates twice the true ones"),
    ("baseline", 1.0, "wrong labels, and the candidate's new documents unjudged"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--draws", type=int, default=150)
    args = parser.parse_args()
    for name, predictor in (("perturb", perturb), ("simex", simex)):
        for pool_from, scale, title in VARIANTS:
            print(f"\n{name}: {title}")
            print("seed  noise  scen.max  scen.mean  bin.max  true  predicted  bar")
            for seed in args.seeds:
                for rate, r in evaluate(predictor, seed, args.draws, pool_from, scale).items():
                    ok = r["scenario_max_gap"] <= BAR and r["bin_max_gap"] <= BAR
                    print(
                        f"{seed:>4}  {rate:>5.2f}  {r['scenario_max_gap']:>8.3f}  "
                        f"{r['scenario_mean_gap']:>9.3f}  {r['bin_max_gap']:>7.3f}  "
                        f"{r['true_flip_rate']:>4.2f}  {r['mean_prediction']:>9.2f}  "
                        f"{'holds' if ok else 'missed'}"
                    )


if __name__ == "__main__":
    main()
