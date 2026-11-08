"""Judge-corrected estimates: prediction-powered inference for a rate or a mean.

A team often has a model judge's verdict on every question and human verdicts on a few. The
judge-only rate is cheap but biased by the judge's errors; the human-only rate is unbiased but
noisy. Prediction-powered inference combines them: it starts from the judge's mean over the
questions humans did not review, then *rectifies* it with the judge's measured error on the
questions humans did review. The interval is valid however wrong the judge is, because the
correction is measured, and it is narrower than the human-only interval when the judge is
informative.

For a mean, with ``n`` labelled units (``Y`` human, ``f`` judge) and ``N`` unlabelled units
(``f`` only), the PPI++ estimator is::

    theta(lambda) = mean(Y) + lambda * (mean(f_unlabelled) - mean(f_labelled))

with variance ``lambda^2 Var(f_unlabelled) / N + Var(Y - lambda f_labelled) / n`` and interval
``theta +- z_{1 - alpha/2} * sqrt(variance)``. ``lambda = 1`` is plain PPI and ``lambda = 0`` the
classical human-only estimate. The power-tuned weight, estimated from the data, is::

    lambda = Cov(Y, f_labelled) / ((1 + n/N) * Var(f_pooled))

For a mean the objective is convex for every ``lambda``, so no clipping is needed (PPI++,
Example 6.1). The interval is valid only when the labelled units are a **uniform random sample**
of the units the judge scored; that is the caller's contract (see
:mod:`vectraudit.evaluation.review_sample`), and nothing here can check it.

Units are whatever the caller passes. Callers that have paraphrase families pass family means,
so a family cannot shrink the interval.

Credit: Angelopoulos, Bates, Fannjiang, Jordan and Zrnic, "Prediction-powered inference"
(Science, 2023); Angelopoulos, Duchi and Zrnic, "PPI++: Efficient prediction-powered inference"
(2023); and ARES (Saad-Falcon, Khattab, Potts and Zaharia, NAACL 2024), which applies it to
retrieval-augmented generation. Our own numpy implementation; see docs/credits.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt

MIN_LABELLED = 30
WARN_LABELLED = 100


class PPIError(ValueError):
    """The estimate cannot be computed; ``hint`` says what to change."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided interval."""

    estimate: float
    ci_low: float
    ci_high: float
    n: int
    method: str

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "method": self.method,
        }


@dataclass(frozen=True)
class PPIEstimate(Interval):
    """A prediction-powered estimate."""

    n_unlabelled: int = 0
    lam: float = 1.0
    """The weight on the judge's correction (0 = human-only, 1 = plain PPI)."""
    alpha: float = 0.05
    clipped: bool = False
    """The interval was cut to ``[0, 1]`` (a rate cannot leave it)."""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            **super().as_dict(),
            "n_unlabelled": self.n_unlabelled,
            "lambda": self.lam,
            "alpha": self.alpha,
            "clipped": self.clipped,
            "warnings": list(self.warnings),
        }


def _array(values: Sequence[float] | npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise PPIError(f"{name} must be a 1-D list of numbers.", "Pass one value per unit.")
    if not np.isfinite(arr).all():
        raise PPIError(f"{name} contains NaN or infinite values.", "Drop or fix those units.")
    return arr


def _z(alpha: float) -> float:
    if not 0 < alpha < 1:
        raise PPIError(f"alpha must be between 0 and 1, got {alpha}.", "Use e.g. 0.05.")
    return NormalDist().inv_cdf(1 - alpha / 2)


def _variance(x: npt.NDArray[np.float64]) -> float:
    return float(np.var(x, ddof=1)) if len(x) > 1 else 0.0


def classical_mean(
    values: Sequence[float] | npt.ArrayLike, *, alpha: float = 0.05, bounded: bool = True
) -> Interval:
    """The human-only mean with a normal interval (``bounded``: cut to ``[0, 1]``).

    The same convention as :func:`ppi_mean` (a z interval on the unit means), so the two are
    comparable width for width.
    """
    y = _array(values, "values")
    if len(y) < 2:
        raise PPIError("Need at least 2 units for an interval.", "Review more questions.")
    half = _z(alpha) * math.sqrt(_variance(y) / len(y))
    mean = float(y.mean())
    low, high = mean - half, mean + half
    if bounded:
        low, high = max(0.0, low), min(1.0, high)
    return Interval(mean, low, high, len(y), "human-only")


def ppi_mean(
    human: Sequence[float] | npt.ArrayLike,
    judge_on_sample: Sequence[float] | npt.ArrayLike,
    judge_on_rest: Sequence[float] | npt.ArrayLike,
    *,
    alpha: float = 0.05,
    power_tuning: bool = True,
    bounded: bool = True,
    min_labelled: int = MIN_LABELLED,
) -> PPIEstimate:
    """The PPI (``power_tuning=False``) or PPI++ estimate of a mean with its interval.

    ``human`` and ``judge_on_sample`` are the human and judge values on the same labelled
    units, in the same order; ``judge_on_rest`` is the judge's value on the units without a
    human value. ``bounded`` cuts the interval to ``[0, 1]``, for rates.

    Raises :class:`PPIError` below ``min_labelled`` labelled units (an interval from so few is
    not worth reading) and warns below :data:`WARN_LABELLED`.
    """
    y = _array(human, "human")
    f_lab = _array(judge_on_sample, "judge_on_sample")
    f_rest = _array(judge_on_rest, "judge_on_rest")
    if len(y) != len(f_lab):
        raise PPIError(
            f"{len(y)} human values but {len(f_lab)} judge values on the same units.",
            "The two lists must describe the same labelled units in the same order.",
        )
    n, big_n = len(y), len(f_rest)
    if n < min_labelled:
        raise PPIError(
            f"Only {n} human-reviewed units; at least {min_labelled} are needed.",
            "Review more questions (a random sample), or use the human-only estimate.",
        )
    if big_n < 1:
        raise PPIError(
            "No units without a human review, so there is nothing to correct.",
            "Use the human-only estimate: every unit already has a human verdict.",
        )
    z = _z(alpha)
    lam = 1.0
    if power_tuning:
        pooled = _variance(np.concatenate([f_lab, f_rest]))
        if pooled > 0 and n > 1:
            covariance = float(np.cov(y, f_lab, ddof=1)[0, 1])
            lam = covariance / ((1 + n / big_n) * pooled)
        else:
            lam = 0.0  # a judge that never varies says nothing: fall back to human-only
    estimate = float(y.mean() + lam * (f_rest.mean() - f_lab.mean()))
    variance = lam**2 * _variance(f_rest) / big_n + _variance(y - lam * f_lab) / n
    half = z * math.sqrt(variance)
    low, high = estimate - half, estimate + half
    clipped = False
    if bounded:
        clipped = low < 0.0 or high > 1.0
        low, high = max(0.0, low), min(1.0, high)
        estimate_out = min(1.0, max(0.0, estimate))
    else:
        estimate_out = estimate
    warnings = []
    if n < WARN_LABELLED:
        warnings.append(
            f"Only {n} human-reviewed units: the interval is approximate below {WARN_LABELLED}."
        )
    return PPIEstimate(
        estimate=estimate_out,
        ci_low=low,
        ci_high=high,
        n=n,
        method="ppi++" if power_tuning else "ppi",
        n_unlabelled=big_n,
        lam=lam,
        alpha=alpha,
        clipped=clipped,
        warnings=tuple(warnings),
    )


def corrected_rate(
    human: Sequence[float] | npt.ArrayLike,
    judge_on_sample: Sequence[float] | npt.ArrayLike,
    judge_on_rest: Sequence[float] | npt.ArrayLike,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The three numbers a report shows side by side for one judge.

    ``judge_only`` is the judge's mean over every unit, labelled or not: cheap, and biased by
    the judge's own error. ``human_only`` uses only the human-reviewed units. ``corrected`` is
    the PPI++ estimate. The agreement is the share of reviewed units where the judge and the
    human give the same 0/1 value (only meaningful for 0/1 values).
    """
    y = _array(human, "human")
    f_lab = _array(judge_on_sample, "judge_on_sample")
    f_rest = _array(judge_on_rest, "judge_on_rest")
    pooled = np.concatenate([f_lab, f_rest])
    return {
        "judge_only": {
            "estimate": float(pooled.mean()),
            "n": len(pooled),
            "note": "The judge's mean over every unit; biased by the judge's own error.",
        },
        "human_only": classical_mean(y, alpha=alpha).as_dict(),
        "corrected": ppi_mean(y, f_lab, f_rest, alpha=alpha).as_dict(),
        "agreement_on_sample": float((np.round(y) == np.round(f_lab)).mean()),
    }
