"""Prediction-powered inference: valid intervals however wrong the judge, tighter when it helps."""

import math

import numpy as np
import pytest

from vectraudit.evaluation import ppi as P

TRUE_RATE = 0.62


def population(rng, n_total, *, flip=0.10, informative=True):
    """Human truth Y (0/1) and a judge f that is Y flipped with probability ``flip``, biased up."""
    y = (rng.random(n_total) < TRUE_RATE).astype(float)
    if informative:
        f = np.where(rng.random(n_total) < flip, 1 - y, y)
        f = np.where((f == 0) & (rng.random(n_total) < 0.10), 1.0, f)  # a lenient judge
    else:
        f = (rng.random(n_total) < 0.5).astype(float)
    return y, f


def draw(rng, n_label=150, n_total=2000, **kw):
    y, f = population(rng, n_total, **kw)
    idx = rng.permutation(n_total)
    lab, rest = idx[:n_label], idx[n_label:]
    return y[lab], f[lab], f[rest], float(y.mean())


# --- the estimator --------------------------------------------------------------------


def test_the_point_estimate_matches_the_formula() -> None:
    y = np.array([1, 0, 1, 1, 0, 1] * 6, dtype=float)
    f_lab = np.array([1, 0, 0, 1, 0, 1] * 6, dtype=float)
    f_rest = np.array([1, 1, 0, 1, 0, 1, 1, 0] * 12, dtype=float)
    plain = P.ppi_mean(y, f_lab, f_rest, power_tuning=False)
    assert plain.lam == 1.0
    assert plain.estimate == pytest.approx(y.mean() + (f_rest.mean() - f_lab.mean()))
    tuned = P.ppi_mean(y, f_lab, f_rest)
    pooled = np.concatenate([f_lab, f_rest])
    lam = np.cov(y, f_lab, ddof=1)[0, 1] / ((1 + len(y) / len(f_rest)) * np.var(pooled, ddof=1))
    assert tuned.lam == pytest.approx(lam)
    assert tuned.estimate == pytest.approx(y.mean() + lam * (f_rest.mean() - f_lab.mean()))
    var = lam**2 * np.var(f_rest, ddof=1) / len(f_rest) + np.var(y - lam * f_lab, ddof=1) / len(y)
    half = 1.959964 * math.sqrt(var)
    assert tuned.ci_high - tuned.ci_low == pytest.approx(
        min(1, tuned.estimate + half) - max(0, tuned.estimate - half), abs=1e-5
    )


def test_a_judge_that_never_varies_falls_back_to_the_human_only_estimate() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(60) < 0.5).astype(float)
    out = P.ppi_mean(y, np.ones(60), np.ones(500))
    assert out.lam == 0.0
    human = P.classical_mean(y)
    assert out.estimate == pytest.approx(human.estimate)
    assert out.width == pytest.approx(human.width)


def test_lambda_is_not_clipped_for_a_mean() -> None:
    # A judge anti-correlated with the humans gets a negative weight: valid for a mean.
    rng = np.random.default_rng(1)
    y = (rng.random(200) < 0.5).astype(float)
    out = P.ppi_mean(y, 1 - y, (rng.random(2000) < 0.5).astype(float))
    assert out.lam < 0


def test_too_few_labelled_units_is_refused_and_a_small_sample_warns() -> None:
    rng = np.random.default_rng(2)
    y, f_lab, f_rest, _ = draw(rng, n_label=29)
    with pytest.raises(P.PPIError, match="at least 30") as e:
        P.ppi_mean(y, f_lab, f_rest)
    assert "random sample" in e.value.hint
    y, f_lab, f_rest, _ = draw(rng, n_label=60)
    assert any("approximate below 100" in w for w in P.ppi_mean(y, f_lab, f_rest).warnings)
    y, f_lab, f_rest, _ = draw(rng, n_label=150)
    assert P.ppi_mean(y, f_lab, f_rest).warnings == ()


def test_inputs_are_validated() -> None:
    ok = np.ones(40)
    with pytest.raises(P.PPIError, match="same units"):
        P.ppi_mean(ok, np.ones(39), np.ones(100))
    with pytest.raises(P.PPIError, match="nothing to correct"):
        P.ppi_mean(ok, ok, [])
    with pytest.raises(P.PPIError, match="NaN"):
        P.ppi_mean(np.r_[ok, np.nan], np.ones(41), np.ones(100))
    with pytest.raises(P.PPIError, match="alpha"):
        P.ppi_mean(ok, ok, ok, alpha=1.5)
    with pytest.raises(P.PPIError, match="1-D"):
        P.ppi_mean(np.ones((4, 10)), ok, ok)


def test_a_rate_stays_inside_zero_and_one() -> None:
    rng = np.random.default_rng(3)
    y = np.ones(40)
    out = P.ppi_mean(y, y, np.ones(400))
    assert 0 <= out.ci_low <= out.estimate <= out.ci_high <= 1
    wide = P.ppi_mean(np.r_[np.ones(20), np.zeros(20)], rng.random(40), rng.random(200))
    assert wide.ci_low >= 0
    assert wide.ci_high <= 1
    unbounded = P.ppi_mean(y, y, np.ones(400), bounded=False)
    assert unbounded.clipped is False


def test_classical_mean_is_a_normal_interval_and_needs_two_units() -> None:
    out = P.classical_mean([1, 0, 1, 1, 0, 0, 1, 1])
    assert out.estimate == pytest.approx(0.625)
    assert out.ci_low < 0.625 < out.ci_high
    assert out.method == "human-only"
    with pytest.raises(P.PPIError):
        P.classical_mean([1.0])


# --- the guarantees, by simulation ----------------------------------------------------


def test_the_interval_covers_the_true_rate_about_95_percent_of_the_time() -> None:
    rng = np.random.default_rng(10)
    covered = judge_covered = 0
    reps = 1500
    for _ in range(reps):
        y, f_lab, f_rest, truth = draw(rng)
        out = P.ppi_mean(y, f_lab, f_rest)
        covered += out.ci_low <= truth <= out.ci_high
        pooled = np.concatenate([f_lab, f_rest])
        judge_covered += abs(pooled.mean() - truth) < 0.005
    assert 0.925 <= covered / reps <= 0.98
    # The judge alone is biased: its rate is almost never the true one.
    assert judge_covered / reps < 0.1


def test_it_is_narrower_than_human_only_when_the_judge_is_informative() -> None:
    rng = np.random.default_rng(11)
    ratios = []
    for _ in range(200):
        y, f_lab, f_rest, _ = draw(rng, flip=0.05)
        ratios.append(P.ppi_mean(y, f_lab, f_rest).width / P.classical_mean(y).width)
    assert np.mean(ratios) < 0.9
    assert max(ratios) < 1.05  # never meaningfully worse (PPI++)


def test_an_uninformative_judge_costs_almost_nothing() -> None:
    rng = np.random.default_rng(12)
    ratios = []
    covered = 0
    for _ in range(600):
        y, f_lab, f_rest, truth = draw(rng, informative=False)
        out = P.ppi_mean(y, f_lab, f_rest)
        ratios.append(out.width / P.classical_mean(y).width)
        covered += out.ci_low <= truth <= out.ci_high
    assert 0.98 < np.mean(ratios) < 1.08
    assert 0.92 <= covered / 600 <= 0.985


def test_plain_ppi_is_valid_but_can_be_wider_than_ppi_plus_plus() -> None:
    rng = np.random.default_rng(13)
    plain_wider = 0
    covered = 0
    for _ in range(400):
        y, f_lab, f_rest, truth = draw(rng, informative=False)
        plain = P.ppi_mean(y, f_lab, f_rest, power_tuning=False)
        tuned = P.ppi_mean(y, f_lab, f_rest)
        plain_wider += plain.width > tuned.width
        covered += plain.ci_low <= truth <= plain.ci_high
    assert plain_wider / 400 > 0.9
    assert covered / 400 > 0.9


def test_corrected_rate_reports_the_three_numbers_and_the_agreement() -> None:
    rng = np.random.default_rng(14)
    y, f_lab, f_rest, _ = draw(rng)
    out = P.corrected_rate(y, f_lab, f_rest)
    assert set(out) == {"judge_only", "human_only", "corrected", "agreement_on_sample"}
    assert out["judge_only"]["n"] == len(f_lab) + len(f_rest)
    assert "biased" in out["judge_only"]["note"]
    assert out["human_only"]["method"] == "human-only"
    assert out["corrected"]["method"] == "ppi++"
    assert out["corrected"]["n"] == len(y)
    assert 0.8 < out["agreement_on_sample"] < 1.0
    assert out["corrected"]["ci_high"] - out["corrected"]["ci_low"] < (
        out["human_only"]["ci_high"] - out["human_only"]["ci_low"]
    )
