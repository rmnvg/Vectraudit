"""Paired randomization, randomized Tukey HSD and Holm: exactness, calibration and power."""

from __future__ import annotations

import numpy as np
import pytest

from vectraudit.evaluation.significance import (
    holm_adjust,
    hsd_for_runs,
    paired_randomization,
    randomized_tukey_hsd,
    wilson_interval,
)


def test_a_consistent_gain_is_significant_and_p_is_never_zero():
    ten = paired_randomization([0.05] * 10, n=10_000, seed=1)
    # Exact two-sided p for ten same-signed deltas is 2 / 2**10.
    assert 0.0005 < ten.p_value < 0.005
    assert (ten.n_families, ten.resamples) == (10, 10_000)
    assert ten.mean_delta == pytest.approx(0.05)
    many = paired_randomization([0.2] * 40, n=10_000)
    assert many.p_value == 1 / 10_001  # (0 + 1) / (n + 1): ranx's b / n would report 0


def test_no_difference_has_p_one_and_the_test_is_two_sided_and_seeded():
    assert paired_randomization([0.0, 0.0, 0.0]).p_value == 1.0
    deltas = [0.04, -0.01, 0.03, 0.02, -0.02, 0.05]
    up = paired_randomization(deltas, seed=4)
    down = paired_randomization([-d for d in deltas], seed=4)
    assert up.p_value == down.p_value
    assert up == paired_randomization(deltas, seed=4)
    assert paired_randomization([0.3]).p_value == pytest.approx(1.0, abs=1e-3)
    # One family can only be flipped one way or the other, so it can never be significant.


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf"), 1.0], [[1.0, 2.0]]])
def test_paired_randomization_refuses_bad_input(bad):
    with pytest.raises(ValueError, match="family_deltas"):
        paired_randomization(bad)
    with pytest.raises(ValueError, match="positive integer"):
        paired_randomization([0.1], n=0)


def test_false_positive_rate_matches_alpha_under_the_null():
    """A/A: when the systems are interchangeable the test rejects about 5% of the time."""
    rng = np.random.default_rng(2024)
    trials = 400
    for draw in (
        lambda: rng.normal(0, 0.1, 25),
        lambda: rng.laplace(0, 0.1, 25),  # heavy tails, still symmetric under the null
    ):
        rejected = sum(
            paired_randomization(draw(), n=500, seed=i).p_value <= 0.05 for i in range(trials)
        )
        expected, sd = trials * 0.05, (trials * 0.05 * 0.95) ** 0.5
        assert abs(rejected - expected) <= 4 * sd, rejected


def test_a_real_difference_is_detected_most_of_the_time():
    rng = np.random.default_rng(5)
    found = sum(
        paired_randomization(rng.normal(0.08, 0.1, 25), n=500, seed=i).p_value <= 0.05
        for i in range(100)
    )
    assert found >= 80


def _systems(rng, effects, families=60, noise=0.05):
    shared = rng.normal(0.5, 0.15, families)  # question difficulty, common to every system
    return {
        name: list(shared + effect + rng.normal(0, noise, families))
        for name, effect in effects.items()
    }


def test_hsd_groups_two_better_systems_apart_from_three_equal_ones():
    rng = np.random.default_rng(11)
    scores = _systems(rng, {"good1": 0.15, "good2": 0.15, "c1": 0, "c2": 0, "c3": 0})
    result = randomized_tukey_hsd(scores, n=5_000, seed=3)
    groups = [sorted(g) for g in result["groups"]]
    assert sorted(groups) == [["c1", "c2", "c3"], ["good1", "good2"]]
    assert result["leader"] in {"good1", "good2"}
    other = ({"good1", "good2"} - {result["leader"]}).pop()
    assert result["indistinguishable"] == [other]
    p = result["p_vs_leader"]
    assert p[result["leader"]] is None
    assert p[other] > 0.05
    assert all(p[c] < 0.01 for c in ("c1", "c2", "c3"))
    assert result["letters"][result["leader"]] == "a"
    assert {result["letters"][c] for c in ("c1", "c2", "c3")} == {"b"}
    assert len(result["pairs"]) == 10
    assert list(result["systems"]) == sorted(result["systems"], key=lambda s: -result["systems"][s])
    assert result == randomized_tukey_hsd(scores, n=5_000, seed=3)


def test_hsd_family_wise_false_winner_rate_is_at_most_alpha():
    """With equal systems, how often does any pair look different? About alpha, not more."""
    rng = np.random.default_rng(8)
    trials = 200
    false_winner = sum(
        any(
            p["p_adjusted"] < 0.05
            for p in randomized_tukey_hsd(
                _systems(rng, dict.fromkeys("abcd", 0.0), families=20), n=300, seed=i
            )["pairs"]
        )
        for i in range(trials)
    )
    expected, sd = trials * 0.05, (trials * 0.05 * 0.95) ** 0.5
    assert false_winner <= expected + 3 * sd, false_winner


def test_hsd_with_identical_systems_is_one_group():
    result = randomized_tukey_hsd({"a": [0.5, 0.6], "b": [0.5, 0.6], "c": [0.5, 0.6]}, n=200)
    assert result["groups"] == [["a", "b", "c"]]
    assert result["indistinguishable"] == ["b", "c"]


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ({"a": [1.0], "b": [2.0]}, "three or more"),
        ({"a": [1.0, 2.0], "b": [1.0], "c": [1.0, 2.0]}, "same families"),
        ({"a": [], "b": [], "c": []}, "same families"),
        ({"a": [1.0], "b": [float("nan")], "c": [1.0]}, "finite"),
    ],
)
def test_hsd_refuses_bad_input(scores, message):
    with pytest.raises(ValueError, match=message):
        randomized_tukey_hsd(scores, n=10)


def test_holm_adjusts_in_the_order_given_and_never_exceeds_bonferroni():
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_adjust([]) == []
    assert holm_adjust([0.2]) == [0.2]
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = list(rng.uniform(0, 1, 6))
        adjusted = holm_adjust(p)
        assert all(a <= min(1.0, len(p) * x) + 1e-12 for a, x in zip(adjusted, p, strict=True))
        assert all(a >= x for a, x in zip(adjusted, p, strict=True))
        order = sorted(range(6), key=lambda i: p[i])
        assert [adjusted[i] for i in order] == sorted(adjusted)
    with pytest.raises(ValueError, match="between 0 and 1"):
        holm_adjust([1.5])


def _run(scores, families=None):
    return {
        "queries": [
            {
                "query_id": f"q{i}",
                **({"family_id": families[i]} if families else {}),
                "metrics": None if s is None else {"ndcg": s},
            }
            for i, s in enumerate(scores)
        ]
    }


def test_hsd_for_runs_uses_shared_questions_and_family_means():
    scores = {
        "a": [0.9, 0.8, 0.9, 0.8, None],  # not scored on the last question
        "b": [0.5, 0.4, 0.5, 0.4, 0.7],
        "c": [0.1, 0.0, 0.1, 0.0, 0.2],
    }
    result = hsd_for_runs({n: _run(s) for n, s in scores.items()}, "ndcg", n=500)
    assert (result["n_queries"], result["n_families"], result["metric"]) == (4, 4, "ndcg")
    assert result["leader"] == "a"
    # Two questions of one family are one unit, not two.
    families = ["f1", "f1", "f2", "f2", "f3"]
    grouped = hsd_for_runs({n: _run(s, families) for n, s in scores.items()}, "ndcg", n=500)
    assert (grouped["n_queries"], grouped["n_families"]) == (4, 2)


def test_hsd_for_runs_needs_three_runs_and_a_shared_question():
    two = {"a": _run([0.5]), "b": _run([0.6])}
    assert hsd_for_runs(two, "ndcg") is None
    disjoint = {"a": _run([0.5, None]), "b": _run([None, 0.5]), "c": _run([0.5, 0.5])}
    assert hsd_for_runs(disjoint, "ndcg") is None


def test_wilson_interval_matches_hand_computed_values_and_stays_in_range():
    low, high = wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)  # zero of ten is not "certainly zero"
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    for n in (1, 2, 7, 40, 500):
        for k in range(n + 1):
            lo, hi = wilson_interval(k, n)
            assert 0 <= lo <= k / n <= hi <= 1
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]
    for k, n in [(-1, 5), (6, 5), (0, -1)]:
        with pytest.raises(ValueError, match="between 0 and n"):
            wilson_interval(k, n)
