"""Many cohorts: count what was judged, and optionally control chance failures (Bonferroni)."""

from __future__ import annotations

import random

import pytest

from tests.test_eval_compare import fake_run
from vectraudit.evaluation.calibration import calibrate
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import PolicyError, parse_policy, policy_sha256
from vectraudit.evaluation.report import render_html, render_markdown

COHORTS, PER_COHORT = 20, 10
TAGS = [[f"tenant-{i // PER_COHORT}"] for i in range(COHORTS * PER_COHORT)]


def rules(control=None, max_drop=0.05):
    cohorts = {"min_queries": 5, "metrics": {"ndcg": {"max_drop": max_drop}}}
    if control:
        cohorts["control"] = control
    return parse_policy(
        {"metrics": {"ndcg": {"max_drop": 0.5}}, "min_queries": 10, "cohorts": cohorts}
    )


def noisy_pair(seed=7, spread=0.25, sunk=None, drop=0.0):
    """An unchanged system measured twice with per-question noise; ``sunk`` cohort drops."""
    rng = random.Random(seed)
    base_values = [0.5 + rng.uniform(-spread, spread) for _ in TAGS]
    cand_values = []
    for i, value in enumerate(base_values):
        noisy = 0.5 + rng.uniform(-spread, spread)
        if sunk is not None and i // PER_COHORT == sunk:
            noisy = max(0.0, value - drop)
        cand_values.append(noisy)
    base = fake_run(base_values, tags=TAGS)
    cand = fake_run(cand_values, run_id="cand", tags=TAGS)
    return base, cand


def cohort_failures(result):
    return [r for r in result["reasons"] if str(r.get("rule", "")).startswith("cohort:")]


def test_chance_cohort_failures_are_counted_and_controlled():
    base, cand = noisy_pair()
    plain = compare_runs(base, cand, rules())
    assert plain["cohort_judging"]["cohorts"] == COHORTS
    assert plain["cohort_judging"]["judged_rules"] == COHORTS
    assert plain["cohort_judging"]["control"] == "none"
    chance = cohort_failures(plain)
    assert len(chance) >= 2  # an unchanged system, yet several cohorts "regress"
    assert plain["cohort_judging"]["failed_rules"] == len(chance)
    assert any("20 cohort rules were judged one by one" in w for w in plain["warnings"])

    controlled = compare_runs(base, cand, rules("bonferroni"))
    assert cohort_failures(controlled) == []
    assert controlled["verdict"] == "pass"
    adjusted = controlled["cohorts"]["tenant-0"]["metrics"]["ndcg"]["ci_adjusted"]
    assert adjusted["confidence"] == pytest.approx(1 - 0.05 / COHORTS)
    assert adjusted["rules"] == COHORTS
    plain_width = plain["cohorts"]["tenant-0"]["metrics"]["ndcg"]
    assert adjusted["high"] - adjusted["low"] > plain_width["ci_high"] - plain_width["ci_low"]
    assert not any("one by one" in w for w in controlled["warnings"])


def test_a_real_cohort_regression_still_fails_under_control():
    base, cand = noisy_pair(spread=0.02, sunk=3, drop=0.4)
    result = compare_runs(base, cand, rules("bonferroni"))
    failures = cohort_failures(result)
    assert [r["rule"] for r in failures] == ["cohort:tenant-3:ndcg.max_drop"]
    assert "adjusted for 20 cohort rules" in failures[0]["message"]
    assert result["verdict"] == "fail"
    assert result["cohort_judging"]["failed_rules"] == 1


def test_one_cohort_rule_is_not_adjusted():
    tags = [["only"]] * 20
    base = fake_run([0.8] * 20, tags=tags)
    cand = fake_run([0.7] * 20, run_id="cand", tags=tags)
    plain = compare_runs(base, cand, rules())
    controlled = compare_runs(base, cand, rules("bonferroni"))
    assert cohort_failures(plain) == cohort_failures(controlled)
    assert "ci_adjusted" not in controlled["cohorts"]["only"]["metrics"]["ndcg"]


def test_policy_control_is_validated_and_keeps_existing_hashes():
    assert policy_sha256(rules()) == policy_sha256(rules("none"))
    assert "control" not in rules().as_dict()["cohorts"]
    assert policy_sha256(rules("bonferroni")) != policy_sha256(rules())
    with pytest.raises(PolicyError, match=r"cohorts\.control"):
        rules("holm")


def test_reports_and_calibration_show_the_effect():
    base, cand = noisy_pair()
    result = compare_runs(base, cand, rules())
    line = "judged on 20 rules"
    assert line in render_markdown(result)
    assert "no multiplicity control" in render_markdown(result)
    assert line in render_html(result)
    controlled = compare_runs(base, cand, rules("bonferroni"))
    assert "Bonferroni" in render_markdown(controlled)
    # A/A: recordings of one unchanged system, compared pairwise.
    runs = []
    for seed in range(4):
        run = noisy_pair(seed=seed)[1]
        run["run_id"] = f"aa-{seed}"
        runs.append(run)
    raw = {"metrics": {"ndcg": {"max_drop": 0.5}}, "min_queries": 10}
    cohorts = {"min_queries": 5, "metrics": {"ndcg": {"max_drop": 0.05}}}
    plain = calibrate(runs, {**raw, "cohorts": cohorts})["current"]
    controlled = calibrate(runs, {**raw, "cohorts": {**cohorts, "control": "bonferroni"}})
    controlled = controlled["current"]
    # Judged one by one, every A/A comparison raises a chance cohort failure. Bonferroni
    # bounds the chance of any false alarm per comparison at about 5%; it does not make it 0.
    assert plain["false_alarms"] == plain["comparisons"] == 12
    assert controlled["false_alarms"] <= 1
