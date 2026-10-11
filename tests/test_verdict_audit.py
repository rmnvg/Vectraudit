"""Adversarial run files never become a pass: each case here once crashed or passed."""

from __future__ import annotations

import copy

import pytest

from tests.test_eval_compare import fake_run
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.runfile import RunFileError, check_run

POLICY = parse_policy({"metrics": {"ndcg": {"max_drop": 0.01}}, "min_queries": 5})
GOOD = [1.0] * 20


def errored(run, n):
    for q in run["queries"][:n]:
        q.update(error={"code": "timeout"}, ids=None, metrics=None)
    return run


def duplicated(run):
    run["queries"][1] = copy.deepcopy(run["queries"][0])
    return run


def scored(run, value):
    for q in run["queries"]:
        q["metrics"]["ndcg"] = value
    return run


def truncated(run):
    run["queries"] = run["queries"][:10]
    return run


@pytest.mark.parametrize(
    ("name", "candidate"),
    [
        ("complete run hiding failed questions", lambda: errored(fake_run(GOOD), 10)),
        (
            "failures hiding the regressed half",
            lambda: errored(fake_run([0.0] * 10 + [1.0] * 10), 10),
        ),
        ("duplicate question rows", lambda: duplicated(fake_run(GOOD))),
        ("NaN metric", lambda: scored(fake_run(GOOD), float("nan"))),
        ("negative infinite metric", lambda: scored(fake_run(GOOD), float("-inf"))),
        ("complete run missing questions", lambda: truncated(fake_run(GOOD))),
    ],
)
def test_invalid_candidates_are_not_comparable_never_pass(name, candidate):
    result = compare_runs(fake_run(GOOD), candidate(), POLICY)
    assert result["verdict"] == "not_comparable", name
    assert any("not a valid run" in r["message"] for r in result["reasons"]), name


def test_a_regression_cannot_drop_out_by_losing_its_scores():
    cand = fake_run([0.0] * 10 + [1.0] * 10)
    for q in cand["queries"][:10]:
        q["metrics"] = None  # the regressed questions, unscored on the candidate side only
    result = compare_runs(fake_run(GOOD), cand, POLICY)
    assert result["verdict"] == "not_comparable"
    assert "scored in only one run" in result["reasons"][0]["message"]


def test_an_invalid_baseline_is_named():
    result = compare_runs(errored(fake_run(GOOD), 20), fake_run(GOOD), POLICY)
    assert result["verdict"] == "not_comparable"
    assert result["reasons"][0]["message"].startswith("The baseline run is not a valid run")


def test_complete_means_every_question_measured_without_failures():
    with pytest.raises(RunFileError, match="cannot contain failed queries"):
        check_run(errored(fake_run(GOOD), 1))
    with pytest.raises(RunFileError, match="must measure every dataset query"):
        check_run(truncated(fake_run(GOOD)))
    partial = errored(fake_run(GOOD), 1)
    partial["status"] = "partial"
    assert check_run(partial)["status"] == "partial"  # honest partial runs still load
