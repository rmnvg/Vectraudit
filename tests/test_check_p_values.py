"""A paired-randomization p-value beside each metric change, for information only."""

from __future__ import annotations

import copy
import random

import pytest

from tests.test_eval_compare import fake_run
from vectraudit.evaluation import compare as compare_module
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.report import P_NOTE, render_html, render_markdown

POLICY = parse_policy({"metrics": {"ndcg": {"max_drop": 0.02}}, "min_queries": 10})


def runs(shift, n=40, seed=3):
    rng = random.Random(seed)
    base = [0.6 + rng.uniform(-0.2, 0.2) for _ in range(n)]
    cand = [min(1.0, max(0.0, v + shift + rng.uniform(-0.02, 0.02))) for v in base]
    return fake_run(base), fake_run(cand, run_id="cand")


def test_a_consistent_drop_has_a_small_p_and_no_change_a_large_one():
    drop = compare_runs(*runs(-0.1), POLICY)
    assert drop["metrics"]["ndcg"]["p_value"] < 0.001
    assert drop["verdict"] == "fail"
    same = compare_runs(*runs(0.0), POLICY)
    assert same["metrics"]["ndcg"]["p_value"] > 0.05
    assert all(0 < m["p_value"] <= 1 for m in drop["metrics"].values())


def test_p_values_are_seeded_and_only_on_the_suite_level_metrics():
    tags = [["a"] if i % 2 else ["b"] for i in range(40)]
    base, cand = runs(-0.05)
    for run in (base, cand):
        for q, t in zip(run["queries"], tags, strict=True):
            q["tags"] = t
    policy = parse_policy(
        {"metrics": {"ndcg": {"max_drop": 0.5}}, "min_queries": 10, "cohorts": {"min_queries": 5}}
    )
    first = compare_runs(base, cand, policy)
    assert (
        first["metrics"]["ndcg"]["p_value"]
        == compare_runs(base, cand, policy)["metrics"]["ndcg"]["p_value"]
    )
    assert all("p_value" not in m for c in first["cohorts"].values() for m in c["metrics"].values())


def test_the_p_value_never_changes_an_interval_or_a_verdict(monkeypatch):
    """It has its own generator, so the bootstrap interval is the same however p is computed."""
    for shift in (-0.1, -0.015, 0.0, 0.05):
        base, cand = runs(shift)
        with_p = compare_runs(base, cand, POLICY)
        monkeypatch.setattr(compare_module, "paired_randomization", _constant_p)
        constant = compare_runs(base, cand, POLICY)
        monkeypatch.undo()
        assert constant["verdict"] == with_p["verdict"]
        assert constant["reasons"] == with_p["reasons"]
        for name, stats in with_p["metrics"].items():
            for key in ("delta", "ci_low", "ci_high"):
                assert constant["metrics"][name][key] == stats[key]


class _Result:
    p_value = 0.5


def _constant_p(*args, **kwargs):
    return _Result()


def test_reports_show_p_beside_the_change_and_say_it_is_not_the_verdict():
    result = compare_runs(*runs(-0.1), POLICY)
    md = render_markdown(result)
    assert "| 95% CI | p | Rows | Families |" in md
    assert "| &lt;0.001 |" in md  # escaped for GitHub's Markdown
    assert P_NOTE in md
    page = render_html(result)
    assert "<th>p</th>" in page
    assert "<td>&lt;0.001</td>" in page
    assert "for information" in page


def test_an_older_check_without_p_values_still_renders():
    result = compare_runs(*runs(-0.1), POLICY)
    old = copy.deepcopy(result)
    for m in old["metrics"].values():
        del m["p_value"]
    md = render_markdown(old)
    assert P_NOTE not in md
    assert "| 95% CI | p | Rows | Families |" in md
    assert "<th>p</th>" in render_html(old)


@pytest.mark.parametrize("n", [1, 2])
def test_few_families_still_give_a_valid_p(n):
    base, cand = fake_run([0.8] * n), fake_run([0.5] * n, run_id="cand")
    p = compare_runs(base, cand, parse_policy({"metrics": {"ndcg": {"max_drop": 0.5}}}))["metrics"][
        "ndcg"
    ]["p_value"]
    assert 0 < p <= 1
