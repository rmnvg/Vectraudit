"""Non-inferiority: pass only when shown within tolerance; inconclusive is not a pass."""

from __future__ import annotations

import pytest

from tests.test_eval_compare import fake_run
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.exceptions import apply_exceptions, failing_rules
from vectraudit.evaluation.policy import PolicyError, parse_policy, policy_sha256
from vectraudit.evaluation.report import render_html, render_markdown

N = 40
BASE = [0.5] * N


def policy(**extra):
    return parse_policy({"metrics": {"ndcg": {"max_drop": 0.02}}, "min_queries": 10, **extra})


NI = {"non_inferiority": True}


def run_pair(deltas):
    return fake_run(BASE), fake_run([b + d for b, d in zip(BASE, deltas, strict=True)], run_id="c")


TIGHT = [0.001] * N  # a tiny gain, measured precisely
WIDE = [0.1, -0.11] * (N // 2)  # mean -0.005: small, but the interval is wide
CLEAR = [-0.1] * N


def ndcg(result):
    return result["metrics"]["ndcg"]


def test_shown_within_tolerance_passes():
    result = compare_runs(*run_pair(TIGHT), policy(**NI))
    assert result["verdict"] == "pass"
    assert ndcg(result)["tolerance"] == {"decision": "within_tolerance", "margin": 0.02}


def test_a_small_drop_with_a_wide_interval_is_inconclusive_not_a_pass():
    base, cand = run_pair(WIDE)
    measured = compare_runs(base, cand, policy())
    assert measured["verdict"] == "pass"  # the measured drop (0.005) is within 0.02
    assert "tolerance" not in ndcg(measured)
    strict = compare_runs(base, cand, policy(**NI))
    assert strict["verdict"] == "fail"
    assert ndcg(strict)["tolerance"]["decision"] == "inconclusive"
    (reason,) = strict["reasons"]
    assert reason["rule"] == "metric:ndcg.max_drop"
    assert reason["inconclusive"] is True
    assert "not shown to be within tolerance (inconclusive)" in reason["message"]
    assert "More query families would narrow it." in reason["message"]


def test_a_clear_drop_is_a_regression_not_inconclusive():
    result = compare_runs(*run_pair(CLEAR), policy(**NI))
    (reason,) = result["reasons"]
    assert "dropped beyond tolerance" in reason["message"]
    assert "inconclusive" not in reason
    assert ndcg(result)["tolerance"]["decision"] == "regression"


def test_policy_option_is_explicit_and_keeps_existing_hashes():
    assert policy_sha256(policy()) == policy_sha256(policy(non_inferiority=False))
    assert "non_inferiority" not in policy().as_dict()
    assert policy_sha256(policy(**NI)) != policy_sha256(policy())
    with pytest.raises(PolicyError, match="opposite rules"):
        policy(non_inferiority=True, require_significance=True)


def test_cohort_rules_keep_their_own_semantics():
    tags = [["billing"]] * N
    base = fake_run(BASE, tags=tags)
    cand = fake_run([b + d for b, d in zip(BASE, WIDE, strict=True)], run_id="c", tags=tags)
    rules = policy(
        non_inferiority=True,
        cohorts={"min_queries": 10, "metrics": {"ndcg": {"max_drop": 0.02}}},
    )
    result = compare_runs(base, cand, rules)
    assert "tolerance" not in result["cohorts"]["billing"]["metrics"]["ndcg"]
    assert failing_rules(result) == ["metric:ndcg.max_drop"]  # the cohort passes as before


def test_an_inconclusive_result_can_be_accepted_like_any_metric_failure():
    rules = policy(non_inferiority=True, exceptions={"allow": ["metric"], "approver": "engineer"})
    result = compare_runs(*run_pair(WIDE), rules)
    exception = {
        "id": "e1",
        "rule": "metric:ndcg.max_drop",
        "accepted_by": "lead@example.test",
        "reason": "Benchmark too small this release; adding questions",
        "expires_at": "2026-11-01T00:00:00+00:00",
    }
    assert apply_exceptions(result, [exception], rules)["verdict"] == "accepted"


def test_reports_state_the_tolerance_decision():
    result = compare_runs(*run_pair(WIDE), policy(**NI))
    line = "Non-inferiority (pass only when the whole 95% interval stays above the allowed drop)"
    md = render_markdown(result)
    assert f"{line}: nDCG@k inconclusive (interval reaches past the allowed drop)." in md
    assert line in render_html(result)
    plain = compare_runs(*run_pair(WIDE), policy())
    assert "Non-inferiority" not in render_markdown(plain)


def test_lead_summary_and_ledger_say_shown_or_inconclusive():
    pytest.importorskip("sqlalchemy")
    from vectraudit.cloud.lead import explain
    from vectraudit.cloud.ledger import summarize_check

    def entry(result):
        return {"verdict": result["verdict"], "summary": summarize_check(result), "check_id": "c"}

    shown = compare_runs(*run_pair(TIGHT), policy(**NI))
    summary = summarize_check(shown)
    assert summary["non_inferiority"] is True
    assert summary["metrics"]["ndcg"]["tolerance"]["decision"] == "within_tolerance"
    good = explain(entry(shown))
    assert any(
        m.startswith("Shown within tolerance under the non-inferiority") for m in good["matters"]
    )
    assert not any("not proof of safety" in u for u in good["uncertain"])
    unsure = explain(entry(compare_runs(*run_pair(WIDE), policy(**NI))))
    assert any(u.startswith("Inconclusive for nDCG") for u in unsure["uncertain"])
    legacy = summarize_check(compare_runs(*run_pair(TIGHT), policy()))
    assert legacy["non_inferiority"] is False
    assert "tolerance" not in legacy["metrics"]["ndcg"]


def test_the_optimizer_does_not_exclude_on_an_inconclusive_screen():
    import json

    from vectraudit.evaluation.dataset import parse_dataset
    from vectraudit.pipelines import Hit, PipelineResult
    from vectraudit.pipelines.optimize import optimize

    rows = [{"query_id": f"e{i}", "text": f"e {i}", "relevant": {f"d{i}": 1}} for i in range(4)]
    rows += [
        {"query_id": f"h{i}", "text": f"h {i}", "relevant": {f"d{i}": 1}, "split": "heldout"}
        for i in range(4)
    ]
    ds = parse_dataset("\n".join(map(json.dumps, rows)))

    def right(request):
        return PipelineResult(hits=[Hit(id="d" + request.query_id[1:]), Hit(id="x")])

    def mixed(request):
        # Better on some questions, worse on others: an exploratory screen cannot show it
        # is within tolerance, but it has not shown a regression either.
        first = int(request.query_id[1:]) % 2 == 0
        hits = ["d" + request.query_id[1:], "x"] if first else ["x", "d" + request.query_id[1:]]
        return PipelineResult(hits=[Hit(id=h) for h in hits])

    rules = parse_policy(
        {"metrics": {"ndcg": {"max_drop": 0.05}}, "min_queries": 2, "non_inferiority": True}
    )
    report = optimize(
        ds, [("base", right, {}), ("mixed", mixed, {})], repeats=1, policy=rules, baseline="base"
    )
    mixed_report = next(c for c in report["candidates"] if c["name"] == "mixed")
    assert mixed_report["policy_screen"]["verdict"] == "fail"
    assert mixed_report["policy_screen"]["inconclusive"] is True
    assert not any("fails the release policy" in r for r in mixed_report["reasons"])


def test_calibration_drafts_a_margin_that_shows_unchanged_runs_within_tolerance():
    from vectraudit.evaluation.calibration import calibrate

    runs = []
    for r in range(4):
        values = [0.5 + 0.05 * (((q + r) % 3) - 1) for q in range(N)]
        run = fake_run(values, run_id=f"aa-{r}")
        runs.append(run)
    raw = {"metrics": {"ndcg": {"max_drop": 0.001}}, "min_queries": 10, "non_inferiority": True}
    report = calibrate(runs, raw)
    assert report["current"]["false_alarms"] > 0
    assert report["draft"]["false_alarms"] == 0
    (change,) = report["changes"]
    assert change["path"] == "metrics.ndcg.max_drop"
