"""Actual calls and spend after a run, beside the plan: retries, failures and unknowns kept."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.test_spend_preview import DATASET, _workspace, candidate, manifest
from vectraudit.cli import app
from vectraudit.pipelines import Hit, PipelineOptions, PipelineResult, execute_pipeline
from vectraudit.pipelines.budget import (
    combine,
    plan,
    planned_for,
    planned_total,
    reconcile,
    reconcile_line,
    usage,
)
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.runner import RetryPolicy

FAST = RetryPolicy(max_attempts=3, max_retries=100, base_delay_s=0, max_delay_s=0)


def run_with(pipeline, repeats=2, retry=FAST, warmup=0):
    return execute_pipeline(
        DATASET,
        pipeline,
        PipelineOptions(k=1, repeats=repeats, label="c", retry=retry, warmup=warmup),
    )


def test_every_call_and_its_reported_cost_is_counted():
    run = run_with(lambda r: PipelineResult(hits=[Hit(id="a")], cost_usd=0.002))
    assert run["queries"][0]["usage"] == {
        "calls": 2,
        "failed_calls": 0,
        "spend_usd": pytest.approx(0.004),
        "unpriced_calls": 0,
    }
    total = run["usage"]
    assert (total["calls"], total["retries"], total["complete"]) == (20, 0, True)
    assert total["spend_usd"] == pytest.approx(0.04)
    # Warm-up calls are calls too, and are paid for.
    warmed = run_with(lambda r: PipelineResult(hits=[Hit(id="a")], cost_usd=0.002), warmup=2)
    assert warmed["usage"]["calls"] == 22
    assert warmed["usage"]["spend_usd"] == pytest.approx(0.044)


def test_retries_and_failures_are_calls_with_unknown_cost():
    seen: dict[str, int] = {}

    def flaky(request):
        seen[request.query_id] = seen.get(request.query_id, 0) + 1
        if request.query_id == "e0" and seen["e0"] == 1:
            raise PipelineCallError("timeout", "slow")
        if request.query_id == "e1":
            raise PipelineCallError("http_auth", "denied")  # not retryable
        return PipelineResult(hits=[Hit(id="a")], cost_usd=0.01)

    run = run_with(flaky)
    by_id = {q["query_id"]: q for q in run["queries"]}
    assert by_id["e0"]["usage"] == {
        "calls": 3,
        "failed_calls": 1,
        "spend_usd": pytest.approx(0.02),
        "unpriced_calls": 1,
    }
    assert by_id["e1"]["usage"]["calls"] == 1
    assert by_id["e1"]["error"] is not None
    total = run["usage"]
    assert (total["calls"], total["failed_calls"], total["retries"]) == (20, 2, 1)
    assert total["complete"] is False  # two calls of unknown cost

    rec = reconcile(planned_for(plan(manifest(candidate("c")), DATASET), "c"), total)
    line = reconcile_line(rec)
    assert line.startswith("20 calls (planned 20) · $0.18 reported, incomplete (planned unknown)")
    assert "1 automatic retries" in line
    assert "2 failed calls (cost unknown)" in line


def test_adapters_that_report_no_cost_stay_unknown_not_free():
    run = run_with(lambda r: PipelineResult(hits=[Hit(id="a")]))
    assert run["usage"]["spend_usd"] == 0
    assert run["usage"]["unpriced_calls"] == 20
    assert run["usage"]["complete"] is False
    rec = reconcile({"calls": 20, "retry_allowance": 40, "spend_usd": None}, run["usage"])
    assert "20 successful calls reported no cost" in rec["notes"]
    assert rec["spend_over_plan_usd"] is None


def test_resumed_attempts_count_and_interrupted_ones_make_totals_a_lower_bound():
    done = {"calls": 2, "failed_calls": 0, "spend_usd": 0.02, "unpriced_calls": 0}
    failed = {"calls": 3, "failed_calls": 3, "spend_usd": 0.0, "unpriced_calls": 3}
    queries = [
        {
            "query_id": "q1",
            "usage": done,
            "attempts": [
                {"attempt": 1, "result": {"usage": failed, "retries": [{}, {}]}},
                {"attempt": 2, "result": None},  # interrupted: calls unknown
                {"attempt": 3, "result": {"usage": done}},
            ],
        },
        {"query_id": "q2", "metrics": {}},  # recorded before usage existed
    ]
    total = usage(queries)
    assert (total["calls"], total["failed_calls"], total["retries"]) == (5, 3, 2)
    assert (total["interrupted_attempts"], total["unrecorded_queries"]) == (1, 1)
    assert total["complete"] is False
    rec = reconcile({"calls": 4, "retry_allowance": 0, "spend_usd": 0.04}, total)
    assert rec["calls_over_plan"] == 1
    assert rec["calls_within_retry_allowance"] is False
    assert any("lower bound" in n for n in rec["notes"])


def test_reconciling_ranges_and_spend_over_plan():
    actual = combine(
        [
            {
                "calls": 10,
                "failed_calls": 0,
                "retries": 0,
                "spend_usd": 0.5,
                "unpriced_calls": 0,
                "interrupted_attempts": 0,
                "unrecorded_queries": 0,
                "complete": True,
            },
            {
                "calls": 7,
                "failed_calls": 1,
                "retries": 1,
                "spend_usd": 0.25,
                "unpriced_calls": 1,
                "interrupted_attempts": 0,
                "unrecorded_queries": 0,
                "complete": False,
            },
        ]
    )
    assert (actual["calls"], actual["spend_usd"], actual["complete"]) == (17, 0.75, False)
    planned = {
        "calls": {"min": 12, "max": 16},
        "retry_allowance": 5,
        "spend_usd": {"min": 0.4, "max": 0.6},
    }
    rec = reconcile(planned, actual)
    assert rec["calls_over_plan"] == 1
    assert rec["calls_within_retry_allowance"] is True
    assert rec["spend_over_plan_usd"] == pytest.approx(0.15)
    assert reconcile_line(rec).startswith(
        "17 calls (planned 12-16) · $0.75 reported, incomplete (planned $0.40-$0.60)"
    )
    unknown = planned_total(
        plan(manifest(candidate("x"), candidate("y")), DATASET, mode="optimize")
    )
    assert unknown["spend_usd"]["max"] is None


def test_cli_run_optimize_and_monitor_report_usage_beside_the_plan(tmp_path):
    path = _workspace(tmp_path, cost_per_call_usd=0.003)
    runner = CliRunner()
    ran = runner.invoke(app, ["workbench", "run", str(path), "--out", str(tmp_path / "runs")])
    assert ran.exit_code == 0, ran.output
    assert "10 calls (planned 10) · $0.030 reported (planned $0.030)" in ran.output
    recorded = json.loads(next((tmp_path / "runs").glob("*.run.json")).read_text())
    assert recorded["usage"]["calls"] == 10
    assert recorded["usage"]["planned"]["cost_source"] == "declared"

    out = tmp_path / "o.json"
    optimized = runner.invoke(app, ["workbench", "optimize", str(path), "--out", str(out)])
    assert "Usage: 10 calls (planned 6-10)" in optimized.output, optimized.output
    result = json.loads(out.read_text())
    assert result["usage"]["actual"]["calls"] == 10  # 6 exploratory + 4 held out
    assert result["usage"]["planned"]["calls"] == {"min": 6, "max": 10}

    base = next((tmp_path / "runs").glob("*.run.json"))
    history = tmp_path / "history"
    monitored = runner.invoke(
        app, ["workbench", "monitor", str(path), str(base), "--history", str(history)]
    )
    lines = monitored.output.strip().splitlines()
    assert lines[-2].startswith("Usage: 10 calls (planned 10)")
    assert json.loads(lines[-1])["verdict"] == "pass"  # the JSON verdict stays last
    report = json.loads(next(history.glob("*.monitor.json")).read_text())
    assert report["usage"]["calls"] == 10
