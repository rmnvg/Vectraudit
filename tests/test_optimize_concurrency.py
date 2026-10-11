"""Optimization runs each candidate with its own concurrency, rate limit and retry policy."""

from __future__ import annotations

import json
import threading
import time

import pytest
from typer.testing import CliRunner

from tests.test_spend_preview import DATASET, _workspace
from vectraudit.cli import app
from vectraudit.evaluation.policy import parse_policy
from vectraudit.pipelines import Hit, PipelineResult
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.optimize import optimize
from vectraudit.pipelines.runner import RetryPolicy


class Tracked:
    """A thread-safe pipeline that records how many calls were in flight at once."""

    def __init__(self, doc: str = "a") -> None:
        self.doc = doc
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def __call__(self, request):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        return PipelineResult(hits=[Hit(id=self.doc)])


def runs(report):
    held = report["heldout"] or {}
    return [c["run"] for c in report["candidates"]] + [
        r for r in (held.get("run"), held.get("baseline_run")) if r
    ]


def test_every_run_of_a_candidate_uses_its_concurrency():
    fast, base = Tracked(), Tracked("b")
    report = optimize(
        DATASET,
        [("fast", fast, {}), ("base", base, {})],
        repeats=1,
        policy=parse_policy({"metrics": {"ndcg": {"max_drop": 1}}}),
        baseline="base",
        execution={"fast": {"concurrency": 4}},
    )
    assert report["selected"] == "fast"
    by_candidate = {r["backend"]["config"]["candidate"]: r for r in runs(report)}
    assert {
        r["backend"]["config"]["candidate"]: r["timing"]["concurrency"] for r in runs(report)
    } == {
        "fast": 4,
        "base": 1,
    }
    assert len(runs(report)) == 4  # two tuning runs, the held-out run and its baseline
    assert by_candidate["fast"]["timing"]["concurrency"] == 4
    assert fast.peak > 1
    assert base.peak == 1
    assert "fast 4 in flight" in report["latency_note"]
    assert "base 1 in flight" in report["latency_note"]


def test_concurrent_results_match_serial_apart_from_timing():
    serial = optimize(DATASET, [("a", Tracked(), {}), ("b", Tracked("b"), {})], repeats=2)
    parallel = optimize(
        DATASET,
        [("a", Tracked(), {}), ("b", Tracked("b"), {})],
        repeats=2,
        execution={"a": {"concurrency": 3}, "b": {"concurrency": 3, "max_rps": 500}},
    )
    assert parallel["selected"] == serial["selected"]
    assert [(c["name"], c["ndcg"]) for c in parallel["candidates"]] == [
        (c["name"], c["ndcg"]) for c in serial["candidates"]
    ]
    assert parallel["latency_note"] is None  # one concurrency for everyone
    assert serial["latency_note"] is None


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="unknown candidates"):
        optimize(DATASET, [("a", Tracked(), {})], execution={"z": {"concurrency": 2}})
    with pytest.raises(ValueError, match="concurrency"):
        optimize(DATASET, [("a", Tracked(), {})], execution={"a": {"concurrency": 99}})

    def unsafe(request):
        return PipelineResult(hits=[Hit(id="a")])

    unsafe.thread_safe = False  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="thread_safe"):
        optimize(DATASET, [("a", unsafe, {})], execution={"a": {"concurrency": 2}})


def test_the_retry_policy_applies_to_optimization_runs():
    seen: set[str] = set()

    def flaky(request):
        if request.query_id not in seen:
            seen.add(request.query_id)
            raise PipelineCallError("timeout", "slow")
        return PipelineResult(hits=[Hit(id="a")])

    fast = RetryPolicy(base_delay_s=0, max_delay_s=0)
    report = optimize(DATASET, [("a", flaky, {})], repeats=1, retry=fast)
    assert report["selected"] == "a"
    assert report["usage"]["retries"] == 10  # one per question, explore and held out
    seen.clear()
    never = optimize(DATASET, [("a", flaky, {})], repeats=1, retry=RetryPolicy(max_attempts=1))
    assert never["selected"] is None  # failures are not retried, so the run is incomplete


def test_cli_optimize_uses_the_manifest_settings(tmp_path):
    path = _workspace(tmp_path, concurrency=2)
    out = tmp_path / "o.json"
    result = CliRunner().invoke(
        app, ["workbench", "optimize", str(path), "--out", str(out), "--max-attempts", "1"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["candidates"][0]["concurrency"] == 2
    assert report["heldout"]["run"]["timing"]["concurrency"] == 2
    assert report["heldout"]["run"]["timing"]["retry"]["max_attempts"] == 1
