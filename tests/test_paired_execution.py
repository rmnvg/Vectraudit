"""Paired execution: two pipelines interleaved per question, so drift falls on both sides."""

from __future__ import annotations

import json
import threading
import time

import pytest
from typer.testing import CliRunner

from tests.test_spend_preview import DATASET
from vectraudit.cli import app
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import DEFAULT_POLICY
from vectraudit.evaluation.report import PAIRED_NOTE, render_markdown
from vectraudit.integrations.sharing import project_run
from vectraudit.pipelines import Hit, PipelineOptions, PipelineResult, execute_pipeline
from vectraudit.pipelines.runner import PairedSide, execute_paired


class DriftingEndpoint:
    """Both pipelines call one endpoint that gets slower with every call it serves."""

    def __init__(self, step_s: float = 0.0015) -> None:
        self.lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []
        self.step_s = step_s

    def pipeline(self, name: str):
        def call(request):
            with self.lock:
                self.calls.append((name, request.query_id))
                served = len(self.calls)
            time.sleep(served * self.step_s)
            return PipelineResult(hits=[Hit(id="a")])

        return call


def options(label, **extra):
    return PipelineOptions(k=1, repeats=1, warmup=0, label=label, **extra)


def p50(run):
    return run["summary"]["latency_ms"]["p50"]


def test_questions_alternate_which_side_goes_first():
    endpoint = DriftingEndpoint(step_s=0)
    a, b = execute_paired(
        DATASET,
        PairedSide(endpoint.pipeline("a"), options("A")),
        PairedSide(endpoint.pipeline("b"), options("B")),
    )
    first_two = [name for name, _ in endpoint.calls[:4]]
    assert first_two == ["a", "b", "b", "a"]  # q0: A then B; q1: B then A
    assert [q for _, q in endpoint.calls[:2]] == ["e0", "e0"]
    assert a["timing"]["paired_with"] == b["run_id"]
    assert b["timing"]["paired_with"] == a["run_id"]
    assert a["timing"]["order"] == b["timing"]["order"] == "interleaved"
    assert (a["status"], b["status"], a["label"], b["label"]) == ("complete", "complete", "A", "B")
    assert a["usage"]["calls"] == b["usage"]["calls"] == len(DATASET)


def test_drift_lands_on_both_sides_only_when_paired():
    sequential = DriftingEndpoint()
    base = execute_pipeline(DATASET, sequential.pipeline("a"), options("A"))
    cand = execute_pipeline(DATASET, sequential.pipeline("b"), options("B"))
    assert p50(cand) / p50(base) > 1.5  # identical pipelines look much slower when run second

    paired = DriftingEndpoint()
    pa, pb = execute_paired(
        DATASET,
        PairedSide(paired.pipeline("a"), options("A")),
        PairedSide(paired.pipeline("b"), options("B")),
    )
    assert 0.8 < p50(pb) / p50(pa) < 1.25


def test_comparisons_and_reports_say_latency_was_paired():
    endpoint = DriftingEndpoint(step_s=0)
    a, b = execute_paired(
        DATASET,
        PairedSide(endpoint.pipeline("a"), options("A")),
        PairedSide(endpoint.pipeline("b"), options("B")),
    )
    result = compare_runs(a, b, DEFAULT_POLICY)
    assert result["latency"]["paired"] is True
    assert PAIRED_NOTE in render_markdown(result)
    hosted = compare_runs(project_run(a), project_run(b), DEFAULT_POLICY)
    assert hosted["latency"]["paired"] is True
    # Two separately recorded runs are not paired, even if each was paired with another.
    other = execute_pipeline(DATASET, endpoint.pipeline("c"), options("C"))
    assert compare_runs(a, other, DEFAULT_POLICY)["latency"]["paired"] is False


def test_different_warm_up_is_called_out():
    endpoint = DriftingEndpoint(step_s=0)
    cold = execute_pipeline(DATASET, endpoint.pipeline("a"), options("A"))
    warm = execute_pipeline(
        DATASET, endpoint.pipeline("b"), PipelineOptions(k=1, repeats=1, warmup=3, label="B")
    )
    warnings = compare_runs(cold, warm, DEFAULT_POLICY)["warnings"]
    assert any("Warm-up differed (baseline 0, candidate 3 calls)" in w for w in warnings)
    same = compare_runs(cold, cold, DEFAULT_POLICY)["warnings"]
    assert not any("Warm-up differed" in w for w in same)


def test_unpairable_settings_are_refused():
    endpoint = DriftingEndpoint(step_s=0)
    with pytest.raises(ValueError, match="one question at a time"):
        execute_paired(
            DATASET,
            PairedSide(endpoint.pipeline("a"), options("A", concurrency=2)),
            PairedSide(endpoint.pipeline("b"), options("B")),
        )
    with pytest.raises(ValueError, match="same k and repeats"):
        execute_paired(
            DATASET,
            PairedSide(endpoint.pipeline("a"), options("A")),
            PairedSide(endpoint.pipeline("b"), PipelineOptions(k=1, repeats=2, warmup=0)),
        )


def _manifest(tmp_path, candidates):
    (tmp_path / "q.jsonl").write_text(
        "\n".join(
            json.dumps({"query_id": f"q{i}", "text": f"t{i}", "relevant": {"a": 1}})
            for i in range(6)
        )
    )
    (tmp_path / "paired_adapter.py").write_text(
        "from vectraudit.pipelines import Hit, PipelineResult\n"
        "def run(request):\n"
        "    return PipelineResult(hits=[Hit(id='a')])\n"
    )
    spec = {
        "dataset": "q.jsonl",
        "repeats": 1,
        "candidates": [
            {"name": name, "adapter": {"kind": "python", "target": "paired_adapter:run"}}
            for name in candidates
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return path


def test_cli_paired_run(tmp_path):
    runner = CliRunner()
    out = tmp_path / "runs"
    path = _manifest(tmp_path, ["current", "candidate"])
    result = runner.invoke(app, ["workbench", "run", str(path), "--out", str(out), "--paired"])
    assert result.exit_code == 0, result.output
    assert result.output.count(", paired)") == 2
    runs = [json.loads(p.read_text()) for p in out.glob("*.run.json")]
    assert {r["label"] for r in runs} == {"current", "candidate"}
    assert {r["timing"]["paired_with"] for r in runs} == {r["run_id"] for r in runs}
    three = _manifest(tmp_path, ["a", "b", "c"])
    refused = runner.invoke(app, ["workbench", "run", str(three), "--out", str(out), "--paired"])
    assert refused.exit_code == 2
    assert "exactly two candidates" in refused.output
    resumed = runner.invoke(
        app, ["workbench", "run", str(path), "--paired", "--resume", str(tmp_path / "x")]
    )
    assert resumed.exit_code == 2
    assert "cannot be resumed" in resumed.output
