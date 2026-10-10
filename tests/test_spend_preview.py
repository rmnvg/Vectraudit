"""Calls and spend before running: honest unknowns, ranges for optimization, refusals first."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines import Hit, PipelineOptions, PipelineResult, execute_pipeline
from vectraudit.pipelines.budget import check_limits, format_usd, measured_cost, plan
from vectraudit.pipelines.manifest import Candidate, Manifest, candidate_config

ROWS = [{"query_id": f"e{i}", "text": f"explore {i}", "relevant": {"a": 1}} for i in range(6)] + [
    {"query_id": f"h{i}", "text": f"held {i}", "relevant": {"a": 1}, "split": "heldout"}
    for i in range(4)
]
DATASET = parse_dataset("\n".join(map(json.dumps, ROWS)))


def candidate(name, **extra):
    return {"name": name, "adapter": {"kind": "http", "target": "https://x.example"}, **extra}


def manifest(*candidates, repeats=2):
    return Manifest.model_validate(
        {"dataset": "q.jsonl", "repeats": repeats, "candidates": list(candidates)}
    )


def costed_run(name, cost, *, missing=0):
    def pipeline(request):
        return PipelineResult(hits=[Hit(id="a")], cost_usd=cost)

    run = execute_pipeline(DATASET, pipeline, PipelineOptions(k=1, repeats=1, label=name))
    for q in run["queries"][:missing]:
        q["cost_usd"] = None
    return run


def test_run_plan_counts_calls_and_never_prices_unknown_as_zero():
    spec = manifest(
        candidate("declared", cost_per_call_usd=0.002),
        candidate("measured", cost_per_call_usd=0.5),  # the observation wins
        candidate("unknown"),
    )
    budget = plan(spec, DATASET, measured=[costed_run("measured", 0.001)])
    rows = {r["name"]: r for r in budget["candidates"]}
    assert {r["calls"] for r in rows.values()} == {20}  # 10 questions x 2 repeats
    assert rows["declared"]["cost_source"] == "declared"
    assert rows["measured"]["cost_source"] == "measured"
    assert rows["measured"]["cost_per_call_usd"] == pytest.approx(0.001)
    assert rows["unknown"]["spend_usd"] is None
    # Each of the 3 runs may retry each call twice, within its own budget of 100.
    assert budget["calls"] == {"min": 60, "max": 60, "retry_allowance": 120}
    assert budget["spend_usd"]["min"] == pytest.approx(20 * 0.002 + 20 * 0.001)
    assert budget["spend_usd"]["complete"] is False
    assert budget["unknown_cost"] == ["unknown"]
    assert any("unknown for unknown" in n for n in budget["notes"])


def test_partial_measurements_yield_to_a_declaration():
    partial = costed_run("p", 0.01, missing=3)
    expected = {"usd": 0.01, "queries": 7, "of": 10, "run_id": partial["run_id"]}
    assert measured_cost(partial) == expected
    budget = plan(
        manifest(candidate("p", cost_per_call_usd=0.02), candidate("q")),
        DATASET,
        measured=[partial, costed_run("q", 0.03, missing=1)],
    )
    rows = {r["name"]: r for r in budget["candidates"]}
    assert rows["p"]["cost_source"] == "declared"
    assert rows["q"]["cost_source"] == "measured (partial)"
    assert any("only part" in n for n in budget["notes"])


def test_optimize_plans_a_range_for_held_out_validation():
    spec = manifest(
        candidate("base", cost_per_call_usd=0.01),
        candidate("cheap", cost_per_call_usd=0.001),
        candidate("dear", cost_per_call_usd=0.1),
    )
    budget = plan(spec, DATASET, mode="optimize", baseline="base")
    explore_calls = 3 * 6 * 2
    assert budget["calls"]["min"] == explore_calls
    assert budget["calls"]["max"] == explore_calls + 2 * 4 * 2  # challenger and baseline
    assert budget["heldout"] == {"questions": 4, "calls_per_run": 8, "runs_up_to": 2}
    explore_spend = 12 * (0.01 + 0.001 + 0.1)
    assert budget["spend_usd"]["min"] == pytest.approx(explore_spend)
    assert budget["spend_usd"]["max"] == pytest.approx(explore_spend + 8 * (0.1 + 0.01))
    alone = plan(spec, DATASET, mode="optimize")
    assert alone["calls"]["max"] == explore_calls + 8
    unknown = plan(manifest(candidate("x"), candidate("y")), DATASET, mode="optimize")
    assert unknown["spend_usd"]["max"] is None
    with pytest.raises(ValueError, match="exactly one"):
        plan(spec, DATASET, mode="monitor")


def test_limits_refuse_before_running():
    budget = plan(manifest(candidate("a", cost_per_call_usd=0.01)), DATASET)  # 20 calls, $0.20
    check_limits(budget, max_calls=20, max_spend=0.2)
    with pytest.raises(ValueError, match="exceed --max-calls 19"):
        check_limits(budget, max_calls=19)
    with pytest.raises(ValueError, match="exceeds --max-spend"):
        check_limits(budget, max_spend=0.19)
    unknown = plan(manifest(candidate("b")), DATASET)
    with pytest.raises(ValueError, match="unknown for b"):
        check_limits(unknown, max_spend=1000)
    check_limits(unknown, max_calls=100)  # calls are always known


def test_a_declared_price_does_not_change_candidate_identity(tmp_path):
    plain = Candidate.model_validate(candidate("a"))
    priced = Candidate.model_validate(candidate("a", cost_per_call_usd=0.004))
    assert candidate_config(plain, tmp_path) == candidate_config(priced, tmp_path)


def test_money_formatting():
    assert format_usd(None) == "unknown"
    assert format_usd(0) == "$0"
    assert format_usd(12.5) == "$12.50"
    assert format_usd(0.0042) == "$0.0042"


def _workspace(tmp_path: Path, **extra) -> Path:
    (tmp_path / "q.jsonl").write_text("\n".join(map(json.dumps, ROWS)))
    # Importing this module would leave a marker: preview and refusals must never import it.
    (tmp_path / "spend_adapter.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('IMPORTED').write_text('yes')\n"
        "from vectraudit.pipelines import Hit, PipelineResult\n"
        "def run(request):\n"
        "    return PipelineResult(hits=[Hit(id='a')], cost_usd=0.003)\n"
    )
    spec = {
        "dataset": "q.jsonl",
        "repeats": 1,
        "candidates": [
            {
                "name": "spendy",
                "adapter": {"kind": "python", "target": "spend_adapter:run"},
                **extra,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return path


def test_cli_preview_and_guards_never_import_the_adapter(tmp_path):
    path = _workspace(tmp_path)
    runner = CliRunner()
    shown = runner.invoke(
        app, ["workbench", "preview", str(path), "--json", str(tmp_path / "plan.json")]
    )
    assert shown.exit_code == 0, shown.output
    assert "spendy: 10 questions x 1 repeats = 10 calls; per call unknown" in shown.output
    assert "Spend: unknown" in shown.output
    assert json.loads((tmp_path / "plan.json").read_text())["calls"]["max"] == 10
    out = ["--out", str(tmp_path / "runs")]
    refused = runner.invoke(app, ["workbench", "run", str(path), *out, "--max-calls", "5"])
    assert refused.exit_code == 2
    assert "exceed --max-calls 5" in refused.output
    unknown = runner.invoke(app, ["workbench", "run", str(path), *out, "--max-spend", "1"])
    assert unknown.exit_code == 2
    assert "unknown for spendy" in unknown.output
    assert not (tmp_path / "IMPORTED").exists()

    ran = runner.invoke(app, ["workbench", "run", str(path), *out, "--max-calls", "10"])
    assert ran.exit_code == 0, ran.output
    recorded = next((tmp_path / "runs").glob("*.run.json"))
    priced = runner.invoke(app, ["workbench", "preview", str(path), "--measured", str(recorded)])
    assert "per call $0.0030 (measured); spend $0.030" in priced.output
    within = runner.invoke(
        app,
        ["workbench", "run", str(path), *out, "--max-spend", "0.05", "--measured", str(recorded)],
    )
    assert within.exit_code == 0, within.output
    over = runner.invoke(
        app,
        ["workbench", "run", str(path), *out, "--max-spend", "0.01", "--measured", str(recorded)],
    )
    assert over.exit_code == 2
    assert "exceeds --max-spend $0.0100" in over.output


def test_optimize_and_monitor_refuse_over_budget_before_importing(tmp_path):
    path = _workspace(tmp_path)
    runner = CliRunner()
    optimized = runner.invoke(
        app,
        ["workbench", "optimize", str(path), "--out", str(tmp_path / "o.json"), "--max-calls", "3"],
    )
    assert optimized.exit_code == 2
    assert "Planned 6 to 10 calls" not in optimized.output  # the maximum is what is checked
    assert "Planned 10 calls exceed --max-calls 3" in optimized.output
    baseline = execute_pipeline(DATASET, lambda r: PipelineResult(hits=[Hit(id="a")]))
    from vectraudit.evaluation.runfile import save_run

    base = save_run(baseline, tmp_path / "base.run.json")
    monitored = runner.invoke(
        app, ["workbench", "monitor", str(path), str(base), "--max-spend", "1"]
    )
    assert monitored.exit_code == 2
    assert "unknown for spendy" in monitored.output
    assert not (tmp_path / "IMPORTED").exists()
