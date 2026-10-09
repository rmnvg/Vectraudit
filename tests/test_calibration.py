"""A/A calibration: false alarms on an unchanged system, and a draft that only absorbs noise."""

from __future__ import annotations

import copy
import json

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.calibration import calibrate
from vectraudit.evaluation.policy import parse_policy, policy_sha256
from vectraudit.evaluation.runfile import save_run

N = 20


def noisy_runs(count: int = 5, *, jitter: float = 0.02, latency=(1.0, 1.0)):
    """``count`` recordings of one unchanged system whose scores and latency wobble."""
    runs = []
    for r in range(count):
        values = [0.7 + jitter * (((q + r) % 3) - 1) for q in range(N)]
        run = fake_run(
            values,
            run_id=f"aa-{r}",
            latency=latency[0] + (latency[1] - latency[0]) * (r % 2),
            tags=[["billing"] if q < 10 else ["auth"] for q in range(N)],
        )
        run["summary"] = {
            "latency_ms": {
                "p50": run["queries"][0]["latency_ms"],
                "p95": run["queries"][0]["latency_ms"],
            }
        }
        runs.append(run)
    return runs


STRICT = {
    "metrics": {"ndcg": {"max_drop": 0.001}},
    "min_queries": 10,
    "max_regressed_queries": 0,
    "cohorts": {"min_queries": 5, "metrics": {"ndcg": {"max_drop": 0.001}}},
}


def test_strict_policy_false_alarms_are_measured_and_the_draft_absorbs_only_noise():
    runs = noisy_runs()
    before = copy.deepcopy(STRICT)
    report = calibrate(runs, STRICT)
    assert before == STRICT  # the input policy is never modified
    current = report["current"]
    assert current["comparisons"] == 5 * 4
    assert current["false_alarms"] > 0
    assert set(current["by_rule"]) >= {"metric:ndcg.max_drop", "regressed_queries"}
    changed = {c["path"]: c for c in report["changes"]}
    assert set(changed) == {
        "metrics.ndcg.max_drop",
        "cohorts.metrics.ndcg.max_drop",
        "max_regressed_queries",
    }
    drop = changed["metrics.ndcg.max_drop"]
    assert drop["to"] >= drop["observed"] * 1.5 - 1e-9
    assert drop["to"] > 0.001
    assert report["draft"]["false_alarms"] == 0
    draft = report["draft_policy"]
    assert draft["min_queries"] == STRICT["min_queries"]  # untouched settings stay as they were
    assert policy_sha256(parse_policy(draft)) != report["policy_sha256"]
    assert report["not_noise"] == []
    assert any("not independent" in note for note in report["notes"])


def test_a_policy_wider_than_the_noise_needs_no_change():
    roomy = {"metrics": {"ndcg": {"max_drop": 0.2}}, "min_queries": 10}
    report = calibrate(noisy_runs(), roomy)
    assert report["current"]["false_alarms"] == 0
    assert report["changes"] == []
    assert report["draft_policy"] is None
    assert report["draft"] == report["current"]


def test_latency_jitter_raises_the_absolute_floor_but_never_the_budget():
    runs = noisy_runs(jitter=0.0, latency=(10.0, 13.0))
    policy = {
        "metrics": {"ndcg": {"max_drop": 0.01}},
        "min_queries": 10,
        "latency": {"p95_max_increase_pct": 5, "min_increase_ms": 0, "p95_max_ms": 12},
    }
    report = calibrate(runs, policy)
    changed = {c["path"]: c for c in report["changes"]}
    assert set(changed) == {"latency.min_increase_ms"}
    assert changed["latency.min_increase_ms"]["observed"] == pytest.approx(3.0)
    assert changed["latency.min_increase_ms"]["to"] == 4.5
    assert report["draft_policy"]["latency"]["p95_max_ms"] == 12  # a budget, not noise
    assert report["not_noise"] == ["latency.p95_max_ms"]
    assert report["draft"]["by_rule"] == {"latency.p95_max_ms": report["draft"]["false_alarms"]}
    assert any("not noise" in note for note in report["notes"])


def test_intermittent_critical_misses_are_reported_not_absorbed():
    runs = noisy_runs(jitter=0.0)
    for run in runs:
        for q in run["queries"]:
            q["repeat_ids"] = [q["ids"]]
        run["timing"] = {"repeats": 1}
    runs[2]["queries"][0]["ids"] = ["x", "y"]
    runs[2]["queries"][0]["repeat_ids"] = [["x", "y"]]
    policy = {
        "metrics": {"ndcg": {"max_drop": 0.5}},
        "min_queries": 10,
        "critical_queries": [{"query_id": "q0", "doc_id": "a", "max_rank": 2}],
    }
    report = calibrate(runs, policy)
    assert report["not_noise"] == ["critical:q0:a"]
    assert report["changes"] == []
    assert report["current"]["by_rule"]["critical:q0:a"] == 4  # runs[2] as the candidate


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda runs: runs[:1], "at least 2"),
        (lambda runs: [runs[0], dict(runs[1], status="partial")], "complete"),
        (
            lambda runs: [runs[0], dict(runs[1], dataset={"sha256": "e" * 64, "n_queries": N})],
            "same dataset",
        ),
        (lambda runs: [runs[0], runs[0]], "separate recording"),
    ],
)
def test_invalid_inputs_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        calibrate(change(noisy_runs(2)), STRICT)


def test_margin_bounds():
    with pytest.raises(ValueError, match="margin"):
        calibrate(noisy_runs(2), STRICT, margin=0.5)


def test_cli_writes_new_files_only(tmp_path):
    paths = [save_run(run, tmp_path / f"aa-{i}.json") for i, run in enumerate(noisy_runs())]
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(STRICT))
    draft, report = tmp_path / "draft.json", tmp_path / "calibration.json"
    runner = CliRunner(env={"COLUMNS": "200"})
    args = ["calibrate", *map(str, paths), "--policy", str(policy)]
    result = runner.invoke(app, [*args, "--draft", str(draft), "--json", str(report)])
    assert result.exit_code == 0, result.output
    assert "20 A/A comparisons" in result.output
    assert "metric:ndcg.max_drop" in result.output
    written = json.loads(draft.read_text())
    assert written["metrics"]["ndcg"]["max_drop"] > STRICT["metrics"]["ndcg"]["max_drop"]
    assert json.loads(report.read_text())["format"] == "vectraudit.calibration"
    assert json.loads(policy.read_text()) == STRICT  # the reviewed policy is untouched
    again = runner.invoke(app, [*args, "--draft", str(draft)])
    assert again.exit_code == 1
    assert "never overwrites" in again.output
    bad = runner.invoke(app, ["calibrate", str(paths[0]), "--policy", str(policy)])
    assert bad.exit_code == 1
    assert "at least 2" in bad.output
