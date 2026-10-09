"""Scheduled monitor runs in hosted history: marked, explained, and one alert per streak."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests.test_cloud_checks import pin, via
from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import sign_in
from vectraudit.cli import app as cli
from vectraudit.cloud import quotas
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import Decision, Run, utcnow
from vectraudit.integrations import cloud as cloud_client

P = "/api/v1/workspaces/acme/projects/search"


def _bad(run_id: str) -> dict:
    run = a_run(run_id, label="nightly")
    for q in run["queries"]:
        q["metrics"] = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
        q["ids"] = ["x", "y"]
    return run


def _status(world) -> dict:
    return world.client.get(f"{P}/monitor", headers=world.as_("acme_viewer")).json()


def test_monitor_checks_are_marked_with_drift_and_open_one_alert(world, cloud_app) -> None:
    client = client_for(world, "acme_runner")
    base = a_run("base")
    base["backend"]["corpus_version"] = "v1"
    client.upload_run(base)
    pin(world, "acme_engineer", "base")
    assert _status(world) == {"state": "none", "last": None, "alert": None}

    ok = a_run("night-1", label="nightly")
    ok["backend"]["corpus_version"] = "v1"
    client.upload_run(ok)
    first = client.create_check("night-1", trigger="monitor")
    assert first["trigger"] == "monitor"
    assert first["result"]["drift"]["corpus"] == "unchanged"
    assert _status(world)["state"] == "healthy"

    failing = []
    for i in (2, 3):
        run = _bad(f"night-{i}")
        run["backend"]["corpus_version"] = "v2"
        client.upload_run(run)
        failing.append(client.create_check(f"night-{i}", trigger="monitor"))
    status = _status(world)
    assert status["state"] == "alerting"
    assert status["alert"]["alert_id"] == failing[0]["id"]
    assert status["alert"]["failing_runs"] == 2
    assert status["last"]["check_id"] == failing[1]["id"]

    summary = world.client.get(
        f"{P}/decisions/{failing[1]['id']}/summary", headers=world.as_("acme_viewer")
    ).json()
    assert summary["trigger"] == "monitor"
    assert summary["headline"].startswith("Scheduled monitor: nightly @ abc123 now breaks")
    assert "Corpus version changed since the baseline." in summary["changes"]
    assert "Returned documents changed for 20 questions." in summary["changes"]

    # A release check of the same run keeps the first check's trigger (identical check).
    again = client.create_check("night-3")
    assert again["id"] == failing[1]["id"]
    assert again["trigger"] == "monitor"

    # Retention removes the runs, not the monitor history or its alert.
    with transaction(cloud_app.state.sessions) as s:
        for run in s.query(Run).filter(Run.run_id.in_(["night-1", "night-2", "night-3"])):
            run.uploaded_at = utcnow() - timedelta(days=40)
    with transaction(cloud_app.state.sessions) as s:
        quotas.purge_expired(s)
        rows = s.query(Decision).filter(Decision.trigger == "monitor").count()
    assert rows == 3
    assert _status(world)["alert"]["alert_id"] == failing[0]["id"]

    fixed = a_run("night-4", label="nightly")
    client.upload_run(fixed)
    resolver = client.create_check("night-4", trigger="monitor")
    healthy = _status(world)
    assert healthy["state"] == "healthy"
    assert healthy["alert"]["alert_id"] == failing[0]["id"]
    assert healthy["alert"]["resolved_by"] == resolver["id"]

    timeline = world.client.get(f"{P}/decisions", headers=world.as_("acme_viewer")).json()
    triggers = {e["check_id"]: e["trigger"] for e in timeline["entries"] if e["kind"] == "decision"}
    assert set(triggers.values()) == {"monitor"}
    assert world.client.get(f"{P}/monitor", headers=world.as_("globex_owner")).status_code == 404


def test_dashboard_separates_monitor_runs_from_release_decisions(world, cloud_app) -> None:
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    client.upload_run(_bad("night-1"))
    alert = client.create_check("night-1", trigger="monitor")
    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    page = browser.get("/app/acme/search").text
    assert "Latest release decision" not in page  # a monitor run is not a release decision
    assert "Scheduled monitor" in page
    assert "Alerting since" in page
    assert f"/app/acme/search/decisions/{alert['id']}" in page
    assert '<span class="muted">Monitor</span>' in page
    client.upload_run(a_run("release-1", label="release"))
    client.create_check("release-1")
    page = browser.get("/app/acme/search").text
    assert "Latest release decision" in page
    assert "release @ abc123 passes" in page
    trends = browser.get("/app/acme/search/trends").text
    assert 'stroke-width="2"' in trends  # the monitor point is hollow
    assert "Scheduled monitor · " in trends


ADAPTER = """
import os

from vectraudit.pipelines import Hit, PipelineResult


def nightly(request):
    good = os.environ.get("MONITOR_MODE", "good") == "good"
    doc = "doc-" + request.query_id
    return PipelineResult(hits=[Hit(id=doc if good else "other"), Hit(id="filler")])
"""


def test_cli_monitor_uploads_alerts_once_and_resolves(world, tmp_path, monkeypatch) -> None:
    from vectraudit.evaluation.dataset import load_dataset
    from vectraudit.pipelines import Hit, PipelineOptions, PipelineResult, execute_pipeline

    rows = [{"query_id": f"q{i}", "text": f"q{i}", "relevant": {f"doc-q{i}": 1}} for i in range(20)]
    (tmp_path / "queries.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "monitor_adapters.py").write_text(ADAPTER)
    candidate = {
        "name": "nightly",
        "adapter": {"kind": "python", "target": "monitor_adapters:nightly"},
    }
    manifest = tmp_path / "monitor.json"
    manifest.write_text(
        json.dumps({"dataset": "queries.jsonl", "k": 2, "repeats": 1, "candidates": [candidate]})
    )
    baseline = execute_pipeline(
        load_dataset(tmp_path / "queries.jsonl"),
        lambda q: PipelineResult(hits=[Hit(id="doc-" + q.query_id), Hit(id="filler")]),
        PipelineOptions(k=2, repeats=1, warmup=0),
    )
    client = client_for(world, "acme_runner")
    client.upload_run(baseline)
    pin(world, "acme_engineer", baseline["run_id"])

    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_runner"])
    runner = CliRunner()
    history = tmp_path / "history"
    args = ["workbench", "monitor", str(manifest), "cloud:main", "--history", str(history)]

    verdicts, states = [], []
    for mode, code in (("good", 0), ("bad", 3), ("bad", 3), ("good", 0)):
        monkeypatch.setenv("MONITOR_MODE", mode)
        result = runner.invoke(cli, [*args, "--upload"])
        assert result.exit_code == code, result.output
        assert "Plain-language summary: vectraudit cloud summary" in result.output
        verdicts.append(json.loads(result.output.strip().splitlines()[-1])["verdict"])
        status = _status(world)
        states.append((status["state"], (status["alert"] or {}).get("alert_id")))
    assert verdicts == ["pass", "fail", "fail", "pass"]
    assert [state for state, _ in states] == ["healthy", "alerting", "alerting", "healthy"]
    assert states[1][1] == states[2][1] == states[3][1]  # one alert, delivered once
    status = _status(world)
    assert status["state"] == "healthy"
    assert status["alert"]["failing_runs"] == 2
    assert len(list(history.glob("*.monitor.json"))) == 4  # local history is still written

    local_only = runner.invoke(cli, ["workbench", "monitor", str(manifest), "x.json", "--upload"])
    assert local_only.exit_code == 2
    assert "pass cloud:<name>" in local_only.output
