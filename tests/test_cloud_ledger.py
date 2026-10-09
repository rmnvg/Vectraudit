"""Release decisions outlive raw-run retention; the timeline shows them with baseline pins."""

from __future__ import annotations

from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")

from tests.test_cloud_checks import pin
from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import sign_in
from vectraudit.cloud import quotas
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import Check, Decision, Run, utcnow

TIMELINE = "/api/v1/workspaces/acme/projects/search/decisions"
PRIVATE = "PRIVATE_QUESTION_TEXT"


def _worse(run_id: str) -> dict:
    run = a_run(run_id, label="candidate-v2")
    for q in run["queries"]:
        q["metrics"] = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    run["queries"][0]["text"] = PRIVATE  # never uploaded without explicit opt-in
    return run


def _age(cloud_app, run_ids: list[str], days: int = 40) -> None:
    with transaction(cloud_app.state.sessions) as s:
        for run in s.query(Run).filter(Run.run_id.in_(run_ids)):
            run.uploaded_at = utcnow() - timedelta(days=days)


@pytest.fixture
def decided(world):
    """A pinned baseline, one failing check and one passing check in acme/search."""
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base", note="first release")
    client.upload_run(_worse("worse"))
    client.upload_run(a_run("same", label="candidate-v3"))
    failing = client.create_check("worse")
    passing = client.create_check("same")
    return client, failing, passing


def test_each_check_records_one_compact_decision(world, cloud_app, decided) -> None:
    client, failing, _ = decided
    again = client.create_check("worse")  # idempotent check: no second decision
    assert again["id"] == failing["id"]
    with transaction(cloud_app.state.sessions) as s:
        rows = {d.candidate_run_id: d for d in s.query(Decision)}
        assert set(rows) == {"worse", "same"}
        worse = rows["worse"]
        assert worse.verdict == "fail"
        assert worse.check_id == failing["id"]
        assert worse.baseline_name == "main"
        assert worse.baseline_run_id == "base"
        assert worse.candidate_label == "candidate-v2"
        assert worse.candidate_commit == "abc123"
        summary = worse.summary
        assert summary["metrics"]["ndcg"]["delta"] < 0
        assert summary["reasons"]
        assert summary["n_queries"] == 20
        assert PRIVATE not in str(summary)
        assert set(summary) == {
            "version",
            "n_queries",
            "n_families",
            "metrics",
            "primary_metric",
            "regressed_queries",
            "latency",
            "critical",
            "reasons",
            "n_reasons",
            "n_warnings",
            "changed_settings",
            "verdict_before_exceptions",
            "accepted",
            "drift",
        }
        assert summary["drift"] is None  # only monitor checks record drift
        assert worse.trigger == "release"


def test_decisions_survive_retention_and_the_timeline_says_so(world, cloud_app, decided) -> None:
    _, failing, _ = decided
    live = world.client.get(TIMELINE, headers=world.as_("acme_viewer")).json()["entries"]
    kinds = [e["kind"] for e in live]
    assert kinds.count("decision") == 2
    assert kinds.count("baseline_pin") == 1
    fail_entry = next(e for e in live if e.get("check_id") == failing["id"])
    assert fail_entry["evidence_available"] is True
    assert fail_entry["report_url"].endswith(f"/checks/{failing['id']}/report")

    _age(cloud_app, ["worse"])
    with transaction(cloud_app.state.sessions) as s:
        assert quotas.purge_expired(s) == {"acme": 1}
    with transaction(cloud_app.state.sessions) as s:
        assert s.query(Check).filter(Check.id == failing["id"]).count() == 0
        assert s.query(Decision).filter(Decision.check_id == failing["id"]).count() == 1

    later = world.client.get(TIMELINE, headers=world.as_("acme_viewer")).json()["entries"]
    expired = next(e for e in later if e.get("check_id") == failing["id"])
    assert expired["verdict"] == "fail"
    assert expired["evidence_available"] is False
    assert expired["report_url"] is None
    assert expired["summary"]["metrics"]["ndcg"]["delta"] < 0
    pin_entry = next(e for e in later if e["kind"] == "baseline_pin")
    assert pin_entry["run_id"] == "base"
    assert pin_entry["note"] == "first release"


def test_purge_backfills_checks_that_predate_the_ledger(world, cloud_app, decided) -> None:
    _, failing, _ = decided
    with transaction(cloud_app.state.sessions) as s:
        s.query(Decision).delete()  # as if the checks were made before the ledger existed
    _age(cloud_app, ["worse"])
    with transaction(cloud_app.state.sessions) as s:
        quotas.purge_expired(s)
    with transaction(cloud_app.state.sessions) as s:
        kept = s.query(Decision).one()
        assert kept.check_id == failing["id"]
        assert kept.verdict == "fail"
        assert kept.baseline_name is None  # unknown for back-filled rows


def test_timeline_pages_newest_first(world, decided) -> None:
    first = world.client.get(TIMELINE, params={"limit": 2}, headers=world.as_("acme_viewer")).json()
    assert len(first["entries"]) == 2
    assert first["next_cursor"]
    rest = world.client.get(
        TIMELINE, params={"cursor": first["next_cursor"]}, headers=world.as_("acme_viewer")
    ).json()
    times = [e["at"] for e in first["entries"] + rest["entries"]]
    assert times == sorted(times, reverse=True)
    assert len(times) == 3


def test_dashboard_shows_the_timeline_after_expiry(world, cloud_app, decided) -> None:
    _age(cloud_app, ["worse"])
    with transaction(cloud_app.state.sessions) as s:
        quotas.purge_expired(s)
    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    page = browser.get("/app/acme/search").text
    assert "Release timeline" in page
    assert "Report expired under retention" in page
    assert "candidate-v2" in page
    assert "Baseline pinned" in page
    assert "ndcg 0.750 → 0.000 (-0.750)" in page
    assert PRIVATE not in page


def test_export_includes_decisions(world, decided) -> None:
    exported = world.client.get(
        "/api/v1/workspaces/acme/export", headers=world.as_("acme_owner")
    ).json()
    search = next(p for p in exported["projects"] if p["slug"] == "search")
    assert {d["candidate_run_id"] for d in search["decisions"]} == {"worse", "same"}


def test_admin_backfill_records_existing_checks_once(
    world, cloud_app, cloud_settings, decided, monkeypatch
) -> None:
    from typer.testing import CliRunner

    from vectraudit.cloud.cli import app

    monkeypatch.setenv("VECTRAUDIT_CLOUD_DATABASE_URL", cloud_settings.database_url)
    monkeypatch.setenv("VECTRAUDIT_CLOUD_SECRET_KEY", cloud_settings.secret_key)
    with transaction(cloud_app.state.sessions) as s:
        s.query(Decision).delete()  # checks made before the ledger existed
    runner = CliRunner()
    first = runner.invoke(app, ["backfill-decisions"])
    assert first.exit_code == 0, first.output
    assert "Recorded 2 decisions" in first.output
    again = runner.invoke(app, ["backfill-decisions"])
    assert "Recorded 0 decisions" in again.output
    with transaction(cloud_app.state.sessions) as s:
        assert s.query(Decision).count() == 2
