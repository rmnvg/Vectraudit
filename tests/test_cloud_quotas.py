"""Limit profiles, upload quotas, usage export and retention."""

from __future__ import annotations

import csv
import io
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests.test_cloud_checks import pin
from tests.test_cloud_runs import a_run, client_for
from vectraudit.cloud import quotas
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import Check, Run, Workspace, utcnow
from vectraudit.integrations.cloud import CloudClientError


def set_limits(app, slug: str, **limits: int) -> None:
    with transaction(app.state.sessions) as s:
        ws = s.query(Workspace).filter(Workspace.slug == slug).one()
        ws.limits = limits


def test_limits_for_plans_and_overrides() -> None:
    ws = Workspace(slug="x1", name="X", plan="large")
    assert quotas.limits_for(ws) == quotas.PLANS["large"]
    ws.limits = {"runs_per_month": 7, "bogus": 3, "stored_runs": -1, "retention_days": "9"}
    lim = quotas.limits_for(ws)
    assert lim.runs_per_month == 7
    assert lim.stored_runs == quotas.PLANS["large"].stored_runs
    assert lim.retention_days == quotas.PLANS["large"].retention_days
    unknown_plan = Workspace(slug="y1", name="Y", plan="mystery")
    assert quotas.limits_for(unknown_plan) == quotas.PLANS["standard"]


def test_upload_quotas(world, cloud_app) -> None:
    client = client_for(world, "acme_runner")
    set_limits(cloud_app, "acme", runs_per_month=2, queries_per_run=15)
    with pytest.raises(CloudClientError) as e:
        client.upload_run(a_run("big"))  # 20 queries
    assert e.value.status == 413
    set_limits(cloud_app, "acme", runs_per_month=2)
    client.upload_run(a_run("r1"))
    client.upload_run(a_run("r2"))
    with pytest.raises(CloudClientError) as e:
        client.upload_run(a_run("r3"))
    assert e.value.status == 429
    assert e.value.code == "quota_exceeded"
    assert client.upload_run(a_run("r2"))["run_id"] == "r2"  # retries are never blocked
    # Other workspaces are unaffected.
    client_for(world, "globex_runner", "globex/search").upload_run(a_run("r3"))
    set_limits(cloud_app, "acme", stored_runs=2)
    with pytest.raises(CloudClientError) as e:
        client.upload_run(a_run("r4"))
    assert e.value.code == "storage_full"


def test_usage_endpoints(world, cloud_app) -> None:
    client_for(world, "acme_runner").upload_run(a_run("u1"))
    status = world.client.get(
        "/api/v1/workspaces/acme/usage", headers=world.as_("acme_viewer")
    ).json()
    assert status["plan"] == "standard"
    assert status["usage"]["runs"] == 1
    assert status["usage"]["queries"] == 20
    assert status["remaining"]["runs_this_month"] == quotas.PLANS["standard"].runs_per_month - 1
    assert (
        world.client.get(
            "/api/v1/workspaces/acme/usage", headers=world.as_("acme_runner")
        ).status_code
        == 403
    )
    assert (
        world.client.get(
            "/api/v1/workspaces/acme/usage", headers=world.as_("globex_owner")
        ).status_code
        == 404
    )
    assert (
        world.client.get("/api/v1/admin/usage", headers=world.as_("acme_owner")).status_code == 403
    )
    assert (
        world.client.get(
            "/api/v1/admin/usage", params={"month": "October"}, headers=world.as_("admin")
        ).status_code
        == 422
    )
    exported = world.client.get("/api/v1/admin/usage", headers=world.as_("admin"))
    assert exported.headers["content-type"].startswith("text/csv")
    rows = {r["workspace"]: r for r in csv.DictReader(io.StringIO(exported.text))}
    assert rows["acme"]["runs"] == "1"
    assert rows["globex"]["runs"] == "0"
    past = world.client.get(
        "/api/v1/admin/usage", params={"month": "2020-01"}, headers=world.as_("admin")
    )
    assert all(r["runs"] == "0" for r in csv.DictReader(io.StringIO(past.text)))


def test_purge_keeps_baselines(world, cloud_app) -> None:
    client = client_for(world, "acme_runner")
    for rid in ("old-base", "old", "fresh"):
        client.upload_run(a_run(rid))
    pin(world, "acme_engineer", "old-base")
    client.create_check("old")
    with transaction(cloud_app.state.sessions) as s:
        for run in s.query(Run).filter(Run.run_id.in_(["old-base", "old"])):
            run.uploaded_at = utcnow() - timedelta(days=40)
    with transaction(cloud_app.state.sessions) as s:
        assert quotas.purge_expired(s) == {"acme": 1}
    with transaction(cloud_app.state.sessions) as s:
        left = {r.run_id for r in s.query(Run).filter(Run.workspace_id.isnot(None))}
        assert left == {"old-base", "fresh"}
        assert s.query(Check).count() == 0


def test_admin_cli(world, cloud_app, cloud_settings, monkeypatch) -> None:
    from vectraudit.cloud.cli import app

    monkeypatch.setenv("VECTRAUDIT_CLOUD_DATABASE_URL", cloud_settings.database_url)
    monkeypatch.setenv("VECTRAUDIT_CLOUD_SECRET_KEY", cloud_settings.secret_key)
    runner = CliRunner()
    r = runner.invoke(app, ["set-plan", "acme", "large", "--limit", "runs_per_month=9"])
    assert r.exit_code == 0, r.output
    assert "runs_per_month=9" in r.output
    assert runner.invoke(app, ["set-plan", "acme", "gold"]).exit_code == 1
    assert runner.invoke(app, ["set-plan", "acme", "large", "--limit", "x=1"]).exit_code == 1
    assert runner.invoke(app, ["set-plan", "nope", "large"]).exit_code == 1
    usage = runner.invoke(app, ["usage"])
    assert usage.exit_code == 0
    assert usage.output.splitlines()[0].startswith("workspace,plan,month")
    client_for(world, "acme_runner").upload_run(a_run("ancient"))
    with transaction(cloud_app.state.sessions) as s:
        s.query(Run).filter(Run.run_id == "ancient").one().uploaded_at = utcnow() - timedelta(
            days=400
        )
    purged = runner.invoke(app, ["purge"])
    assert purged.exit_code == 0, purged.output
    assert "Purged 1 expired runs: acme 1" in purged.output
    audit = world.client.get("/api/v1/workspaces/acme/audit", headers=world.as_("acme_owner"))
    actions = [e["action"] for e in audit.json()]
    assert "retention.purge" in actions
    assert "workspace.plan" in actions


def test_dashboard_shows_limits(world, cloud_app) -> None:
    from tests.test_cloud_web import sign_in

    page = sign_in(cloud_app, world.tokens["acme_viewer"]).get("/app/acme").text
    assert "Limits <strong>standard</strong>" in page
    assert "runs kept 30 days" in page
