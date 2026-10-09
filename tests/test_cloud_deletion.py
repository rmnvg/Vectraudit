"""Workspace export and deletion of projects, workspaces and users."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import func, select
from typer.testing import CliRunner

from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import form_csrf, sign_in
from vectraudit.cloud import service
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import (
    ApiToken,
    AuditEvent,
    Baseline,
    Check,
    Membership,
    Project,
    Run,
    WebSession,
    Workspace,
    utcnow,
)

TENANT_TABLES = (Project, Run, Baseline, Check, Membership, AuditEvent)


@pytest.fixture
def populated(world):
    for ws in ("acme", "globex"):
        client = client_for(world, f"{ws}_runner", f"{ws}/search")
        client.upload_run(a_run("r1"))
        client.upload_run(a_run("r2"))
        world.client.put(
            f"/api/v1/workspaces/{ws}/projects/search/baselines/main",
            json={"run_id": "r1"},
            headers=world.as_(f"{ws}_owner"),
        )
        client.create_check("r2")
    return world


def rows(app, model, workspace_slug: str) -> int:
    with transaction(app.state.sessions) as s:
        ws_id = s.scalar(select(Workspace.id).where(Workspace.slug == workspace_slug))
        if ws_id is None:
            return 0
        return int(
            s.scalar(select(func.count()).select_from(model).where(model.workspace_id == ws_id))
            or 0
        )


def test_export(populated, cloud_app) -> None:
    w = populated
    r = w.client.get("/api/v1/workspaces/acme/export", headers=w.as_("acme_owner"))
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    data = r.json()
    assert data["format"] == "vectraudit.workspace-export"
    assert data["workspace"]["slug"] == "acme"
    assert {m["email"] for m in data["members"]} == {
        "owner@acme.test",
        "engineer@acme.test",
        "viewer@acme.test",
    }
    search = next(p for p in data["projects"] if p["slug"] == "search")
    assert [run["run"]["run_id"] for run in search["runs"]] == ["r1", "r2"]
    assert search["baselines"][0]["run_id"] == "r1"
    assert search["checks"][0]["candidate_run_id"] == "r2"
    assert search["runner_tokens"][0]["prefix"].startswith("va_run_")
    text = json.dumps(data)
    assert all(token not in text for token in w.tokens.values())
    assert "globex" not in text
    for who in ("acme_engineer", "acme_runner", "globex_owner"):
        assert w.client.get("/api/v1/workspaces/acme/export", headers=w.as_(who)).status_code in (
            403,
            404,
        )
    audit = w.client.get("/api/v1/workspaces/acme/audit", headers=w.as_("acme_owner")).json()
    assert "workspace.export" in [e["action"] for e in audit]


def test_delete_project(populated, cloud_app) -> None:
    w = populated
    url = "/api/v1/workspaces/acme/projects/search"
    assert w.client.request("DELETE", url, json={}, headers=w.as_("acme_owner")).status_code == 422
    wrong = w.client.request("DELETE", url, json={"confirm": "acme"}, headers=w.as_("acme_owner"))
    assert wrong.json()["error_code"] == "confirmation_required"
    assert (
        w.client.request(
            "DELETE", url, json={"confirm": "search"}, headers=w.as_("acme_engineer")
        ).status_code
        == 403
    )
    done = w.client.request("DELETE", url, json={"confirm": "search"}, headers=w.as_("acme_owner"))
    assert done.status_code == 200
    assert done.json() == {
        "shared_records": 0,
        "decisions": 1,
        "checks": 1,
        "baselines": 1,
        "runs": 2,
        "runner_tokens": 1,
    }
    assert w.client.get(url, headers=w.as_("acme_owner")).status_code == 404
    assert w.client.get("/api/v1/workspaces/acme", headers=w.as_("acme_runner")).status_code == 401
    assert rows(cloud_app, Run, "globex") == 2
    audit = w.client.get("/api/v1/workspaces/acme/audit", headers=w.as_("acme_owner")).json()
    assert audit[0]["action"] == "project.delete"


def test_delete_workspace_leaves_nothing(populated, cloud_app) -> None:
    w = populated
    before = {m.__name__: rows(cloud_app, m, "globex") for m in TENANT_TABLES}
    r = w.client.request(
        "DELETE", "/api/v1/workspaces/acme", json={"confirm": "acme"}, headers=w.as_("acme_owner")
    )
    assert r.status_code == 200
    assert r.json()["projects"] == 2
    with transaction(cloud_app.state.sessions) as s:
        assert s.scalar(select(Workspace).where(Workspace.slug == "acme")) is None
        for model in (*TENANT_TABLES, ApiToken):
            assert (
                s.scalar(
                    select(func.count())
                    .select_from(model)
                    .where(model.workspace_id.notin_(select(Workspace.id)))
                )
                == 0
            ), model.__name__
        record = s.scalars(select(AuditEvent).where(AuditEvent.action == "workspace.delete")).one()
        assert record.workspace_id is None
        assert record.details["slug"] == "acme"
    after = {m.__name__: rows(cloud_app, m, "globex") for m in TENANT_TABLES}
    assert after == before  # the other tenant is untouched
    # Members keep their accounts and tokens; they just have no workspace now.
    me = w.client.get("/api/v1/me", headers=w.as_("acme_engineer")).json()
    assert me["workspaces"] == []


def test_delete_workspace_from_dashboard(populated, cloud_app) -> None:
    owner = sign_in(cloud_app, populated.tokens["acme_owner"])
    page = owner.get("/app/acme").text
    assert "Delete this workspace" in page
    csrf = form_csrf(page)
    wrong = owner.post("/app/acme/delete", data={"csrf": csrf, "confirm": "nope"})
    assert wrong.status_code == 422
    done = owner.post("/app/acme/delete", data={"csrf": csrf, "confirm": "acme"})
    assert done.status_code == 303
    assert rows(cloud_app, Run, "acme") == 0
    viewer = sign_in(cloud_app, populated.tokens["globex_owner"])
    assert "Delete this workspace" in viewer.get("/app/globex").text
    engineer = sign_in(cloud_app, populated.tokens["acme_engineer"])
    assert engineer.get("/app/acme").status_code == 404


def test_delete_user(populated, cloud_app, cloud_settings, monkeypatch) -> None:
    from vectraudit.cloud.cli import app

    monkeypatch.setenv("VECTRAUDIT_CLOUD_DATABASE_URL", cloud_settings.database_url)
    monkeypatch.setenv("VECTRAUDIT_CLOUD_SECRET_KEY", cloud_settings.secret_key)
    runner = CliRunner()
    sign_in(cloud_app, populated.tokens["acme_engineer"])
    assert runner.invoke(app, ["delete-user", "engineer@acme.test"]).exit_code == 1  # no --yes
    last = runner.invoke(app, ["delete-user", "owner@acme.test", "--yes"])
    assert last.exit_code == 1
    assert "last owner of acme" in last.output
    done = runner.invoke(app, ["delete-user", "engineer@acme.test", "--yes"])
    assert done.exit_code == 0, done.output
    assert "removed from: acme" in done.output
    assert (
        populated.client.get("/api/v1/me", headers=populated.as_("acme_engineer")).status_code
        == 401
    )
    with transaction(cloud_app.state.sessions) as s:
        assert s.scalar(select(func.count()).select_from(WebSession)) == 0
    assert runner.invoke(app, ["delete-user", "ghost@acme.test", "--yes"]).exit_code == 1


def test_purge_old_sessions(world, cloud_app) -> None:
    sign_in(cloud_app, world.tokens["acme_viewer"])
    sign_in(cloud_app, world.tokens["acme_owner"])
    with transaction(cloud_app.state.sessions) as s:
        first = s.scalars(select(WebSession)).first()
        first.expires_at = utcnow() - timedelta(days=31)
    with transaction(cloud_app.state.sessions) as s:
        assert service.purge_sessions(s) == 1
        assert s.scalar(select(func.count()).select_from(WebSession)) == 1
