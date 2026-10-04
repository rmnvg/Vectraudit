"""Accounts, roles, tokens, audit and tenant isolation in the hosted control plane."""

from __future__ import annotations

from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import ApiToken, utcnow


def test_me_and_workspaces(world) -> None:
    me = world.client.get("/api/v1/me", headers=world.as_("acme_engineer")).json()
    assert me["user"]["email"] == "engineer@acme.test"
    assert me["workspaces"] == [
        {"slug": "acme", "name": "Acme", "plan": "pilot", "role": "engineer"}
    ]
    assert world.client.get("/api/v1/me").status_code == 401
    bad = world.client.get("/api/v1/me", headers={"Authorization": "Bearer va_pat_nope"})
    assert bad.status_code == 401
    assert bad.json()["error_code"] == "unauthorized"
    runner = world.client.get("/api/v1/me", headers=world.as_("acme_runner"))
    assert runner.status_code == 403


ACME_ENDPOINTS = [
    ("GET", "/api/v1/workspaces/acme"),
    ("GET", "/api/v1/workspaces/acme/members"),
    ("POST", "/api/v1/workspaces/acme/members"),
    ("GET", "/api/v1/workspaces/acme/projects"),
    ("POST", "/api/v1/workspaces/acme/projects"),
    ("GET", "/api/v1/workspaces/acme/projects/search"),
    ("GET", "/api/v1/workspaces/acme/projects/search/runner-tokens"),
    ("POST", "/api/v1/workspaces/acme/projects/search/runner-tokens"),
    ("GET", "/api/v1/workspaces/acme/audit"),
]


@pytest.mark.parametrize(("method", "path"), ACME_ENDPOINTS)
@pytest.mark.parametrize("intruder", ["globex_owner", "globex_runner", "admin"])
def test_other_tenants_get_not_found(world, method, path, intruder) -> None:
    body = {"email": "x@y.test", "role": "viewer", "slug": "x1", "name": "x"}
    response = world.client.request(method, path, json=body, headers=world.as_(intruder))
    assert response.status_code == 404, (intruder, path, response.json())
    assert response.json()["error_code"] == "not_found"


def test_unknown_workspace_looks_the_same_as_a_foreign_one(world) -> None:
    foreign = world.client.get("/api/v1/workspaces/acme", headers=world.as_("globex_owner"))
    missing = world.client.get("/api/v1/workspaces/nope", headers=world.as_("globex_owner"))
    assert foreign.status_code == missing.status_code == 404
    assert (
        foreign.json()
        == missing.json()
        == {
            "error_code": "not_found",
            "message": "Workspace not found.",
            "hint": None,
        }
    )


def test_runner_tokens_are_bound_to_their_project(world) -> None:
    h = world.as_("acme_runner")
    assert world.client.get("/api/v1/workspaces/acme/projects/search", headers=h).status_code == 200
    assert world.client.get("/api/v1/workspaces/acme/projects/chat", headers=h).status_code == 404
    listed = world.client.get("/api/v1/workspaces/acme/projects", headers=h).json()
    assert [p["slug"] for p in listed] == ["search"]
    for method, path in [
        ("GET", "/api/v1/workspaces/acme/members"),
        ("POST", "/api/v1/workspaces/acme/projects"),
        ("GET", "/api/v1/workspaces/acme/projects/search/runner-tokens"),
        ("GET", "/api/v1/workspaces/acme/audit"),
    ]:
        r = world.client.request(method, path, json={"slug": "x1", "name": "x"}, headers=h)
        assert r.status_code == 403, path


def test_roles(world) -> None:
    c = world.client
    viewer, engineer, owner = (world.as_(f"acme_{r}") for r in ("viewer", "engineer", "owner"))
    assert c.get("/api/v1/workspaces/acme/members", headers=viewer).status_code == 200
    new = {"slug": "recs", "name": "Recs"}
    assert c.post("/api/v1/workspaces/acme/projects", json=new, headers=viewer).status_code == 403
    assert c.post("/api/v1/workspaces/acme/projects", json=new, headers=engineer).status_code == 201
    member = {"email": "new@acme.test", "role": "viewer"}
    denied = c.post("/api/v1/workspaces/acme/members", json=member, headers=engineer)
    assert denied.status_code == 403
    assert "owner role" in denied.json()["message"]
    added = c.post("/api/v1/workspaces/acme/members", json=member, headers=owner)
    assert added.status_code == 201
    dup = c.post("/api/v1/workspaces/acme/members", json=member, headers=owner)
    assert dup.status_code == 409
    uid = added.json()["user"]["id"]
    assert (
        c.patch(
            f"/api/v1/workspaces/acme/members/{uid}", json={"role": "engineer"}, headers=owner
        ).json()["role"]
        == "engineer"
    )
    bad = c.patch(f"/api/v1/workspaces/acme/members/{uid}", json={"role": "god"}, headers=owner)
    assert bad.status_code == 422
    assert c.delete(f"/api/v1/workspaces/acme/members/{uid}", headers=owner).status_code == 204
    assert c.get("/api/v1/workspaces/acme/audit", headers=engineer).status_code == 403


def test_last_owner_is_protected(world) -> None:
    owner = world.as_("acme_owner")
    me = world.client.get("/api/v1/me", headers=owner).json()["user"]["id"]
    r = world.client.patch(
        f"/api/v1/workspaces/acme/members/{me}", json={"role": "viewer"}, headers=owner
    )
    assert r.status_code == 409
    assert r.json()["error_code"] == "last_owner"
    r = world.client.delete(f"/api/v1/workspaces/acme/members/{me}", headers=owner)
    assert r.status_code == 409


def test_removed_members_lose_access(world) -> None:
    owner, viewer = world.as_("acme_owner"), world.as_("acme_viewer")
    members = world.client.get("/api/v1/workspaces/acme/members", headers=owner).json()
    vid = next(m["user"]["id"] for m in members if m["role"] == "viewer")
    assert (
        world.client.delete(f"/api/v1/workspaces/acme/members/{vid}", headers=owner).status_code
        == 204
    )
    assert world.client.get("/api/v1/workspaces/acme", headers=viewer).status_code == 404


def test_runner_token_lifecycle(world) -> None:
    c, engineer = world.client, world.as_("acme_engineer")
    base = "/api/v1/workspaces/acme/projects/search/runner-tokens"
    created = c.post(base, json={"name": "github", "scopes": ["runs:write"]}, headers=engineer)
    assert created.status_code == 201
    body = created.json()
    assert body["token"].startswith("va_run_")
    assert body["scopes"] == ["runs:write"]
    listed = c.get(base, headers=engineer).json()
    assert all("token" not in t for t in listed)
    assert body["prefix"] in [t["prefix"] for t in listed]
    runner = {"Authorization": f"Bearer {body['token']}"}
    # This token has only runs:write: reading the project is outside its scopes.
    assert c.get("/api/v1/workspaces/acme/projects/search", headers=runner).status_code == 403
    assert c.delete(f"{base}/{body['id']}", headers=engineer).status_code == 204
    assert c.get("/api/v1/workspaces/acme", headers=runner).status_code == 401
    bad_scope = c.post(base, json={"name": "x", "scopes": ["members:write"]}, headers=engineer)
    assert bad_scope.status_code == 422
    foreign = c.delete(f"{base}/{body['id']}", headers=world.as_("globex_owner"))
    assert foreign.status_code == 404


def test_personal_tokens(world, cloud_app) -> None:
    c, viewer = world.client, world.as_("acme_viewer")
    created = c.post(
        "/api/v1/tokens", json={"name": "laptop", "expires_in_days": 30}, headers=viewer
    ).json()
    assert created["token"].startswith("va_pat_")
    assert created["expires_at"] is not None
    new = {"Authorization": f"Bearer {created['token']}"}
    assert c.get("/api/v1/me", headers=new).status_code == 200
    tokens = c.get("/api/v1/tokens", headers=new).json()
    assert {t["name"] for t in tokens} == {"viewer", "laptop"}
    assert all("token" not in t for t in tokens)
    assert (
        c.delete(f"/api/v1/tokens/{created['id']}", headers=world.as_("acme_owner")).status_code
        == 404
    )
    assert c.delete(f"/api/v1/tokens/{created['id']}", headers=viewer).status_code == 204
    assert c.get("/api/v1/me", headers=new).status_code == 401
    with transaction(cloud_app.state.sessions) as s:
        row = s.query(ApiToken).filter(ApiToken.name == "viewer").one()
        row.expires_at = utcnow() - timedelta(seconds=1)
    expired = c.get("/api/v1/me", headers=viewer)
    assert expired.status_code == 401
    assert "expired" in expired.json()["message"]


def test_workspace_creation_is_admin_only(world) -> None:
    body = {"slug": "initech", "name": "Initech", "owner_email": "Boss@Initech.test"}
    assert (
        world.client.post(
            "/api/v1/workspaces", json=body, headers=world.as_("acme_owner")
        ).status_code
        == 403
    )
    created = world.client.post("/api/v1/workspaces", json=body, headers=world.as_("admin"))
    assert created.status_code == 201
    again = world.client.post("/api/v1/workspaces", json=body, headers=world.as_("admin"))
    assert again.status_code == 409
    bad = world.client.post(
        "/api/v1/workspaces", json={**body, "slug": "No Spaces"}, headers=world.as_("admin")
    )
    assert bad.status_code == 422
    # The admin created it but is not a member: no access to its data.
    assert (
        world.client.get("/api/v1/workspaces/initech", headers=world.as_("admin")).status_code
        == 404
    )


def test_audit_log_records_changes_without_secrets(world) -> None:
    owner = world.as_("acme_owner")
    world.client.post(
        "/api/v1/workspaces/acme/projects/search/runner-tokens", json={"name": "gh"}, headers=owner
    )
    events = world.client.get("/api/v1/workspaces/acme/audit", headers=owner).json()
    actions = [e["action"] for e in events]
    for action in ("workspace.create", "project.create", "member.add", "token.create"):
        assert action in actions
    text = str(events)
    assert all(secret not in text for secret in world.tokens.values())


def test_cli_bootstrap(cloud_settings, monkeypatch) -> None:
    from vectraudit.cloud.cli import app
    from vectraudit.cloud.migrate import upgrade

    upgrade(cloud_settings.database_url)
    monkeypatch.setenv("VECTRAUDIT_CLOUD_DATABASE_URL", cloud_settings.database_url)
    monkeypatch.setenv("VECTRAUDIT_CLOUD_SECRET_KEY", cloud_settings.secret_key)
    runner = CliRunner()
    result = runner.invoke(
        app, ["create-workspace", "acme", "--name", "Acme", "--owner-email", "owner@acme.test"]
    )
    assert result.exit_code == 0, result.output
    token = result.output.strip().splitlines()[-1]
    assert token.startswith("va_pat_")
    again = runner.invoke(
        app, ["create-workspace", "acme", "--name", "Acme", "--owner-email", "owner@acme.test"]
    )
    assert again.exit_code == 1
    assert "already exists" in again.output
    assert runner.invoke(app, ["create-admin", "ops@v.test"]).exit_code == 0
    assert runner.invoke(app, ["create-token", "owner@acme.test"]).exit_code == 0
    assert runner.invoke(app, ["create-token", "nobody@acme.test"]).exit_code == 1

    from vectraudit.cloud.app import create_app

    client = TestClient(create_app(cloud_settings))
    me = client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["workspaces"][0]["role"] == "owner"
    client.app.state.engine.dispose()  # type: ignore[attr-defined]
