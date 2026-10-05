"""Project restrictions, export controls, SSO boundaries, and schema upgrade."""

import dataclasses
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient

from vectraudit.cloud import sso
from vectraudit.cloud.app import create_app


def test_project_allowlist_caps_existing_memberships(world):
    base = "/api/v1/workspaces/acme/projects/search"
    members = world.client.get(
        "/api/v1/workspaces/acme/members", headers=world.as_("acme_owner")
    ).json()
    engineer = next(m["user"]["id"] for m in members if m["role"] == "engineer")
    result = world.client.put(
        base + "/access", json={"roles": {engineer: "viewer"}}, headers=world.as_("acme_owner")
    )
    assert result.status_code == 200, result.text
    assert world.client.get(base, headers=world.as_("acme_viewer")).status_code == 404
    assert world.client.get(base, headers=world.as_("acme_engineer")).status_code == 200
    assert (
        world.client.put(
            base + "/access", json={"roles": None}, headers=world.as_("acme_engineer")
        ).status_code
        == 403
    )
    projects = world.client.get(
        "/api/v1/workspaces/acme/projects", headers=world.as_("acme_viewer")
    ).json()
    assert "search" not in [p["slug"] for p in projects]
    assert world.client.get(base, headers=world.as_("acme_owner")).status_code == 200
    assert world.client.get(base, headers=world.as_("globex_owner")).status_code == 404


def test_governance_is_owner_only(world):
    url = "/api/v1/workspaces/acme/governance"
    body = {"exports_enabled": False, "retention_days": 14}
    assert world.client.put(url, json=body, headers=world.as_("acme_engineer")).status_code == 403
    assert world.client.put(url, json=body, headers=world.as_("acme_owner")).status_code == 200
    assert (
        world.client.get(
            "/api/v1/workspaces/acme/export", headers=world.as_("acme_owner")
        ).status_code
        == 403
    )
    assert (
        world.client.put(
            url, json={**body, "retention_days": 0}, headers=world.as_("acme_owner")
        ).status_code
        == 422
    )


def sso_browser(cloud_app):
    settings = dataclasses.replace(
        cloud_app.state.settings,
        public_url="https://app.test",
        sso={
            "client_id": "client",
            "client_secret": "secret",
            "authorize_url": "https://id.test/authorize",
            "token_url": "https://id.test/token",
            "userinfo_url": "https://id.test/userinfo",
        },
    )
    app = create_app(settings, cloud_app.state.engine)
    return TestClient(app, base_url="https://app.test", follow_redirects=False)


def test_sso_pkce_verified_invited_user(world, cloud_app, monkeypatch):
    client = sso_browser(cloud_app)
    start = client.get("/auth/sso?next=https://evil.test")
    query = parse_qs(urlsplit(start.headers["location"]).query)
    assert query["code_challenge_method"] == ["S256"]
    calls = []

    def provider(url, headers, data=None):
        calls.append((url, headers, data))
        if url.endswith("/token"):
            assert b"code_verifier=" in data
            return {"access_token": "token", "token_type": "Bearer"}
        return {"email": "owner@acme.test", "email_verified": True, "sub": "subject"}

    monkeypatch.setattr(sso, "_request", provider)
    response = client.get("/auth/sso/callback", params={"code": "code", "state": query["state"][0]})
    assert response.headers["location"] == "/app"
    assert "va_session=" in response.headers["set-cookie"]
    assert len(calls) == 2
    assert client.get("/app/acme").status_code == 200
    assert client.get("/app/globex").status_code == 404


@pytest.mark.parametrize(
    ("verified", "email"), [(False, "owner@acme.test"), (True, "unknown@test")]
)
def test_sso_rejects_unverified_or_uninvited(world, cloud_app, monkeypatch, verified, email):
    client = sso_browser(cloud_app)
    start = client.get("/auth/sso")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    monkeypatch.setattr(
        sso,
        "_request",
        lambda url, headers, data=None: (
            {"access_token": "token", "token_type": "Bearer"}
            if data
            else {"email": email, "email_verified": verified, "sub": "x"}
        ),
    )
    response = client.get("/auth/sso/callback", params={"code": "x", "state": state})
    assert response.headers["location"].startswith("/login?error=")
    assert client.cookies.get("va_session") is None


def test_sso_tampered_state_never_calls_provider(world, cloud_app, monkeypatch):
    client = sso_browser(cloud_app)
    client.get("/auth/sso")

    def forbidden(*args):
        pytest.fail("Provider must not be called with invalid state")

    monkeypatch.setattr(sso, "_request", forbidden)
    response = client.get("/auth/sso/callback?code=x&state=invalid")
    assert response.headers["location"].startswith("/login?error=")


def test_upgrade_and_backup_restore_preserve_governance(tmp_path):
    import sqlite3

    from sqlalchemy import create_engine, text

    from vectraudit.cloud.migrate import current_revision, head_revision, upgrade

    source = tmp_path / "source.db"
    restored = tmp_path / "restored.db"
    url = f"sqlite:///{source}"
    upgrade(url, "0004")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO workspaces (id,slug,name,plan,created_at) "
                "VALUES ('w','acme','Acme','pilot',CURRENT_TIMESTAMP)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO projects (id,workspace_id,slug,name,created_at) "
                "VALUES ('p','w','search','Search',CURRENT_TIMESTAMP)"
            )
        )
    upgrade(url)
    with engine.begin() as conn:
        assert conn.execute(text("SELECT exports_enabled FROM workspaces")).scalar() == 1
        assert conn.execute(text("SELECT access_roles FROM projects")).scalar() is None
        conn.execute(text("UPDATE workspaces SET exports_enabled=0"))
        conn.execute(text("UPDATE projects SET access_roles='{}'"))
    engine.dispose()
    with sqlite3.connect(source) as src, sqlite3.connect(restored) as target:
        src.backup(target)
    recovered = create_engine(f"sqlite:///{restored}")
    assert current_revision(recovered) == head_revision()
    with recovered.connect() as conn:
        assert conn.execute(text("SELECT exports_enabled FROM workspaces")).scalar() == 0
        assert conn.execute(text("SELECT access_roles FROM projects")).scalar() == "{}"
    recovered.dispose()
