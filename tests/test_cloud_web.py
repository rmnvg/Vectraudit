"""The dashboard: sign-in, sessions, pages, forms, CSRF and escaping."""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from tests.test_cloud_checks import pin
from tests.test_cloud_runs import a_run, client_for
from vectraudit.cloud.app import create_app
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import WebSession, utcnow

P = "/app/acme/search"
EVIL = "<script>alert(1)</script>"


def browser(app) -> TestClient:
    return TestClient(app, follow_redirects=False)


def sign_in(app, token: str) -> TestClient:
    b = browser(app)
    b.get("/login")
    csrf = b.cookies.get("va_login_csrf")
    r = b.post("/login/token", data={"csrf": csrf, "token": token, "next": "/app"})
    assert r.status_code == 303, r.text
    assert r.headers["location"] == "/app"
    return b


def form_csrf(page: str) -> str:
    match = re.search(r'name="csrf" value="([0-9a-f]{64})"', page)
    assert match, "no CSRF field on the page"
    return match.group(1)


@pytest.fixture
def project(world):
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base", label="main"))
    client.upload_run(a_run("cand", label=EVIL))
    pin(world, "acme_engineer", "base")
    check = client.create_check("cand")
    return check


def test_signed_out_redirects_to_login(cloud_app) -> None:
    r = browser(cloud_app).get("/app/acme")
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=/app/acme"
    assert browser(cloud_app).get("/").headers["location"] == "/app"


def test_token_sign_in(world, cloud_app) -> None:
    b = browser(cloud_app)
    login = b.get("/login")
    assert "default-src 'none'" in login.headers["content-security-policy"]
    assert "GitHub" not in login.text  # not configured
    bad_csrf = b.post("/login/token", data={"csrf": "x", "token": world.tokens["acme_viewer"]})
    assert "expired" in bad_csrf.headers["location"]
    csrf = b.cookies.get("va_login_csrf")
    for token, message in [("va_pat_nope", "not+valid"), (world.tokens["acme_runner"], "Runner")]:
        r = b.post("/login/token", data={"csrf": csrf, "token": token})
        assert (
            message.replace("+", "%20") in r.headers["location"] or message in r.headers["location"]
        )
    r = b.post(
        "/login/token",
        data={
            "csrf": csrf,
            "token": world.tokens["acme_viewer"],
            "next": "https://evil.example/steal",
        },
    )
    assert r.headers["location"] == "/app"  # no open redirect
    cookie = r.headers["set-cookie"]
    assert "va_session=" in cookie
    assert "HttpOnly" in cookie
    assert "samesite=lax" in cookie.lower()
    home = b.get("/app")
    assert home.status_code == 200
    assert "Acme" in home.text
    assert "Globex" not in home.text


def test_pages_respect_tenancy_and_escape(world, cloud_app, project) -> None:
    viewer = sign_in(cloud_app, world.tokens["acme_viewer"])
    ws = viewer.get("/app/acme")
    assert "search" in ws.text
    assert "viewer@acme.test" in ws.text
    page = viewer.get(P)
    assert page.status_code == 200
    assert EVIL not in page.text
    assert "&lt;script&gt;" in page.text
    assert "Pin as main" not in page.text  # viewers can't pin
    assert "engineer@acme.test" in page.text  # pinned by, shown as a person, not an id
    assert "baseline main" in page.text
    assert "Runner tokens" not in page.text
    assert viewer.get("/app/globex").status_code == 404
    assert viewer.get("/app/globex/search").status_code == 404
    check = viewer.get(f"{P}/checks/{project['id']}")
    assert check.status_code == 200
    assert 'sandbox=""' in check.text
    frame = viewer.get(f"{P}/checks/{project['id']}/report")
    assert "frame-ancestors 'self'" in frame.headers["content-security-policy"]
    assert EVIL not in frame.text
    outsider = sign_in(cloud_app, world.tokens["globex_owner"])
    assert outsider.get(f"{P}/checks/{project['id']}").status_code == 404


def test_forms_need_csrf_and_roles(world, cloud_app, project) -> None:
    engineer = sign_in(cloud_app, world.tokens["acme_engineer"])
    page = engineer.get(P).text
    assert "Pin as main" in page
    csrf = form_csrf(page)
    no_csrf = engineer.post(f"{P}/pin", data={"run_id": "cand", "name": "main"})
    assert no_csrf.status_code == 403
    pinned = engineer.post(f"{P}/pin", data={"csrf": csrf, "run_id": "cand", "name": "main"})
    assert pinned.status_code == 303
    assert "Baseline%20main%20now%20points%20to%20run%20cand" in pinned.headers["location"]
    created = engineer.post(f"{P}/runner-tokens", data={"csrf": csrf, "name": "gh"})
    assert created.status_code == 201
    token = re.search(r"va_run_[A-Za-z0-9]{8}_[A-Za-z0-9_-]+", created.text).group(0)
    assert (
        TestClient(cloud_app)
        .get(
            "/api/v1/workspaces/acme/projects/search",
            headers={"Authorization": f"Bearer {token}"},
        )
        .status_code
        == 200
    )
    page = engineer.get(P).text
    token_id = re.search(r"/runner-tokens/([0-9a-f]{32})/revoke", page).group(1)
    revoked = engineer.post(f"{P}/runner-tokens/{token_id}/revoke", data={"csrf": csrf})
    assert revoked.status_code == 303
    viewer = sign_in(cloud_app, world.tokens["acme_viewer"])
    viewer_csrf = form_csrf(viewer.get("/app").text)
    denied = viewer.post(f"{P}/pin", data={"csrf": viewer_csrf, "run_id": "base"})
    assert denied.status_code == 403


def test_cookie_reads_api_but_never_writes(world, cloud_app, project) -> None:
    b = sign_in(cloud_app, world.tokens["acme_engineer"])
    assert b.get("/api/v1/me").status_code == 200
    write = b.post("/api/v1/workspaces/acme/projects", json={"slug": "x2", "name": "X"})
    assert write.status_code == 401


def test_logout_and_expiry(world, cloud_app) -> None:
    b = sign_in(cloud_app, world.tokens["acme_viewer"])
    csrf = form_csrf(b.get("/app").text)
    raw = b.cookies.get("va_session")
    out = b.post("/logout", data={"csrf": csrf})
    assert out.status_code == 303
    stale = browser(cloud_app)
    stale.cookies.set("va_session", raw)
    assert stale.get("/app").status_code == 303
    b2 = sign_in(cloud_app, world.tokens["acme_viewer"])
    with transaction(cloud_app.state.sessions) as s:
        for row in s.query(WebSession).filter(WebSession.revoked_at.is_(None)):
            row.expires_at = utcnow() - timedelta(seconds=1)
    assert b2.get("/app").status_code == 303


def test_github_sign_in(world, cloud_app) -> None:
    settings = dataclasses.replace(
        cloud_app.state.settings, github_client_id="gh-id", github_client_secret="gh-secret"
    )
    app = create_app(settings, engine=cloud_app.state.engine)
    calls = []
    emails = [
        {"email": "Engineer@Acme.test", "verified": True, "primary": True},
        {"email": "owner@acme.test", "verified": False, "primary": False},
    ]

    def fake_github(method, url, headers, body):
        calls.append((method, url, body))
        if url.endswith("/access_token"):
            return 200, json.dumps({"access_token": "gho_x"}).encode()
        return 200, json.dumps(emails).encode()

    app.state.github_http = fake_github
    b = browser(app)
    assert "Sign in with GitHub" in b.get("/login").text
    start = b.get("/auth/github", params={"next": "/app/acme"})
    assert start.status_code == 303
    assert start.headers["location"].startswith("https://github.com/login/oauth/authorize?")
    state = re.search(r"state=([^&]+)", start.headers["location"]).group(1)
    wrong = b.get("/auth/github/callback", params={"code": "c", "state": "nope"})
    assert "expired" in wrong.headers["location"]
    ok = b.get("/auth/github/callback", params={"code": "c", "state": state})
    assert ok.status_code == 303
    assert ok.headers["location"] == "/app/acme"
    assert b.get("/api/v1/me").json()["user"]["email"] == "engineer@acme.test"
    assert b"client_secret=gh-secret" in calls[0][2]

    emails[:] = [
        {"email": "stranger@else.test", "verified": True, "primary": True},
        {"email": "owner@acme.test", "verified": False, "primary": False},
    ]
    b2 = browser(app)
    start = b2.get("/auth/github")
    state = re.search(r"state=([^&]+)", start.headers["location"]).group(1)
    denied = b2.get("/auth/github/callback", params={"code": "c", "state": state})
    assert "No%20invitation%20for%20stranger%40else.test" in denied.headers["location"]
    assert browser(cloud_app).get("/auth/github").status_code == 404
