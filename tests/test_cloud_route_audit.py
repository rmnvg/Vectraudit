"""Every route is either tenant-scoped (and proven to reject other tenants) or reviewed.

Workspace routes are called with another tenant's token and session using *real* ids from
the victim workspace, so a missing authorization check shows up as a success, not a 404.
Routes outside workspaces must be listed in UNSCOPED with a reason; a new route that is
neither fails this test until someone reviews it.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.routing import APIRoute

from tests.test_cloud_checks import pin
from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import sign_in

UNSCOPED = {
    ("GET", "/healthz"): "liveness, no data",
    ("GET", "/readyz"): "readiness, no tenant data",
    ("GET", "/"): "redirect",
    ("GET", "/app"): "lists the caller's own workspaces",
    ("GET", "/login"): "sign-in page",
    ("POST", "/login/token"): "sign-in",
    ("GET", "/auth/github"): "sign-in",
    ("GET", "/auth/github/callback"): "sign-in",
    ("POST", "/logout"): "own session",
    ("GET", "/api/v1/me"): "the caller",
    ("GET", "/api/v1/tokens"): "the caller's tokens",
    ("POST", "/api/v1/tokens"): "the caller's tokens",
    ("DELETE", "/api/v1/tokens/{token_id}"): "the caller's tokens (owner-scoped query)",
    ("GET", "/api/v1/workspaces"): "the caller's memberships",
    ("POST", "/api/v1/workspaces"): "platform admins only",
    ("GET", "/api/v1/admin/usage"): "platform admins only",
    ("GET", "/r/{check_id}"): "HMAC-signed share link",
}
DOCS = {"/api/docs", "/api/openapi.json", "/api/docs/oauth2-redirect"}


def _flatten(routes, prefix: str = "") -> list[APIRoute | tuple[str, APIRoute]]:
    """API routes, including those inside FastAPI's lazily included routers."""
    out = []
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            out += _flatten(included.routes, prefix + route.include_context.prefix)
        elif isinstance(route, APIRoute):
            out.append((prefix + route.path, route))
    return out


def _routes(app) -> list[tuple[str, str]]:
    out = []
    for path, route in _flatten(app.routes):
        if path in DOCS:
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            out.append((method, path))
    return out


@pytest.fixture
def victim(world, cloud_app):
    """Real ids of objects in acme/search for the intruder to aim at."""
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("victim-run"))
    pin(world, "acme_engineer", "victim-run")
    check = client.create_check("victim-run")
    members = world.client.get(
        "/api/v1/workspaces/acme/members", headers=world.as_("acme_owner")
    ).json()
    tokens = world.client.get(
        "/api/v1/workspaces/acme/projects/search/runner-tokens", headers=world.as_("acme_owner")
    ).json()
    return {
        "workspace": "acme",
        "project": "search",
        "run_id": "victim-run",
        "name": "main",
        "check_id": check["id"],
        "token_id": tokens[0]["id"],
        "user_id": next(m["user"]["id"] for m in members if m["role"] == "engineer"),
    }


def test_every_route_is_scoped_or_reviewed(cloud_app) -> None:
    unreviewed = [
        (m, p) for m, p in _routes(cloud_app) if "{workspace}" not in p and (m, p) not in UNSCOPED
    ]
    assert unreviewed == [], "New unscoped routes need review (add to UNSCOPED with a reason)"
    stale = set(UNSCOPED) - set(_routes(cloud_app))
    assert stale == set(), f"UNSCOPED lists routes that no longer exist: {stale}"


@pytest.mark.parametrize("intruder", ["globex_owner", "globex_runner"])
def test_workspace_routes_reject_other_tenants(world, cloud_app, victim, intruder) -> None:
    body = {
        "email": "x@y.test",
        "role": "owner",
        "slug": "x1",
        "name": "x",
        "run_id": "victim-run",
        "candidate_run_id": "victim-run",
        "metrics": {"ndcg": {"min": 0}},
        "days": 1,
        "confirm": "acme",
    }
    checked = 0
    for method, path in _routes(cloud_app):
        if "{workspace}" not in path:
            continue
        url = re.sub(r"{(\w+)}", lambda m: victim[m.group(1)], path)
        r = world.client.request(
            method, url, json=body, headers=world.as_(intruder), follow_redirects=False
        )
        to_login = r.status_code == 303 and r.headers["location"].startswith("/login")
        assert r.status_code in (401, 403, 404) or to_login, (intruder, method, path, r.status_code)
        checked += 1
    assert checked >= 30  # every API and dashboard workspace route


def test_dashboard_routes_reject_other_tenants(world, cloud_app, victim) -> None:
    browser = sign_in(cloud_app, world.tokens["globex_owner"])
    for method, path in _routes(cloud_app):
        if not path.startswith("/app/{workspace}"):
            continue
        url = re.sub(r"{(\w+)}", lambda m: victim[m.group(1)], path)
        r = browser.request(method, url, data={"run_id": "victim-run", "name": "main"})
        assert r.status_code in (403, 404), (method, path, r.status_code)
        assert "victim-run" not in r.text or method == "POST", path
