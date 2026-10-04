"""Dashboard member management, invite links and personal token settings."""

from __future__ import annotations

import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from tests.test_cloud_web import form_csrf, sign_in
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import Invite, utcnow

LINK = re.compile(r"http://localhost:8800(/invite/[A-Za-z0-9_-]{40,})")


def invite(owner: TestClient, email: str, role: str = "viewer") -> str:
    csrf = form_csrf(owner.get("/app/acme").text)
    r = owner.post("/app/acme/members", data={"csrf": csrf, "email": email, "role": role})
    assert r.status_code == 201, r.text
    return LINK.search(r.text).group(1)


def test_invite_link_signs_in_once(world, cloud_app) -> None:
    owner = sign_in(cloud_app, world.tokens["acme_owner"])
    path = invite(owner, "New.Person@Acme.test", "engineer")
    newcomer = TestClient(cloud_app, follow_redirects=False)
    r = newcomer.get(path)
    assert r.status_code == 303
    assert r.headers["location"] == "/app/acme"
    me = newcomer.get("/api/v1/me").json()
    assert me["user"]["email"] == "new.person@acme.test"
    assert me["workspaces"][0]["role"] == "engineer"
    again = TestClient(cloud_app, follow_redirects=False).get(path)
    assert "invalid%2C%20used%20or%20expired" in again.headers["location"]


def test_expired_and_bogus_links(world, cloud_app) -> None:
    owner = sign_in(cloud_app, world.tokens["acme_owner"])
    path = invite(owner, "late@acme.test")
    with transaction(cloud_app.state.sessions) as s:
        for row in s.query(Invite):
            row.expires_at = utcnow() - timedelta(minutes=1)
    stranger = TestClient(cloud_app, follow_redirects=False, client=("192.0.2.50", 1))
    assert "expired" in stranger.get(path).headers["location"]
    for _ in range(19):
        stranger.get("/invite/" + "x" * 43)
    blocked = stranger.get("/invite/" + "y" * 43)
    assert "Too%20many%20attempts" in blocked.headers["location"]


def test_roles_and_removal_from_the_dashboard(world, cloud_app) -> None:
    owner = sign_in(cloud_app, world.tokens["acme_owner"])
    page = owner.get("/app/acme").text
    csrf = form_csrf(page)
    viewer_id = re.search(r"/members/([0-9a-f]{32})/role", page.split("viewer@acme.test")[1]).group(
        1
    )
    r = owner.post(f"/app/acme/members/{viewer_id}/role", data={"csrf": csrf, "role": "engineer"})
    assert r.status_code == 303
    roles = {
        m["user"]["email"]: m["role"] for m in owner.get("/api/v1/workspaces/acme/members").json()
    }
    assert roles["viewer@acme.test"] == "engineer"
    me = owner.get("/api/v1/me").json()["user"]["id"]
    last = owner.post(f"/app/acme/members/{me}/role", data={"csrf": csrf, "role": "viewer"})
    assert last.status_code == 409
    assert "at least one owner" in last.text
    fresh = owner.post(f"/app/acme/members/{viewer_id}/invite", data={"csrf": csrf})
    assert LINK.search(fresh.text)
    gone = owner.post(f"/app/acme/members/{viewer_id}/remove", data={"csrf": csrf})
    assert gone.status_code == 303
    assert (
        world.client.get("/api/v1/workspaces/acme", headers=world.as_("acme_viewer")).status_code
        == 404
    )
    assert owner.post("/app/acme/members", data={"email": "x@acme.test"}).status_code == 403


def test_non_owners_see_no_member_controls(world, cloud_app) -> None:
    engineer = sign_in(cloud_app, world.tokens["acme_engineer"])
    page = engineer.get("/app/acme").text
    assert "Invite" not in page
    assert "/remove" not in page
    csrf = form_csrf(page)
    r = engineer.post(
        "/app/acme/members", data={"csrf": csrf, "email": "x@acme.test", "role": "owner"}
    )
    assert r.status_code == 403


def test_personal_tokens_page(world, cloud_app) -> None:
    viewer = sign_in(cloud_app, world.tokens["acme_viewer"])
    page = viewer.get("/account")
    assert page.status_code == 200
    assert "viewer@acme.test" in page.text
    csrf = form_csrf(page.text)
    created = viewer.post("/account/tokens", data={"csrf": csrf, "name": "laptop", "days": "30"})
    assert created.status_code == 201
    token = re.search(r"va_pat_[A-Za-z0-9]{8}_[A-Za-z0-9_-]+", created.text).group(0)
    api = TestClient(cloud_app)
    assert api.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    listing = viewer.get("/account").text
    assert "laptop" in listing
    assert token not in listing
    token_id = re.search(
        r'/account/tokens/([0-9a-f]{32})/revoke"', listing.split("laptop")[1]
    ).group(1)
    other = sign_in(cloud_app, world.tokens["acme_owner"])
    foreign = other.post(
        f"/account/tokens/{token_id}/revoke", data={"csrf": form_csrf(other.get("/account").text)}
    )
    assert foreign.status_code == 404
    revoked = viewer.post(f"/account/tokens/{token_id}/revoke", data={"csrf": csrf})
    assert revoked.status_code == 303
    assert api.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert viewer.post("/account/tokens", data={"name": "x"}).status_code == 403
