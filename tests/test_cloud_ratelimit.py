"""Brute-force protection on authentication."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from vectraudit.cloud.ratelimit import FailureLimiter


def test_sliding_window() -> None:
    now = [1000.0]
    limiter = FailureLimiter(limit=3, window_s=60, clock=lambda: now[0])
    for _ in range(3):
        assert limiter.retry_after("ip") is None
        limiter.failure("ip")
    assert limiter.retry_after("ip") == 60
    assert limiter.retry_after("other") is None
    now[0] += 30
    assert limiter.retry_after("ip") == 30
    now[0] += 31
    assert limiter.retry_after("ip") is None


def test_bad_tokens_get_rate_limited(world, cloud_app, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="vectraudit.cloud.security")
    attacker = TestClient(cloud_app, client=("203.0.113.9", 4000))
    guess = "va_pat_ABCDEFGH_" + "x" * 43
    for _ in range(20):
        assert (
            attacker.get("/api/v1/me", headers={"Authorization": f"Bearer {guess}"}).status_code
            == 401
        )
    blocked = attacker.get("/api/v1/me", headers=world.as_("acme_viewer"))
    assert blocked.status_code == 429
    assert blocked.json()["error_code"] == "rate_limited"
    assert int(blocked.headers["retry-after"]) > 0
    # Other addresses are unaffected; requests without credentials don't count.
    neighbour = TestClient(cloud_app, client=("198.51.100.7", 4000))
    assert neighbour.get("/api/v1/me", headers=world.as_("acme_viewer")).status_code == 200
    for _ in range(30):
        neighbour.get("/api/v1/me")
    assert neighbour.get("/api/v1/me", headers=world.as_("acme_viewer")).status_code == 200
    failures = [r for r in caplog.records if r.getMessage() == "auth_failed"]
    assert len(failures) == 20
    assert failures[0].fields["token_prefix"] == "va_pat_ABCDEFGH"
    assert guess not in caplog.text


def test_sign_in_form_is_rate_limited(world, cloud_app) -> None:
    browser = TestClient(cloud_app, client=("203.0.113.10", 4000), follow_redirects=False)
    browser.get("/login")
    csrf = browser.cookies.get("va_login_csrf")
    for _ in range(20):
        r = browser.post("/login/token", data={"csrf": csrf, "token": "va_pat_nope"})
        assert "not%20valid" in r.headers["location"]
    r = browser.post("/login/token", data={"csrf": csrf, "token": world.tokens["acme_viewer"]})
    assert "Too%20many%20attempts" in r.headers["location"]
