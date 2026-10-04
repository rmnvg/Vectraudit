"""Incident containment commands."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests.test_cloud_web import sign_in


@pytest.fixture
def cli(cloud_settings, monkeypatch):
    from vectraudit.cloud.cli import app

    monkeypatch.setenv("VECTRAUDIT_CLOUD_DATABASE_URL", cloud_settings.database_url)
    monkeypatch.setenv("VECTRAUDIT_CLOUD_SECRET_KEY", cloud_settings.secret_key)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def status(world, who: str) -> int:
    return world.client.get(
        "/api/v1/me"
        if "runner" not in who
        else "/api/v1/workspaces/acme/projects/search"
        if who.startswith("acme")
        else "/api/v1/workspaces/globex/projects/search",
        headers=world.as_(who),
    ).status_code


def test_revoke_one_workspace(world, cloud_app, cli) -> None:
    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    assert cli("revoke-access", "--workspace", "acme").exit_code == 1  # needs --yes
    r = cli("revoke-access", "--workspace", "acme", "--yes")
    assert r.exit_code == 0, r.output
    assert "for workspace acme" in r.output
    for who in ("acme_owner", "acme_engineer", "acme_viewer", "acme_runner"):
        assert status(world, who) == 401, who
    assert browser.get("/app").status_code == 303  # session revoked
    for who in ("globex_owner", "globex_runner", "admin"):
        assert status(world, who) == 200, who
    events = [json.loads(line) for line in cli("audit", "acme").output.splitlines()]
    assert events[0]["action"] == "incident.revoke_access"
    assert events[0]["details"]["sessions"] == 1


def test_revoke_everything(world, cli) -> None:
    r = cli("revoke-access", "--yes")
    assert r.exit_code == 0
    assert "for the whole service" in r.output
    for who in ("acme_owner", "globex_runner", "admin"):
        assert status(world, who) == 401, who


def test_audit_command(world, cli) -> None:
    lines = cli("audit", "acme", "--limit", "2").output.splitlines()
    assert len(lines) == 2
    assert all("actor" in json.loads(line) for line in lines)
    assert cli("audit", "nope").exit_code == 1
    assert cli("revoke-access", "--workspace", "nope", "--yes").exit_code == 1
