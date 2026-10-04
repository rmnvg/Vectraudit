"""Hosted history, private reports and signed share links."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from tests.test_cloud_checks import POLICY, pin
from tests.test_cloud_runs import a_run, client_for
from vectraudit.cloud.app import create_app
from vectraudit.cloud.routes.history import _signature

P = "/api/v1/workspaces/acme/projects/search"


@pytest.fixture
def history(world):
    client = client_for(world, "acme_runner")
    for i in range(5):
        run = a_run(f"run-{i}", label="main" if i % 2 == 0 else "pr")
        run["provenance"]["commit_sha"] = f"c0ffee{i}"
        if i == 3:
            run["status"] = "partial"
        client.upload_run(run)
    world.client.put(f"{P}/policy", json=POLICY, headers=world.as_("acme_engineer"))
    pin(world, "acme_engineer", "run-0")
    checks = {rid: client.create_check(rid) for rid in ("run-1", "run-3")}
    return checks


def test_run_history_pagination_and_filters(world, history) -> None:
    h = world.as_("acme_viewer")
    seen, cursor = [], None
    for _ in range(5):
        params = {"limit": 2, **({"cursor": cursor} if cursor else {})}
        page = world.client.get(f"{P}/runs", params=params, headers=h).json()
        seen += [r["run_id"] for r in page["runs"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == ["run-4", "run-3", "run-2", "run-1", "run-0"]

    def by(**q: str) -> list[str]:
        page = world.client.get(f"{P}/runs", params=q, headers=h).json()
        return [r["run_id"] for r in page["runs"]]

    assert by(status="partial") == ["run-3"]
    assert by(label="pr") == ["run-3", "run-1"]
    assert by(commit="c0ffee2") == ["run-2"]
    assert by(backend="faiss") == seen
    assert by(backend="qdrant") == []
    bad = world.client.get(f"{P}/runs", params={"cursor": "!!"}, headers=h)
    assert bad.status_code == 422
    for who in ("globex_owner", "globex_runner"):
        assert world.client.get(f"{P}/runs", headers=world.as_(who)).status_code == 404


def test_check_history(world, history) -> None:
    page = world.client.get(f"{P}/checks", headers=world.as_("acme_viewer")).json()
    assert [c["candidate_run_id"] for c in page["checks"]] == ["run-3", "run-1"]
    errored = page["checks"][0]
    assert errored["verdict"] == "error"
    assert "candidate run is partial" in errored["reasons"][0]
    assert errored["report_url"].endswith(f"/checks/{errored['id']}/report")
    only = world.client.get(
        f"{P}/checks", params={"verdict": "pass"}, headers=world.as_("acme_viewer")
    ).json()["checks"]
    assert [c["candidate_run_id"] for c in only] == ["run-1"]


def test_private_report(world, history) -> None:
    check_id = history["run-1"]["id"]
    url = f"{P}/checks/{check_id}/report"
    assert world.client.get(url).status_code == 401
    r = world.client.get(url, headers=world.as_("acme_viewer"))
    assert r.status_code == 200
    assert r.text.startswith("<!doctype html>")
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "no-store" in r.headers["cache-control"]
    assert world.client.get(url, headers=world.as_("acme_runner")).status_code == 200
    for who in ("globex_owner", "globex_runner"):
        assert world.client.get(url, headers=world.as_(who)).status_code == 404


def test_share_links(world, history, cloud_app) -> None:
    check_id = history["run-1"]["id"]
    share = f"{P}/checks/{check_id}/share"
    disabled = world.client.post(share, json={"days": 3}, headers=world.as_("acme_engineer"))
    assert disabled.status_code == 403
    assert disabled.json()["error_code"] == "share_links_disabled"

    settings = dataclasses.replace(cloud_app.state.settings, share_links=True)
    app = create_app(settings, engine=cloud_app.state.engine)
    client = TestClient(app)
    assert client.post(share, json={}, headers=world.as_("acme_viewer")).status_code == 403
    assert (
        client.post(share, json={"days": 90}, headers=world.as_("acme_engineer")).status_code == 422
    )
    made = client.post(share, json={"days": 3}, headers=world.as_("acme_engineer"))
    assert made.status_code == 201
    link = made.json()["url"].removeprefix(settings.public_url)
    opened = client.get(link)
    assert opened.status_code == 200
    assert "Retrieval check" in opened.text
    assert client.get(link[:-2] + "xx").status_code == 404
    other = history["run-3"]["id"]
    assert client.get(link.replace(check_id, other)).status_code == 404
    past = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    expired = f"/r/{check_id}?e={past}&s={_signature(settings.secret_key, check_id, past)}"
    assert client.get(expired).status_code == 404
    # Turning share links off again disables existing links too.
    assert world.client.get(link).status_code == 404
    events = world.client.get("/api/v1/workspaces/acme/audit", headers=world.as_("acme_owner"))
    assert "report.share" in [e["action"] for e in events.json()]


def test_cli_cloud_runs(world, history, monkeypatch) -> None:
    from typer.testing import CliRunner

    from tests.test_cloud_runs import via
    from vectraudit.cli import app as cli
    from vectraudit.integrations import cloud as cloud_client

    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_viewer"])
    result = CliRunner().invoke(
        cli, ["cloud", "runs", "--project", "acme/search", "--status", "partial"]
    )
    assert result.exit_code == 0, result.output
    assert "run-3" in result.output
    assert "run-1" not in result.output
