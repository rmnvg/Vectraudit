"""The lead's view: plain-language decisions and trends, served from the ledger after expiry."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests import test_cloud_ledger
from tests.test_cloud_checks import via
from tests.test_cloud_ledger import PRIVATE, _age
from tests.test_cloud_web import sign_in
from vectraudit.cli import app as cli
from vectraudit.cloud import quotas
from vectraudit.cloud.db import transaction
from vectraudit.integrations import cloud as cloud_client

P = "/api/v1/workspaces/acme/projects/search"
decided = test_cloud_ledger.decided  # a pinned baseline, a failing and a passing check


def test_summary_explains_a_failure_and_survives_retention(world, cloud_app, decided) -> None:
    _, failing, passing = decided
    url = f"{P}/decisions/{failing['id']}/summary"
    live = world.client.get(url, headers=world.as_("acme_viewer")).json()
    assert live["verdict"] == "fail"
    assert live["headline"].startswith("candidate-v2 @ abc123 is blocked")
    assert live["changes"][0].startswith("nDCG (ranking quality) worsened from 0.750 to 0.000")
    assert live["report_url"].endswith(f"/checks/{failing['id']}/report")
    assert PRIVATE not in json.dumps(live)

    _age(cloud_app, ["worse"])
    with transaction(cloud_app.state.sessions) as s:
        assert quotas.purge_expired(s) == {"acme": 1}
    later = world.client.get(url, headers=world.as_("acme_viewer")).json()
    assert later["headline"] == live["headline"]
    assert later["report_url"] is None
    assert any("expired under retention" in u for u in later["uncertain"])

    ok = world.client.get(
        f"{P}/decisions/{passing['id']}/summary", headers=world.as_("acme_viewer")
    )
    assert ok.json()["verdict"] == "pass"
    missing = world.client.get(f"{P}/decisions/nope/summary", headers=world.as_("acme_viewer"))
    assert missing.status_code == 404
    for who in ("globex_owner", "globex_runner"):
        assert world.client.get(url, headers=world.as_(who)).status_code == 404


def test_trends_read_the_ledger_per_benchmark_version(world, cloud_app, decided) -> None:
    _, failing, passing = decided
    _age(cloud_app, ["worse"])
    with transaction(cloud_app.state.sessions) as s:
        quotas.purge_expired(s)
    body = world.client.get(f"{P}/trends", headers=world.as_("acme_viewer")).json()
    assert body["decisions"] == 2
    (series,) = body["series"]
    assert series["decisions"] == 2
    assert series["verdicts"] == {"fail": 1, "pass": 1}
    points = series["metrics"]["ndcg"]["points"]
    assert [p["check_id"] for p in points] == [failing["id"], passing["id"]]  # oldest first
    assert [p["value"] for p in points] == [0.0, 0.75]
    assert points[0]["baseline"] == 0.75
    assert series["policy_changes"] == []
    assert world.client.get(f"{P}/trends", headers=world.as_("globex_owner")).status_code == 404


def test_dashboard_gives_the_lead_a_readable_decision(world, cloud_app, decided) -> None:
    _, failing, _ = decided
    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    project = browser.get("/app/acme/search").text
    assert "Latest release decision" in project
    assert "passes the release policy" in project  # the newest check passed
    assert f"/app/acme/search/decisions/{failing['id']}" in project
    assert 'href="/app/acme/search/trends"' in project

    page = browser.get(f"/app/acme/search/decisions/{failing['id']}")
    assert page.status_code == 200
    assert "is blocked" in page.text
    assert "What changed" in page.text
    assert "Download as Markdown" in page.text
    assert "<script" not in page.text
    assert PRIVATE not in page.text
    download = browser.get(f"/app/acme/search/decisions/{failing['id']}/summary.md")
    assert download.headers["content-type"].startswith("text/markdown")
    assert "attachment" in download.headers["content-disposition"]
    assert download.text.startswith("**Fail**")

    trends = browser.get("/app/acme/search/trends")
    assert trends.status_code == 200
    assert "<svg" in trends.text
    assert "<script" not in trends.text
    assert "nDCG (ranking quality)" in trends.text
    assert "candidate-v2" in trends.text

    check = browser.get(f"/app/acme/search/checks/{failing['id']}").text
    assert "Plain-language summary" in check
    outsider = sign_in(cloud_app, world.tokens["globex_owner"])
    assert outsider.get(f"/app/acme/search/decisions/{failing['id']}").status_code == 404
    assert outsider.get("/app/acme/search/trends").status_code == 404


def test_empty_project_has_no_latest_card_and_an_empty_trend(world, cloud_app) -> None:
    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    assert "Latest release decision" not in browser.get("/app/acme/search").text
    assert "No release decisions yet" in browser.get("/app/acme/search/trends").text
    empty = world.client.get(f"{P}/trends", headers=world.as_("acme_viewer")).json()
    assert empty == {"series": [], "decisions": 0}


def test_cli_prints_markdown_for_a_ci_job_summary(world, decided, monkeypatch) -> None:
    _, failing, _ = decided
    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_runner"])
    runner = CliRunner()
    result = runner.invoke(cli, ["cloud", "summary", failing["id"]])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("**Fail**: candidate-v2 @ abc123 is blocked")
    assert "**What changed**" in result.output
    as_json = runner.invoke(cli, ["cloud", "summary", failing["id"], "--json"])
    assert json.loads(as_json.output)["verdict"] == "fail"
    missing = runner.invoke(cli, ["cloud", "summary", "nope"])
    assert missing.exit_code == 1
    assert "Decision not found" in missing.output
