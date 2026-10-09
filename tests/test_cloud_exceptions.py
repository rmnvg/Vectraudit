"""Hosted audited exceptions: who may accept what, for how long, and how every view shows it."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests.test_cloud_checks import pin
from tests.test_cloud_runs import a_run, client_for, via
from tests.test_cloud_web import sign_in
from vectraudit.cli import app as cli
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import AuditEvent, Decision, PolicyException, utcnow
from vectraudit.evaluation.runfile import save_run
from vectraudit.integrations import cloud as cloud_client

P = "/api/v1/workspaces/acme/projects/search"
QUALITY = {
    "metrics": {"ndcg": {"max_drop": 0.01}},
    "min_queries": 10,
    "exceptions": {"allow": ["metric", "latency"], "approver": "engineer", "max_days": 30},
}


def soon(days: int = 7) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def worse(run_id: str) -> dict:
    run = a_run(run_id, label=run_id)
    for q in run["queries"]:
        q["metrics"] = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    return run


def set_policy(world, policy: dict, who: str = "acme_owner"):
    return world.client.put(f"{P}/policy", json=policy, headers=world.as_(who))


def accept(world, check_id: str, rule: str, who: str = "acme_engineer", **extra):
    body = {"rule": rule, "reason": "Known trade-off", "owner": "lead@acme.test"}
    body.update({"expires_at": soon(), **extra})
    return world.client.post(f"{P}/checks/{check_id}/exceptions", json=body, headers=world.as_(who))


@pytest.fixture
def failing(world):
    """acme/search with a pinned baseline and one check failing on nDCG."""
    assert set_policy(world, QUALITY).status_code == 200
    runner = client_for(world, "acme_runner")
    runner.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    runner.upload_run(worse("worse-1"))
    check = runner.create_check("worse-1")
    assert check["verdict"] == "fail"
    return runner, check


def test_granting_requires_policy_role_failing_rule_and_bounded_expiry(world, failing):
    _, check = failing
    rule = "metric:ndcg.max_drop"
    assert accept(world, check["id"], rule, who="acme_viewer").status_code == 403
    assert accept(world, check["id"], rule, who="acme_runner").status_code == 403
    not_failing = accept(world, check["id"], "latency.p95_max_ms")
    assert not_failing.status_code == 422
    assert not_failing.json()["error_code"] == "rule_not_failing"
    assert accept(world, check["id"], rule, expires_at=soon(60)).status_code == 422
    assert accept(world, check["id"], rule, expires_at=soon(-1)).status_code == 422
    assert accept(world, check["id"], rule, reason=" ").status_code == 422
    granted = accept(world, check["id"], rule)
    assert granted.status_code == 201, granted.text
    assert granted.json()["status"] == "active"
    assert granted.json()["accepted_by"] == "engineer@acme.test"
    assert accept(world, check["id"], rule).status_code == 409  # one active per rule
    with transaction(world_app(world).state.sessions) as s:
        audit = s.query(AuditEvent).filter(AuditEvent.action == "exception.grant").one()
        assert audit.details["rule"] == rule


def world_app(world):
    return world.client.app


def test_origin_check_and_its_decision_become_accepted_not_pass(world, failing):
    _, check = failing
    assert accept(world, check["id"], "metric:ndcg.max_drop").status_code == 201
    stored = world.client.get(f"{P}/checks/{check['id']}", headers=world.as_("acme_viewer"))
    result = stored.json()["result"]
    assert stored.json()["verdict"] == "accepted"
    assert result["verdict_before_exceptions"] == "fail"
    accepted = [r for r in result["reasons"] if "accepted" in r]
    assert [r["rule"] for r in accepted] == ["metric:ndcg.max_drop"]
    with transaction(world_app(world).state.sessions) as s:
        decision = s.query(Decision).filter(Decision.check_id == check["id"]).one()
        assert decision.verdict == "accepted"
        assert decision.summary["verdict_before_exceptions"] == "fail"
        assert decision.summary["accepted"][0]["rule"] == "metric:ndcg.max_drop"
        assert s.query(PolicyException).one().uses == 1
    report = world.client.get(
        f"{P}/checks/{check['id']}/report", headers=world.as_("acme_viewer")
    ).text
    assert "Fail, accepted by exception" in report
    assert "Known trade-off" in report


def test_later_checks_are_covered_but_new_failures_still_block(world, failing):
    runner, check = failing
    assert accept(world, check["id"], "metric:ndcg.max_drop").status_code == 201
    runner.upload_run(worse("worse-2"))
    assert runner.create_check("worse-2")["verdict"] == "accepted"
    # A new, different failure on top of the accepted one keeps the check failing.
    slow = worse("worse-3")
    for q in slow["queries"]:
        q["latency_ms"] = 500.0
    runner.upload_run(slow)
    assert (
        set_policy(world, {**QUALITY, "latency": {"p95_max_ms": 100}}, who="acme_owner").status_code
        == 200
    )
    result = runner.create_check("worse-3")
    assert result["verdict"] == "fail"
    marks = {r["rule"]: "accepted" in r for r in result["result"]["reasons"]}
    assert marks == {"metric:ndcg.max_drop": True, "latency.p95_max_ms": False}


def test_expiry_use_limits_and_revocation_make_failures_resurface(world, cloud_app, failing):
    runner, check = failing
    granted = accept(world, check["id"], "metric:ndcg.max_drop", max_checks=2).json()
    runner.upload_run(worse("worse-2"))
    assert runner.create_check("worse-2")["verdict"] == "accepted"  # second (last) use
    runner.upload_run(worse("worse-3"))
    assert runner.create_check("worse-3")["verdict"] == "fail"
    listed = world.client.get(f"{P}/exceptions", headers=world.as_("acme_viewer")).json()
    assert listed[0]["status"] == "used"
    assert (
        world.client.get(f"{P}/exceptions?active=true", headers=world.as_("acme_viewer")).json()
        == []
    )

    second = accept(world, check["id"], "metric:ndcg.max_drop").json()
    with transaction(cloud_app.state.sessions) as s:
        s.get(PolicyException, second["id"]).expires_at = utcnow() - timedelta(seconds=1)
    runner.upload_run(worse("worse-4"))
    assert runner.create_check("worse-4")["verdict"] == "fail"

    third = accept(world, check["id"], "metric:ndcg.max_drop").json()
    revoked = world.client.post(
        f"{P}/exceptions/{third['id']}/revoke", headers=world.as_("acme_engineer")
    )
    assert revoked.json()["status"] == "revoked"
    runner.upload_run(worse("worse-5"))
    assert runner.create_check("worse-5")["verdict"] == "fail"
    # Earlier decisions keep what applied at the time.
    again = world.client.get(f"{P}/checks/{check['id']}", headers=world.as_("acme_viewer"))
    assert again.json()["verdict"] == "accepted"
    assert granted["id"] in {
        a["exception_id"] for a in again.json()["result"]["exceptions_applied"]
    }


def test_safety_failures_need_owner_or_cannot_be_accepted(world):
    safety = {"metrics": {"ndcg": {"max_drop": 0.5}}, "min_queries": 10}
    assert set_policy(world, {**safety, "exceptions": {"allow": ["metric"]}}).status_code == 200
    runner = client_for(world, "acme_runner")
    base = a_run("base")
    base["queries"][0]["permission_checked"] = True
    base["queries"][0]["forbidden_ids"] = []
    runner.upload_run(base)
    pin(world, "acme_engineer", "base")
    leak = a_run("leak")
    leak["queries"][0].update(permission_checked=True, forbidden_ids=["secret-doc"])
    runner.upload_run(leak)
    check = runner.create_check("leak")
    rule = "safety.forbidden:q0"
    assert check["verdict"] == "fail"
    assert accept(world, check["id"], rule, who="acme_owner").status_code == 403  # never
    opened = {**safety, "exceptions": {"allow": ["metric"], "safety": "owner"}}
    assert set_policy(world, opened).status_code == 200
    assert accept(world, check["id"], rule, who="acme_engineer").status_code == 403
    assert accept(world, check["id"], rule, who="acme_owner").status_code == 201


def test_only_owners_change_who_may_accept(world):
    assert set_policy(world, {"metrics": {"ndcg": {"max_drop": 0.01}}}).status_code == 200
    by_engineer = set_policy(world, QUALITY, who="acme_engineer")
    assert by_engineer.status_code == 403
    assert set_policy(world, QUALITY, who="acme_owner").status_code == 200
    # Other policy edits that keep the exceptions block stay engineer-level.
    tighter = {**QUALITY, "metrics": {"ndcg": {"max_drop": 0.005}}}
    assert set_policy(world, tighter, who="acme_engineer").status_code == 200


def test_ci_exit_code_follows_active_exceptions(world, failing, tmp_path, monkeypatch):
    _, check = failing
    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_runner"])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    cli_runner = CliRunner()
    candidate = save_run(worse("ci-1"), tmp_path / "ci-1.json")
    args = ["check", "cloud:main", str(candidate), "--upload", "--no-github-summary"]
    blocked = cli_runner.invoke(cli, args)
    assert blocked.exit_code == 3, blocked.output
    assert accept(world, check["id"], "metric:ndcg.max_drop").status_code == 201
    later = save_run(worse("ci-2"), tmp_path / "ci-2.json")
    passed = cli_runner.invoke(
        cli, ["check", "cloud:main", str(later), "--upload", "--no-github-summary"]
    )
    assert passed.exit_code == 0, passed.output
    flat = " ".join(passed.output.split())
    assert "FAIL, ACCEPTED" in flat
    assert "accepted by" in flat
    assert "the hosted verdict is" not in flat  # the hosted verdict agrees with the local one


def test_cli_grant_list_and_revoke(world, failing, monkeypatch):
    _, check = failing
    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_engineer"])
    monkeypatch.setenv("COLUMNS", "250")  # keep the rule column readable in the table
    runner = CliRunner()
    until = (datetime.now(timezone.utc) + timedelta(days=5)).date().isoformat()
    granted = runner.invoke(
        cli,
        [
            "cloud",
            "accept",
            check["id"],
            "metric:ndcg.max_drop",
            "--reason",
            "Planned trade-off",
            "--owner",
            "lead@acme.test",
            "--expires",
            until,
        ],
    )
    assert granted.exit_code == 0, granted.output
    listed = runner.invoke(cli, ["cloud", "exceptions"])
    assert "metric:ndcg.max_drop" in listed.output
    exception_id = world.client.get(f"{P}/exceptions", headers=world.as_("acme_viewer")).json()[0][
        "id"
    ]
    revoked = runner.invoke(cli, ["cloud", "revoke-exception", exception_id])
    assert revoked.exit_code == 0, revoked.output
    assert "No exceptions." in runner.invoke(cli, ["cloud", "exceptions"]).output
    assert "revoked" in runner.invoke(cli, ["cloud", "exceptions", "--all"]).output


def test_timeline_and_dashboard_show_exceptions(world, cloud_app, failing):
    _, check = failing
    assert accept(world, check["id"], "metric:ndcg.max_drop").status_code == 201
    entries = world.client.get(f"{P}/decisions", headers=world.as_("acme_viewer")).json()["entries"]
    grant = next(e for e in entries if e["kind"] == "exception")
    assert grant["action"] == "granted"
    assert grant["rule"] == "metric:ndcg.max_drop"
    decision = next(e for e in entries if e["kind"] == "decision" and e["check_id"] == check["id"])
    assert decision["verdict"] == "accepted"
    viewer = sign_in(cloud_app, world.tokens["acme_viewer"])
    project_page = viewer.get("/app/acme/search").text
    assert "Accepted regressions" in project_page
    assert "Exception granted" in project_page
    assert "Fail, accepted" in project_page
    check_page = viewer.get(f"/app/acme/search/checks/{check['id']}").text
    assert "Accepted by" in check_page
    engineer = sign_in(cloud_app, world.tokens["acme_engineer"])
    runner = client_for(world, "acme_runner")
    runner.upload_run(worse("worse-x"))
    other = runner.create_check("worse-x")
    page = engineer.get(f"/app/acme/search/checks/{other['id']}").text
    assert "Failing rules" in page
