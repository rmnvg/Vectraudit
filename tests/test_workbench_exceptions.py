"""Hosted exceptions beside a workbench comparison: shown, never applied to the local verdict."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_policy_exceptions import failing_pair, policy
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import policy_sha256
from vectraudit.evaluation.runfile import save_run
from vectraudit.pipelines.hosted_exceptions import hosted_view
from vectraudit.pipelines.web import create_workbench

OPEN = {"allow": ["metric", "latency", "cohort"], "approver": "engineer"}


def exception(rule: str, status: str = "active", **extra) -> dict:
    return {
        "id": f"{rule}-{status}",
        "rule": rule,
        "kind": rule.split(":")[0].split(".")[0],
        "reason": "Known trade-off",
        "owner": "lead@acme.test",
        "accepted_by": "eng@acme.test",
        "created_at": "2026-10-01T00:00:00+00:00",
        "expires_at": "2026-10-20T00:00:00+00:00",
        "max_checks": None,
        "uses": 0,
        "status": status,
        "origin_check_id": "c" * 32,
        **extra,
    }


def checked(rules):
    base, cand = failing_pair()
    return compare_runs(base, cand, rules)


def test_coverage_names_active_lapsed_and_missing_exceptions():
    rules = policy(exceptions=OPEN)
    check = checked(rules)
    listed = [
        exception("metric:ndcg.max_drop"),
        exception("latency.p95_max_ms", "expired"),
        exception("latency.p95_increase", "revoked"),
        exception("latency.p95_increase", "used"),  # older; the newest one is shown
        exception("safety.forbidden:q1"),  # active, but safety is never open here
    ]
    view = hosted_view(check, listed, rules, hosted_default=False)
    coverage = {c["rule"]: c for c in view["coverage"]}
    assert coverage["metric:ndcg.max_drop"]["status"] == "covered"
    assert coverage["metric:ndcg.max_drop"]["exception"]["accepted_by"] == "eng@acme.test"
    assert coverage["latency.p95_max_ms"]["status"] == "expired"
    assert coverage["latency.p95_increase"]["status"] == "revoked"
    assert coverage["safety.forbidden:q1"]["status"] == "not_open"
    assert coverage["critical:q0:a"] == {
        "rule": "critical:q0:a",
        "status": "none",
        "exception": None,
    }
    assert view["same_policy"] is True
    assert view["hosted_policy_sha256"] == policy_sha256(rules)
    # Other failures still block: the hosted project would record a plain fail.
    assert view["hosted_verdict"] == "fail"
    assert check["verdict"] == "fail"  # the local result is untouched


def test_full_coverage_reads_accepted_only_under_the_same_policy():
    rules = policy(
        exceptions={
            "allow": ["metric", "latency", "cohort", "critical", "regressed_queries"],
            "safety": "owner",
            "approver": "engineer",
        }
    )
    check = checked(rules)
    listed = [
        exception(rule) for rule in (r["rule"] for r in check["reasons"] if r["verdict"] == "fail")
    ]
    view = hosted_view(check, listed, rules, hosted_default=False)
    assert {c["status"] for c in view["coverage"]} == {"covered"}
    assert view["hosted_verdict"] == "accepted"
    other = policy(exceptions={"allow": ["metric"], "approver": "owner"}, min_queries=5)
    differs = hosted_view(check, listed, other, hosted_default=True)
    assert differs["same_policy"] is False
    assert differs["hosted_verdict"] is None  # different rules: no guess
    assert differs["hosted_policy_default"] is True
    assert {c["rule"] for c in differs["coverage"] if c["status"] == "not_open"} >= {
        "latency.p95_max_ms",
        "cohort:billing:ndcg.max_drop",
    }


def test_errors_are_never_covered():
    rules = policy(exceptions=OPEN)
    check = checked(rules)
    check["verdict"] = "error"
    view = hosted_view(check, [exception("metric:ndcg.max_drop")], rules, hosted_default=False)
    assert {c["rule"]: c["status"] for c in view["coverage"]}["metric:ndcg.max_drop"] == (
        "not_applicable"
    )
    assert view["hosted_verdict"] is None


# --- the workbench routes ----------------------------------------------------------------


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    for name in ("VECTRAUDIT_CLOUD_URL", "VECTRAUDIT_CLOUD_TOKEN", "VECTRAUDIT_CLOUD_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    from tests.test_cloud_exceptions import QUALITY, worse
    from tests.test_cloud_runs import a_run

    ids = {"base": "1" * 32, "worse": "2" * 32}
    for name, run in (("base", a_run("base")), ("worse", worse("worse"))):
        run["run_id"] = ids[name]
        save_run(run, tmp_path / f"{ids[name]}.run.json")
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        saved = client.post("/api/guided/policies", json={"name": "Release", "policy": QUALITY})
        assert saved.status_code == 200, saved.text
        compared = client.post(
            "/api/guided/compare",
            json={"baseline": ids["base"], "candidate": ids["worse"], "policy": saved.json()["id"]},
        )
        assert compared.status_code == 200, compared.text
        yield client, tmp_path, compared.json()


def test_comparisons_say_who_may_accept_each_failure(workbench):
    _, _, comparison = workbench
    assert comparison["check"]["verdict"] == "fail"
    assert comparison["acceptance"] == [
        {
            "rule": "metric:ndcg.max_drop",
            "kind": "metric",
            "message": comparison["check"]["reasons"][0]["message"],
            "role": "engineer",
            "max_days": 30,
            "why": None,
        }
    ]


def test_hosted_lookup_needs_configuration_and_a_known_comparison(workbench):
    client, _, comparison = workbench
    check_id = comparison["files"]["check"].removesuffix(".check.json")
    unconfigured = client.post("/api/guided/exceptions/hosted", json={"check": check_id})
    assert unconfigured.status_code == 422
    assert unconfigured.json()["detail"].startswith("Not connected to a hosted project")
    assert "VECTRAUDIT_CLOUD_TOKEN" in unconfigured.json()["detail"]
    assert client.post("/api/guided/exceptions/hosted", json={"check": "../x"}).status_code == 422
    missing = client.post("/api/guided/exceptions/hosted", json={"check": "f" * 32})
    assert missing.status_code == 404
    client.headers.pop("X-Workbench-Token")
    denied = client.post("/api/guided/exceptions/hosted", json={"check": check_id})
    assert denied.status_code in (401, 403)


def test_hosted_exceptions_are_shown_but_never_change_the_local_verdict(
    workbench, world, monkeypatch
):
    pytest.importorskip("sqlalchemy")
    from tests.test_cloud_checks import pin, via
    from tests.test_cloud_exceptions import QUALITY, accept, set_policy, worse
    from tests.test_cloud_runs import a_run, client_for
    from vectraudit.integrations import cloud as cloud_client

    client, root, comparison = workbench
    assert set_policy(world, QUALITY).status_code == 200
    runner = client_for(world, "acme_runner")
    runner.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    runner.upload_run(worse("worse-1"))
    hosted_check = runner.create_check("worse-1")
    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_viewer"])
    check_id = comparison["files"]["check"].removesuffix(".check.json")
    check_file = root / comparison["files"]["check"]
    before = check_file.read_text()

    none = client.post("/api/guided/exceptions/hosted", json={"check": check_id})
    assert none.status_code == 200, none.text
    view = none.json()
    assert view["same_policy"] is True
    assert view["coverage"][0]["status"] == "none"
    assert view["hosted_verdict"] == "fail"
    assert view["destination"]["project"] == "acme/search"
    assert world.tokens["acme_viewer"] not in json.dumps(view)

    assert accept(world, hosted_check["id"], "metric:ndcg.max_drop").status_code == 201
    view = client.post("/api/guided/exceptions/hosted", json={"check": check_id}).json()
    assert view["coverage"][0]["status"] == "covered"
    assert view["coverage"][0]["exception"]["reason"] == "Known trade-off"
    assert view["hosted_verdict"] == "accepted"
    assert view["exceptions"][0]["origin_check_id"] == hosted_check["id"]
    # Reading sends nothing and changes nothing locally.
    assert check_file.read_text() == before
    assert json.loads(before)["verdict"] == "fail"

    exception_id = view["exceptions"][0]["id"]
    revoked = world.client.post(
        f"/api/v1/workspaces/acme/projects/search/exceptions/{exception_id}/revoke",
        headers=world.as_("acme_engineer"),
    )
    assert revoked.status_code == 200, revoked.text
    view = client.post("/api/guided/exceptions/hosted", json={"check": check_id}).json()
    assert view["coverage"][0]["status"] == "revoked"  # the failure blocks again
    assert view["hosted_verdict"] == "fail"
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", "wrong")
    assert client.post("/api/guided/exceptions/hosted", json={"check": check_id}).status_code == 401
