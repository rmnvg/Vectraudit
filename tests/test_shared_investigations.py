"""Investigations shared with the hosted project: states and times only, linked to the
recurring problems they investigated."""

from __future__ import annotations

import json

import pytest

from vectraudit.integrations.shared_records import project_record, validate_record
from vectraudit.pipelines.investigation_records import (
    close,
    link_result,
    new_record,
    record_protection,
    request_evidence,
    set_diagnosis,
)

PRIVATE = "PLANTED-PRIVATE-REASON"


def investigation(query_id: str = "q1") -> dict:
    record = new_record("a" * 32, "b" * 32, query_id, None, "Asha Private-Name", ["not_observed"])
    set_diagnosis(record, "disputed", "Asha Private-Name", PRIVATE)
    request_evidence(record, f"{PRIVATE} corpus snapshot", "Ben")
    set_diagnosis(record, "accepted", "Ben", None)
    link_result(
        record,
        {
            "kind": "trial",
            "id": "c" * 32,
            "fix": {"name": "deeper"},
            "recovered_focus": 1,
            "regressed": [],
        },
        "Ben",
    )
    link_result(
        record,
        {
            "kind": "validation",
            "id": "d" * 32,
            "fix": {"name": "deeper"},
            "status": "approved_under_policy",
            "approved": True,
        },
        "Ben",
    )
    close(record, "fixed", "Ben", f"{PRIVATE} note")
    record_protection(
        record,
        {"id": "e" * 32, "name": "Release", "version": 2, "sha256": "f" * 64},
        [{"kind": "rank"}],
        "Ben",
    )
    record["revision"] = "9" * 32
    return record


def test_projection_carries_states_and_times_but_no_names_or_free_text():
    shared = project_record("investigation", investigation())
    data = shared["data"]
    assert (data["diagnosis"], data["outcome"], data["evidence_requests"]) == (
        "accepted",
        "fixed",
        1,
    )
    assert data["validations"][0]["approved"] is True
    assert data["protections"] == [
        {
            "policy_id": "e" * 32,
            "version": 2,
            "sha256": "f" * 64,
            "rules": ["rank"],
            "at": data["protections"][0]["at"],
        }
    ]
    assert data["minutes_to_accepted_diagnosis"] is not None
    text = json.dumps(shared)
    for private in (PRIVATE, "Asha", "Ben"):
        assert private not in text
    assert validate_record(shared) == shared


def test_unknown_or_tampered_investigation_fields_are_refused():
    shared = project_record("investigation", investigation())
    tampered = json.loads(json.dumps(shared))
    tampered["data"]["closing_note"] = PRIVATE
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        validate_record(tampered)


def test_the_share_panel_never_sends_text_for_investigations(tmp_path):
    from vectraudit.pipelines.guided import GuidedWorkspace
    from vectraudit.pipelines.share import preview, projection
    from vectraudit.pipelines.workflows import LocalRecords

    workspace = GuidedWorkspace(tmp_path)
    LocalRecords(workspace, "investigations").write(investigation())
    projected = projection(workspace, "investigation", "a" * 32, share_query_text=True)
    assert projected["sharing"]["query_text"] is False
    summary = preview(projected)["summary"]
    assert "reviewer names" in summary["withheld"]
    assert summary["question_text_count"] == 0


sqlalchemy = pytest.importorskip("sqlalchemy")


def test_recurring_problems_link_their_investigation(world, cloud_app):
    from tests.test_cloud_checks import pin
    from tests.test_cloud_monitor import _bad
    from tests.test_cloud_runs import a_run, client_for
    from tests.test_cloud_web import sign_in

    path = "/api/v1/workspaces/acme/projects/search"
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    for name in ("r1", "r2"):
        client.upload_run(_bad(name))
        client.create_check(name)
    stored = client.upload_record(project_record("investigation", investigation()))
    found = world.client.get(f"{path}/recurring", headers=world.as_("acme_viewer")).json()
    q1 = next(q for q in found["questions"] if q["query_id"] == "q1")
    (linked,) = q1["investigations"]
    assert (linked["outcome"], linked["diagnosis"], linked["protected"]) == (
        "fixed",
        "accepted",
        True,
    )
    assert next(q for q in found["questions"] if q["query_id"] == "q2")["investigations"] == []

    browser = sign_in(cloud_app, world.tokens["acme_viewer"])
    recurring = browser.get("/app/acme/search/recurring").text
    assert "fixed, protected when shared" in recurring
    listing = browser.get("/app/acme/search/records?kind=investigation").text
    assert "q1: investigation closed, fixed" in listing
    page = browser.get(f"/app/acme/search/records/{stored['id']}").text
    assert "Investigation of q1" in page
    assert "approved_under_policy" in page
    assert PRIVATE not in page
    assert "Asha" not in page
