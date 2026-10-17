"""Investigation records: diagnoses accepted or disputed, fixes verified, times measured."""

from __future__ import annotations

from typer.testing import CliRunner

from tests import test_investigation_web as web
from tests.test_investigation_web import evaluate, save, wait
from vectraudit.cli import app
from vectraudit.pipelines.investigation_records import measures, new_record

FIX = "use the phrase reranker"
client = web.client  # the workbench fixture with a startup policy


def start(client, path, body):
    response = client.post(f"/api/guided/{path}", json=body)
    assert response.status_code == 202, response.text
    return wait(client, response.json()["id"])


def opened(client, run, by="Asha"):
    response = client.post(
        "/api/guided/investigations", json={"run": run["run_id"], "query_id": "refund", "by": by}
    )
    assert response.status_code == 200, response.text
    return response.json()


def post(client, record, action, body, status=200):
    response = client.post(f"/api/guided/investigations/{record['id']}/{action}", json=body)
    assert response.status_code == status, response.text
    return response.json()


def test_the_repair_loop_is_recorded_and_measured(client, tmp_path):
    run = evaluate(client, save(client))
    record = opened(client, run)
    assert record["losses"] == ["lost_downstream"]
    assert opened(client, run, by="Someone else")["id"] == record["id"]  # one open record

    post(client, record, "diagnosis", {"status": "disputed", "by": "Asha"}, 422)  # needs why
    post(client, record, "diagnosis", {"status": "disputed", "by": "Asha", "reason": "stale"})
    post(client, record, "requests", {"what": "corpus version of the index", "by": "Asha"})
    record = post(client, record, "diagnosis", {"status": "accepted", "by": "Asha"})
    assert record["diagnosis"]["status"] == "accepted"

    post(client, record, "close", {"outcome": "fixed", "by": "Asha"}, 422)  # nothing verified
    trial = start(client, "fixes/try", {"run": run["run_id"], "fix": FIX, "query_ids": ["refund"]})
    record = post(client, record, "results", {"result": trial["id"], "by": "Asha"})
    assert record["trials"][0]["recovered_focus"] == 1
    post(client, record, "close", {"outcome": "fixed", "by": "Asha"}, 422)  # a trial is not proof
    post(client, record, "results", {"result": trial["id"], "by": "Asha"}, 422)  # linked once

    body = {"run": run["run_id"], "fix": FIX, "policy": "startup", "confirm_heldout": True}
    first = start(client, "fixes/validate", body)
    second = start(client, "fixes/validate", body)
    post(client, record, "results", {"result": first["id"], "by": "Asha"})
    record = post(client, record, "results", {"result": second["id"], "by": "Ben"})
    record = post(client, record, "close", {"outcome": "fixed", "by": "Asha", "note": "ok"})
    post(client, record, "diagnosis", {"status": "accepted", "by": "Asha"}, 409)  # closed

    found = client.get(f"/api/guided/investigations/{record['id']}").json()["measures"]
    assert found["diagnosis"] == "accepted"
    assert found["fix_verified"] is True
    assert found["reproduced_by_second_engineer"] is True
    assert found["evidence_requests"] == 1
    assert found["minutes_to_accepted_diagnosis"] is not None
    summary = client.get("/api/guided/investigations").json()
    assert summary["investigations"] == 1
    assert summary["fixes_verified"] == 1
    assert summary["minutes_to_verified_fix"]["n"] == 1

    text = (tmp_path / ".workbench" / "investigations" / f"{record['id']}.json").read_text()
    assert "annual refund" not in text  # IDs, names and reasons only; no question text

    printed = CliRunner().invoke(app, ["workbench", "investigations", str(tmp_path)])
    assert printed.exit_code == 0, printed.output
    assert "Diagnoses: 1 of 1 accepted" in printed.output
    assert "reproduced by a second engineer: 1" in printed.output


def test_results_from_another_run_cannot_be_linked(client):
    draft = save(client)
    run, other = evaluate(client, draft), evaluate(client, draft)
    record = opened(client, run)
    trial = start(
        client, "fixes/try", {"run": other["run_id"], "fix": FIX, "query_ids": ["refund"]}
    )
    post(client, record, "results", {"result": trial["id"], "by": "Asha"}, 422)
    post(client, record, "results", {"result": "0" * 32, "by": "Asha"}, 404)


def test_reproduction_needs_a_second_person():
    record = new_record("a" * 32, "r", "q", None, "Asha", [])
    for by in ("Asha", "Asha"):
        record["validations"].append(
            {"id": by, "fix": "f", "by": by, "approved": True, "at": record["opened_at"]}
        )
    assert measures(record)["reproduced_by_second_engineer"] is False
    record["validations"].append(
        {"id": "b", "fix": "other", "by": "Ben", "approved": True, "at": record["opened_at"]}
    )
    assert measures(record)["reproduced_by_second_engineer"] is False  # a different fix
    record["validations"].append(
        {"id": "c", "fix": "f", "by": "Ben", "approved": True, "at": record["opened_at"]}
    )
    assert measures(record)["reproduced_by_second_engineer"] is True


def fixed_record(client):
    run = evaluate(client, save(client))
    record = opened(client, run)
    post(client, record, "diagnosis", {"status": "accepted", "by": "Asha"})
    body = {"run": run["run_id"], "fix": FIX, "policy": "startup", "confirm_heldout": True}
    validation = start(client, "fixes/validate", body)
    post(client, record, "results", {"result": validation["id"], "by": "Asha"})
    return post(client, record, "close", {"outcome": "fixed", "by": "Asha"})


def test_a_verified_fix_protects_its_question_in_a_new_policy_version(client):
    policy = client.post(
        "/api/guided/policies",
        json={"name": "Refunds release", "policy": {"metrics": {"ndcg": {"max_drop": 0.0}}}},
    ).json()
    record = fixed_record(client)
    rank = {"policy": policy["id"], "by": "Asha", "doc_id": "b", "max_rank": 1}
    result = post(client, record, "protect", rank)
    assert result["added"] == [{"kind": "rank", "query_id": "refund", "doc_id": "b", "max_rank": 1}]
    assert (result["policy"]["name"], result["policy"]["version"]) == ("Refunds release", 2)
    assert result["policy"]["supersedes"] == policy["id"]
    assert result["policy"]["policy"]["critical_queries"] == [
        {"query_id": "refund", "doc_id": "b", "max_rank": 1}
    ]
    assert result["policy"]["policy"]["metrics"] == {"ndcg": {"max_drop": 0.0}}  # kept
    assert "investigation" in result["policy"]["note"]
    assert result["record"]["protections"][0]["version"] == 2

    again = post(client, record, "protect", {**rank, "policy": result["policy"]["id"]})
    assert again["added"] == []  # already protected: no duplicate rule, same version
    assert again["policy"]["version"] == 2
    post(
        client, record, "protect", {**rank, "max_rank": 2, "policy": result["policy"]["id"]}, 422
    )  # a conflicting rank is changed in the editor, not silently replaced

    # The built-in pipeline records no model context: a context rule could never be judged.
    post(client, record, "protect", {"policy": policy["id"], "by": "Asha", "context": True}, 422)
    post(
        client,
        record,
        "protect",
        {"policy": "startup", "by": "Asha", "doc_id": "b", "max_rank": 1},
        422,
    )
    post(client, record, "protect", {**rank, "max_rank": 5}, 422)  # beyond the run's k


def test_only_a_verified_fix_can_be_protected(client):
    policy = client.post(
        "/api/guided/policies",
        json={"name": "P", "policy": {"metrics": {"ndcg": {"max_drop": 0.0}}}},
    ).json()
    record = opened(client, evaluate(client, save(client)))
    post(
        client,
        record,
        "protect",
        {"policy": policy["id"], "by": "Asha", "doc_id": "b", "max_rank": 1},
        409,
    )


def test_protection_rules_add_without_duplicating():
    from vectraudit.pipelines.investigation_records import protection_policy

    raw = {"metrics": {}, "critical_context": [{"query_id": "q"}]}
    policy, added = protection_policy(raw, "q", context=True, rank=None)
    assert added == []
    assert policy["critical_context"] == [{"query_id": "q"}]
    policy, added = protection_policy(raw, "r", context=True, rank=("d", 3))
    assert [a["kind"] for a in added] == ["context", "rank"]
    assert raw["critical_context"] == [{"query_id": "q"}]  # the input is not modified
