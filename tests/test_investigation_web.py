"""The workbench investigation API: explain a question, try a fix, validate it."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from vectraudit.pipelines.web import create_workbench

# k=1: "b" holds the exact phrase but "a" outranks it on keywords, so the cut at k drops it.
CORPUS = "\n".join(
    json.dumps(d)
    for d in [
        {"doc_id": "a", "text": "refund refund annual annual"},
        {
            "doc_id": "b",
            "text": "the annual refund rule is described here at length for every customer account",
        },
        {"doc_id": "c", "text": "shipping times"},
    ]
)
ROWS = [
    {"query_id": "refund", "text": "annual refund", "relevant": {"b": 1}},
    {"query_id": "shipping", "text": "shipping times", "relevant": {"c": 1}},
    {"query_id": "final", "text": "annual refund rule", "relevant": {"b": 1}, "split": "heldout"},
]


@pytest.fixture
def client(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"metrics": {"ndcg": {"max_drop": 0.0}}, "min_queries": 1}))
    with TestClient(create_workbench(tmp_path, policy=policy)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        yield client


def save(client, rows=ROWS, **extra):
    response = client.post(
        "/api/guided/drafts",
        json={
            "name": "Refunds",
            "dataset": "\n".join(map(json.dumps, rows)),
            "corpus": CORPUS,
            **extra,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def wait(client, identifier):
    assert client.app.state.guided.jobs.get(identifier).wait(20)
    state = client.get(f"/api/guided/jobs/{identifier}").json()
    assert state["status"] == "done", state
    return state["run"]


def evaluate(client, draft):
    response = client.post(
        "/api/guided/jobs",
        json={
            "draft_id": draft["id"],
            "revision": draft["revision"],
            "corpus": CORPUS,
            "k": 1,
            "depth": 10,
            "label": "Current",
        },
    )
    assert response.status_code == 202, response.text
    return wait(client, response.json()["id"])


def explain(client, run, query="refund"):
    response = client.post(
        "/api/guided/investigate", json={"run": run["run_id"], "query_id": query}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_investigate_try_and_validate_a_fix_in_the_workbench(client, tmp_path):
    run = evaluate(client, save(client))
    found = explain(client, run)
    assert found["question"] == "annual refund"
    assert found["losses"][0] == {
        "class": "lost_downstream",
        "doc_id": "b",
        "stage": "final",
        "stage_kind": "truncate",
    }
    assert "use the phrase reranker" in [f["name"] for f in found["fixes"]]
    assert found["trial"] == {
        "available": True,
        "reason": None,
        "explore_questions": 2,
        "heldout_questions": 1,
        "repeats": 1,
        "k": 1,
    }

    started = client.post(
        "/api/guided/fixes/try",
        json={"run": run["run_id"], "fix": "use the phrase reranker", "query_ids": ["refund"]},
    )
    assert started.status_code == 202, started.text
    trial = wait(client, started.json()["id"])
    assert (trial["kind"], trial["status"], trial["recovered_focus"]) == ("trial", "exploratory", 1)
    assert trial["regressed"] == []
    assert client.get(f"/api/guided/fixes/{trial['id']}").json()["recovered_focus"] == 1
    assert sorted(p.name for p in (tmp_path / ".workbench" / "fixes" / trial["id"]).iterdir()) == [
        "current.run.json",
        "fixed.run.json",
    ]
    assert len(list(tmp_path.glob("*.run.json"))) == 1  # trial runs stay out of the run list

    body = {"run": run["run_id"], "fix": "use the phrase reranker", "policy": "startup"}
    assert client.post("/api/guided/fixes/validate", json=body).status_code == 422
    started = client.post("/api/guided/fixes/validate", json={**body, "confirm_heldout": True})
    assert started.status_code == 202, started.text
    validation = wait(client, started.json()["id"])
    assert (validation["status"], validation["approved"]) == ("approved_under_policy", True)
    assert validation["promote"]["params"]["reranker"] == "phrase"


def test_a_changed_benchmark_or_held_out_question_cannot_be_tried(client):
    draft = save(client)
    run = evaluate(client, draft)
    save(client, id=draft["id"], revision=draft["revision"], name="Refunds v2")
    found = explain(client, run)
    assert found["trial"]["available"] is False
    assert "benchmark changed" in found["trial"]["reason"]
    refused = client.post(
        "/api/guided/fixes/try",
        json={"run": run["run_id"], "fix": "use the phrase reranker", "query_ids": ["refund"]},
    )
    assert refused.status_code == 409

    unknown = client.post(
        "/api/guided/fixes/try",
        json={"run": run["run_id"], "fix": "make it better", "query_ids": ["refund"]},
    )
    assert unknown.status_code == 422


def test_held_out_questions_are_reserved_for_validation(client):
    draft = save(client)
    response = client.post(
        "/api/guided/jobs",
        json={
            "draft_id": draft["id"],
            "revision": draft["revision"],
            "corpus": CORPUS,
            "k": 1,
            "mode": "heldout",
            "confirm_heldout": True,
        },
    )
    run = wait(client, response.json()["id"])
    found = explain(client, run, "final")
    assert found["trial"]["available"] is False
    assert "reserved for validation" in found["trial"]["reason"]


def test_investigation_requires_the_session_token(client):
    client.headers.pop("X-Workbench-Token")
    response = client.post("/api/guided/investigate", json={"run": "0" * 32, "query_id": "q"})
    assert response.status_code == 403
