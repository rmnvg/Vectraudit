"""Passage queues use real saved runs, exact versions and human label states."""

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_guided_workbench import CORPUS, run, save
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.models import Hit, PipelineResult
from vectraudit.pipelines.web import create_workbench


@pytest.fixture
def review(tmp_path, monkeypatch):
    def pipeline(request):
        if request.context.get("tenant") == "other":
            return PipelineResult(hits=[Hit(id="other")])
        ids = ["span", "positive", "negative", "forbidden", "common"]
        ids += ["base-only" if request.params["reranker"] == "none" else "candidate-only"]
        return PipelineResult(hits=[Hit(id=doc) for doc in ids])

    monkeypatch.setattr("vectraudit.pipelines.guided.hybrid_pipeline", lambda _: pipeline)
    rows = [
        {
            "query_id": "q",
            "text": "invoice",
            "context": {"tenant": "acme"},
            "relevant": {"positive": 2},
            "relevant_spans": [{"doc_id": "span", "start": 0, "end": 1}],
            "forbidden": ["forbidden"],
        },
        {"query_id": "other", "text": "invoice", "context": {"tenant": "other"}},
        {"query_id": "heldout", "text": "invoice", "split": "heldout"},
    ]
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        draft = save(client, rows=rows, corpus=CORPUS, irrelevant={"q": ["negative"]}).json()
        baseline = run(client, draft, k=10)
        candidate = run(client, draft, k=10, reranker="phrase")
        body = {
            "draft_id": draft["id"],
            "revision": draft["revision"],
            "baseline": baseline["run_id"],
            "candidate": candidate["run_id"],
            "query_id": "q",
        }
        yield client, draft, body, tmp_path


def test_pool_prioritizes_disagreements_preserves_labels_and_context(review):
    client, draft, body, _ = review
    response = client.post("/api/guided/review-pool", json=body)
    assert response.status_code == 200, response.text
    pool = response.json()
    assert [i["doc_id"] for i in pool["items"]] == ["base-only", "candidate-only", "common"]
    assert [i["priority"] for i in pool["items"]] == [2, 2, 1]
    assert pool["items"][0]["baseline_rank"] == 6
    assert pool["items"][0]["candidate_rank"] is None
    assert pool["baseline_measured"]
    assert pool["candidate_measured"]
    assert client.get(f"/api/guided/drafts/{draft['id']}").json() == draft
    other = client.post("/api/guided/review-pool", json={**body, "query_id": "other"}).json()
    assert [i["doc_id"] for i in other["items"]] == ["other"]


def test_pool_orders_by_decision_and_shows_when_judging_can_stop(review):
    client, _, body, _ = review
    pool = client.post("/api/guided/review-pool", json=body).json()
    decision = pool["decision"]
    # Both runs return a forbidden document: that failure holds whatever is judged.
    assert decision["verdict"] == "fail"
    assert decision["state"] == "settled"
    assert "safety.forbidden:q fails whatever is judged" in decision["message"]
    assert decision["n_could_change"] == 0
    assert decision["next_questions"] == []
    assert "about a third fewer, not half" in decision["saving"]
    # Only this question's documents leave the server, each with its swing, largest first.
    assert "items" not in decision
    swings = [i["swing"] for i in pool["items"]]
    assert swings == sorted(swings, reverse=True)
    # Shared documents at the same rank move no difference; one-sided ones do.
    by_doc = {i["doc_id"]: i["swing"] for i in pool["items"]}
    assert by_doc["common"] <= min(by_doc["base-only"], by_doc["candidate-only"])


@pytest.mark.parametrize("query_id", ["heldout", "unknown"])
def test_pool_excludes_heldout_and_unknown_questions(review, query_id):
    client, _, body, _ = review
    assert (
        client.post("/api/guided/review-pool", json={**body, "query_id": query_id}).status_code
        == 422
    )


def test_pool_requires_exact_saved_revision_and_benchmark_hash(review):
    client, draft, body, _ = review
    assert (
        client.post("/api/guided/review-pool", json={**body, "revision": "stale"}).status_code
        == 409
    )
    rows = [json.loads(line) for line in draft["dataset"].splitlines()]
    rows[0]["relevant"]["candidate-only"] = 1
    changed = save(
        client, rows=rows, id=draft["id"], revision=draft["revision"], corpus=CORPUS
    ).json()
    response = client.post(
        "/api/guided/review-pool", json={**body, "revision": changed["revision"]}
    )
    assert response.status_code == 422
    assert "exact saved benchmark" in response.text


def test_pool_excludes_saved_negatives_on_reload_without_changing_dataset(review):
    client, draft, body, _ = review
    rows = [json.loads(line) for line in draft["dataset"].splitlines()]
    changed = save(
        client,
        rows=rows,
        id=draft["id"],
        revision=draft["revision"],
        corpus=CORPUS,
        irrelevant={"q": ["negative", "base-only"]},
    ).json()
    response = client.post(
        "/api/guided/review-pool", json={**body, "revision": changed["revision"]}
    )
    assert response.status_code == 200
    assert [i["doc_id"] for i in response.json()["items"]] == ["candidate-only", "common"]


def test_failed_query_is_unknown_not_a_disagreement(review):
    client, _, body, root = review
    path = root / f"{body['candidate']}.run.json"
    candidate = json.loads(path.read_text())
    candidate["status"] = "partial"
    candidate["queries"][0].update(error={"message": "failed"}, ids=None)
    path.write_text(json.dumps(candidate))
    response = client.post("/api/guided/review-pool", json=body)
    assert response.status_code == 200
    pool = response.json()
    assert pool["baseline_measured"]
    assert not pool["candidate_measured"]
    assert all(i["priority"] == 1 for i in pool["items"])
    assert all(i["candidate_rank"] is None for i in pool["items"])


def test_pool_rejects_same_run_cancelled_runs_and_untrusted_paths(review):
    client, _, body, root = review
    assert (
        client.post(
            "/api/guided/review-pool", json={**body, "candidate": body["baseline"]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/guided/review-pool", json={**body, "candidate": "../elsewhere"}
        ).status_code
        == 404
    )
    path = root / f"{body['candidate']}.run.json"
    candidate = json.loads(path.read_text())
    candidate["status"] = "cancelled"
    path.write_text(json.dumps(candidate))
    assert client.post("/api/guided/review-pool", json=body).status_code == 422


def test_pool_requires_session_and_same_origin(review):
    client, _, body, _ = review
    for headers in ({"X-Workbench-Token": ""}, {"Origin": "https://evil.example"}):
        assert client.post("/api/guided/review-pool", json=body, headers=headers).status_code == 403


def test_audit_excludes_spans_zero_grades_negatives_and_heldout():
    dataset = parse_dataset(
        json.dumps(
            {
                "query_id": "q",
                "text": "a",
                "relevant": {"zero": 0, "positive": 1},
                "relevant_spans": [{"doc_id": "span", "start": 0, "end": 1}],
            }
        )
        + "\n"
        + json.dumps({"query_id": "final", "text": "b", "split": "heldout"})
    )
    recorded = {
        "dataset": {"sha256": dataset.sha256},
        "queries": [
            {"query_id": "q", "ids": ["zero", "span", "negative", "new"]},
            {"query_id": "final", "ids": ["secret"]},
        ],
    }
    result = audit_benchmark(dataset, [recorded], irrelevant={"q": ["negative"]})
    assert [(i["query_id"], i["doc_id"]) for i in result["judgment_queue"]] == [("q", "new")]
