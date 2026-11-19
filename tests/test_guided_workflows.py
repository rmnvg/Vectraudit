"""Real local APIs: import -> annotation -> freeze, permission runs and case lifecycle."""

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_guided_workbench import CORPUS, ROWS, run, save
from vectraudit.pipelines.models import Hit, PipelineResult
from vectraudit.pipelines.web import create_workbench


@pytest.fixture
def client(tmp_path):
    with TestClient(create_workbench(tmp_path)) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        yield client


def post(client, path, body):
    response = client.post("/api/guided/" + path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_csv_preview_optional_ids_and_quoted_passages(client):
    result = post(
        client,
        "import",
        {
            "format": "csv",
            "content": 'question,labels\n"where, invoice?",a:2\n',
            "text_column": "question",
            "relevant_column": "labels",
        },
    )
    assert result["count"] == 1
    row = json.loads(result["dataset"].splitlines()[1])
    assert row["text"] == "where, invoice?"
    assert row["relevant"] == {"a": 2}
    assert row["query_id"].startswith("q-")
    assert (
        client.post("/api/guided/import", json={"format": "json", "content": "[1]"}).status_code
        == 422
    )


def test_incidents_strip_labels_deduplicate_with_context_and_keep_provenance(client):
    records = [
        {
            "query_id": name,
            "inputs": {"question": "invoice"},
            "context": {"tenant": tenant},
            "relevant": {"a": 2},
            "forbidden": ["b"],
            "no_answer": True,
            "feedback": "bad",
            "split": "heldout",
        }
        for name, tenant in [("r1", "a"), ("r2", "a"), ("r3", "b")]
    ]
    result = post(
        client,
        "import",
        {
            "format": "json",
            "content": json.dumps({"items": records}),
            "text_column": "inputs.question",
            "incident": True,
        },
    )
    rows = [json.loads(line) for line in result["dataset"].splitlines()[1:]]
    assert result["count"] == 2
    assert result["duplicates"] == 1
    assert all(
        not r.get("relevant") and not r.get("no_answer") and not r.get("forbidden") for r in rows
    )
    assert all(r["split"] == "explore" for r in rows)
    assert result["references"][rows[0]["query_id"]] == ["r1", "r2"]
    reordered = post(
        client,
        "import",
        {
            "format": "json",
            "content": json.dumps(records[::-1]),
            "text_column": "inputs.question",
            "incident": True,
        },
    )
    assert {r["query_id"] for r in rows} == {
        json.loads(line)["query_id"] for line in reordered["dataset"].splitlines()[1:]
    }
    assert "feedback" not in result["dataset"]


def test_permission_matrix_real_context_forwarding_and_validation(client):
    result = post(
        client,
        "permissions",
        {
            "dataset": json.dumps(ROWS[0]),
            "query_id": "allowed",
            "scenarios": [
                {
                    "name": "Acme",
                    "context": {"tenant": "acme"},
                    "allowed": ["a"],
                    "forbidden": ["private"],
                },
                {
                    "name": "Other",
                    "context": {"tenant": "other"},
                    "allowed": ["private"],
                    "forbidden": ["a"],
                },
                {"name": "Revoked", "context": {"tenant": "revoked"}, "no_answer": True},
            ],
        },
    )
    draft = save(client, rows=result["rows"]).json()
    evaluated = run(client, draft)
    assert [q["ids"] for q in evaluated["queries"]] == [["a"], ["private"], []]
    assert all(not q["forbidden_ids"] for q in evaluated["queries"])
    assert evaluated["queries"][2]["no_answer_correct"] is True
    for scenario in [
        {"name": "bad", "context": {}, "allowed": ["a"]},
        {"name": "bad", "context": {"tenant": "a"}, "allowed": ["a"], "forbidden": ["a"]},
        {"name": "bad", "context": {"tenant": "a"}, "allowed": ["a"], "no_answer": True},
    ]:
        assert (
            client.post(
                "/api/guided/permissions",
                json={
                    "dataset": json.dumps(ROWS[0]),
                    "query_id": "allowed",
                    "scenarios": [scenario],
                },
            ).status_code
            == 422
        )
    assert (
        client.post(
            "/api/guided/permissions",
            json={
                "dataset": json.dumps(ROWS[3]),
                "query_id": "final",
                "scenarios": [{"name": "bad", "context": {"tenant": "a"}, "allowed": ["a"]}],
            },
        ).status_code
        == 422
    )


def test_freeze_preserves_negatives_spans_and_detects_changed_deleted_no_answer(client, tmp_path):
    rows = [
        {
            "query_id": "span",
            "text": "invoice",
            "relevant_spans": [{"doc_id": "a", "start": 0, "end": 7}],
        },
        {"query_id": "empty", "text": "volcano", "no_answer": True},
        {"query_id": "negative", "text": "invoice"},
    ]
    draft = save(client, rows=rows, corpus=CORPUS, irrelevant={"negative": ["private"]}).json()
    frozen = post(client, "snapshots", {"draft_id": draft["id"], "revision": draft["revision"]})
    assert frozen["irrelevant"] == {"negative": ["private"]}
    assert (
        post(client, "freshness", {"snapshot": frozen["id"], "corpus": CORPUS})["status"]
        == "unchanged"
    )
    replacement = json.dumps({"doc_id": "a", "text": "volcano new answer"})
    result = post(client, "freshness", {"snapshot": frozen["id"], "corpus": replacement})
    assert result["changed"] == ["a"]
    assert result["deleted"] == ["private"]
    assert {i["query_id"] for i in result["items"]} == {"span", "empty", "negative"}
    revised = save(
        client, rows=rows, id=draft["id"], revision=draft["revision"], corpus=replacement
    ).json()
    assert revised["revision"] != draft["revision"]
    assert client.get("/api/guided/snapshots/" + frozen["id"]).json() == frozen
    assert (
        client.post(
            "/api/guided/snapshots", json={"draft_id": draft["id"], "revision": draft["revision"]}
        ).status_code
        == 409
    )
    assert (
        post(client, "freshness", {"snapshot": frozen["id"], "corpus": ""})["status"] == "unknown"
    )
    with TestClient(create_workbench(tmp_path)) as other:
        other.headers["X-Workbench-Token"] = other.get("/api/guided/session").json()["token"]
        assert other.get("/api/guided/snapshots/" + frozen["id"]).json() == frozen


def test_negative_judgments_cannot_contradict_positive_or_span_labels(client):
    assert save(client, corpus=CORPUS, irrelevant={"allowed": ["a"]}).status_code == 422
    assert save(client, irrelevant={"nonexistent": ["a"]}).status_code == 422
    assert (
        save(
            client,
            rows=[
                {
                    "query_id": "span",
                    "text": "a",
                    "relevant_spans": [{"doc_id": "a", "start": 0, "end": 1}],
                }
            ],
            irrelevant={"span": ["a"]},
        ).status_code
        == 422
    )
    assert save(client, corpus=CORPUS + "\n" + CORPUS).status_code == 422


def test_case_verified_only_by_new_full_comparable_run_and_reopens(client, monkeypatch, tmp_path):
    def pipeline(request):
        return PipelineResult(hits=[Hit(id="a" if request.params["reranker"] == "none" else "b")])

    monkeypatch.setattr("vectraudit.pipelines.guided.hybrid_pipeline", lambda _: pipeline)
    rows = [{"query_id": f"q{i}", "text": "invoice", "relevant": {"a": 2}} for i in range(35)]
    draft = save(client, rows=rows).json()
    baseline = run(client, draft, k=1)
    broken = run(client, draft, k=1, reranker="phrase")
    body = {"baseline": baseline["run_id"], "candidate": broken["run_id"], "query_id": "q0"}
    case = post(client, "cases", body)
    assert post(client, "cases", body) == case
    path = "cases/" + case["id"]
    assert (
        client.post(
            "/api/guided/" + path, json={"revision": case["revision"], "status": "verified"}
        ).status_code
        == 422
    )
    updated = post(
        client,
        path,
        {
            "revision": case["revision"],
            "status": "fix_proposed",
            "owner": "Engineer",
            "note": "Restore relevant document",
        },
    )
    assert (
        client.post(
            "/api/guided/" + path, json={"revision": case["revision"], "status": "open"}
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/guided/" + path + "/verify",
            json={"revision": updated["revision"], "candidate": baseline["run_id"]},
        ).status_code
        == 422
    )
    subset = save(client, rows=rows[:2]).json()
    short = run(client, subset, k=1)
    assert (
        client.post(
            "/api/guided/" + path + "/verify",
            json={"revision": updated["revision"], "candidate": short["run_id"]},
        ).status_code
        == 422
    )
    fixed = run(client, draft, k=1)
    verified = post(
        client, path + "/verify", {"revision": updated["revision"], "candidate": fixed["run_id"]}
    )
    assert verified["status"] == "verified"
    assert verified["history"][-1]["verdict"] == "pass"
    check_path = tmp_path / verified["history"][-1]["check_file"]
    persisted_check = json.loads(check_path.read_text())
    assert persisted_check["verdict"] == "pass"
    assert persisted_check["policy"]

    recurrence = run(client, draft, k=1, reranker="phrase")
    reopened = post(
        client,
        path + "/verify",
        {"revision": verified["revision"], "candidate": recurrence["run_id"]},
    )
    assert reopened["status"] == "open"
    assert reopened["history"][-1]["verified"] is False
    with TestClient(create_workbench(tmp_path)) as other:
        other.headers["X-Workbench-Token"] = other.get("/api/guided/session").json()["token"]
        assert other.get("/api/guided/cases").json() == [reopened]
        assert len(other.get("/api/guided/runs").json()) == 5


def test_all_workflow_endpoints_reject_cross_origin_or_missing_token(client, tmp_path):
    for route in ["import", "permissions", "snapshots", "freshness", "cases"]:
        assert (
            client.post(
                "/api/guided/" + route, json={}, headers={"X-Workbench-Token": ""}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/guided/" + route, json={}, headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )
    for route in ["snapshots", "cases", "runs"]:
        assert (
            client.get("/api/guided/" + route, headers={"X-Workbench-Token": ""}).status_code == 403
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".workbench").mkdir()
    (tmp_path / ".workbench" / "cases").symlink_to(outside)
    assert client.get("/api/guided/cases").status_code == 400


def test_native_preview_preserves_spans_filters_vectors_and_header(client):
    source = "\n".join(
        map(
            json.dumps,
            [
                {
                    "dataset": {
                        "name": "Existing",
                        "embedding_model": "local-model",
                        "provenance": {"original": "human"},
                    }
                },
                {
                    "query_id": "existing",
                    "text": "invoice",
                    "vector": [1.0, 0.0],
                    "filter": {"current": True},
                    "context": {"tenant": "acme"},
                    "split": "heldout",
                    "relevant_spans": [{"doc_id": "a", "start": 0, "end": 7, "grade": 2}],
                },
            ],
        )
    )
    preview = post(client, "import", {"format": "jsonl", "content": source})
    header, row = map(json.loads, preview["dataset"].splitlines())
    assert header["dataset"]["embedding_model"] == "local-model"
    assert header["dataset"]["provenance"]["original"] == "human"
    assert row["relevant_spans"][0]["doc_id"] == "a"
    assert row["filter"] == {"current": True}
    assert row["context"] == {"tenant": "acme"}
    assert row["split"] == "heldout"
    assert row["vector"] == [1.0, 0.0]


def test_incident_dedup_keeps_distinct_filters_and_nested_ids(client):
    records = [
        {"trace": {"id": name}, "text": "invoice", "filter": {"current": current}}
        for name, current in [("t1", True), ("t2", False)]
    ]
    preview = post(
        client,
        "import",
        {
            "format": "json",
            "content": json.dumps(records),
            "id_column": "trace.id",
            "incident": True,
        },
    )
    assert preview["count"] == 2
    assert sorted(preview["references"].values()) == [["t1"], ["t2"]]


def test_permission_generation_keeps_source_unchanged(client):
    source = json.dumps(ROWS[0])
    result = post(
        client,
        "permissions",
        {
            "dataset": source,
            "query_id": "allowed",
            "scenarios": [{"name": "New", "context": {"tenant": "other"}, "allowed": ["private"]}],
        },
    )
    assert result["rows"][0]["relevant"] == {"private": 1}
    assert not result["rows"][0]["forbidden"]
    assert json.loads(source)["forbidden"] == ["private"]
