"""Exercise the guided workflow against real persistence and evaluation jobs."""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from vectraudit.pipelines.models import PipelineResult
from vectraudit.pipelines.web import create_workbench

ROWS = [
    {
        "query_id": "allowed",
        "text": "invoice",
        "relevant": {"a": 2},
        "context": {"tenant": "acme"},
        "forbidden": ["private"],
        "tags": ["permission"],
    },
    {"query_id": "none", "text": "volcano", "no_answer": True},
    {"query_id": "missing", "text": "invoice"},
    {"query_id": "final", "text": "invoice", "relevant": {"a": 1}, "split": "heldout"},
]
CORPUS = "\n".join(
    json.dumps(d)
    for d in [
        {"doc_id": "a", "text": "invoice", "metadata": {"tenant": "acme"}},
        {"doc_id": "private", "text": "invoice", "metadata": {"tenant": "other"}},
    ]
)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_workbench(tmp_path)) as client:
        session = client.get("/api/guided/session")
        assert session.headers["cache-control"] == "no-store"
        client.headers["X-Workbench-Token"] = session.json()["token"]
        yield client


def save(client, rows=None, **changes):
    return client.post(
        "/api/guided/drafts",
        json={
            "name": "Benchmark",
            "dataset": "\n".join(map(json.dumps, rows or ROWS)),
            **changes,
        },
    )


def options(draft, **changes):
    return {
        "draft_id": draft["id"],
        "revision": draft["revision"],
        "corpus": CORPUS,
        "k": 2,
        **changes,
    }


def run(client, draft, **changes):
    response = client.post("/api/guided/jobs", json=options(draft, **changes))
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    assert client.app.state.guided.jobs.get(identifier).wait(10)
    state = client.get(f"/api/guided/jobs/{identifier}").json()
    assert state["status"] == "done", state
    return state["run"]


def test_drafts_roundtrip_revision_and_restart(client, tmp_path):
    draft = save(client).json()
    assert draft["audit"]["n_labelled"] == 3
    assert draft["audit"]["unannotated"] == ["missing"]
    assert client.get(f"/api/guided/drafts/{draft['id']}").json() == draft
    revised = save(client, id=draft["id"], revision=draft["revision"], name="Changed")
    assert revised.status_code == 200
    assert revised.json()["revision"] != draft["revision"]
    assert save(client, id=draft["id"], revision=draft["revision"]).status_code == 409
    assert client.post("/api/guided/jobs", json=options(draft)).status_code == 409
    with TestClient(create_workbench(tmp_path)) as reopened:
        session = reopened.get("/api/guided/session").json()
        assert session["drafts"][0]["name"] == "Changed"
        assert session["token"] != client.headers["X-Workbench-Token"]


def test_real_evaluation_splits_and_comparison(client, tmp_path):
    draft = save(client).json()
    baseline = run(client, draft, label="Baseline")
    assert baseline["status"] == "complete"
    assert [q["query_id"] for q in baseline["queries"]] == ["allowed", "none", "missing"]
    assert baseline["queries"][0]["forbidden_ids"] == []
    assert baseline["queries"][1]["no_answer_correct"] is True
    assert baseline["queries"][2]["metrics"] is None
    assert baseline["queries"][0]["evidence"]["stages"]
    assert baseline["benchmark_source"]["split"] == "explore"
    assert (tmp_path / f"{baseline['run_id']}.run.json").exists()
    candidate = run(client, draft, label="Candidate", reranker="phrase")
    compared = client.post(
        "/api/guided/compare",
        json={
            "baseline": baseline["run_id"],
            "candidate": candidate["run_id"],
        },
    )
    assert compared.status_code == 200, compared.text
    assert compared.json()["check"]["verdict"] != "not_comparable"
    files = compared.json()["files"]
    assert files["baseline"] == f"{baseline['run_id']}.run.json"
    assert (tmp_path / files["check"]).exists()
    findings = compared.json()["triage"]
    assert findings["format"] == "vectraudit.triage"
    assert findings["candidate_run"] == candidate["run_id"]
    assert findings["n_paired"] == 3
    assert client.post("/api/guided/jobs", json=options(draft, mode="heldout")).status_code == 422
    heldout = run(client, draft, mode="heldout", confirm_heldout=True)
    assert [q["query_id"] for q in heldout["queries"]] == ["final"]
    incompatible = client.post(
        "/api/guided/compare",
        json={
            "baseline": baseline["run_id"],
            "candidate": heldout["run_id"],
        },
    ).json()
    assert incompatible["check"]["verdict"] == "not_comparable"


def test_permissions_and_input_boundaries(client, tmp_path):
    body = {"name": "Test", "dataset": json.dumps(ROWS[0])}
    assert (
        client.post("/api/guided/drafts", json=body, headers={"X-Workbench-Token": ""}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/guided/drafts", json=body, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/guided/drafts", json=body, headers={"Sec-Fetch-Site": "cross-site"}
        ).status_code
        == 403
    )
    assert client.get("/api/guided/session", headers={"Host": "evil.example"}).status_code == 400
    assert (
        client.post("/api/guided/drafts", content=b"x" * (5 * 1024 * 1024 + 1)).status_code == 413
    )
    assert save(client, rows=[{**ROWS[0], "no_answer": True}]).status_code == 422
    draft = save(client).json()
    assert (
        client.post("/api/guided/jobs", json=options(draft, pipeline="os:system")).status_code
        == 422
    )
    assert (
        client.post(
            "/api/guided/jobs", json=options(draft, target="https://example.com")
        ).status_code
        == 422
    )
    identifier = "a" * 32
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret":"hidden"}')
    (tmp_path / ".workbench" / "drafts" / f"{identifier}.json").symlink_to(outside)
    assert client.get(f"/api/guided/drafts/{identifier}").status_code == 400
    assert (
        client.post(
            "/api/guided/compare", json={"baseline": "../outside", "candidate": identifier}
        ).status_code
        == 404
    )


def test_registered_http_requires_consent_and_redacts_failure(tmp_path, monkeypatch):
    manifest = tmp_path / "pipelines.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "unused.jsonl",
                "candidates": [
                    {
                        "name": "Remote",
                        "adapter": {"kind": "http", "target": "https://example.com/retrieve"},
                    },
                ],
            }
        )
    )

    def fail(*args):
        raise RuntimeError("SECRET")

    monkeypatch.setattr("vectraudit.pipelines.guided.open_adapter", fail)
    with TestClient(create_workbench(tmp_path, manifest)) as client:
        session = client.get("/api/guided/session").json()
        client.headers["X-Workbench-Token"] = session["token"]
        draft = save(client).json()
        body = options(draft, pipeline="configured-0")
        assert client.post("/api/guided/jobs", json=body).status_code == 422
        response = client.post("/api/guided/jobs", json={**body, "confirm_remote": True})
        identifier = response.json()["id"]
        assert client.app.state.guided.jobs.get(identifier).wait(10)
        result = client.get(f"/api/guided/jobs/{identifier}")
        assert result.json()["status"] == "failed"
        assert "SECRET" not in result.text


def test_cancellation_and_capacity_do_not_save_partial_results(client, monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()

    def pipeline(request):
        entered.set()
        assert release.wait(10)
        return PipelineResult(hits=[])

    monkeypatch.setattr("vectraudit.pipelines.guided.hybrid_pipeline", lambda _: pipeline)
    draft = save(client).json()
    first = client.post("/api/guided/jobs", json=options(draft)).json()["id"]
    try:
        assert entered.wait(5)
        # Two evaluations may run at once; the third waits for capacity.
        second = client.post("/api/guided/jobs", json=options(draft))
        assert second.status_code == 202
        third = client.post("/api/guided/jobs", json=options(draft))
        assert third.status_code == 429
        assert "2 evaluations are already running" in third.json()["detail"]
        for job in (first, second.json()["id"]):
            assert client.post(f"/api/guided/jobs/{job}/cancel").json()["cancel_requested"]
    finally:
        release.set()
    for job in (first, second.json()["id"]):
        assert client.app.state.guided.jobs.get(job).wait(10)
        assert client.get(f"/api/guided/jobs/{job}").json()["status"] == "cancelled"
    assert not list(tmp_path.glob("*.run.json"))


def test_saved_evaluation_resumes_after_workspace_restart(client, monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def pipeline(request):
        calls.append(request.query_id)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
        return PipelineResult(hits=[])

    monkeypatch.setattr("vectraudit.pipelines.guided.hybrid_pipeline", lambda _: pipeline)
    draft = save(client).json()
    identifier = client.post("/api/guided/jobs", json=options(draft)).json()["id"]
    try:
        assert entered.wait(5)
        before = client.get("/api/guided/evaluations").json()[0]
        assert before["running"]
        assert (
            client.post(f"/api/guided/evaluations/{identifier}/resume", json={}).status_code == 429
        )
        assert client.get("/api/guided/evaluations").json()[0]["job_id"] == identifier
        client.post(f"/api/guided/jobs/{identifier}/cancel")
    finally:
        release.set()
    assert client.app.state.guided.jobs.get(identifier).wait(10)
    assert client.get("/api/guided/evaluations").json()[0]["finished"] == 1
    # Fresh server state, same local directory: no JobRunner memory survives.
    with TestClient(create_workbench(tmp_path)) as restarted:
        restarted.headers["X-Workbench-Token"] = restarted.get("/api/guided/session").json()[
            "token"
        ]
        state = restarted.get("/api/guided/evaluations").json()[0]
        assert not state["running"]
        started = restarted.post(f"/api/guided/evaluations/{identifier}/resume", json={})
        assert started.status_code == 202
        job_id = started.json()["id"]
        assert restarted.app.state.guided.jobs.get(job_id).wait(10)
        run = restarted.get(f"/api/guided/jobs/{job_id}").json()["run"]
        assert run["status"] == "complete"
        assert calls == ["allowed", "none", "missing"]
        assert run["benchmark_source"]["revision"] == draft["revision"]
        assert restarted.get("/api/guided/evaluations").json()[0]["status"] == "complete"


def test_resume_rejects_changed_draft_and_requires_heldout_consent(client):
    draft = save(client).json()
    run(client, draft, mode="heldout", confirm_heldout=True)
    identifier = client.get("/api/guided/evaluations").json()[0]["id"]
    assert client.post(f"/api/guided/evaluations/{identifier}/resume", json={}).status_code == 422
    assert (
        save(client, id=draft["id"], revision=draft["revision"], name="Changed").status_code == 200
    )
    rejected = client.post(
        f"/api/guided/evaluations/{identifier}/resume", json={"confirm_heldout": True}
    )
    assert rejected.status_code == 409


def test_resume_reexports_if_process_stopped_after_journal_commit(client, tmp_path):
    draft = save(client).json()
    initial = run(client, draft)
    (tmp_path / f"{initial['run_id']}.run.json").unlink()
    state = client.get("/api/guided/evaluations").json()[0]
    assert state["status"] == "partial"
    started = client.post(f"/api/guided/evaluations/{state['id']}/resume", json={}).json()["id"]
    assert client.app.state.guided.jobs.get(started).wait(10)
    recovered = client.get(f"/api/guided/jobs/{started}").json()["run"]
    assert recovered["run_id"] == initial["run_id"]
    assert recovered["queries"] == initial["queries"]
    assert (tmp_path / f"{initial['run_id']}.run.json").exists()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_damaged_guided_journal_never_silently_starts_over(client, tmp_path, damage):
    draft = save(client).json()
    initial = run(client, draft)
    identifier = client.get("/api/guided/evaluations").json()[0]["id"]
    journal = tmp_path / ".workbench" / "evaluations" / f"{identifier}.sqlite"
    if damage == "missing":
        journal.unlink()
    else:
        journal.write_bytes(b"broken")
    state = client.get("/api/guided/evaluations").json()[0]
    assert state["status"] == "unavailable"
    assert state["resume_blocker"]
    response = client.post(f"/api/guided/evaluations/{identifier}/resume", json={})
    if damage == "missing":
        assert response.status_code == 409
    else:
        job_id = response.json()["id"]
        assert client.app.state.guided.jobs.get(job_id).wait(10)
        assert client.get(f"/api/guided/jobs/{job_id}").json()["status"] == "failed"
    assert json.loads((tmp_path / f"{initial['run_id']}.run.json").read_text()) == initial
