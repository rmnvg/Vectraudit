"""Real resolver execution, bounded passage access and snapshot/context preservation."""

import json
import textwrap

import pytest
from fastapi import HTTPException

from tests.test_guided_workbench import run, save
from tests.test_passage_review import review  # noqa: F401
from vectraudit.pipelines.document_lookup import (
    LookupConfig,
    LookupRequest,
    LookupResult,
    LookupRunner,
)
from vectraudit.pipelines.manifest import Manifest

RESOLVER = """
import json
def lookup(request):
    print("SECRET adapter output")
    return {"corpus_version": request.corpus_version, "documents": [
        {"doc_id": doc, "status": "found", "text": json.dumps(request.model_dump()),
         "title": "A passage", "version": "content-v1"}
        for doc in request.doc_ids
    ]}
"""


@pytest.fixture
def lookup(request):
    client, draft, body, root = request.getfixturevalue("review")
    module = root / "customer_documents.py"
    module.write_text(textwrap.dedent(RESOLVER))
    client.app.state.guided.manifest = Manifest.model_validate(
        {
            "dataset": "unused.jsonl",
            "candidates": [
                {"name": "existing", "adapter": {"kind": "python", "target": "unused:retrieve"}}
            ],
            "document_lookup": {"target": "customer_documents:lookup", "timeout_s": 2},
        }
    )
    return (
        client,
        draft,
        {
            **body,
            "source_run": body["candidate"],
            "doc_ids": ["base-only", "common"],
            "confirm_lookup": True,
        },
        module,
    )


def test_lookup_forwards_saved_context_snapshot_and_never_persists_text(lookup):
    client, draft, body, module = lookup
    before = set(module.parent.rglob("*.json"))
    response = client.post("/api/guided/review-documents", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["version_verified"]
    assert len(result["documents"]) == 2
    assert "SECRET" not in response.text
    request = json.loads(result["documents"][0]["text"])
    assert request["context"] == {"tenant": "acme"}
    assert request["filters"] == {}
    assert request["corpus_version"] == result["corpus_version"]
    assert request["run_id"] == body["candidate"]
    assert "text" not in request  # No question text or full corpus sent to the source.
    assert len(result["documents"][0]["sha256"]) == 64
    assert client.get(f"/api/guided/drafts/{draft['id']}").json() == draft
    assert set(module.parent.rglob("*.json")) == before


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"confirm_lookup": False}, 422),
        ({"revision": "stale"}, 409),
        ({"query_id": "heldout"}, 422),
        ({"query_id": "other"}, 422),
        ({"doc_ids": ["forbidden"]}, 422),
        ({"doc_ids": ["not-in-pool"]}, 422),
        ({"doc_ids": ["common"] * 2}, 422),
        ({"doc_ids": ["common"] * 11}, 422),
        ({"source_run": "../arbitrary"}, 422),
        ({"target": "evil:lookup"}, 422),
        ({"context": {"tenant": "other"}}, 422),
    ],
)
def test_lookup_rejects_untrusted_or_stale_selection_before_execution(lookup, change, code):
    client, _, body, module = lookup
    module.write_text("raise RuntimeError('resolver must not run')")
    response = client.post("/api/guided/review-documents", json={**body, **change})
    assert response.status_code == code, response.text


@pytest.mark.parametrize("headers", [{"X-Workbench-Token": ""}, {"Origin": "https://evil.example"}])
def test_lookup_requires_token_and_same_origin(lookup, headers):
    client, _, body, _ = lookup
    assert (
        client.post("/api/guided/review-documents", json=body, headers=headers).status_code == 403
    )


@pytest.mark.parametrize("status", ["missing", "forbidden", "version_unavailable"])
def test_explicit_unavailable_status_has_no_text(lookup, status):
    client, _, body, module = lookup
    module.write_text(f'''def lookup(request):
    return {{"corpus_version": request.corpus_version, "documents": [
        {{"doc_id": d, "status": "{status}"}} for d in request.doc_ids]}}
''')
    response = client.post("/api/guided/review-documents", json=body)
    assert response.status_code == 200, response.text
    assert all(d["status"] == status and d["text"] is None for d in response.json()["documents"])


@pytest.mark.parametrize(
    ("replacement", "code"),
    [
        ('return {"corpus_version": "wrong", "documents": []}', 409),
        ('return {"corpus_version": request.corpus_version, "documents": []}', 502),
        ('raise RuntimeError("SECRET exception")', 502),
        ("import time; time.sleep(5)", 504),
    ],
)
def test_invalid_response_exception_and_timeout_do_not_expose_content(lookup, replacement, code):
    client, _, body, module = lookup
    module.write_text("def lookup(request):\n    " + replacement + "\n")
    response = client.post("/api/guided/review-documents", json=body)
    assert response.status_code == code, response.text
    assert "SECRET" not in response.text


def test_unknown_snapshot_is_explicit(lookup):
    client, _, body, module = lookup
    path = module.parent / f"{body['source_run']}.run.json"
    run = json.loads(path.read_text())
    run["backend"].update(corpus_version=None, snapshot=None)
    path.write_text(json.dumps(run))
    result = client.post("/api/guided/review-documents", json=body).json()
    assert result["version_verified"] is False


def test_saved_filters_and_other_tenant_are_forwarded(lookup):
    client, draft, body, _ = lookup
    rows = [json.loads(line) for line in draft["dataset"].splitlines() if line.strip()]
    next(row for row in rows if row["query_id"] == "other")["filter"] = {"category": "support"}
    draft = save(
        client,
        rows=rows,
        id=draft["id"],
        revision=draft["revision"],
        corpus=draft["corpus"],
        irrelevant=draft["irrelevant"],
    ).json()
    base = run(client, draft, k=10)
    candidate = run(client, draft, k=10, reranker="phrase")
    response = client.post(
        "/api/guided/review-documents",
        json={
            **body,
            "revision": draft["revision"],
            "baseline": base["run_id"],
            "candidate": candidate["run_id"],
            "source_run": base["run_id"],
            "query_id": "other",
            "doc_ids": ["other"],
        },
    )
    assert response.status_code == 200, response.text
    request = json.loads(response.json()["documents"][0]["text"])
    assert request["context"] == {"tenant": "other"}
    assert request["filters"] == {"category": "support"}


@pytest.mark.parametrize(
    "documents",
    [
        [{"doc_id": "unexpected", "status": "found", "text": "SECRET", "version": "v1"}],
        [{"doc_id": "common", "status": "missing"}] * 2,
        [{"doc_id": "common", "status": "forbidden", "text": "SECRET"}],
        [{"doc_id": "common", "status": "found", "text": "SECRET"}],
    ],
)
def test_unexpected_duplicate_or_invalid_documents_fail_closed(lookup, documents):
    client, _, body, module = lookup
    module.write_text(
        'def lookup(request):\n    return {"corpus_version": request.corpus_version, '
        f'"documents": {documents!r}}}\n'
    )
    response = client.post("/api/guided/review-documents", json=body)
    assert response.status_code == 502
    assert "SECRET" not in response.text


def test_concurrent_lookup_is_bounded(tmp_path):
    runner = LookupRunner()
    runner._lock.acquire()
    try:
        with pytest.raises(HTTPException) as exc:
            runner.run(
                LookupConfig(target="unused:lookup"),
                tmp_path,
                LookupRequest(query_id="q", doc_ids=["d"], run_id="r"),
            )
        assert exc.value.status_code == 429
    finally:
        runner._lock.release()


def test_changed_draft_during_lookup_drops_response(lookup, monkeypatch):
    client, draft, body, _ = lookup

    def changed(self, config, root, request):
        path = client.app.state.guided._draft_path(draft["id"])
        current = json.loads(path.read_text())
        current["revision"] = "changed-during-lookup"
        path.write_text(json.dumps(current))
        return LookupResult(corpus_version=request.corpus_version, documents=[])

    monkeypatch.setattr(LookupRunner, "run", changed)
    assert client.post("/api/guided/review-documents", json=body).status_code == 409


def test_example_sqlite_resolver_scopes_tenant_snapshot_and_filters(tmp_path, monkeypatch):
    import importlib.util
    import sqlite3
    from pathlib import Path

    path = Path(__file__).parents[1] / "examples/workbench/document_source.py"
    spec = importlib.util.spec_from_file_location("document_source", path)
    assert spec is not None
    assert spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    db = tmp_path / "passages.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE passages "
            "(snapshot, tenant, doc_id, text, title, content_version, category)"
        )
        connection.executemany(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("s1", "acme", "a", "Acme text", "A", "v1", "support"),
                ("s1", "acme", "b", "Billing text", "B", "v1", "billing"),
                ("s1", "other", "c", "Other tenant", "C", "v1", "support"),
                ("s0", "acme", "d", "Old snapshot", "D", "v0", "support"),
            ],
        )
    monkeypatch.setenv("DOCUMENT_LOOKUP_DB", str(db))

    def statuses(**fields):
        request = LookupRequest(
            query_id="q", doc_ids=["a", "b", "c", "d"], run_id="r", corpus_version="s1", **fields
        )
        return {d.doc_id: d.status for d in example.lookup(request).documents}

    assert statuses(context={"tenant": "acme"}) == {
        "a": "found",
        "b": "found",
        "c": "missing",
        "d": "missing",
    }
    assert statuses(context={"tenant": "acme"}, filters={"category": "support"})["b"] == "missing"
    assert set(statuses().values()) == {"forbidden"}
    with pytest.raises(ValueError, match="Unsupported filter"):
        statuses(context={"tenant": "acme"}, filters={"region": "eu"})
