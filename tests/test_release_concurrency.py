"""Release checks run with the approved execution protocol; guided suites reach 1,000."""

import json

import pytest

from tests.test_guided_workbench import save
from tests.test_release_configuration import CORPUS, LENIENT, ROWS, _client, post, record
from vectraudit.evaluation.runfile import load_run
from vectraudit.pipelines.release import (
    check_release,
    load_release,
    release_json,
    run_release,
    validate_release,
)

RELEASE = "retrieval/release/release.json"


@pytest.fixture
def registered(tmp_path):
    """A workbench whose registered hybrid candidate runs 3 queries at a time."""
    root = tmp_path / "config"
    root.mkdir()
    (root / "corpus.jsonl").write_text(CORPUS + "\n")
    (root / "queries.jsonl").write_text("\n".join(map(json.dumps, ROWS)))
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "queries.jsonl",
                "candidates": [
                    {
                        "name": "Search",
                        "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"},
                        "concurrency": 3,
                        "max_rps": 500,
                    }
                ],
            }
        )
    )
    client = _client(tmp_path / "workspace", manifest=manifest)
    yield client
    client.__exit__(None, None, None)


def _export(client, *, pipeline="builtin"):
    draft = save(client, rows=ROWS, corpus=CORPUS).json()
    baseline = record(client, draft, label="Baseline", pipeline=pipeline)
    candidate = record(client, draft, label="Candidate", pipeline=pipeline)
    policy = post(client, "policies", {"name": "Release", "policy": LENIENT})
    created = post(
        client,
        "releases",
        {
            "baseline": baseline["run_id"],
            "candidate": candidate["run_id"],
            "policy": policy["id"],
            "name": "Support search",
            "repeats": 1,
            "include_corpus": True,
        },
    )
    return candidate, created["files"]


def _checkout(path, files, **release_changes):
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    if release_changes:
        release = json.loads((path / RELEASE).read_text())
        release.update(release_changes)
        (path / RELEASE).write_text(json.dumps(release))
    return path / RELEASE


def test_serial_exports_stay_byte_compatible(tmp_path):
    client = _client(tmp_path / "workspace")
    try:
        candidate, files = _export(client)
    finally:
        client.__exit__(None, None, None)
    assert candidate["timing"]["concurrency"] == 1
    text = files[RELEASE]
    assert "concurrency" not in text
    assert "max_rps" not in text
    release = load_release(_checkout(tmp_path / "repo", files))
    assert (release.concurrency, release.max_rps) == (1, None)
    assert release_json(release) == text


def test_export_repeats_the_measured_protocol_and_ci_applies_the_approved_one(registered, tmp_path):
    candidate, files = _export(registered, pipeline="configured-0")
    assert candidate["timing"]["concurrency"] == 3
    exported = json.loads(files[RELEASE])
    assert (exported["concurrency"], exported["max_rps"]) == (3, 500)
    assert "3 queries in flight at most 500 calls/s" in files["retrieval/release/README.md"]

    approved = _checkout(tmp_path / "main", files)
    # The pull request proposes a different protocol; the approved one measures both sides.
    proposed = _checkout(tmp_path / "pr", files, concurrency=1, max_rps=None)
    base = run_release(approved)
    cand = run_release(proposed, approved_path=approved)
    assert base["timing"]["concurrency"] == cand["timing"]["concurrency"] == 3
    assert cand["timing"]["max_rps"] == 500
    result = check_release(approved, base, cand, proposed_path=proposed)
    assert result["verdict"] == "pass"
    assert result["latency"]["compared"] is True
    assert any("concurrency or rate limits" in w for w in result["warnings"])
    assert result["release"]["proposed_changes"]


def test_validate_warns_when_python_adapters_run_concurrently(registered, tmp_path):
    _, files = _export(registered, pipeline="configured-0")
    path = _checkout(tmp_path / "repo", files)
    release = json.loads(path.read_text())
    release["pipeline"] = {"name": "Search", "kind": "python", "target": "adapter:search"}
    path.write_text(json.dumps(release))
    messages = [i["message"] for i in validate_release(path)]
    assert any("thread-safe" in m for m in messages)


def test_run_list_returns_summaries_and_runs_load_on_demand(tmp_path):
    client = _client(tmp_path / "workspace")
    try:
        draft = save(client, rows=ROWS, corpus=CORPUS).json()
        run = record(client, draft, label="Baseline")
        listed = client.get("/api/guided/runs").json()
        assert [r["run_id"] for r in listed] == [run["run_id"]]
        assert "queries" not in listed[0]
        assert listed[0]["queries_omitted"] == len(ROWS)
        assert listed[0]["summary"] == run["summary"]
        full = client.get(f"/api/guided/runs/{run['run_id']}").json()
        assert full["queries"] == run["queries"]
        # A changed file is re-read rather than served from the summary cache.
        path = tmp_path / "workspace" / f"{run['run_id']}.run.json"
        changed = load_run(path)
        changed["label"] = "Relabelled baseline"
        path.write_text(json.dumps(changed))
        assert client.get("/api/guided/runs").json()[0]["label"] == "Relabelled baseline"
        assert client.get("/api/guided/runs/" + "0" * 32).status_code == 404
    finally:
        client.__exit__(None, None, None)


def test_run_size_limit_applies_to_listing_and_loading(tmp_path, monkeypatch):
    client = _client(tmp_path / "workspace")
    try:
        draft = save(client, rows=ROWS, corpus=CORPUS).json()
        run = record(client, draft, label="Baseline")
        monkeypatch.setattr("vectraudit.pipelines.guided.MAX_RUN_BYTES", 1000)
        assert client.get(f"/api/guided/runs/{run['run_id']}").status_code == 413
        assert client.get("/api/guided/runs").json() == []
    finally:
        client.__exit__(None, None, None)


def test_guided_benchmarks_accept_1000_questions(tmp_path):
    rows = [{"query_id": f"q{i}", "text": f"invoice {i}"} for i in range(1001)]
    client = _client(tmp_path / "workspace")
    try:
        assert save(client, rows=rows[:1000], corpus=CORPUS).status_code == 200
        over = save(client, rows=rows, corpus=CORPUS)
        assert over.status_code == 422
        assert "1000" in over.text
        content = "\n".join(
            json.dumps({"query_id": r["query_id"], "text": r["text"]}) for r in rows
        )
        imported = client.post("/api/guided/import", json={"content": content, "format": "jsonl"})
        assert imported.status_code == 422
        assert "1000" in imported.text
        accepted = client.post(
            "/api/guided/import",
            json={"content": content.rsplit("\n", 1)[0], "format": "jsonl"},
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["count"] == 1000
    finally:
        client.__exit__(None, None, None)
