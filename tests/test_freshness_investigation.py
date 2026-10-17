"""Investigations flag content a change log retired, and stop offering retrieval fixes for
labelled content that is not indexed yet."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.test_investigation_web import CORPUS, wait
from tests.test_lineage_investigation import ROWS
from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.freshness import Change, question_freshness
from vectraudit.evaluation.lineage import LineageError, SourceRecord
from vectraudit.pipelines.investigate import investigate
from vectraudit.pipelines.manifest import Fix
from vectraudit.pipelines.web import create_workbench

DEPTH = Fix(name="deeper search", addresses=["not_observed"], params={"depth": 100})
UPDATE = Change(
    source_id="policy", change="update", to_revision="r9", changed_at="2026-10-18T09:00:00Z"
)


def sources(*chunks: tuple[str, str, bool]) -> dict[str, SourceRecord]:
    return {
        "policy": SourceRecord.model_validate(
            {
                "source_id": "policy",
                "revision": "r9",
                "chunks": [
                    {"chunk_id": c, "start": 0, "end": 10, "indexed_revision": r, "indexed": i}
                    for c, r, i in chunks
                ],
            }
        )
    }


def run(ids, context=(), failures=(), at="2026-10-18T12:00:00+00:00"):
    return {
        "run_id": "r",
        "created_at": at,
        "queries": [
            {
                "query_id": "refund",
                "error": None,
                "ids": list(ids),
                "forbidden_ids": [],
                "model_context": {
                    "coverage": "observed",
                    "items": [{"id": i, "truncated": False} for i in context],
                },
                "evidence": {
                    "stages": [{"name": "search", "kind": "vector", "hits": []}],
                    "failures": [
                        {"doc_id": d, "stage": None, "kind": "not_observed", "reason": "x"}
                        for d in failures
                    ],
                },
            }
        ],
    }


def dataset(relevant, tags=()):
    return parse_dataset(
        json.dumps(
            {"query_id": "refund", "text": "refund", "relevant": relevant, "tags": list(tags)}
        )
    )


def test_stale_content_in_context_and_conflicting_versions_become_losses():
    index = sources(("policy#0@r8", "r8", True), ("policy#0@r9", "r9", True))
    recorded = run(["policy#0@r8", "policy#0@r9"], context=["policy#0@r8"])
    fresh = question_freshness(
        [UPDATE], index, recorded, "refund", dataset=dataset({"policy#0@r9": 1})
    )
    assert fresh["as_of"] == "2026-10-18T12:00:00+00:00"  # judged when the run was recorded
    assert [r["status"] for r in fresh["changes"]] == ["lingering"]
    result = investigate(recorded, "refund", [DEPTH], freshness=fresh)
    stale = next(x for x in result["losses"] if x["class"] == "stale_content")
    assert (stale["doc_id"], stale["detail"]) == (
        "policy#0@r8",
        "update of policy; reached the context",
    )
    conflict = next(x for x in result["losses"] if x["class"] == "conflicting_versions")
    assert conflict["detail"] == "revisions r8, r9"
    assert any("Re-index or purge" in c for c in result["cautions"])
    assert result["unaddressed"] == ["conflicting_versions", "stale_content"]
    assert result["freshness"] == fresh


def test_a_labelled_document_waiting_to_be_indexed_gets_no_retrieval_fix():
    index = sources(("policy#0@r8", "r8", True))
    recorded = run(["faq"], failures=["policy"])
    late = question_freshness([UPDATE], index, recorded, "refund", dataset=dataset({"policy": 1}))
    assert late["unindexed_labels"] == ["policy"]
    result = investigate(recorded, "refund", [DEPTH], freshness=late)
    assert result["fixes"] == []
    assert any("new content is not indexed yet" in h for h in result["hypotheses"])
    early = question_freshness(
        [UPDATE],
        index,
        run(["faq"], failures=["policy"], at="2026-10-18T09:20:00+00:00"),
        "refund",
        dataset=dataset({"policy": 1}),
    )
    assert early["changes"][0]["status"] == "pending"
    pending = investigate(recorded, "refund", [DEPTH], freshness=early)
    assert any("inside its propagation window" in u for u in pending["uncertainty"])


def test_historical_questions_may_reach_old_content():
    index = sources(("policy#0@r8", "r8", True))
    recorded = run(["policy#0@r8"], context=["policy#0@r8"])
    fresh = question_freshness(
        [UPDATE], index, recorded, "refund", dataset=dataset({"policy": 1}, ["historical"])
    )
    assert (fresh["stale"], fresh["historical"]) == ([], True)
    result = investigate(recorded, "refund", freshness=fresh)
    assert not any(x["class"] == "stale_content" for x in result["losses"])
    assert any("tagged historical" in u for u in result["uncertainty"])


def test_source_ids_find_removed_sources_but_say_revisions_are_unknown():
    deleted = Change(source_id="policy", change="delete", changed_at="2026-10-18T09:00:00Z")
    index = sources(("policy#0", "r8", True))
    fresh = question_freshness([deleted], index, run(["policy"]), "refund", ids="source")
    assert fresh["stale"][0]["chunk_id"] == "policy"
    updated = question_freshness([UPDATE], index, run(["policy"]), "refund", ids="source")
    assert updated["stale"] == []
    result = investigate(run(["policy"]), "refund", freshness=updated)
    assert any("cannot be told apart" in u for u in result["uncertainty"])


def write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_the_workbench_shows_stale_content_in_the_investigation(tmp_path):
    lineage = write(
        tmp_path,
        "lineage.jsonl",
        [
            {"source_id": s, "chunks": [{"chunk_id": f"{s}#0", "start": 0, "end": 10}]}
            for s in ("a", "b")
        ],
    )
    changes = write(
        tmp_path,
        "changes.jsonl",
        [
            {
                "source_id": "a",
                "change": "replace",
                "replaced_by": "a-2026",
                "changed_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    app_ = create_workbench(tmp_path, lineage=lineage, lineage_ids="source", changes=changes)
    with TestClient(app_) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        draft = client.post(
            "/api/guided/drafts",
            json={"name": "Refunds", "dataset": "\n".join(map(json.dumps, ROWS)), "corpus": CORPUS},
        ).json()
        started = client.post(
            "/api/guided/jobs",
            json={
                "draft_id": draft["id"],
                "revision": draft["revision"],
                "corpus": CORPUS,
                "k": 1,
                "depth": 10,
            },
        )
        found = client.post(
            "/api/guided/investigate",
            json={"run": wait(client, started.json()["id"])["run_id"], "query_id": "refund"},
        ).json()
    assert found["freshness"]["changes"][0]["status"] == "lingering"
    stale = next(x for x in found["losses"] if x["class"] == "stale_content")
    assert (stale["doc_id"], stale["detail"]) == ("a", "replace of a; reached the results")


def test_a_change_log_without_lineage_is_refused(tmp_path):
    changes = write(
        tmp_path,
        "changes.jsonl",
        [{"source_id": "b", "change": "delete", "changed_at": "2026-01-01T00:00:00Z"}],
    )
    with pytest.raises(LineageError, match="needs the lineage export"):
        create_workbench(tmp_path, changes=changes)


def test_cli_investigate_reads_the_change_log(tmp_path):
    from tests.test_try_fix import setup
    from vectraudit.pipelines.manifest import load_manifest, open_adapter
    from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline

    manifest = setup(tmp_path, module="freshness_cli_app")
    spec = load_manifest(manifest)
    recorded = execute_pipeline(
        parse_dataset((tmp_path / "queries.jsonl").read_text()),
        open_adapter(spec.candidates[0].adapter, tmp_path),
        PipelineOptions(k=2, warmup=0, label="current"),
    )
    (tmp_path / "run.json").write_text(json.dumps(recorded))
    delivered = next(q for q in recorded["queries"] if q["query_id"] == "target")["ids"][0]
    lineage = write(
        tmp_path,
        "lineage.jsonl",
        [{"source_id": delivered, "chunks": [{"chunk_id": "x#0", "start": 0, "end": 1}]}],
    )
    changes = write(
        tmp_path,
        "changes.jsonl",
        [{"source_id": delivered, "change": "delete", "changed_at": "2026-01-01T00:00:00Z"}],
    )
    base = ["workbench", "investigate", str(manifest), str(tmp_path / "run.json"), "target"]
    refused = CliRunner().invoke(app, [*base, "--changes", str(changes)])
    assert refused.exit_code == 2
    assert "needs the --lineage export" in refused.output
    shown = CliRunner().invoke(
        app,
        [*base, "--lineage", str(lineage), "--lineage-ids", "source", "--changes", str(changes)],
    )
    assert shown.exit_code == 0, shown.output
    assert f"lost {delivered}: stale content (delete of {delivered}; reached the results)" in (
        shown.output
    )
