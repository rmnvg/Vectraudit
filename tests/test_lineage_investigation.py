"""Investigations name ingestion losses from a lineage export, and stop offering retrieval
fixes that cannot recover text that was never indexed intact."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.test_investigation_web import CORPUS, wait
from vectraudit.cli import app
from vectraudit.evaluation.lineage import LineageError
from vectraudit.pipelines.investigate import investigate
from vectraudit.pipelines.manifest import Fix
from vectraudit.pipelines.web import create_workbench

DEPTH = Fix(name="deeper search", addresses=["not_observed"], params={"depth": 100})
RUN = {
    "run_id": "r",
    "queries": [
        {
            "query_id": "q",
            "error": None,
            "ids": ["other"],
            "forbidden_ids": [],
            "evidence": {
                "stages": [{"name": "search", "kind": "vector", "hits": [{"id": "other"}]}],
                "failures": [
                    {"doc_id": "policy", "stage": None, "kind": "not_observed", "reason": "x"}
                ],
            },
        }
    ],
}
SPLIT = {
    "query_id": "q",
    "source_id": "policy",
    "start": 450,
    "end": 550,
    "loss": "split_across_chunks",
    "chunks": ["policy#0", "policy#1"],
}


def test_an_ingestion_loss_explains_a_missing_document_and_withholds_retrieval_fixes():
    plain = investigate(RUN, "q", [DEPTH])
    assert [f["name"] for f in plain["fixes"]] == ["deeper search"]
    with_lineage = investigate(RUN, "q", [DEPTH], lineage=[SPLIT])
    ingestion = next(x for x in with_lineage["losses"] if x["class"] == "ingestion")
    assert (ingestion["stage"], ingestion["detail"]) == (
        "split_across_chunks",
        "characters 450 to 550",
    )
    assert with_lineage["fixes"] == []  # deeper search cannot fetch a passage no chunk holds
    assert with_lineage["unaddressed"] == ["ingestion", "not_observed"]
    assert any("chunk boundary" in h for h in with_lineage["hypotheses"])
    assert any("Fix extraction, chunking or indexing first" in c for c in with_lineage["cautions"])
    intact = investigate(RUN, "q", [DEPTH], lineage=[{**SPLIT, "loss": None}])
    assert [f["name"] for f in intact["fixes"]] == ["deeper search"]  # indexed intact: try it


ROWS = [
    {
        "query_id": "refund",
        "text": "annual refund",
        "relevant": {"b": 1},
        "relevant_spans": [{"doc_id": "b", "start": 4, "end": 30, "grade": 1}],
    },
    {"query_id": "shipping", "text": "shipping times", "relevant": {"c": 1}},
]


def lineage_file(tmp_path, chunks):
    path = tmp_path / "lineage.jsonl"
    path.write_text(json.dumps({"source_id": "b", "extracted_chars": 80, "chunks": chunks}))
    return path


def test_the_workbench_shows_lineage_in_the_investigation(tmp_path):
    split = lineage_file(
        tmp_path,
        [{"chunk_id": "b#0", "start": 0, "end": 10}, {"chunk_id": "b#1", "start": 10, "end": 80}],
    )
    with TestClient(create_workbench(tmp_path, lineage=split, lineage_ids="source")) as client:
        client.headers["X-Workbench-Token"] = client.get("/api/guided/session").json()["token"]
        draft = client.post(
            "/api/guided/drafts",
            json={
                "name": "Refunds",
                "dataset": "\n".join(map(json.dumps, ROWS)),
                "corpus": CORPUS,
            },
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
        run = wait(client, started.json()["id"])
        found = client.post(
            "/api/guided/investigate", json={"run": run["run_id"], "query_id": "refund"}
        ).json()
    (row,) = found["lineage"]
    assert (row["source_id"], row["loss"]) == ("b", "split_across_chunks")
    assert any(x["class"] == "ingestion" for x in found["losses"])
    # b was still retrieved as a document; the reranker fix for its truncation stays offered.
    assert "use the phrase reranker" in [f["name"] for f in found["fixes"]]


def test_a_broken_lineage_export_stops_the_workbench_at_start(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}")
    with pytest.raises(LineageError):
        create_workbench(tmp_path, lineage=bad)
    with pytest.raises(LineageError, match="chunk or source"):
        create_workbench(tmp_path, lineage=lineage_file(tmp_path, []), lineage_ids="rows")


def test_cli_investigate_reads_the_lineage_export(tmp_path):
    from tests.test_try_fix import setup

    manifest = setup(tmp_path, module="lineage_cli_app")
    rows = [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text().splitlines()]
    rows[0]["relevant_spans"] = [{"doc_id": "rel", "start": 0, "end": 50, "grade": 1}]
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "lineage.jsonl").write_text(
        json.dumps({"source_id": "rel", "extracted_chars": 0, "chunks": []})
    )
    from vectraudit.evaluation.dataset import parse_dataset
    from vectraudit.pipelines.manifest import load_manifest, open_adapter
    from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline

    spec = load_manifest(manifest)
    run = execute_pipeline(
        parse_dataset((tmp_path / "queries.jsonl").read_text()),
        open_adapter(spec.candidates[0].adapter, tmp_path),
        PipelineOptions(k=2, warmup=0, label="current"),
    )
    (tmp_path / "run.json").write_text(json.dumps(run))
    shown = CliRunner().invoke(
        app,
        [
            "workbench",
            "investigate",
            str(manifest),
            str(tmp_path / "run.json"),
            "target",
            "--lineage",
            str(tmp_path / "lineage.jsonl"),
            "--lineage-ids",
            "source",
        ],
    )
    assert shown.exit_code == 0, shown.output
    assert "lost rel: ingestion at empty extraction" in shown.output
    assert "Fixes to try" not in shown.output  # depth cannot fetch text extraction never produced
