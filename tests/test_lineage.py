"""Source-to-context lineage: the first stage that lost each required passage."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.lineage import (
    LineageError,
    check_lineage,
    ingestion_stage,
    read_manifest,
)
from vectraudit.pipelines import (
    ContextItem,
    Hit,
    ModelContext,
    PipelineOptions,
    PipelineRequest,
    PipelineResult,
    StageTrace,
    execute_pipeline,
)

SHA = "a" * 64


def source(**fields) -> dict:
    return {
        "source_id": "policy",
        "revision": "r2",
        "extracted_chars": 1000,
        "chunks": [
            {"chunk_id": "policy#0", "start": 0, "end": 500, "indexed_revision": "r2"},
            {"chunk_id": "policy#1", "start": 500, "end": 1000, "indexed_revision": "r2"},
        ],
        **fields,
    }


def write(tmp_path, *records, name="lineage.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def dataset(*spans: tuple[str, int, int], query="refund") -> object:
    row = {
        "query_id": query,
        "text": "refund",
        "relevant_spans": [{"doc_id": d, "start": s, "end": e, "grade": 1} for d, s, e in spans],
    }
    return parse_dataset(json.dumps(row))


def loss(tmp_path, span, **fields):
    sources = read_manifest(write(tmp_path, source(**fields)))
    return check_lineage(dataset(("policy", *span)), sources)["rows"][0]


@pytest.mark.parametrize(
    ("span", "fields", "expected"),
    [
        ((100, 200), {}, None),
        ((100, 200), {"extracted_chars": 0, "chunks": []}, "empty_extraction"),
        ((1100, 1200), {}, "beyond_extraction"),
        ((450, 550), {}, "split_across_chunks"),
        (
            (100, 200),
            {"chunks": [{"chunk_id": "policy#0", "start": 300, "end": 900}]},
            "not_chunked",
        ),
        (
            (100, 200),
            {"chunks": [{"chunk_id": "policy#0", "start": 0, "end": 500, "indexed": False}]},
            "not_indexed",
        ),
        (
            (100, 200),
            {
                "chunks": [
                    {"chunk_id": "policy#0", "start": 0, "end": 500, "indexed_revision": "r1"}
                ]
            },
            "stale_index",
        ),
    ],
)
def test_each_ingestion_stage_is_named_as_the_first_loss(tmp_path, span, fields, expected):
    assert loss(tmp_path, span, **fields)["loss"] == expected


def test_a_source_missing_from_the_export_is_a_loss_not_a_pass(tmp_path):
    sources = read_manifest(write(tmp_path, source()))
    row = check_lineage(dataset(("other", 0, 10)), sources)["rows"][0]
    assert row["loss"] == "source_missing"


def test_unverified_revisions_are_reported_not_assumed():
    from vectraudit.evaluation.lineage import SourceRecord

    record = SourceRecord.model_validate(source(revision=None))
    assert ingestion_stage(record, 100, 200)["revision_unverified"] is True


def test_a_chunk_boundary_defect_is_found_and_its_repair_recovers_the_passage(tmp_path):
    """A reproducible finding, then a verified repair."""
    passage = ("policy", 450, 550)  # the exception straddles the boundary at 500
    broken = check_lineage(dataset(passage), read_manifest(write(tmp_path, source())))
    assert broken["first_loss"] == {"split_across_chunks": 1}
    assert broken["rows"][0]["chunks"] == ["policy#0", "policy#1"]
    assert any("embedding model" in line for line in broken["advice"])
    overlapping = source(
        chunks=[
            {"chunk_id": "policy#0", "start": 0, "end": 600, "indexed_revision": "r2"},
            {"chunk_id": "policy#1", "start": 400, "end": 1000, "indexed_revision": "r2"},
        ]
    )
    repaired = check_lineage(
        dataset(passage), read_manifest(write(tmp_path, overlapping, name="b"))
    )
    assert repaired["first_loss"] == {}
    assert repaired["rows"][0]["chunks"] == ["policy#0", "policy#1"]


class Assistant:
    """Retrieves policy#1 then reranks; ``mode`` decides where the passage is lost."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        first = ["policy#1", "faq#0"] if self.mode != "unretrieved" else ["faq#0"]
        delivered = [h for h in first if not (self.mode == "dropped" and h == "policy#1")]
        stages = [
            StageTrace(name="search", kind="vector", hits=[Hit(id=i) for i in first], latency_ms=0),
            StageTrace(
                name="rerank",
                kind="rerank",
                parents=["search"],
                hits=[Hit(id=i) for i in delivered],
                latency_ms=0,
            ),
        ]
        items = [
            ContextItem(id=i, truncated=(self.mode == "trimmed" and i == "policy#1"))
            for i in delivered
            if not (self.mode == "omitted" and i == "policy#1")
        ]
        return PipelineResult(
            hits=[Hit(id=i) for i in delivered], stages=stages, context=ModelContext(items=items)
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("ok", None),
        ("unretrieved", "not_retrieved"),
        ("dropped", "lost_downstream"),
        ("trimmed", "lost_in_context"),
        ("omitted", "lost_in_context"),
    ],
)
def test_a_run_continues_the_path_to_the_model(tmp_path, mode, expected):
    data = dataset(("policy", 600, 700))
    run = execute_pipeline(data, Assistant(mode), PipelineOptions(k=2, warmup=0))
    report = check_lineage(data, read_manifest(write(tmp_path, source())), run)
    row = report["rows"][0]
    assert row["loss"] == expected
    if mode == "dropped":
        assert row["last_stage"] == "search"
    assert report["ingestion_losses"] == 0


def test_source_ids_in_results_are_matched_by_source(tmp_path):
    data = dataset(("policy", 600, 700))
    run = {"run_id": "r", "queries": [{"query_id": "refund", "ids": ["policy"], "error": None}]}
    row = check_lineage(data, read_manifest(write(tmp_path, source())), run, ids="source")["rows"][
        0
    ]
    assert (row["loss"], row["delivered"], row["unknown"]) == (None, ["policy"], "model context")


def test_corpus_findings_stale_chunks_and_duplicates_crowding_results(tmp_path):
    records = [
        source(
            chunks=[
                {
                    "chunk_id": "policy#0",
                    "start": 0,
                    "end": 500,
                    "indexed_revision": "r2",
                    "text_sha256": SHA,
                },
                {"chunk_id": "policy#0-old", "start": 0, "end": 500, "indexed_revision": "r1"},
            ]
        ),
        {
            "source_id": "copy",
            "extracted_chars": 500,
            "chunks": [{"chunk_id": "copy#0", "start": 0, "end": 500, "text_sha256": SHA}],
        },
        {"source_id": "scan", "extracted_chars": 0, "chunks": []},
    ]
    sources = read_manifest(write(tmp_path, *records))
    run = {
        "run_id": "r",
        "queries": [
            {"query_id": "refund", "ids": ["policy#0", "copy#0"], "error": None},
        ],
    }
    corpus = check_lineage(dataset(("policy", 0, 10)), sources, run)["corpus"]
    assert corpus["empty_extraction"] == ["scan"]
    assert corpus["stale_chunks_indexed"] == [
        {"source_id": "policy", "revision": "r2", "chunks": ["policy#0-old"]}
    ]
    assert corpus["duplicate_chunks"] == [["copy#0", "policy#0"]]
    assert corpus["duplicates_in_results"] == [
        {"query_id": "refund", "slots_taken_by_duplicates": 1}
    ]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([source(), source()], "appears twice"),
        (
            [
                source(),
                {"source_id": "b", "chunks": [{"chunk_id": "policy#0", "start": 0, "end": 1}]},
            ],
            "also belongs to policy",
        ),
        ([source(extracted_chars=100)], "past the extracted text"),
        ([source(chunks=[{"chunk_id": "c", "start": 5, "end": 5}])], "start must be before end"),
        ([source(notes="private")], "notes"),
    ],
)
def test_invalid_manifests_are_refused_with_a_reason(tmp_path, records, message):
    with pytest.raises(LineageError, match=message):
        read_manifest(write(tmp_path, *records))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(LineageError, match="lists no sources"):
        read_manifest(empty)


def test_cli_reports_and_can_fail_ci_on_ingestion_losses(tmp_path):
    manifest = write(tmp_path, source())
    data = tmp_path / "questions.jsonl"
    data.write_text(
        json.dumps(
            {
                "query_id": "refund",
                "text": "refund",
                "relevant_spans": [{"doc_id": "policy", "start": 450, "end": 550, "grade": 1}],
            }
        )
    )
    shown = CliRunner().invoke(
        app,
        ["workbench", "lineage", str(manifest), str(data), "--json", str(tmp_path / "out.json")],
    )
    assert shown.exit_code == 0, shown.output
    assert "1 first lost at split across chunks" in shown.output
    assert json.loads((tmp_path / "out.json").read_text())["ingestion_losses"] == 1
    gated = CliRunner().invoke(
        app, ["workbench", "lineage", str(manifest), str(data), "--fail-on-ingestion"]
    )
    assert gated.exit_code == 3
    bad = CliRunner().invoke(
        app, ["workbench", "lineage", str(manifest), str(data), "--ids", "rows"]
    )
    assert bad.exit_code == 2
