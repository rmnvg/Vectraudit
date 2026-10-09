"""Structured requests preserve identity and cannot silently lose context or evidence gaps."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from vectraudit.contracts import matches_filter
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.evaluation.dataset_tools import (
    ColumnMap,
    dataset_rows,
    diff_datasets,
    import_rows,
    render_dataset,
)
from vectraudit.evaluation.runfile import RunFileError, check_run
from vectraudit.pipelines import (
    EvidenceCoverage,
    Hit,
    PipelineOptions,
    PipelineResult,
    SourceReference,
    execute_pipeline,
)
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.models import PipelineRequest, StageTrace
from vectraudit.pipelines.runner import failure_analysis


def row():
    return {
        "query_id": "q",
        "text": "What about the exception?",
        "relevant": {"doc": 1},
        "context": {"groups": ["support", "billing"], "tenant": "acme"},
        "filter": {"language": ["en", "hi"], "year": {"gte": 2024, "lt": 2027}},
        "history": [{"role": "user", "content": "How do I recover my account?"}],
    }


def test_legacy_hashes_and_empty_history_are_unchanged():
    raw = {"query_id": "q", "text": "Question", "context": {"tenant": "a"}}
    canonical = [
        {
            "query_id": "q",
            "vector": None,
            "text": "Question",
            "relevant": {},
            "filter": {},
            "tags": [],
            "context": {"tenant": "a"},
        }
    ]
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert parse_dataset(json.dumps(raw)).sha256 == expected
    assert parse_dataset(json.dumps({**raw, "history": []})).sha256 == expected


def test_structured_values_roundtrip_and_history_changes_identity():
    source = row()
    ds = parse_dataset(json.dumps(source))
    assert dataset_rows(ds) == [source]
    assert parse_dataset(render_dataset(dataset_rows(ds))).sha256 == ds.sha256
    assert import_rows([source], ColumnMap())[0]["history"] == source["history"]
    changed = copy.deepcopy(source)
    changed["history"][0]["content"] = "A different prior question"
    other = parse_dataset(json.dumps(changed))
    assert ds.sha256 != other.sha256
    assert "history" in json.dumps(diff_datasets(ds, other))


@pytest.mark.parametrize(
    "change",
    [
        {"context": {"groups": [["nested"]]}},
        {"context": {"groups": ["a"] * 257}},
        {"filter": {"year": {"gte": 2027, "lt": 2024}}},
        {"filter": {"year": {"gt": True}}},
        {"filter": {"year": {"native": "backend expression"}}},
        {"history": [{"role": "tool", "content": "not supported"}]},
        {"history": [{"role": "user", "content": "a" * 8193}]},
        {"history": [{"role": "user", "content": "a"}] * 65},
    ],
)
def test_invalid_or_unbounded_inputs_fail_before_execution(change):
    with pytest.raises(DatasetError, match="Line 1:"):
        parse_dataset(json.dumps({**row(), **change}))


def test_request_isolation_and_checkpoint_history_identity(tmp_path):
    ds = parse_dataset(json.dumps(row()))
    seen = []

    def pipeline(request):
        seen.append(request.model_dump())
        request.context["groups"].append("mutation")
        return PipelineResult(hits=[Hit(id="doc")])

    checkpoint = tmp_path / "run.sqlite"
    options = PipelineOptions(warmup=1, repeats=2)
    # Resuming custom adapters requires declared pipeline and corpus versions.
    pinned = {"config": {"pipeline_version": "v1"}, "corpus_version": "v1"}
    run = execute_pipeline(ds, pipeline, options, checkpoint=checkpoint, **pinned)
    assert run["status"] == "complete"
    assert len(seen) == 3
    assert all(value["context"]["groups"] == ["support", "billing"] for value in seen)
    assert all(value["history"] == row()["history"] for value in seen)
    assert "How do I recover" not in json.dumps(run)
    execute_pipeline(ds, pipeline, options, checkpoint=checkpoint, resume=True, **pinned)
    assert len(seen) == 3
    changed = row()
    changed["history"][0]["content"] = "different"
    with pytest.raises(ValueError, match=r"identity|match|changed"):
        execute_pipeline(
            parse_dataset(json.dumps(changed)),
            pipeline,
            options,
            checkpoint=checkpoint,
            resume=True,
            **pinned,
        )


def test_portable_membership_and_range_filters():
    assert matches_filter("en", ["en", "hi"])
    assert matches_filter(2025, {"gte": 2024, "lt": 2027})
    assert not matches_filter(2027, {"gte": 2024, "lt": 2027})
    assert not matches_filter(True, {"gte": 0})
    pipeline = hybrid_pipeline(
        [
            {"doc_id": "a", "text": "billing", "metadata": {"lang": "en", "year": 2025}},
            {"doc_id": "b", "text": "billing", "metadata": {"lang": "de", "year": 2025}},
        ]
    )
    result = pipeline(
        PipelineRequest(
            query_id="q", text="billing", filters={"lang": ["en", "hi"], "year": {"gte": 2024}}
        )
    )
    assert [hit.id for hit in result.hits] == ["a"]
    unsupported = execute_pipeline(parse_dataset(json.dumps(row())), pipeline)
    assert unsupported["status"] == "partial"
    assert unsupported["queries"][0]["error"]["code"] == "context_unsupported"


def test_source_identity_and_complete_observations_survive_repeats():
    ds = parse_dataset(json.dumps({"query_id": "q", "text": "q", "relevant": {"label-id": 1}}))
    hit = Hit(
        id="label-id",
        source=SourceReference(
            source_id="source", source_revision="v2", chunk_id="c", start=2, end=10
        ),
    )
    run = execute_pipeline(
        ds,
        lambda request: PipelineResult(
            hits=[hit], coverage=EvidenceCoverage(source_mapping="observed")
        ),
        PipelineOptions(warmup=0, repeats=2),
    )
    query = check_run(run)["queries"][0]
    assert query["metrics"]["recall"] == 1
    assert query["ids"] == ["label-id"]
    assert len(query["observations"]) == 2
    assert query["observations"][1]["hits"][0]["source"]["chunk_id"] == "c"


@pytest.mark.parametrize("state", ["partial", "unrecorded", "unsupported"])
def test_incomplete_ranking_never_passes_or_becomes_abstention(state):
    ds = parse_dataset('{"query_id":"q","text":"q","no_answer":true}')
    run = execute_pipeline(
        ds, lambda request: PipelineResult(coverage=EvidenceCoverage(ranked_output=state))
    )
    assert run["status"] == "partial"
    assert run["queries"][0]["error"]["code"] == "incomplete_evidence"
    assert run["queries"][0]["observations"][0]["coverage"]["ranked_output"] == state
    assert compare_runs(run, run)["verdict"] == "error"
    tampered = copy.deepcopy(run)
    tampered["queries"][0]["error"] = None
    with pytest.raises(RunFileError, match="incomplete ranked observations"):
        check_run(tampered)


def test_partial_stage_trace_does_not_claim_a_proven_loss():
    stages = [
        StageTrace(name="retrieve", kind="vector", hits=[Hit(id="doc")], latency_ms=1),
        StageTrace(name="filter", kind="filter", parents=["retrieve"], hits=[], latency_ms=1),
    ]
    result = PipelineResult(stages=stages, coverage=EvidenceCoverage(stage_trace="partial"))
    assert failure_analysis(result, {"doc": 1}) == []
    assert failure_analysis(PipelineResult(stages=stages), {"doc": 1})[0]["stage"] == "filter"
