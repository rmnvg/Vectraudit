"""Permission boundaries upstream of the results: where a forbidden document travelled."""

from __future__ import annotations

import copy
import json

import pytest

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.exceptions import acceptance
from vectraudit.evaluation.policy import PolicyError, parse_policy, policy_sha256
from vectraudit.evaluation.runfile import RunFileError, check_run
from vectraudit.evaluation.triage import triage
from vectraudit.integrations.sharing import project_run
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
from vectraudit.pipelines.investigate import investigate
from vectraudit.pipelines.models import SourceReference

DATA = parse_dataset(
    json.dumps(
        {
            "query_id": "invoice",
            "text": "invoice export",
            "relevant": {"guide": 1},
            "forbidden": ["private"],
            "context": {"tenant": "acme"},
        }
    )
)
METRIC = {"metrics": {"ndcg": {"max_drop": 0.5}}}


def stage(name, kind, ids, parents=()):
    return StageTrace(
        name=name, kind=kind, parents=list(parents), hits=[Hit(id=i) for i in ids], latency_ms=0
    )


class Assistant:
    """Search finds the other tenant's guide too; ``order`` decides when it is filtered."""

    def __init__(self, order="filter_first", context=("guide",), record_context=True):
        self.order, self.context, self.record_context = order, context, record_context

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        found = ["guide", "private"]
        if self.order == "filter_first":
            stages = [
                stage("search", "vector", found),
                stage("acl", "filter", ["guide"], ["search"]),
                stage("rerank", "rerank", ["guide"], ["acl"]),
            ]
        else:  # the reranker sees every candidate before access control runs
            stages = [
                stage("search", "vector", found),
                stage("rerank", "rerank", found, ["search"]),
                stage("acl", "filter", ["guide"], ["rerank"]),
            ]
        context = (
            ModelContext(items=[ContextItem(id=i) for i in self.context])
            if self.record_context
            else None
        )
        return PipelineResult(hits=[Hit(id="guide")], stages=stages, context=context)


def run(pipeline, repeats=1):
    return execute_pipeline(DATA, pipeline, PipelineOptions(k=2, warmup=0, repeats=repeats))


def test_the_path_records_every_stage_a_forbidden_document_reached():
    (query,) = run(Assistant())["queries"]
    (doc,) = query["permission_path"]["documents"]
    assert doc["doc_id"] == "private"
    assert [(s["name"], s["as"]) for s in doc["stages"]] == [("search", "output"), ("acl", "input")]
    assert (doc["kinds"], doc["results"], doc["context"]) == (["filter", "vector"], False, False)
    assert query["forbidden_ids"] == []


def test_the_policy_boundary_catches_a_reranker_that_sees_forbidden_documents():
    """The planted defect: access control moved after the reranker. Results look clean."""
    boundary = parse_policy({**METRIC, "forbidden_stages": ["rerank"]})
    safe = run(Assistant())
    assert compare_runs(safe, safe, boundary)["verdict"] == "pass"
    leaky = run(Assistant(order="rerank_first"))
    assert leaky["queries"][0]["forbidden_ids"] == []
    assert compare_runs(leaky, leaky, parse_policy(METRIC))["verdict"] == "pass"
    result = compare_runs(safe, leaky, boundary)
    assert result["verdict"] == "fail"
    failed = [r for r in result["reasons"] if r["verdict"] == "fail"]
    assert [r["rule"] for r in failed] == ["safety.forbidden:invoice"]
    assert "reached a rerank stage" in failed[0]["message"]


def test_a_forbidden_document_in_the_model_context_fails_even_when_not_delivered():
    leaky = run(Assistant(context=("guide", "private")))
    result = compare_runs(run(Assistant()), leaky, parse_policy(METRIC))
    assert result["verdict"] == "fail"
    assert any("reached the model context" in r["message"] for r in result["reasons"])
    exceptions = parse_policy({**METRIC, "exceptions": {"allow": ["metric"]}})
    (who,) = acceptance(compare_runs(leaky, leaky, exceptions), exceptions)
    assert (who["role"], who["why"]) == (None, "never")  # a safety failure, never waived
    (item,) = triage(run(Assistant()), leaky)["items"]
    (finding,) = [f for f in item["findings"] if f["kind"] == "permission"]
    assert finding["context_doc_ids"] == ["private"]
    loss = investigate(leaky, "invoice")["losses"]
    assert {"class": "permission", "doc_id": "private", "detail": "reached the model context"} in (
        loss
    )


def test_unrecorded_context_is_unknown_and_a_boundary_needs_a_trace():
    (query,) = run(Assistant(record_context=False))["queries"]
    assert query["permission_path"]["documents"][0]["context"] is None

    def untraced(request: PipelineRequest) -> PipelineResult:
        return PipelineResult(hits=[Hit(id="guide")])

    bare = run(untraced)
    assert bare["queries"][0]["permission_path"] == {
        "stages_recorded": False,
        "context_recorded": False,
        "documents": [],
    }
    assert compare_runs(bare, bare, parse_policy(METRIC))["verdict"] == "pass"
    boundary = parse_policy({**METRIC, "forbidden_stages": ["rerank"]})
    assert compare_runs(bare, bare, boundary)["verdict"] == "not_comparable"


def test_a_forbidden_source_is_caught_through_its_chunks():
    """Labels name the source document; results return chunk IDs with a source reference."""

    def chunked(request: PipelineRequest) -> PipelineResult:
        chunk = Hit(id="private#3", source=SourceReference(source_id="private"))
        return PipelineResult(hits=[Hit(id="guide"), chunk])

    (query,) = run(chunked, repeats=2)["queries"]
    assert query["forbidden_ids"] == ["private"]
    assert [r["forbidden_ids"] for r in query["repeat_safety"]] == [["private"], ["private"]]
    assert query["permission_path"]["documents"][0]["results"] is True


def test_the_hosted_projection_reaches_the_local_verdict_without_stage_names():
    boundary = parse_policy({**METRIC, "forbidden_stages": ["rerank"]})
    safe, leaky = run(Assistant()), run(Assistant(order="rerank_first"))
    shared = project_run(leaky)
    path = shared["queries"][0]["permission_path"]
    assert path["documents"] == [
        {
            "doc_id": "private",
            "results": False,
            "context": False,
            "kinds": ["filter", "rerank", "vector"],
        }
    ]
    assert "search" not in json.dumps(path)
    assert compare_runs(project_run(safe), shared, boundary)["verdict"] == "fail"


def test_malformed_paths_and_policies_are_refused():
    broken = copy.deepcopy(run(Assistant()))
    broken["queries"][0]["permission_path"]["documents"][0]["kinds"] = "rerank"
    with pytest.raises(RunFileError, match="permission_path"):
        check_run(broken)
    with pytest.raises(PolicyError, match="forbidden_stages"):
        parse_policy({**METRIC, "forbidden_stages": ["llm"]})
    assert policy_sha256(parse_policy({**METRIC, "forbidden_stages": ["rerank"]})) != (
        policy_sha256(parse_policy(METRIC))
    )
