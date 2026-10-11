"""Model-context evidence: retrieved, delivered, and what actually reached the model."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from vectraudit.evaluation.dataset import parse_dataset
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
    execute_pipeline,
)

DATA = parse_dataset(
    "\n".join(
        json.dumps(row)
        for row in [
            {"query_id": "refund", "text": "refund an annual plan", "relevant": {"policy": 1}},
            {"query_id": "pw", "text": "reset password", "relevant": {"guide": 1, "faq": 1}},
        ]
    )
)


class Packed:
    """Delivers both documents; ``context`` decides what the model receives."""

    def __init__(self, context: dict[str, ModelContext | None]) -> None:
        self.context = context

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        ids = {"refund": ["policy", "other"], "pw": ["guide", "faq"]}[request.query_id]
        return PipelineResult(
            hits=[Hit(id=i) for i in ids], context=self.context.get(request.query_id)
        )


def run(context: dict[str, ModelContext | None]) -> dict:
    return execute_pipeline(DATA, Packed(context), PipelineOptions(k=2, warmup=0))


def observed(*items: ContextItem) -> ModelContext:
    return ModelContext(items=list(items))


def by_id(record: dict) -> dict:
    return {q["query_id"]: q for q in record["queries"]}


def test_contract_rejects_incoherent_context():
    with pytest.raises(ValidationError, match="cannot list documents"):
        ModelContext(coverage="unrecorded", items=[ContextItem(id="a")])
    with pytest.raises(ValidationError, match="twice"):
        ModelContext(items=[ContextItem(id="a"), ContextItem(id="a")])


def test_omitted_and_truncated_documents_are_separate_from_recall():
    record = run(
        {
            "refund": observed(ContextItem(id="other", truncated=False)),
            "pw": observed(ContextItem(id="guide", truncated=True), ContextItem(id="faq")),
        }
    )
    check_run(record)
    queries = by_id(record)
    assert queries["refund"]["metrics"]["recall"] == 1.0  # retrieval found it…
    assert queries["refund"]["evidence"]["context"]["losses"] == [
        {
            "doc_id": "policy",
            "kind": "omitted",
            "reason": "delivered by retrieval; absent from the model context",
        }
    ]  # …the model never received it
    pw = queries["pw"]["evidence"]["context"]
    assert [(loss["doc_id"], loss["kind"]) for loss in pw["losses"]] == [("guide", "truncated")]
    assert pw["reached"] == ["faq"]  # completeness unknown (truncated None) still counts
    assert queries["pw"]["model_context"]["items"] == [
        {"id": "guide", "truncated": True},
        {"id": "faq"},
    ]


def test_partial_or_missing_context_proves_nothing_about_absence():
    record = run({"refund": ModelContext(coverage="partial"), "pw": None})
    queries = by_id(record)
    assert queries["refund"]["evidence"]["context"]["losses"] == []
    assert queries["refund"]["evidence"]["context"]["unknown"] == ["policy"]
    assert "model_context" not in queries["pw"]
    assert "context" not in queries["pw"]["evidence"]


def test_triage_names_a_new_context_loss_as_a_regression():
    full = observed(ContextItem(id="policy", truncated=False))
    baseline = run({"refund": full, "pw": observed(ContextItem(id="guide", truncated=True))})
    candidate = run(
        {"refund": observed(), "pw": observed(ContextItem(id="guide"), ContextItem(id="faq"))}
    )
    items = {item["query_id"]: item for item in triage(baseline, candidate)["items"]}
    [lost] = items["refund"]["findings"]
    assert (lost["kind"], lost["severity"], lost["doc_ids"]) == (
        "context_lost",
        "regression",
        ["policy"],
    )
    assert "as they did in the baseline" in lost["message"]
    kinds = {f["kind"] for f in items["pw"]["findings"]}
    assert "context_recovered" in kinds  # guide now reaches the model in full


def test_unknown_baseline_context_is_not_called_a_regression():
    baseline = run({})
    candidate = run({"refund": observed()})
    [finding] = next(i for i in triage(baseline, candidate)["items"] if i["query_id"] == "refund")[
        "findings"
    ]
    assert (finding["kind"], finding["severity"]) == ("context_loss", "info")


def test_context_stays_local_and_is_validated():
    record = run({"refund": observed(ContextItem(id="policy"))})
    shared = project_run(record)
    assert all("model_context" not in q for q in shared["queries"])
    broken = json.loads(json.dumps(record))
    broken["queries"][0]["model_context"]["items"].append({"id": "policy"})
    with pytest.raises(RunFileError, match="invalid model context"):
        check_run(broken)
