"""Critical context: required evidence must reach the model on every execution."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.exceptions import rule_kind
from vectraudit.evaluation.policy import PolicyError, parse_policy, policy_sha256
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

ROOT = Path(__file__).resolve().parents[1]
DATA = parse_dataset(
    json.dumps(
        {
            "query_id": "recovery",
            "text": "recover my account",
            "relevant": {"guide": 1, "exception": 1},
            "required_evidence": [["guide", "exception"]],
        }
    )
)
POLICY = parse_policy({"critical_context": [{"query_id": "recovery"}], "min_queries": 1})


class Packing:
    """Delivers both documents; ``trim`` decides on which executions the exception is cut."""

    def __init__(self, trim: set[int] | None = None, context: bool = True) -> None:
        self.trim, self.context, self.calls = trim or set(), context, 0

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        self.calls += 1
        items = [
            ContextItem(id="guide", truncated=False),
            ContextItem(id="exception", truncated=self.calls in self.trim),
        ]
        return PipelineResult(
            hits=[Hit(id="guide"), Hit(id="exception")],
            context=ModelContext(items=items) if self.context else None,
        )


def run(pipeline: Packing) -> dict:
    return execute_pipeline(DATA, pipeline, PipelineOptions(k=2, warmup=0, repeats=3))


def test_every_execution_must_deliver_the_evidence_to_the_model():
    baseline = run(Packing())
    assert compare_runs(baseline, run(Packing()), POLICY)["verdict"] == "pass"
    intermittent = compare_runs(baseline, run(Packing(trim={2})), POLICY)
    assert intermittent["verdict"] == "fail"
    [result] = intermittent["critical_context"]
    assert result["context"] == ["sufficient", "insufficient", "sufficient"]
    assert "on 1 of 3 executions" in result["detail"]
    [reason] = [r for r in intermittent["reasons"] if r["verdict"] == "fail"]
    assert reason["rule"] == "critical_context:recovery"
    assert rule_kind(reason["rule"]) == "critical"  # exceptions treat it as a critical rule


UNGROUPED = parse_dataset(
    json.dumps({"query_id": "recovery", "text": "x", "relevant": {"guide": 1}})
)


def ungrouped() -> dict:
    return execute_pipeline(UNGROUPED, Packing(), PipelineOptions(k=2, warmup=0, repeats=3))


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        (lambda: run(Packing()), lambda: run(Packing(context=False)), "unknown on 3 of 3"),
        (ungrouped, ungrouped, "no per-execution evidence sufficiency"),
    ],
)
def test_unknown_context_is_never_a_pass(baseline, candidate, message):
    result = compare_runs(baseline(), candidate(), POLICY)
    assert result["verdict"] == "not_comparable"
    assert any(message in r["message"] for r in result["reasons"])


def test_hosted_projection_reaches_the_same_verdict():
    baseline, candidate = run(Packing()), run(Packing(trim={3}))
    local = compare_runs(baseline, candidate, POLICY)
    hosted = compare_runs(project_run(baseline), project_run(candidate), POLICY)
    assert hosted["verdict"] == local["verdict"] == "fail"
    shared = project_run(candidate)["queries"][0]["repeat_sufficiency"]
    assert shared == [
        {"delivered": "sufficient", "context": c}
        for c in ("sufficient", "sufficient", "insufficient")
    ]


def test_policy_rules_are_validated_and_existing_hashes_kept():
    plain = {"metrics": {"ndcg": {"max_drop": 0.1}}}
    assert policy_sha256(parse_policy(plain)) == policy_sha256(
        parse_policy({**plain, "critical_context": []})
    )
    assert policy_sha256(parse_policy(plain)) != policy_sha256(
        parse_policy({**plain, "critical_context": [{"query_id": "q"}]})
    )
    for bad, message in [
        ([{"query_id": "q"}, {"query_id": "q"}], "duplicate"),
        ([{"query_id": "q", "doc_id": "d"}], "unknown"),
        ([{"query_id": ""}], "nonempty"),
    ]:
        with pytest.raises(PolicyError, match=message):
            parse_policy({**plain, "critical_context": bad})


def test_old_runs_without_per_execution_sufficiency_are_not_comparable():
    candidate = run(Packing())
    old = copy.deepcopy(candidate)
    old["queries"][0].pop("repeat_sufficiency")
    result = compare_runs(candidate, old, POLICY)
    assert result["verdict"] == "not_comparable"


def test_the_context_story_demo_runs_end_to_end(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "context_story_demo", ROOT / "examples" / "context_story_demo.py"
    )
    assert spec is not None
    assert spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    assert demo.main(["--out", str(tmp_path / "story")]) == 0
    page = (tmp_path / "story" / "context-story.html").read_text()
    assert "Synthetic demonstration" in page
    assert "<script" not in page
    facts = demo.rehearse(tmp_path / "again")
    assert facts["recall"] == 1.0
    assert facts["sufficiency"]["context"] == "insufficient"
    assert facts["diagnosis"]["class"] == "lost_in_context"
    assert facts["trial"]["recovered_focus"] == 1
    assert facts["trial"]["regressed"] == []
    assert (facts["validation"], facts["approved"], facts["blocked"]) == (
        "approved_under_policy",
        "pass",
        "fail",
    )


def test_ledger_counts_critical_context_failures():
    from vectraudit.cloud.ledger import failed_questions, summarize_check

    result = compare_runs(run(Packing()), run(Packing(trim={1})), POLICY)
    assert failed_questions(result)["critical"] == ["recovery"]
    assert summarize_check(result)["critical"] == {"total": 1, "failed": 1}
