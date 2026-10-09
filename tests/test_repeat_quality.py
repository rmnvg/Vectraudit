"""Repeated executions affect quality without inventing independent queries."""

import copy
import json

import pytest

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.metrics import repeat_quality
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.report import render_html, render_markdown
from vectraudit.evaluation.runfile import RunFileError, check_run
from vectraudit.evaluation.runner import RunOptions, execute_run
from vectraudit.evaluation.triage import triage
from vectraudit.integrations.sharing import project_run
from vectraudit.pipelines import Hit, PythonPipeline, Stage, execute_pipeline


def record(ids, *, evidence=True):
    outputs = iter(ids)
    pipeline = PythonPipeline(
        [
            Stage("retrieve", "keyword", lambda q, p: [Hit(id="a")]),
            Stage("final", "truncate", lambda q, p: [Hit(id=next(outputs))], ("retrieve",)),
        ]
    )
    ds = parse_dataset(json.dumps({"query_id": "q", "text": "test", "relevant": {"a": 1}}))
    return execute_pipeline(
        ds, pipeline, RunOptions(k=1, warmup=0, repeats=len(ids), evidence=evidence)
    )


POLICY = parse_policy({"min_queries": 1, "metrics": {"ndcg": {"max_drop": 0.1}}})


def test_intermittent_loss_changes_release_decision_and_trace():
    base, candidate = record(["a", "a"]), record(["a", "b"])
    q = check_run(candidate)["queries"][0]
    assert q["ids"] == ["a"]
    assert [m["ndcg"] for m in q["repeat_metrics"]] == [1, 0]
    assert q["metrics"]["ndcg"] == candidate["summary"]["metrics"]["ndcg"] == 0.5
    assert q["evidence"]["failures"] == []
    assert q["repeat_evidence"]["2"]["failures"][0]["stage"] == "final"
    result = compare_runs(base, candidate, POLICY)
    assert result["verdict"] == "fail"
    assert result["n_queries"] == result["n_families"] == 1
    assert (
        compare_runs(project_run(base), project_run(candidate), POLICY)["metrics"]
        == result["metrics"]
    )
    assert "repeat_evidence" not in project_run(candidate)["queries"][0]
    for render in (render_html, render_markdown):
        assert "Quality averages every repeat" in render(result)
        assert "show repeat 1" in render(result)
    finding = triage(base, candidate)["items"][0]["findings"][0]
    assert "quality averages repeats" in finding["message"]
    assert "same documents are ranked lower" not in finding["message"]


def test_connector_averages_semantic_and_exact_recall():
    from tests.test_eval_runner import Scripted, _two_dim

    connector = Scripted(lambda call, request: ["a"] if request.exact or call == 1 else ["b"])
    run = execute_run(_two_dim(1), connector, RunOptions(k=1, warmup=0, repeats=2))
    query = check_run(run)["queries"][0]
    assert query["metrics"]["ndcg"] == query["ann_recall"] == 0.5
    assert query["repeat_ann_recall"] == [1, 0]


def test_failed_later_execution_has_no_partial_quality_average():
    from tests.test_eval_runner import Scripted, _two_dim
    from vectraudit.connectors.base import ConnectorError, ErrorCode

    def fail_second(call, request):
        if call == 2:
            raise ConnectorError(ErrorCode.TIMEOUT, "timeout")
        return ["a"]

    run = execute_run(_two_dim(1), Scripted(fail_second), RunOptions(k=1, warmup=0, repeats=2))
    assert check_run(run)["status"] == "partial"
    assert run["summary"]["metrics"] is None
    assert compare_runs(run, run, POLICY)["verdict"] == "error"


def test_deterministic_traces_are_sparse_and_evidence_can_be_disabled():
    assert record(["a", "a"])["queries"][0]["repeat_evidence"] == {}
    assert "repeat_evidence" not in record(["a", "b"], evidence=False)["queries"][0]


def test_order_invariance_and_ann_reference():
    forward = repeat_quality([["a"], ["b"]], {"a": 1}, 1, ["a"])
    reverse = repeat_quality([["b"], ["a"]], {"a": 1}, 1, ["a"])
    assert forward["metrics"] == reverse["metrics"]
    assert forward["ann_recall"] == reverse["ann_recall"] == 0.5
    assert forward["repeat_ann_recall"] == [1, 0]
    assert repeat_quality([["a"], ["a"]], {}, 1)["metrics"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeat_metrics", []),
        ("repeat_ids", [["a"]]),
        ("repeat_ann_recall", [1, 1]),
        ("metrics", {"ndcg": 1, "recall": 1, "precision": 1, "mrr": 1}),
        ("repeat_evidence", {"3": {}}),
    ],
)
def test_corrupt_repeat_records_cannot_pass(field, value):
    base = record(["a", "a"])
    bad = record(["a", "b"])
    bad["queries"][0][field] = value
    with pytest.raises(RunFileError):
        check_run(bad)
    assert compare_runs(base, bad, POLICY)["verdict"] == "not_comparable"


def test_old_contracts_remain_readable_but_require_rerecording_to_compare():
    new = record(["a", "a"])
    old = copy.deepcopy(new)
    old["metric_contract"]["version"] = 1
    old["metric_contract"].pop("repeat_aggregation")
    for key in ("repeat_metrics", "repeat_ann_recall", "repeat_evidence"):
        old["queries"][0].pop(key)
    check_run(old)
    assert compare_runs(old, old, POLICY)["verdict"] == "pass"
    assert compare_runs(old, new, POLICY)["verdict"] == "not_comparable"
    assert compare_runs(new, record(["a"]), POLICY)["verdict"] == "not_comparable"
