"""DeepEval test results as answer judgments."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.test_answer_judgments import DATA, SECRET, run
from vectraudit.cli import app
from vectraudit.evaluation.answers import attach_file
from vectraudit.evaluation.deepeval_import import (
    DeepEvalImportError,
    load_test_run,
    metric_names,
    to_judgments,
)

CORRECTNESS = "Correctness [GEval]"


def metric(name, success, score=None, **extra):
    return {
        "name": name, "threshold": 0.5, "success": success, "score": score,
        "reason": f"{SECRET} judge reasoning", "strictMode": False, "flaky": False,
        "evaluationModel": "gpt-4.1", **extra,
    }  # fmt: skip


def case(name, input_text, metrics, metadata=None):
    out = {
        "name": name, "input": input_text, "actualOutput": f"{SECRET} answer",
        "retrievalContext": [f"{SECRET} passage"], "success": True, "metricsData": metrics,
        "runDuration": 1.2, "order": 0,
    }  # fmt: skip
    if metadata is not None:
        out["metadata"] = metadata
    return out


def deepeval_results(wrapped=True):
    test_run = {
        "testFile": "test_rag.py",
        "testCases": [
            case("by id", "anything", [metric(CORRECTNESS, False, 0.2),
                                      metric("Faithfulness", True, 0.9),
                                      metric("Contextual Precision", True, 1.0)],
                 metadata={"query_id": "wrong"}),
            case("by text", "Question  RIGHT", [metric(CORRECTNESS, True, 0.8)]),
            case("by trace", "x", [metric(CORRECTNESS, True, 0.7)], metadata={"trace_id": "t-9"}),
            case("unknown", "a question nobody asked", [metric(CORRECTNESS, True, 1.0)]),
            case("errored", "question packed",
                 [metric(CORRECTNESS, None, error="judge timed out")]),
            case("pending", "question dropped", [metric(CORRECTNESS, None)]),
        ],
        "metricsScores": [],
        "testPassed": 4,
        "testFailed": 1,
    }  # fmt: skip
    return {"testRunData": test_run} if wrapped else test_run


QUESTIONS = {q.query_id: q.text for q in DATA.queries}


def write(tmp_path, data, name="latest_test_run.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_deepeval_decisions_become_verdicts_matched_by_id_trace_or_text():
    rows, counts = to_judgments(deepeval_results()["testRunData"], [CORRECTNESS], QUESTIONS)
    assert rows == [
        {"query_id": "wrong", "evaluator": CORRECTNESS, "kind": "model", "verdict": "incorrect",
         "score": 0.2, "evaluator_version": "gpt-4.1"},
        {"query_id": "right", "evaluator": CORRECTNESS, "kind": "model", "verdict": "correct",
         "score": 0.8, "evaluator_version": "gpt-4.1"},
        {"trace_id": "t-9", "evaluator": CORRECTNESS, "kind": "model", "verdict": "correct",
         "score": 0.7, "evaluator_version": "gpt-4.1"},
    ]  # fmt: skip
    assert counts == {
        "matched_by_query_id": 1, "matched_by_text": 3, "matched_by_trace_id": 1,
        "unmatched": 1, "metric_error": 1, "not_evaluated": 1, "judgments": 3,
    }  # fmt: skip
    assert SECRET not in json.dumps(rows)


def test_only_chosen_metrics_count_and_retrieval_metrics_are_refused():
    run_data = deepeval_results()["testRunData"]
    rows, _ = to_judgments(run_data, ["Faithfulness"], QUESTIONS)
    assert [(r["query_id"], r["evaluator"]) for r in rows] == [("wrong", "Faithfulness")]
    for retrieval in ("Contextual Precision", "Contextual Recall", "Contextual Relevancy"):
        with pytest.raises(DeepEvalImportError, match="judge the retrieved context"):
            to_judgments(run_data, [CORRECTNESS, retrieval], QUESTIONS)
    with pytest.raises(DeepEvalImportError, match="--metric"):
        to_judgments(run_data, [], QUESTIONS)
    assert metric_names(run_data) == {
        CORRECTNESS: 6, "Faithfulness": 1, "Contextual Precision": 1,
    }  # fmt: skip


def test_a_text_matching_several_questions_is_not_guessed():
    run_data = {"testCases": [case("dup", "same text", [metric(CORRECTNESS, True)])]}
    rows, counts = to_judgments(run_data, [CORRECTNESS], {"a": "Same text", "b": "same  TEXT"})
    assert rows == []
    assert counts["ambiguous_text"] == 1


def test_the_judgments_attach_and_diagnose(tmp_path):
    rows, _ = to_judgments(deepeval_results()["testRunData"], [CORRECTNESS], QUESTIONS)
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    out = attach_file(run(), path)
    answers = {q["query_id"]: q["answer"] for q in out["queries"] if "answer" in q}
    assert answers["wrong"]["verdict"] == "incorrect"
    assert answers["wrong"]["diagnosis"]["class"] == "answered_wrong_despite_evidence"
    assert answers["right"]["verdict"] == "correct"
    assert out["answers"]["unmatched_judgments"] == 1  # the trace ID is not in this run


def test_both_file_layouts_load_and_other_files_are_refused(tmp_path):
    assert load_test_run(write(tmp_path, deepeval_results()))["testFile"] == "test_rag.py"
    assert load_test_run(write(tmp_path, deepeval_results(wrapped=False), "b.json"))["testCases"]
    for bad in ({"testCases": "no"}, ["not", "a", "run"], {"cases": []}):
        with pytest.raises(DeepEvalImportError, match="not a DeepEval test run"):
            load_test_run(write(tmp_path, bad, "bad.json"))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(DeepEvalImportError, match="not JSON"):
        load_test_run(broken)


def test_cli_lists_metrics_writes_judgments_and_refuses(tmp_path):
    results = write(tmp_path, deepeval_results())
    data = tmp_path / "questions.jsonl"
    rows = [
        {"query_id": q.query_id, "text": q.text, "relevant": dict(q.relevant)} for q in DATA.queries
    ]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "judgments.jsonl"
    runner = CliRunner()
    args = ["workbench", "deepeval-judgments", str(results), str(data), "--out", str(out)]
    listing = runner.invoke(app, args)
    assert listing.exit_code == 2
    assert f"In this file: {CORRECTNESS} (6), Faithfulness (1), Contextual Precision (1)" in (
        " ".join(listing.output.split())
    )
    assert not out.exists()
    done = runner.invoke(app, [*args, "--metric", CORRECTNESS])
    assert done.exit_code == 0, done.output
    assert "Wrote 3 judgments" in done.output
    assert "1 test cases matching no question" in done.output
    assert len(out.read_text().splitlines()) == 3
    refused = runner.invoke(app, [*args, "--metric", "Contextual Recall"])
    assert refused.exit_code == 2
    assert "judge the retrieved context" in refused.output
