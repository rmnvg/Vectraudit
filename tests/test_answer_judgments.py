"""Answer judgments from existing tools, evidence sufficiency, and per-question diagnosis."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.answers import (
    AnswerJudgmentError,
    attach_answers,
    attach_file,
    read_judgments,
)
from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.runfile import check_run
from vectraudit.evaluation.triage import triage
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

SECRET = "PLANTED-PRIVATE-EXPLANATION"
POLICY = {"policy": 1, "exception": 1, "summary": 1}
GROUPS = [["policy", "exception"], ["summary"]]  # both together, or the summary alone


def question(qid: str) -> dict:
    return {
        "query_id": qid,
        "text": f"question {qid}",
        "relevant": POLICY,
        "required_evidence": GROUPS,
    }


DATA = parse_dataset(
    "\n".join(
        json.dumps(question(q))
        for q in ["packed", "wrong", "right", "dropped", "unseen", "unknown"]
    )
)


def full(*ids: str) -> ModelContext:
    return ModelContext(items=[ContextItem(id=i, truncated=False) for i in ids])


def stage(name: str, ids: list[str], parents: list[str] | None = None) -> StageTrace:
    return StageTrace(
        name=name,
        kind="custom",
        parents=parents or [],
        hits=[Hit(id=i) for i in ids],
        latency_ms=1.0,
    )


class App:
    """Six planted outcomes, one per question."""

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        both = ["policy", "exception"]
        if request.query_id == "packed":  # retrieved both; packing dropped the exception
            return PipelineResult(hits=[Hit(id=i) for i in both], context=full("policy"))
        if request.query_id in ("wrong", "right"):  # everything reached the model
            return PipelineResult(hits=[Hit(id=i) for i in both], context=full(*both))
        if request.query_id == "dropped":  # the reranker cut the exception
            stages = [stage("search", both), stage("rerank", ["policy"], ["search"])]
            return PipelineResult(hits=[Hit(id="policy")], stages=stages, context=full("policy"))
        if request.query_id == "unseen":  # never retrieved at all
            stages = [stage("search", ["other"])]
            return PipelineResult(hits=[Hit(id="other")], stages=stages, context=full("other"))
        return PipelineResult(hits=[Hit(id=i) for i in both])  # context not recorded


def run() -> dict:
    return execute_pipeline(DATA, App(), PipelineOptions(k=3, warmup=0))


def judged(verdicts: dict[str, str], **extra) -> list:
    rows = "\n".join(
        json.dumps({"query_id": q, "evaluator": "review", "kind": "human", "verdict": v, **extra})
        for q, v in verdicts.items()
    )
    return rows


def write(tmp_path, text: str, name: str = "judgments.jsonl"):
    path = tmp_path / name
    path.write_text(text)
    return path


ALL_WRONG = {q: "incorrect" for q in ["packed", "wrong", "dropped", "unseen", "unknown"]}


def test_required_evidence_groups_are_validated_and_round_trip():
    row = question("q")
    again = parse_dataset(render_dataset(dataset_rows(parse_dataset(json.dumps(row)))))
    assert again.queries[0].required_evidence == (("exception", "policy"), ("summary",))
    for bad, message in [
        ({"required_evidence": [["missing"]]}, "must be relevant"),
        ({"required_evidence": []}, "1-16 groups"),
        ({"required_evidence": [[]]}, "1-32 document ids"),
    ]:
        with pytest.raises(DatasetError, match=message):
            parse_dataset(json.dumps({**row, **bad}))
    plain = {"query_id": "q", "text": "x", "relevant": POLICY}
    assert parse_dataset(json.dumps(plain)).sha256 != parse_dataset(json.dumps(row)).sha256


def test_missing_exception_is_detected_although_document_recall_passes():
    queries = {q["query_id"]: q for q in run()["queries"]}
    packed = queries["packed"]
    assert packed["metrics"]["recall"] == pytest.approx(2 / 3)  # both group documents found
    assert packed["sufficiency"] == {
        "groups": 2,
        "delivered": "sufficient",
        "missing_from_delivered": [],
        "context": "insufficient",
    }
    assert queries["wrong"]["sufficiency"]["context"] == "sufficient"
    assert queries["unknown"]["sufficiency"]["context"] == "unrecorded"


def test_context_omission_and_answer_failure_are_diagnosed_separately(tmp_path):
    rows = read_judgments(write(tmp_path, judged({**ALL_WRONG, "right": "correct"})))
    result = attach_answers(run(), rows)
    check_run(result)
    classes = {q["query_id"]: q["answer"]["diagnosis"]["class"] for q in result["queries"]}
    assert classes == {
        "packed": "lost_in_context",
        "wrong": "answered_wrong_despite_evidence",
        "right": "answered_correctly",
        "dropped": "lost_downstream",
        "unseen": "not_observed",
        "unknown": "context_unknown",  # mixed and unknown outcomes stay unknown
    }
    dropped = next(q for q in result["queries"] if q["query_id"] == "dropped")
    assert "exception dropped at rerank" in dropped["answer"]["diagnosis"]["facts"]
    assert result["answers"]["diagnoses"]["lost_in_context"] == 1


def test_humans_decide_disagreements_are_disputes_and_models_are_calibrated(tmp_path):
    lines = [
        {"query_id": "wrong", "evaluator": "review", "kind": "human", "verdict": "incorrect"},
        {"query_id": "wrong", "evaluator": "ragas.answer_correctness", "score": 0.9},
        {"query_id": "right", "evaluator": "review", "kind": "human", "verdict": "correct"},
        {"query_id": "right", "evaluator": "ragas.answer_correctness", "score": 0.8},
        {"query_id": "packed", "evaluator": "review", "kind": "human", "verdict": "correct"},
        {"query_id": "packed", "evaluator": "second", "kind": "human", "verdict": "incorrect"},
        {
            "query_id": "unseen",
            "evaluator": "ragas.answer_correctness",
            "score": 0.1,
            "explanation": SECRET,
        },
    ]
    path = write(tmp_path, "\n".join(json.dumps(x) for x in lines))
    result = attach_file(run(), path, {"ragas.answer_correctness": 0.7})
    answers = {q["query_id"]: q["answer"] for q in result["queries"] if "answer" in q}
    assert (answers["wrong"]["verdict"], answers["wrong"]["decided_by"]) == ("incorrect", "human")
    assert answers["packed"]["verdict"] == "disputed"
    assert answers["packed"]["diagnosis"]["class"] == "disputed"
    assert (answers["unseen"]["verdict"], answers["unseen"]["decided_by"]) == ("incorrect", "model")
    assert result["answers"]["agreement_with_human"]["ragas.answer_correctness"] == {
        "questions": 2,
        "agreed": 1,
        "disagreed_on": ["wrong"],
    }
    assert SECRET not in json.dumps(result)

    unthresholded = attach_file(run(), path)
    unseen = next(q for q in unthresholded["queries"] if q["query_id"] == "unseen")
    assert unseen["answer"]["diagnosis"]["class"] == "unjudged"  # no default threshold


def test_abstention_and_handoff_are_not_wrong_answers(tmp_path):
    lines = [
        {"query_id": "wrong", "evaluator": "review", "kind": "human", "outcome": "abstained"},
        {"query_id": "packed", "evaluator": "review", "kind": "human", "outcome": "abstained"},
        {"query_id": "right", "evaluator": "review", "kind": "human", "outcome": "handoff"},
    ]
    result = attach_file(run(), write(tmp_path, "\n".join(json.dumps(x) for x in lines)))
    classes = {
        q["query_id"]: q["answer"]["diagnosis"]["class"] for q in result["queries"] if "answer" in q
    }
    assert classes == {
        "wrong": "abstained_despite_evidence",
        "packed": "abstained",
        "right": "handoff",
    }


def test_judgments_attach_by_trace_id_and_unknown_ones_are_counted(tmp_path):
    record = run()
    for q in record["queries"]:
        q["trace_id"] = f"trace-{q['query_id']}"
    lines = [
        {"trace_id": "trace-wrong", "evaluator": "phoenix.qa", "verdict": "incorrect"},
        {"trace_id": "trace-elsewhere", "evaluator": "phoenix.qa", "verdict": "correct"},
        {"query_id": "not-in-run", "evaluator": "phoenix.qa", "verdict": "correct"},
    ]
    result = attach_file(record, write(tmp_path, "\n".join(json.dumps(x) for x in lines)))
    assert result["answers"]["judged_questions"] == 1
    assert result["answers"]["unmatched_judgments"] == 2


def test_triage_reports_an_answer_that_became_wrong(tmp_path):
    baseline = attach_file(run(), write(tmp_path, judged({"packed": "correct"}), "b.jsonl"))
    candidate = attach_file(run(), write(tmp_path, judged({"packed": "incorrect"}), "c.jsonl"))
    item = next(i for i in triage(baseline, candidate)["items"] if i["query_id"] == "packed")
    [finding] = [f for f in item["findings"] if f["kind"].startswith("answer")]
    assert (finding["kind"], finding["severity"], finding["diagnosis"]) == (
        "answer_regressed",
        "regression",
        "lost_in_context",
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ({"evaluator": "x", "verdict": "correct"}, "exactly one of query_id or trace_id"),
        ({"query_id": "q", "evaluator": "x"}, "needs a verdict, a score, claims or a"),
        ({"query_id": "q", "evaluator": "x", "verdict": "great"}, "verdict must be"),
        ({"query_id": "q", "evaluator": "x", "kind": "robot", "verdict": "correct"}, "kind"),
        ({"query_id": "q", "evaluator": "x", "score": "high"}, "score must be"),
    ],
)
def test_invalid_judgments_are_refused(tmp_path, line, message):
    with pytest.raises(AnswerJudgmentError, match=message):
        read_judgments(write(tmp_path, json.dumps(line)))


def test_cli_attaches_into_a_new_file(tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run()))
    path = write(tmp_path, judged({"packed": "incorrect"}))
    out = tmp_path / "judged.json"
    result = CliRunner().invoke(
        app, ["workbench", "attach-answers", str(run_path), str(path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "1: lost in context." in result.output
    assert json.loads(out.read_text())["answers"]["judged_questions"] == 1
    same = CliRunner().invoke(
        app, ["workbench", "attach-answers", str(run_path), str(path), "--out", str(run_path)]
    )
    assert same.exit_code == 2


def cited(doc_ids: dict[str, list[str]], **extra) -> str:
    return "\n".join(
        json.dumps(
            {
                "query_id": q,
                "evaluator": "review",
                "kind": "human",
                "verdict": "incorrect",
                "citations": ids,
                **extra,
            }
        )
        for q, ids in doc_ids.items()
    )


def test_citations_are_placed_against_delivery_and_model_context(tmp_path):
    path = write(
        tmp_path,
        cited(
            {
                "packed": ["policy", "exception", "invented"],  # exception was packed out
                "right": ["policy"],
                "unknown": ["policy"],  # context not recorded
            }
        ),
    )
    out = attach_file(run(), path)
    by = {q["query_id"]: q["answer"]["citations"] for q in out["queries"] if "answer" in q}
    assert by["packed"]["classes"] == {
        "exception": "not_in_context",
        "invented": "not_delivered",
        "policy": "in_context",
    }
    assert by["right"]["classes"] == {"policy": "in_context"}
    assert by["unknown"]["classes"] == {"policy": "delivered_context_unknown"}
    assert out["answers"]["citations"] == {
        "questions_with_citations": 3,
        "questions_citing_undelivered": 1,
        "questions_citing_outside_context": 1,
        "questions_with_unsupported_citation": 0,
        "questions_with_conflicting_citations": 0,
    }


def test_citations_do_not_change_verdicts_or_diagnoses(tmp_path):
    plain = attach_file(run(), write(tmp_path, judged(ALL_WRONG)))
    with_cites = attach_file(
        run(), write(tmp_path, cited({q: ["policy", "ghost"] for q in ALL_WRONG}), "c.jsonl")
    )
    assert with_cites["answers"]["verdicts"] == plain["answers"]["verdicts"]
    assert with_cites["answers"]["diagnoses"] == plain["answers"]["diagnoses"]


def test_citation_support_follows_human_over_model_and_disputes(tmp_path):
    rows = [
        {"query_id": "right", "evaluator": "tool", "kind": "model", "verdict": "correct",
         "citations": ["policy", "exception"],
         "citation_support": {"policy": "supported", "exception": "supported"}},
        {"query_id": "right", "evaluator": "review", "kind": "human", "verdict": "correct",
         "citations": ["policy", "exception"],
         "citation_support": {"policy": "unsupported"}},
        {"query_id": "wrong", "evaluator": "a", "kind": "human", "verdict": "incorrect",
         "citations": ["policy"], "citation_support": {"policy": "supported"}},
        {"query_id": "wrong", "evaluator": "b", "kind": "human", "verdict": "incorrect",
         "citations": ["policy"], "citation_support": {"policy": "unsupported"}},
    ]  # fmt: skip
    out = attach_file(run(), write(tmp_path, "\n".join(map(json.dumps, rows))))
    by = {q["query_id"]: q["answer"]["citations"] for q in out["queries"] if "answer" in q}
    assert by["right"]["support"] == {
        "policy": {"verdict": "unsupported", "decided_by": "human"},
        "exception": {"verdict": "supported", "decided_by": "model"},
    }
    assert by["wrong"]["support"]["policy"]["verdict"] == "disputed"
    assert out["answers"]["citations"]["questions_with_unsupported_citation"] == 1


def test_conflicting_citation_lists_are_not_analysed(tmp_path):
    rows = [
        {"query_id": "right", "evaluator": "a", "kind": "human", "verdict": "correct",
         "citations": ["policy"]},
        {"query_id": "right", "evaluator": "b", "kind": "model", "verdict": "correct",
         "citations": ["summary"]},
    ]  # fmt: skip
    out = attach_file(run(), write(tmp_path, "\n".join(map(json.dumps, rows))))
    got = next(q for q in out["queries"] if q["query_id"] == "right")["answer"]["citations"]
    assert got == {"status": "conflicting", "cited": ["policy", "summary"]}
    assert "classes" not in got
    assert out["answers"]["citations"]["questions_with_conflicting_citations"] == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"citations": "policy"},
        {"citations": [""]},
        {"citations": [7]},
        {"citations": [f"d{i}" for i in range(101)]},
        {"citation_support": {"policy": "supported"}},
        {"citations": ["policy"], "citation_support": {"other": "supported"}},
        {"citations": ["policy"], "citation_support": {"policy": "maybe"}},
        {"citations": ["policy"], "citation_support": {}},
    ],
)
def test_invalid_citation_input_is_refused(tmp_path, extra):
    row = {"query_id": "right", "evaluator": "r", "kind": "human", "verdict": "correct", **extra}
    with pytest.raises(AnswerJudgmentError):
        read_judgments(write(tmp_path, json.dumps(row)))


def test_citations_on_failed_questions_and_unmatched_rows_are_not_analysed(tmp_path):
    bad = run()
    bad["queries"][0]["error"] = {"code": "pipeline_error"}
    row = cited({bad["queries"][0]["query_id"]: ["policy"], "nope": ["policy"]})
    out = attach_answers(bad, read_judgments(write(tmp_path, row)))
    assert "citations" not in out["queries"][0]["answer"]
    assert out["answers"]["citations"]["questions_with_citations"] == 0
    assert out["answers"]["unmatched_judgments"] == 1


def test_cli_reports_citations_and_keeps_private_text_out(tmp_path):
    base = tmp_path / "run.json"
    base.write_text(json.dumps(run()))
    path = write(
        tmp_path,
        cited({"packed": ["policy", "ghost"]}, explanation=SECRET),
    )
    out = tmp_path / "out.json"
    result = CliRunner().invoke(
        app, ["workbench", "attach-answers", str(base), str(path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "Citations on 1 questions: 1 cite a document retrieval never delivered" in result.output
    assert SECRET not in out.read_text()
    check_run(json.loads(out.read_text()))
