"""Langfuse scores and LangSmith feedback attach as answer judgments, keyed by trace."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.test_answer_judgments import run
from tests.test_trace_formats import docs, langsmith_trace
from tests.test_trace_import import dataset
from vectraudit.cli import app
from vectraudit.evaluation.answers import AnswerJudgmentError, attach_file, read_judgment_file
from vectraudit.evaluation.runfile import check_run

SECRET = "PLANTED-REVIEWER-COMMENT"


def traced_run() -> dict:
    record = run()
    for q in record["queries"]:
        q["trace_id"] = f"trace-{q['query_id']}"
    return record


def score(trace, name, source, data_type, value=None, string_value=None, **extra):
    """One score as the Langfuse Scores API returns it."""
    return {
        "id": f"{trace}-{name}-{source}",
        "traceId": trace,
        "observationId": None,
        "name": name,
        "source": source,
        "dataType": data_type,
        "value": value,
        "stringValue": string_value,
        "comment": SECRET,
        "authorUserId": "user-42",
        "timestamp": "2026-10-20T10:00:00.000Z",
        **extra,
    }


def write(tmp_path, value, name="scores.json"):
    path = tmp_path / name
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    return path


def answers_of(result):
    return {q["query_id"]: q["answer"] for q in result["queries"] if "answer" in q}


def test_langfuse_annotations_decide_and_llm_judges_are_measured_against_them(tmp_path):
    page = {
        "data": [
            score("trace-wrong", "correctness", "ANNOTATION", "CATEGORICAL", 0, "Incorrect"),
            score("trace-wrong", "correctness", "EVAL", "NUMERIC", 0.9),
            score("trace-packed", "correctness", "EVAL", "NUMERIC", 0.2),
            score("trace-wrong", "user-feedback", "API", "NUMERIC", 1),
            score(None, "correctness", "ANNOTATION", "CATEGORICAL", 1, "Correct", sessionId="s1"),
            score("trace-right", "correctness", "ANNOTATION", "TEXT", None, "looks fine"),
        ],
        "meta": {"page": 1, "limit": 50, "totalItems": 6, "totalPages": 1},
    }
    result = attach_file(
        traced_run(),
        write(tmp_path, page),
        {"correctness": 0.5},
        evaluators={"correctness": None},
    )
    check_run(result)
    answers = answers_of(result)
    assert (answers["wrong"]["verdict"], answers["wrong"]["decided_by"]) == ("incorrect", "human")
    assert (answers["packed"]["verdict"], answers["packed"]["decided_by"]) == ("incorrect", "model")
    summary = result["answers"]
    assert summary["source_format"] == "langfuse"
    assert summary["agreement_with_human"]["correctness"] == {
        "questions": 1,
        "agreed": 0,
        "disagreed_on": ["wrong"],
    }
    assert summary["skipped_judgments"] == {
        "not_on_a_trace": {"correctness": 1},
        "not_selected": {"user-feedback": 1},
        "text_score": {"correctness": 1},
    }
    assert SECRET not in json.dumps(result)
    assert "user-42" not in json.dumps(result)


def test_who_judged_is_declared_where_the_export_does_not_say(tmp_path):
    scores = [
        score("trace-wrong", "review", "API", "CATEGORICAL", None, "Partially correct"),
        score("trace-packed", "review", "API", "CATEGORICAL", None, "Needs work"),
    ]
    path = write(tmp_path, "\n".join(json.dumps(s) for s in scores), "scores.jsonl")
    with pytest.raises(
        AnswerJudgmentError, match="does not say whether a person or a model judged: 1; categorical"
    ):
        read_judgment_file(
            path, evaluators={"review": None}, labels={"partially correct": "partial"}
        )
    read = read_judgment_file(
        path, evaluators={"review": "human"}, labels={"Partially  Correct": "partial"}
    )
    [(key_kind, key, judgment)] = read.rows
    assert (key_kind, key, judgment.kind, judgment.verdict) == (
        "trace_id",
        "trace-wrong",
        "human",
        "partial",
    )
    assert read.skipped == {"label_unmapped": {"review": 1}}


def langsmith_feedback(run_id, key, score=None, value=None, source="model", **extra):
    """One feedback record as LangSmith's list_feedback returns it."""
    return {
        "id": f"fb-{run_id}-{key}-{source}",
        "run_id": run_id,
        "session_id": "project-1",
        "key": key,
        "score": score,
        "value": value,
        "comment": SECRET,
        "correction": {"outputs": SECRET},
        "feedback_source": {"type": source, "user_id": "user-42"},
        "created_at": "2026-10-20T10:00:00.000000",
        **extra,
    }


def test_langsmith_feedback_attaches_to_a_run_imported_from_the_same_traces(tmp_path):
    runs = langsmith_trace(1, "How do I reset my password?", docs("d1"))
    runs += langsmith_trace(2, "Where is the invoice?", docs("d5"), start=10)
    root1, root2 = runs[0]["id"], runs[3]["id"]
    traces = write(tmp_path, runs, "runs.json")
    data = tmp_path / "questions.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": q.query_id, "text": q.text, "relevant": q.relevant})
            for q in dataset().queries
        )
    )
    imported = tmp_path / "traces.run.json"
    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "workbench", "import-traces", str(data), str(traces),
            "--id-field", "doc_id", "--out", str(imported),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    feedback = [
        langsmith_feedback(root1, "correctness", score=True),
        langsmith_feedback(root2, "correctness", score=0),
        langsmith_feedback(root2, "correctness", value="incorrect", source="app"),
        langsmith_feedback(runs[1]["id"], "correctness", score=1),  # a child run: unmatched
        langsmith_feedback("other", "correctness", score=1, trace_id=root1),
    ]
    judgments = write(tmp_path, feedback, "feedback.json")
    out = tmp_path / "judged.run.json"
    result = cli.invoke(
        app,
        [
            "workbench", "attach-answers", str(imported), str(judgments), "--out", str(out),
            "--evaluator", "correctness=human", "--threshold", "correctness=1",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    judged = json.loads(out.read_text())
    answers = answers_of(judged)
    assert (answers["q1"]["verdict"], answers["q1"]["decided_by"]) == ("correct", "model")
    assert (answers["q2"]["verdict"], answers["q2"]["decided_by"]) == ("incorrect", "human")
    assert judged["answers"]["unmatched_judgments"] == 1
    assert len(answers["q1"]["judgments"]) == 2  # trace_id wins over a child run_id
    assert SECRET not in out.read_text()
    assert "user-42" not in out.read_text()


def test_exports_need_named_evaluators_and_formats_do_not_mix(tmp_path):
    page = {
        "data": [
            score("t", "correctness", "EVAL", "NUMERIC", 1),
            score("t", "tox", "EVAL", "NUMERIC", 0),
        ]
    }
    with pytest.raises(AnswerJudgmentError, match=r"scores named correctness \(1\), tox \(1\)"):
        read_judgment_file(write(tmp_path, page))
    native = write(
        tmp_path, json.dumps({"query_id": "q", "evaluator": "x", "verdict": "correct"}), "j.jsonl"
    )
    with pytest.raises(AnswerJudgmentError, match="apply to Langfuse and LangSmith exports"):
        read_judgment_file(native, evaluators={"x": None})
    mixed = [score("t", "c", "EVAL", "NUMERIC", 1), langsmith_feedback("r", "c", score=1)]
    with pytest.raises(AnswerJudgmentError, match="mixes formats"):
        read_judgment_file(write(tmp_path, mixed, "mixed.json"), evaluators={"c": None})
    with pytest.raises(AnswerJudgmentError, match="verdict must be one of"):
        read_judgment_file(write(tmp_path, page), evaluators={"c": None}, labels={"ok": "great"})
    with pytest.raises(AnswerJudgmentError, match="contains no judgments"):
        read_judgment_file(write(tmp_path, page), evaluators={"absent": None})
