"""Production intake: new traces become recurrences or reviewed tests, never labels."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.test_trace_import import dataset, trace, write_otlp
from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.intake import intake, read_feedback, read_routes, review_dataset
from vectraudit.pipelines.traces import TraceImportError

ACME = {"metadata.tenant": "acme"}
BETA = {"metadata.tenant": "beta"}


def first_batch():
    return (
        trace("t1", "How do I reset my password?", retrieved=["d1", "d9"], start=0)
        # The labelled d2 is missing from production retrieval: a likely recurrence.
        + trace("t2", "Where is the invoice?", retrieved=["d5", "d6"], start=1000)
        + trace("t4", "How do I export data?", retrieved=["d8"], start=2000, root_attrs=ACME)
        + trace("t5", "how do i EXPORT data?", retrieved=["d8"], start=3000, root_attrs=ACME)
        + trace("t6", "Delete my account", retrieved=["d7"], start=4000, root_attrs=BETA)
    )


def run(tmp_path, spans, data=None, **kwargs):
    path = write_otlp(tmp_path / "traces.jsonl", spans)
    return intake(path, data or dataset(), tmp_path / "state", context_fields=["tenant"], **kwargs)


def queue(tmp_path):
    lines = (tmp_path / "state" / "queue.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_traces_become_recurrences_and_a_deduplicated_review_queue(tmp_path):
    report = run(
        tmp_path,
        first_batch(),
        feedback={"t6": {"negative": 1}},
        routes=[{"when": {"tenant": "acme"}, "reviewer": "ana"}],
    )
    assert (report["traces_taken"], report["matched_benchmark"]) == (5, 2)
    (recurrence,) = report["likely_recurrences"]
    assert (recurrence["query_id"], recurrence["trace_id"]) == ("q2", "t2")
    assert recurrence["delivered"] == ["d5", "d6"]
    rows = queue(tmp_path)
    assert [r["text"] for r in rows] == ["Delete my account", "How do I export data?"]
    delete, export = rows
    assert delete["signals"] == {"negative": 1}  # feedback raises priority...
    assert (export["count"], export["trace_ids"]) == (2, ["t4", "t5"])
    assert (export["reviewer"], delete["reviewer"]) == ("ana", None)
    assert report["queue"]["unassigned"] == 1
    review = parse_dataset(review_dataset(tmp_path / "state"))
    assert all(not q.relevant and not q.forbidden for q in review.queries)  # ...never labels


def test_the_cursor_takes_only_new_traces_from_overlapping_exports(tmp_path):
    run(tmp_path, first_batch())
    again = run(tmp_path, first_batch())
    assert (again["traces_taken"], again["traces_already_taken"]) == (0, 5)
    later = run(
        tmp_path,
        first_batch()
        + trace("t7", "Delete my account", retrieved=["d7"], start=4000, root_attrs=BETA)
        + trace("t8", "Delete my account", retrieved=["d7"], start=5000, root_attrs=BETA),
    )
    # t7 shares the cursor's instant but was not taken before; t8 is newer.
    assert later["traces_taken"] == 2
    assert next(r for r in queue(tmp_path) if r["text"] == "Delete my account")["count"] == 3


def test_a_question_added_to_the_benchmark_is_promoted_on_the_next_run(tmp_path):
    run(tmp_path, first_batch())
    rows = [
        {"query_id": "q1", "text": "How do I reset my password?", "relevant": {"d1": 1}},
        {"query_id": "export", "text": "How do I export data?", "relevant": {"d8": 1}},
    ]
    data = parse_dataset("\n".join(json.dumps(r) for r in rows))
    report = run(tmp_path, first_batch(), data=data)
    assert report["queue"]["promoted_now"] == 1
    statuses = {r["text"]: r["status"] for r in queue(tmp_path)}
    assert statuses["How do I export data?"] == "promoted"


def test_a_recurrence_names_its_case_and_asks_to_reopen_a_verified_one(tmp_path):
    cases = tmp_path / "ws" / ".workbench" / "cases"
    cases.mkdir(parents=True)
    (cases / "c1.json").write_text(json.dumps({"id": "c1", "query_id": "q2", "status": "verified"}))
    report = run(tmp_path, first_batch(), workspace=tmp_path / "ws")
    (recurrence,) = report["likely_recurrences"]
    assert recurrence["cases"] == [{"case": "c1", "status": "verified"}]
    assert recurrence["reopen"] == ["c1"]


def test_an_independent_sample_is_chosen_before_signals_and_survives_a_capped_export(tmp_path):
    spans = []
    for i in range(20):
        spans += trace(f"n{i}", f"new question {i}", retrieved=["d1"], start=i * 1000)
    feedback = {f"n{i}": {"escalated": 1} for i in range(10)}
    report = run(tmp_path, spans, feedback=feedback, sample_rate=0.25, seed=3)
    sampled = {r["text"] for r in queue(tmp_path) if r["sampled"]}
    assert report["queue"]["independent_sample"] == len(sampled) > 0
    again = intake(
        write_otlp(tmp_path / "other.jsonl", spans),
        dataset(),
        tmp_path / "other-state",
        sample_rate=0.25,
        seed=3,
    )
    assert again["queue"]["independent_sample"] == len(sampled)  # signals do not change it
    exported = parse_dataset(review_dataset(tmp_path / "state", limit=4)).queries
    assert any("independent-sample" in q.tags for q in exported)
    assert len(exported) == 4


def test_text_shared_by_several_benchmark_questions_is_neither_new_nor_judged(tmp_path):
    rows = [
        {"query_id": "a", "text": "Where is the invoice?", "relevant": {"d2": 1}},
        {"query_id": "b", "text": "where is the invoice?", "relevant": {"d3": 1}},
    ]
    data = parse_dataset("\n".join(json.dumps(r) for r in rows))
    report = run(tmp_path, trace("t2", "Where is the invoice?", retrieved=["d5"]), data=data)
    assert (report["ambiguous"], report["likely_recurrences"], report["queue"]["pending"]) == (
        1,
        [],
        0,
    )


def test_feedback_and_routes_are_validated(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text(json.dumps({"trace_id": "t1", "signal": "relevant"}))
    with pytest.raises(TraceImportError, match="signal"):
        read_feedback(path)
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps([{"when": {"tenant": "acme"}}]))
    with pytest.raises(TraceImportError, match="reviewer"):
        read_routes(routes)
    with pytest.raises(TraceImportError, match="sample_rate"):
        run(tmp_path, first_batch(), sample_rate=2)


def test_cli_takes_traces_exports_review_and_can_fail_on_recurrence(tmp_path):
    traces = write_otlp(tmp_path / "traces.jsonl", first_batch())
    data = tmp_path / "questions.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": q.query_id, "text": q.text, "relevant": q.relevant})
            for q in dataset().queries
        )
    )
    out = tmp_path / "review.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "workbench", "intake", str(traces), str(data),
            "--state", str(tmp_path / "state"),
            "--context-field", "tenant",
            "--export-review", str(out),
            "--fail-on-recurrence",
        ],
    )  # fmt: skip
    assert result.exit_code == 3, result.output
    assert "Took 5 new trace(s)" in result.output
    assert "likely recurrence: q2" in result.output
    assert len(parse_dataset(out.read_text()).queries) == 2
