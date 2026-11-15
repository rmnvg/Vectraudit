"""RAGChecker claim-level results as judgments, tested on RAGChecker's own example output."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.answers import AnswerJudgmentError, attach_file, read_judgments
from vectraudit.evaluation.ragchecker_import import (
    CLAIMS_EVALUATOR,
    RagCheckerImportError,
    derive,
    load_output,
    metric_names,
    run_documents,
    to_judgments,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ragchecker" / "checking_outputs.json"
ROWS = load_output(FIXTURE)
QUESTIONS = {"q0": ROWS[0]["query"], "q1": ROWS[1]["query"].upper()}
DOCUMENTS = {"q0": {"000", "001", "002", "003", "extra"}, "q1": {"000", "001", "002"}}
MODELS = {"extractor": "llama3-70b", "checker": "llama3-70b"}
ALL_METRICS = [
    "precision", "recall", "f1", "claim_recall", "context_precision", "context_utilization",
    "faithfulness", "hallucination", "self_knowledge", "noise_sensitivity_in_relevant",
    "noise_sensitivity_in_irrelevant",
]  # fmt: skip


def lines(metrics=("f1",), rows=ROWS, documents=DOCUMENTS, **kwargs):
    return to_judgments(rows, list(metrics), QUESTIONS, run_documents=documents, **MODELS, **kwargs)


@pytest.mark.parametrize("row", ROWS, ids=["river", "second"])
def test_counts_reproduce_ragcheckers_own_metrics(row):
    facts, _ = derive(row)
    m = row["metrics"]
    assert set(m) == set(ALL_METRICS)
    for name, expected in m.items():
        assert facts.metrics[name] == pytest.approx(expected, abs=1e-12), name
    c = facts.claims
    assert c["unsupported_incorrect"] / c["response"] == pytest.approx(m["hallucination"])
    assert c["unsupported_correct"] / c["response"] == pytest.approx(m["self_knowledge"])
    assert c["reference_in_context"] / c["reference"] == pytest.approx(m["claim_recall"])
    assert c["useful_documents"] / c["context_documents"] == pytest.approx(m["context_precision"])
    relevant = c["from_relevant_incorrect"] / c["response"]
    assert relevant == pytest.approx(m["noise_sensitivity_in_relevant"])
    irrelevant = c["from_irrelevant_incorrect"] / c["response"]
    assert irrelevant == pytest.approx(m["noise_sensitivity_in_irrelevant"])
    # Every incorrect response claim is unsupported, or entailed by a useful or a useless chunk.
    split = (
        c["unsupported_incorrect"] + c["from_relevant_incorrect"] + c["from_irrelevant_incorrect"]
    )
    assert split == c["response_incorrect"]
    assert c["reference"] == len(row["response2answer"])
    assert c["response"] == len(row["answer2response"])


def test_a_delivered_document_that_contradicts_the_reference_is_counted_on_its_own():
    _, ids = derive(ROWS[0])
    claims = {j["query_id"]: j for j in lines()[0] if j["evaluator"] == CLAIMS_EVALUATOR}
    support = claims["q0"]["claim_support"]
    assert list(support) == ids == ["000", "001", "002", "003"]
    assert {k: v for k, v in support["000"].items() if k != "reference_claims"} == {
        "reference": 3, "response": 1, "response_incorrect": 0, "contradicts_reference": 1,
    }  # fmt: skip
    quiet = {k: v for k, v in support["002"].items() if k != "reference_claims"}
    assert quiet == dict.fromkeys(quiet, 0)  # supports nothing: noise
    assert support["003"]["response_incorrect"] == 1  # entailed an incorrect response claim
    # Which reference claims each document entails, by position only (never the claim text).
    for doc in support.values():
        assert len(doc["reference_claims"]) == doc["reference"]
        assert doc["reference_claims"] == sorted(set(doc["reference_claims"]))
        assert all(0 <= i < 22 for i in doc["reference_claims"])
    assert support["002"]["reference_claims"] == []


def test_scores_come_from_ragchecker_and_claim_lines_carry_the_counts():
    out, counts = lines(("f1", "ragchecker.faithfulness", "hallucination"))
    scores = [j for j in out if j["evaluator"] != CLAIMS_EVALUATOR]
    names = ("f1", "faithfulness", "hallucination")
    assert [(j["query_id"], j["evaluator"]) for j in scores] == [
        (q, f"ragchecker.{m}") for q in ("q0", "q1") for m in names
    ]
    assert scores[0]["score"] == ROWS[0]["metrics"]["f1"]
    assert all(j["kind"] == "model" and "verdict" not in j for j in out)
    assert {j["evaluator_version"] for j in out} == {
        "ragchecker; extractor=llama3-70b; checker=llama3-70b"
    }
    assert counts["judgments"] == 6
    assert counts["claim_lines"] == 2
    assert counts["matched_by_text"] == 2  # the second question differs only in case
    claim_line = next(j for j in out if j["evaluator"] == CLAIMS_EVALUATOR)
    assert set(claim_line) == {
        "query_id", "evaluator", "kind", "evaluator_version", "claims", "claim_support",
    }  # fmt: skip
    # Without a chosen metric only the claim lines are written.
    only_claims, _ = lines(())
    assert {j["evaluator"] for j in only_claims} == {CLAIMS_EVALUATOR}


def test_verdicts_come_only_from_declared_pass_marks_and_noise_is_lower_is_better():
    out, _ = lines(
        ("f1", "hallucination"), thresholds={"ragchecker.f1": 0.6, "hallucination": 0.05}
    )
    verdicts = {(j["query_id"], j["evaluator"]): j["verdict"] for j in out if "verdict" in j}
    assert verdicts == {
        ("q0", "ragchecker.f1"): "incorrect",  # 0.593 < 0.6
        ("q1", "ragchecker.f1"): "correct",  # 0.774
        ("q0", "ragchecker.hallucination"): "incorrect",  # 0.091 > 0.05
        ("q1", "ragchecker.hallucination"): "correct",  # 0.0
    }
    undeclared, _ = lines(("f1",))
    assert not any("verdict" in j for j in undeclared)


@pytest.mark.parametrize(
    ("metrics", "thresholds", "message"),
    [
        (("claim_recall",), None, "measure retrieval"),
        (("context_precision", "f1"), None, "measure retrieval"),
        (("bleu",), None, "Unknown metric"),
        (("f1",), {"ragchecker.recall": 0.5}, "not a chosen metric"),
        (("self_knowledge",), {"self_knowledge": 0.5}, "neither good nor bad"),
        (("f1",), {"f1": float("nan")}, "finite"),
    ],
)
def test_unusable_metrics_and_pass_marks_are_refused(metrics, thresholds, message):
    with pytest.raises(RagCheckerImportError, match=message):
        lines(metrics, thresholds=thresholds)
    with pytest.raises(RagCheckerImportError, match="must name the model"):
        to_judgments(ROWS, [], QUESTIONS, run_documents=DOCUMENTS, extractor=" ", checker="x")


def _row(**changes):
    row = copy.deepcopy(ROWS[1])
    row.update(changes)
    return row


@pytest.mark.parametrize(
    "bad",
    [
        {"answer2response": ["Entailment", "Maybe", "Neutral", "Neutral", "Neutral"]},
        {"answer2response": ["Entailment"]},  # no longer one label per response claim
        {"retrieved2answer": [["Neutral"] * 2] * 8},  # not one column per chunk
        {"retrieved2response": "Entailment"},
        {"retrieved_context": "text"},
        {"response2answer": None},
        {"retrieved2response": [["Neutral", "Neutral", None]] * 5},
    ],
)
def test_malformed_rows_are_refused_and_counted(bad):
    out, counts = lines(rows=[ROWS[0], _row(**bad)])
    assert counts["malformed"] == 1
    assert {j["query_id"] for j in out} == {"q0"}


def _third(chunk):
    return {"retrieved_context": [{"doc_id": "000"}, {"doc_id": "001"}, chunk]}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (_third({"doc_id": "zzz"}), "unknown_doc_id"),
        (_third({"text": "x"}), "missing_doc_id"),
        (_third({"doc_id": "001"}), "duplicate_doc_id"),
    ],
)
def test_chunks_that_do_not_match_the_run_disable_chunk_evidence_but_keep_scores(change, reason):
    out, counts = lines(rows=[_row(**change)])
    assert [j["evaluator"] for j in out] == ["ragchecker.f1"]
    assert counts[f"chunk_evidence_disabled_{reason}"] == 1


def test_without_a_run_or_for_a_question_missing_from_it_only_scores_are_written():
    out, counts = lines(documents=None)
    assert {j["evaluator"] for j in out} == {"ragchecker.f1"}
    assert counts["chunk_evidence_disabled_no_run"] == 2
    out, counts = lines(documents={"q1": DOCUMENTS["q1"]})
    assert [j["evaluator"] for j in out if j["query_id"] == "q0"] == ["ragchecker.f1"]
    assert counts["chunk_evidence_disabled_not_in_run"] == 1
    no_chunks = _row(retrieved_context=[], retrieved2answer=[[]] * 8, retrieved2response=[[]] * 5)
    _, counts = lines(rows=[no_chunks])
    assert counts["chunk_evidence_disabled_no_context"] == 1


def test_questions_match_by_id_then_unique_text_and_unmatched_rows_are_counted():
    unknown = copy.deepcopy(ROWS[0]) | {"query_id": "zzz", "query": "nope"}
    out, counts = lines(rows=[_row(query_id="q1"), unknown])
    assert counts["matched_by_query_id"] == 1
    assert counts["unmatched"] == 1
    assert {j["query_id"] for j in out} == {"q1"}
    twins = {"a": "same question", "b": "Same  Question"}
    _, counts = to_judgments(
        [_row(query="SAME question", query_id="x")], ["f1"], twins, run_documents=None, **MODELS
    )
    assert counts["ambiguous_text"] == 1
    out, counts = lines(rows=[ROWS[0], ROWS[0]])
    assert counts["duplicate_question"] == 1
    assert len({j["query_id"] for j in out}) == 1


def test_metric_names_list_what_ragchecker_recorded():
    assert metric_names(ROWS) == dict.fromkeys(ALL_METRICS, 2)
    assert not metric_names([{"query_id": "x"}])


def test_run_documents_are_the_delivered_and_model_context_ids():
    run = fake_run([1.0, 1.0])
    run["queries"][0]["ids"] = ["a", "b"]
    run["queries"][1]["model_context"] = {"coverage": "observed", "items": [{"id": "m"}]}
    run["queries"][1]["ids"] = ["c"]
    assert run_documents(run) == {"q0": {"a", "b"}, "q1": {"c", "m"}}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def private_strings():
    private = []
    for row in ROWS:
        for key in ("query", "gt_answer", "response", "response_claims", "gt_answer_claims"):
            private += [s for s in _strings(row[key]) if len(s) >= 8]
        private += [c["text"] for c in row["retrieved_context"]]
    assert len(private) > 30
    return private


def test_no_claim_response_or_chunk_text_reaches_a_judgment():
    out, _ = lines(tuple(ALL_METRICS[:3] + ALL_METRICS[6:]))
    written = json.dumps(out)
    assert not [s for s in private_strings() if s in written]


def _world(tmp_path):
    ds = tmp_path / "benchmark.jsonl"
    ds.write_text(
        "\n".join(
            json.dumps({"query_id": qid, "text": text, "relevant": {"000": 1}})
            for qid, text in QUESTIONS.items()
        )
    )
    run = fake_run([0.9, 0.8])
    run["search"]["k"] = run["metric_contract"]["k"] = 4
    delivered = (["000", "001", "002", "003"], ["000", "001", "002"])
    for q, docs in zip(run["queries"], delivered, strict=True):
        q["ids"] = docs
    run_path = tmp_path / "current.run.json"
    run_path.write_text(json.dumps(run))
    return ds, run_path


def invoke(*args):
    return CliRunner().invoke(app, ["workbench", *args])


def test_the_round_trip_attaches_claim_evidence_without_changing_a_verdict(tmp_path):
    ds, run_path = _world(tmp_path)
    out = tmp_path / "judgments.jsonl"
    result = invoke(
        "ragchecker-judgments", str(FIXTURE), "--dataset", str(ds), "--run", str(run_path),
        "--extractor", "llama3-70b", "--checker", "llama3-70b",
        "--metrics", "f1,faithfulness,hallucination", "--threshold", "ragchecker.f1=0.6",
        "--out", str(out),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Wrote 8 judgments (2 claim lines)" in result.output
    rows = read_judgments(out)
    assert sum(j.claims is not None for _, _, j in rows) == 2

    run = json.loads(run_path.read_text())
    attached = attach_file(run, out)
    q0, q1 = attached["queries"]
    assert attached["answers"]["claim_evidence"]["questions"] == 2
    assert q0["answer"]["claim_evidence"]["claims"]["unsupported_incorrect"] == 1
    assert q0["answer"]["claim_evidence"]["claim_support"]["000"]["contradicts_reference"] == 1
    assert q0["answer"]["claim_evidence"]["evaluator"] == CLAIMS_EVALUATOR
    assert (q0["answer"]["verdict"], q1["answer"]["verdict"]) == ("incorrect", "correct")
    # The claim line decides nothing: the same file without it gives the same verdicts.
    plain = tmp_path / "plain.jsonl"
    plain.write_text(
        "\n".join(line for line in out.read_text().splitlines() if CLAIMS_EVALUATOR not in line)
    )
    without = attach_file(run, plain)
    assert [q["answer"]["verdict"] for q in without["queries"]] == ["incorrect", "correct"]
    assert [q["answer"].get("diagnosis") for q in without["queries"]] == [
        q["answer"].get("diagnosis") for q in attached["queries"]
    ]
    # Nothing private reaches the judgments file or the attached run.
    for text in (out.read_text(), json.dumps(attached)):
        assert not [s for s in private_strings() if s in text]


def test_nothing_private_reaches_the_audit_report_or_an_accepted_dataset(tmp_path):
    """The end of the chain: findings, review queue, proposals and the new dataset version."""
    from vectraudit.evaluation.dataset import parse_dataset
    from vectraudit.pipelines.benchmark import audit_benchmark
    from vectraudit.pipelines.claim_review import accept_evidence

    ds, run_path = _world(tmp_path)
    out = tmp_path / "judgments.jsonl"
    result = invoke(
        "ragchecker-judgments", str(FIXTURE), "--dataset", str(ds), "--run", str(run_path),
        "--extractor", "x", "--checker", "y", "--metrics", "f1", "--out", str(out),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    # The benchmark labels two documents per question, so proposals can be accepted.
    rows = [
        {"query_id": qid, "text": text, "relevant": {"000": 1, "001": 1, "003": 1}}
        for qid, text in QUESTIONS.items()
    ]
    benchmark = parse_dataset("\n".join(json.dumps(r) for r in rows))
    run = json.loads(run_path.read_text())
    run["dataset"]["sha256"] = benchmark.sha256
    attached = attach_file(run, out)
    audit = audit_benchmark(benchmark, [attached])
    assert audit["judgment_queue"]
    assert audit["required_evidence_proposals"]
    text, _ = accept_evidence(benchmark, audit, ["q0"])
    for written in (json.dumps(audit), json.dumps(attached)):
        assert not [s for s in private_strings() if s in written]
    # A dataset holds its own question text; accepting evidence must add none of the rest.
    own = "\n".join(json.dumps(r) for r in rows)
    added = [s for s in private_strings() if s not in own]
    assert len(added) > 30
    assert not [s for s in added if s in text]


def test_cli_reports_what_it_could_not_use_and_refuses_bad_input(tmp_path):
    ds, _ = _world(tmp_path)
    base = [
        "ragchecker-judgments", str(FIXTURE), "--dataset", str(ds),
        "--extractor", "x", "--checker", "y", "--out", str(tmp_path / "o.jsonl"),
    ]  # fmt: skip
    no_run = invoke(*base, "--metrics", "f1")
    assert no_run.exit_code == 0
    assert "2 rows scored without chunk evidence (no --run)" in no_run.output
    for args, message in [
        (["--metrics", "claim_recall"], "measure retrieval"),
        (["--metrics", "f1", "--threshold", "f1"], "expects ragchecker.METRIC=VALUE"),
        (["--metrics", "f1", "--run", str(tmp_path / "missing.json")], "Run file not found"),
    ]:
        failed = invoke(*base, *args)
        assert failed.exit_code == 2, failed.output
        assert message in failed.output
    garbage = tmp_path / "garbage.json"
    garbage.write_text('{"nothing": 1}')
    bad = invoke(
        "ragchecker-judgments", str(garbage), "--dataset", str(ds),
        "--extractor", "x", "--checker", "y", "--out", str(tmp_path / "o.jsonl"),
    )  # fmt: skip
    assert bad.exit_code == 2
    assert "not a RAGChecker output file" in bad.output


CLAIMS = {
    "reference": 4, "response": 5, "reference_in_context": 3, "reference_in_response": 2,
    "unsupported_incorrect": 1, "unsupported_correct": 1, "from_irrelevant_incorrect": 0,
    "from_relevant_incorrect": 1,
}  # fmt: skip
SUPPORT = {"doc-12": {"reference": 2, "response": 3, "contradicts_reference": 0}}


def line(**fields):
    base = {"query_id": "q", "evaluator": "ragchecker.claims", "kind": "model"}
    return json.dumps({**base, **fields})


def read(tmp_path, *text):
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(text))
    return read_judgments(path)


def test_the_reader_accepts_a_claims_only_line_as_the_plan_shows_it(tmp_path):
    ((_, _, judgment),) = read(tmp_path, line(claims=CLAIMS, claim_support=SUPPORT))
    assert judgment.verdict is None
    assert judgment.score is None
    assert dict(judgment.claims) == CLAIMS
    assert dict(judgment.claim_support[0][1]) == SUPPORT["doc-12"]


def _support(**changes):
    return {"claims": CLAIMS, "claim_support": {"d": {**SUPPORT["doc-12"], **changes}}}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"claim_support": SUPPORT}, "claim_support needs claims"),
        ({"claims": {**CLAIMS, "extra": 1}}, "unknown extra"),
        ({"claims": {k: v for k, v in CLAIMS.items() if k != "response"}}, "lacks response"),
        ({"claims": {**CLAIMS, "reference": -1}}, "whole number"),
        ({"claims": {**CLAIMS, "reference": True}}, "whole number"),
        ({"claims": {**CLAIMS, "reference": 1.5}}, "whole number"),
        ({"claims": {**CLAIMS, "reference": 10**9}}, "whole number"),
        ({"claims": "many"}, "object of counts"),
        ({"claims": {**CLAIMS, "reference_in_context": 9}}, "reference_in_context exceeds"),
        ({"claims": {**CLAIMS, "unsupported_incorrect": 5, "unsupported_correct": 5}}, "exceed"),
        ({"claims": {**CLAIMS, "response_incorrect": 1}}, "counted more than once"),
        (_support(reference=9), "more claims than were extracted"),
        ({"claims": CLAIMS, "claim_support": {"d": {"reference": 1}}}, "lacks"),
        ({"claims": CLAIMS, "claim_support": {"": SUPPORT["doc-12"]}}, "claim_support document ID"),
        ({"claims": CLAIMS, "claim_support": [1]}, "must map"),
    ],
)
def test_the_reader_bounds_and_checks_claim_evidence(tmp_path, fields, message):
    with pytest.raises(AnswerJudgmentError, match=message):
        read(tmp_path, line(**fields))


def _positions(positions, reference=2):
    doc = {"reference": reference, "response": 0, "contradicts_reference": 0}
    return {"claims": CLAIMS, "claim_support": {"d": {**doc, "reference_claims": positions}}}


def test_the_reader_keeps_the_positions_of_the_reference_claims_a_document_entails(tmp_path):
    ((_, _, judgment),) = read(tmp_path, line(**_positions([3, 1])))
    assert judgment.claim_support[0][1]["reference_claims"] == (1, 3)


@pytest.mark.parametrize(
    "positions",
    [[1, 1], [-1, 0], [4, 5], [True, 0], ["a", "b"], "all", [0], list(range(1001))],
)
def test_reference_claim_positions_must_be_distinct_and_match_the_count(tmp_path, positions):
    with pytest.raises(AnswerJudgmentError, match=r"reference_claims|lists"):
        read(tmp_path, line(**_positions(positions)))


def test_too_many_documents_are_refused(tmp_path):
    empty = {"reference": 0, "response": 0, "contradicts_reference": 0}
    support = {f"d{i}": empty for i in range(501)}
    with pytest.raises(AnswerJudgmentError, match="at most 500"):
        read(tmp_path, line(claims=CLAIMS, claim_support=support))


def test_two_claim_judgments_for_one_question_are_refused(tmp_path):
    path = tmp_path / "j.jsonl"
    path.write_text(
        "\n".join(
            [line(claims=CLAIMS, evaluator_version="a"), line(claims=CLAIMS, evaluator_version="b")]
        )
    )
    run = fake_run([1.0])
    run["queries"][0]["query_id"] = "q"
    with pytest.raises(AnswerJudgmentError, match="2 claim-evidence judgments"):
        attach_file(run, path)
