"""Ragas evaluation results as answer judgments, tested on real Ragas 0.4.3 output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.ragas_import import (
    RagasImportError,
    is_retrieval_metric,
    load_results,
    metric_names,
    render,
    to_judgments,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ragas"
QUESTIONS = {
    "q1": "How do I reset my password?",
    "q2": "Where is the invoice?",
    "q3": "cancel  MY plan",
}
METRICS = ["exact_match", "string_present", "non_llm_string_similarity", "bleu_score",
           "rouge_score(mode=fmeasure)", "non_llm_context_recall"]  # fmt: skip


@pytest.mark.parametrize("name", ["results.csv", "results.jsonl", "results.json"])
def test_every_saved_layout_reads_the_same_rows(name):
    rows = load_results(FIXTURES / name)
    assert len(rows) == 4
    assert list(metric_names(rows)) == METRICS
    assert all(n == 4 for n in metric_names(rows).values())
    judgments, counts = to_judgments(rows, ["exact_match"], QUESTIONS)
    assert [(j["query_id"], j["score"]) for j in judgments] == [
        ("q1", 1.0),
        ("q2", 0.0),
        ("q3", 0.0),
    ]
    assert counts["matched_by_text"] == 3
    assert counts["unmatched"] == 1  # the refund question is not in the benchmark


def test_judgments_carry_only_the_score_and_never_the_text():
    rows = load_results(FIXTURES / "results.jsonl")
    judgments, _ = to_judgments(rows, ["ragas.bleu_score", "rouge_score(mode=fmeasure)"],
                                QUESTIONS)  # fmt: skip
    assert {j["evaluator"] for j in judgments} == {
        "ragas.bleu_score",
        "ragas.rouge_score(mode=fmeasure)",
    }
    for j in judgments:
        assert set(j) == {"query_id", "evaluator", "kind", "score"}
        assert j["kind"] == "model"
    text = json.dumps(judgments)
    assert "Settings" not in text
    assert "Billing" not in text


def test_failed_metrics_are_skipped_and_counted_in_csv_and_jsonl():
    # A real run whose LLM judge was unreachable: Ragas leaves the cell empty (CSV) or null.
    for name in ("failed.csv", "failed.jsonl"):
        rows = load_results(FIXTURES / name)
        assert metric_names(rows) == {"exact_match": 2, "faithfulness": 0}
        judgments, counts = to_judgments(rows, ["exact_match", "faithfulness"], QUESTIONS)
        assert [j["evaluator"] for j in judgments] == ["ragas.exact_match"] * 2
        assert counts["not_scored"] == 2


def test_context_metrics_are_refused_and_unknown_metrics_listed():
    rows = load_results(FIXTURES / "results.csv")
    for refused in ("non_llm_context_recall", "context_precision", "ragas.context_utilization"):
        assert is_retrieval_metric(refused.removeprefix("ragas."))
        with pytest.raises(RagasImportError, match="judge the retrieved context"):
            to_judgments(rows, [refused], QUESTIONS)
    assert not is_retrieval_metric("faithfulness")
    with pytest.raises(RagasImportError, match="No numeric metric column faithfulness"):
        to_judgments(rows, ["faithfulness"], QUESTIONS)
    with pytest.raises(RagasImportError, match="No numeric metric column response"):
        to_judgments(rows, ["response"], QUESTIONS)  # a sample field, not a metric
    with pytest.raises(RagasImportError, match="Choose the metrics"):
        to_judgments(rows, [], QUESTIONS)


def test_ids_added_before_saving_win_over_text_and_ambiguity_is_counted():
    rows = load_results(FIXTURES / "results.jsonl")
    rows[0] = {**rows[0], "query_id": "q2"}
    rows[1] = {**rows[1], "trace_id": "t-9"}
    rows[2] = {**rows[2], "user_input": [{"content": "hi", "type": "human"}]}
    twice = {**QUESTIONS, "q4": "What is the  refund window?", "q5": "what is the refund window"}
    judgments, counts = to_judgments(rows, ["exact_match"], twice)
    assert [j.get("query_id") or j.get("trace_id") for j in judgments] == ["q2", "t-9"]
    assert counts["matched_by_query_id"] == 1
    assert counts["matched_by_trace_id"] == 1
    assert counts["multi_turn"] == 1
    assert counts["ambiguous_text"] == 1


def test_render_names_the_thresholds_attach_answers_needs():
    text = render({"matched_by_text": 3, "unmatched": 1, "not_scored": 2}, 6, "j.jsonl",
                  ["exact_match", "ragas.bleu_score"])  # fmt: skip
    assert "Wrote 6 score judgments to j.jsonl." in text
    assert "1 rows matching no question" in text
    assert "2 scores missing or failed (skipped)" in text
    assert "--threshold ragas.exact_match=VALUE --threshold ragas.bleu_score=VALUE" in text


def test_scores_become_verdicts_only_with_a_declared_threshold(tmp_path):
    from tests.test_answer_judgments import run
    from vectraudit.evaluation.answers import attach_answers, read_judgments

    rows = load_results(FIXTURES / "results.csv")
    judgments, _ = to_judgments(rows, ["non_llm_string_similarity"],
                                {"right": "Where is the invoice?"})  # fmt: skip
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in judgments))

    def verdict(thresholds):
        out = attach_answers(run(), read_judgments(path, thresholds))
        return next(q for q in out["queries"] if q["query_id"] == "right")["answer"]["verdict"]

    assert verdict({}) not in ("correct", "incorrect")
    assert verdict({"ragas.non_llm_string_similarity": 0.5}) == "correct"  # score is 0.5
    assert verdict({"ragas.non_llm_string_similarity": 0.8}) == "incorrect"


def test_other_files_are_refused(tmp_path):
    cases = {
        "a.json": json.dumps({"testCases": []}),
        "b.json": json.dumps([]),
        "c.jsonl": json.dumps({"input": "x", "score": 1}),
        "d.csv": "input,score\nx,1\n",
    }
    for name, text in cases.items():
        path = tmp_path / name
        path.write_text(text)
        with pytest.raises(RagasImportError, match="not a Ragas result table"):
            load_results(path)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{nope")
    with pytest.raises(RagasImportError, match="not JSON"):
        load_results(broken)
    with pytest.raises(RagasImportError, match="Cannot read"):
        load_results(tmp_path / "missing.csv")


def test_cli_lists_metrics_then_writes(tmp_path):
    data = tmp_path / "questions.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": q, "text": t, "relevant": {"d1": 1}})
            for q, t in QUESTIONS.items()
        )
    )
    out = tmp_path / "j.jsonl"
    runner = CliRunner()
    args = ["workbench", "ragas-judgments", str(FIXTURES / "results.csv"), str(data),
            "--out", str(out)]  # fmt: skip
    listing = runner.invoke(app, args)
    assert listing.exit_code == 2
    text = " ".join(listing.output.split())
    assert "exact_match (4), string_present (4)" in text
    refused = runner.invoke(app, [*args, "--metric", "non_llm_context_recall"])
    assert refused.exit_code == 2
    assert "judge the retrieved context" in " ".join(refused.output.split())
    done = runner.invoke(app, [*args, "--metric", "exact_match", "--metric", "bleu_score"])
    assert done.exit_code == 0, done.output
    assert "Wrote 6 score judgments" in done.output
    assert "--threshold ragas.exact_match=VALUE" in done.output
    assert len(out.read_text().splitlines()) == 6
