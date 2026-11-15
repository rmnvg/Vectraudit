"""promptfoo eval results as answer judgments, tested on a real promptfoo results file."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.promptfoo_import import (
    PromptfooImportError,
    assertion_names,
    load_results,
    to_judgments,
    variants,
)

RESULTS = Path(__file__).parent / "fixtures" / "promptfoo" / "results.json"
QUESTIONS = {
    "q1": "How do I reset my password?",
    "q2": "Where is the invoice?",
    "q3": "Cancel my plan",
}


def real():
    return load_results(RESULTS)


def test_the_real_file_holds_two_variants_three_tests_and_two_repeats():
    results = real()
    assert len(results) == 12
    assert variants(results) == {("echo-model", "concise"): 6, ("echo-model", "detailed"): 6}
    assert assertion_names(results) == {"Correctness": 12, "Brevity": 4}


def test_one_variant_is_required_when_the_eval_compares_several():
    with pytest.raises(PromptfooImportError, match="compare several variants"):
        to_judgments(real(), ["Correctness"], QUESTIONS)
    with pytest.raises(PromptfooImportError, match="No results match"):
        to_judgments(real(), ["Correctness"], QUESTIONS, prompt="missing")


def test_failures_are_imported_not_mistaken_for_errors():
    # promptfoo sets "error" to the assertion message on every ordinary failure; reading it
    # as an error would drop all failing answers and leave only passes.
    rows, counts = to_judgments(real(), ["Correctness"], QUESTIONS, prompt="detailed")
    verdicts = sorted((r["query_id"], r["verdict"]) for r in rows)
    assert verdicts == [("q1", "incorrect"), ("q1", "incorrect"), ("q2", "incorrect"),
                        ("q2", "incorrect")]  # fmt: skip
    assert counts["errored"] == 0
    assert counts["unmatched"] == 2  # the question not in the benchmark, twice


def test_the_concise_prompt_passes_where_it_mentions_settings():
    rows, counts = to_judgments(real(), ["Correctness", "Brevity"], QUESTIONS, prompt="concise")
    by = {(r["query_id"], r["evaluator"]) for r in rows if r["verdict"] == "correct"}
    assert ("q1", "Correctness") in by
    assert ("q1", "Brevity") in by
    assert counts["matched_by_query_id"] == 2  # q1 carries metadata.query_id
    assert counts["matched_by_text"] == 2  # q2 is matched by its question variable
    assert all(set(r) <= {"query_id", "evaluator", "kind", "verdict", "score"} for r in rows)


def test_overall_uses_the_tests_own_pass_but_never_with_a_retrieval_assertion():
    rows, _ = to_judgments(real(), ["overall"], QUESTIONS, prompt="concise")
    assert {r["evaluator"] for r in rows} == {"promptfoo"}
    with_retrieval = copy.deepcopy(real())
    for result in with_retrieval:
        result["gradingResult"]["componentResults"].append(
            {"pass": True, "score": 1, "assertion": {"type": "context-recall", "value": "x"}}
        )
    rows, counts = to_judgments(with_retrieval, ["overall"], QUESTIONS, prompt="concise")
    assert rows == []
    assert counts["overall_includes_retrieval"] == 4
    for refused in ("context-recall", "context-relevance"):
        with pytest.raises(PromptfooImportError, match="judge the retrieved context"):
            to_judgments(real(), [refused], QUESTIONS, prompt="concise")


def test_a_real_error_is_skipped_and_counted():
    results = copy.deepcopy(real())
    for result in results:
        result["failureReason"] = 2
        result["error"] = "provider timed out"
    rows, counts = to_judgments(results, ["Correctness"], QUESTIONS, prompt="concise")
    assert rows == []
    assert counts["errored"] == 6


def test_the_judgments_attach_with_disagreeing_repeats_as_disputes(tmp_path):
    from tests.test_answer_judgments import run
    from vectraudit.evaluation.answers import attach_answers, read_judgments

    rows, _ = to_judgments(real(), ["Correctness"], {"right": "Where is the invoice?"},
                           prompt="concise")  # fmt: skip
    rows[0] = {**rows[0], "verdict": "correct"}  # one repeat says correct, the other not
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    out = attach_answers(run(), read_judgments(path))
    right = next(q for q in out["queries"] if q["query_id"] == "right")
    assert right["answer"]["verdict"] == "disputed"


def test_other_files_are_refused(tmp_path):
    for bad in ({"results": []}, {"results": {"version": 3}}, ["x"]):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(PromptfooImportError, match="not promptfoo eval output"):
            load_results(path)
    broken = tmp_path / "broken.json"
    broken.write_text("{nope")
    with pytest.raises(PromptfooImportError, match="not JSON"):
        load_results(broken)


def test_cli_lists_choices_requires_a_variant_and_writes(tmp_path):
    data = tmp_path / "questions.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": q, "text": t, "relevant": {"d1": 1}})
            for q, t in QUESTIONS.items()
        )
    )
    out = tmp_path / "j.jsonl"
    runner = CliRunner()
    args = ["workbench", "promptfoo-judgments", str(RESULTS), str(data), "--out", str(out)]
    listing = runner.invoke(app, args)
    assert listing.exit_code == 2
    text = " ".join(listing.output.split())
    assert "Assertions: Correctness (12), Brevity (4)" in text
    assert "echo-model / concise (6)" in text
    several = runner.invoke(app, [*args, "--assertion", "Correctness"])
    assert several.exit_code == 2
    assert "compare several variants" in " ".join(several.output.split())
    done = runner.invoke(app, [*args, "--assertion", "Correctness", "--prompt", "detailed"])
    assert done.exit_code == 0, done.output
    assert "Wrote 4 judgments" in done.output
    assert len(out.read_text().splitlines()) == 4
