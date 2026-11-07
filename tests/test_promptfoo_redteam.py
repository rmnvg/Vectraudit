"""promptfoo red-team results, and fabricated citations by inventory lookup.

There is no offline way to run ``promptfoo redteam run`` (it needs a model to generate and to
grade), so the red-team files here are built from a REAL promptfoo eval result
(tests/fixtures/promptfoo/results.json): its structure is promptfoo's own, and only the test
metadata (``pluginId``, ``strategyId``) and the outcomes are set.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.answers import (
    AnswerJudgmentError,
    attach_file,
    read_judgments,
    render_summary,
)
from vectraudit.evaluation.promptfoo_redteam_import import (
    PLUGINS,
    import_redteam,
    load_results,
    plugin_counts,
)

REAL = json.loads((Path(__file__).parent / "fixtures/promptfoo/results.json").read_text())
QUESTIONS = {"q1": "How do I reset my password?", "q2": "Where is the invoice?"}


def redteam_file(tmp_path, specs):
    """specs: (plugin, strategy, outcome, key) with outcome pass, fail or error."""
    template = REAL["results"]["results"][0]
    results = []
    for plugin, strategy, outcome, key in specs:
        result = copy.deepcopy(template)
        metadata = {"pluginId": plugin, **({"strategyId": strategy} if strategy else {})}
        if key:
            metadata[key[0]] = key[1]
        result["testCase"]["metadata"] = metadata
        result["metadata"] = dict(metadata)
        result["success"] = outcome == "pass"
        result["gradingResult"]["pass"] = outcome == "pass"
        result["failureReason"] = 2 if outcome == "error" else (0 if outcome == "pass" else 1)
        results.append(result)
    data = copy.deepcopy(REAL)
    data["results"]["results"] = results
    path = tmp_path / "redteam.json"
    path.write_text(json.dumps(data))
    return path


SPECS = [
    ("rag-poisoning", "basic", "fail", ("query_id", "q1")),
    ("rag-poisoning", "base64", "pass", None),
    ("rag-poisoning", "base64", "fail", None),
    ("rag-poisoning", None, "error", None),
    ("indirect-prompt-injection", "basic", "pass", ("trace_id", "abc123")),
    ("prompt-extraction", "rot13", "pass", ("query_id", "q2")),
    ("harmful:hate", "basic", "fail", None),  # not a RAG plugin
]


def run_import(tmp_path, specs=SPECS):
    return import_redteam(load_results(redteam_file(tmp_path, specs)), QUESTIONS)


def test_counts_per_plugin_and_strategy_with_a_failure_interval(tmp_path):
    _, report = run_import(tmp_path)
    assert report["format"] == "vectraudit.promptfoo_redteam"
    assert report["evidence"] == "supported"
    poisoning = report["plugins"]["rag-poisoning"]
    assert (poisoning["tests"], poisoning["passed"], poisoning["failed"], poisoning["errored"]) == (
        3,
        1,
        2,
        1,
    )
    assert poisoning["failure_rate"] == pytest.approx(2 / 3)
    assert 0 < poisoning["ci_low"] < 2 / 3 < poisoning["ci_high"] < 1
    assert poisoning["by_strategy"] == {
        "base64": {"tests": 2, "failed": 1},
        "basic": {"tests": 1, "failed": 1},
    }
    assert report["plugins"]["indirect-prompt-injection"]["failed"] == 0
    assert set(report["plugins"]) == {
        "rag-poisoning",
        "indirect-prompt-injection",
        "prompt-extraction",
    }
    assert report["summary"]["skipped_other_plugins"] == {"harmful:hate": 1}
    assert report["summary"]["errored"] == 1
    assert report["summary"]["results"] == len(SPECS)


def test_only_a_named_question_or_trace_becomes_a_per_question_record(tmp_path):
    records, report = run_import(tmp_path)
    assert {(r.get("query_id") or r.get("trace_id")) for r in records} == {"q1", "abc123", "q2"}
    first = next(r for r in records if r.get("query_id") == "q1")
    assert first == {
        "query_id": "q1",
        "evaluator": "promptfoo.redteam.rag-poisoning",
        "kind": "safety",
        "evaluator_version": "promptfoo red team; pass means the system resisted",
        "redteam": {"plugin": "rag-poisoning", "strategy": "basic", "passed": False},
    }
    s = report["summary"]
    assert (s["matched_by_query_id"], s["matched_by_trace_id"], s["unmatched"]) == (2, 1, 2)
    assert report["plugins"]["rag-poisoning"]["matched_to_questions"] == 1


def test_generated_prompts_and_unknown_ids_are_never_matched_by_text_or_guesses(tmp_path):
    specs = [
        ("rag-poisoning", "basic", "fail", ("query_id", "not-in-the-benchmark")),
        ("rag-poisoning", "basic", "fail", ("question", QUESTIONS["q1"])),  # text never matches
    ]
    records, report = run_import(tmp_path, specs)
    assert records == []
    assert report["summary"]["unmatched"] == 2


def test_nothing_private_from_the_results_is_written(tmp_path):
    records, report = run_import(tmp_path)
    written = json.dumps([records, report])
    template = REAL["results"]["results"][0]
    private = [
        template["gradingResult"]["reason"],
        template["response"]["output"],
        json.dumps(template["prompt"]),
        template["testCase"]["vars"]["question"],
    ]
    assert all(len(p) > 10 for p in private)
    assert not [p for p in private if p in written]


def test_a_file_without_a_rag_plugin_is_refused_and_lists_what_it_has(tmp_path):
    path = redteam_file(tmp_path, [("harmful:hate", "basic", "fail", None)] * 2)
    with pytest.raises(ValueError, match=r"harmful:hate \(2\)"):
        import_redteam(load_results(path), QUESTIONS)
    assert plugin_counts(load_results(path)) == {"harmful:hate": 2}
    assert set(PLUGINS) == {
        "rag-poisoning",
        "rag-document-exfiltration",
        "indirect-prompt-injection",
        "rag-source-attribution",
        "prompt-extraction",
    }


def test_records_attach_as_safety_apart_from_the_verdict_and_summarise_per_plugin(tmp_path):
    records, _ = run_import(tmp_path)
    judgments = tmp_path / "safety.jsonl"
    human = {"query_id": "q1", "evaluator": "review", "kind": "human", "verdict": "correct"}
    judgments.write_text("\n".join(json.dumps(r) for r in [*records, human]))
    run = fake_run([1.0, 1.0])
    run["queries"][0]["query_id"] = "q1"
    run["queries"][1]["query_id"] = "q2"
    attached = attach_file(run, judgments)
    q1 = attached["queries"][0]["answer"]
    assert q1["verdict"] == "correct"  # a failed red-team test is not an answer verdict
    assert q1["redteam"] == [
        {
            "plugin": "rag-poisoning",
            "strategy": "basic",
            "passed": False,
            "evaluator": "promptfoo.redteam.rag-poisoning",
            "evaluator_version": "promptfoo red team; pass means the system resisted",
        }
    ]
    assert attached["answers"]["redteam"] == {
        "prompt-extraction": {"tests": 1, "failed": 0},
        "rag-poisoning": {"tests": 1, "failed": 1},
    }
    text = render_summary(attached)
    assert "Red-team rag-poisoning: 1 of 1 tests matched to questions failed" in text


def line(**fields):
    base = {"query_id": "q", "evaluator": "x", "kind": "safety"}
    return json.dumps({**base, **fields})


def read(tmp_path, *text):
    path = tmp_path / "j.jsonl"
    path.write_text("\n".join(text))
    return read_judgments(path)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"redteam": {"plugin": "rag-poisoning", "passed": True}, "safety": {}}, "exactly one"),
        ({}, "exactly one"),
        (
            {"redteam": {"plugin": "rag-poisoning", "passed": True}, "verdict": "correct"},
            "no verdict",
        ),
        (
            {"kind": "model", "redteam": {"plugin": "p", "passed": True}, "verdict": "correct"},
            "nothing else",
        ),
        ({"redteam": {"plugin": "rag-poisoning"}}, "plugin and passed"),
        ({"redteam": {"plugin": "rag-poisoning", "passed": "no"}}, "true or false"),
        ({"redteam": {"plugin": "Bad Plugin!", "passed": True}}, "plugin ID"),
        ({"redteam": {"plugin": "p", "passed": True, "strategy": "bad strategy"}}, "strategy ID"),
        ({"redteam": {"plugin": "p", "passed": True, "extra": 1}}, "plugin and passed"),
        ({"redteam": "x"}, "plugin and passed"),
    ],
)
def test_the_reader_checks_red_team_records_and_keeps_them_out_of_verdicts(
    tmp_path, fields, message
):
    with pytest.raises(AnswerJudgmentError, match=message):
        read(tmp_path, line(**fields))


def test_the_reader_accepts_a_red_team_record_without_a_strategy(tmp_path):
    ((_, _, judgment),) = read(tmp_path, line(redteam={"plugin": "rag-poisoning", "passed": True}))
    assert judgment.redteam == {"plugin": "rag-poisoning", "strategy": None, "passed": True}
    assert judgment.verdict is None


def test_the_cli_writes_records_and_a_report_and_refuses_bad_input(tmp_path):
    results = redteam_file(tmp_path, SPECS)
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"query_id": q, "text": t, "relevant": {"d": 1}})
            for q, t in QUESTIONS.items()
        )
    )
    out, report = tmp_path / "safety.jsonl", tmp_path / "redteam.json"
    run = CliRunner().invoke(
        app,
        [
            "workbench",
            "promptfoo-redteam",
            str(results),
            str(dataset),
            "--out",
            str(out),
            "--report",
            str(report),
        ],
    )
    assert run.exit_code == 0, run.output
    assert "Kept 6 of 7 red-team results; wrote 3 per-question safety records" in run.output
    assert "rag-poisoning: 2 of 3 failed (67%" in run.output
    assert "exploratory evidence, not a release approval" in run.output
    assert len(read_judgments(out)) == 3
    assert json.loads(report.read_text())["format"] == "vectraudit.promptfoo_redteam"
    plain = tmp_path / "eval.json"
    plain.write_text(json.dumps(REAL))
    refused = CliRunner().invoke(
        app, ["workbench", "promptfoo-redteam", str(plain), str(dataset), "--out", str(out)]
    )
    assert refused.exit_code == 2
    assert "No result is from a RAG red-team plugin" in refused.output
    missing = CliRunner().invoke(
        app,
        [
            "workbench",
            "promptfoo-redteam",
            str(tmp_path / "x.json"),
            str(dataset),
            "--out",
            str(out),
        ],
    )
    assert missing.exit_code == 2


def citations_run(tmp_path, cited):
    run = fake_run([1.0])
    run["queries"][0].update(query_id="q", ids=["d1", "d2"])
    path = tmp_path / "j.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "evaluator": "r",
                "kind": "human",
                "verdict": "correct",
                "citations": cited,
            }
        )
    )
    return run, path


def test_a_cited_id_outside_the_inventory_is_a_fabricated_citation_found_by_lookup(tmp_path):
    run, path = citations_run(tmp_path, ["d1", "memory-doc", "ghost-doc"])
    inventory = {"d1", "d2", "memory-doc"}
    attached = attach_file(run, path, inventory=inventory)
    classes = attached["queries"][0]["answer"]["citations"]["classes"]
    assert classes == {
        "d1": "delivered_context_unknown",
        "ghost-doc": "not_in_corpus",
        "memory-doc": "not_delivered",  # in the corpus, so legitimate memory, not fabricated
    }
    cites = attached["answers"]["citations"]
    assert cites["questions_citing_outside_corpus"] == 1
    assert cites["inventory_documents"] == 3
    assert cites["questions_citing_undelivered"] == 1
    text = render_summary(attached)
    assert "1 questions cite an ID absent from the 3-document corpus inventory: fabricated" in text


def test_without_an_inventory_citations_are_exactly_as_before(tmp_path):
    run, path = citations_run(tmp_path, ["d1", "ghost-doc"])
    attached = attach_file(run, path)
    classes = attached["queries"][0]["answer"]["citations"]["classes"]
    assert classes == {"d1": "delivered_context_unknown", "ghost-doc": "not_delivered"}
    assert "questions_citing_outside_corpus" not in attached["answers"]["citations"]
    assert "inventory" not in render_summary(attached)


def test_the_cli_reads_the_corpus_inventory_and_refuses_a_bad_one(tmp_path):
    run, judgments = citations_run(tmp_path, ["d1", "ghost-doc"])
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run))
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text('{"id": "d1"}\n{"id": "d2", "sha256": "ab"}\n')
    out = tmp_path / "out.json"
    base = ["workbench", "attach-answers", str(run_path), str(judgments), "--out", str(out)]
    ok = CliRunner().invoke(app, [*base, "--inventory", str(inventory)])
    assert ok.exit_code == 0, ok.output
    assert "fabricated citations" in ok.output
    classes = json.loads(out.read_text())["queries"][0]["answer"]["citations"]["classes"]
    assert classes["ghost-doc"] == "not_in_corpus"
    inventory.write_text('{"id": "d1", "text": "private"}\n')
    bad = CliRunner().invoke(app, [*base, "--inventory", str(inventory)])
    assert bad.exit_code == 2
    assert "no text" in bad.output
