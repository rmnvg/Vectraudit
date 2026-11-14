"""The component scoreboard places each question's first loss on the recorded path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_answer_judgments import ALL_WRONG, judged, write
from tests.test_answer_judgments import run as planted_run
from vectraudit.evaluation.answers import attach_file
from vectraudit.evaluation.components import (
    attribute,
    compare_components,
    component_scoreboard,
)
from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.evaluation.dataset_tools import render_dataset
from vectraudit.pipelines import (
    Hit,
    PipelineOptions,
    PipelineRequest,
    PipelineResult,
    StageTrace,
    execute_pipeline,
)

RELEVANT = {"a": 1, "b": 1}


def rows(component_tags=None):
    """Ten questions; q0-q1 form one family. The rest are described in BrokenReranker."""
    out = []
    for i in range(10):
        row = {"query_id": f"q{i}", "text": f"question {i}", "relevant": RELEVANT}
        if i < 2:
            row["family_id"] = "fam"
        if i in (6, 7):
            row["tags"] = ["multi_hop"]
        out.append(row)
    out.append({"query_id": "none", "text": "out of scope", "no_answer": True})
    out.append({"query_id": "plain", "text": "no labels yet"})
    meta = {"component_tags": component_tags} if component_tags else None
    return parse_dataset(render_dataset(out, meta))


def stage(name, kind, ids, parents=()):
    return StageTrace(
        name=name, kind=kind, parents=list(parents), hits=[Hit(id=i) for i in ids], latency_ms=1.0
    )


class BrokenReranker:
    """Search finds the relevant documents; the reranker throws them away.

    q0-q5: retrieved, then dropped by the reranker. q6: never retrieved. q7: kept.
    q8-q9: no stage trace and nothing relevant delivered, so not observed.
    """

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        i = request.query_id
        if i in ("q8", "q9"):
            return PipelineResult(hits=[Hit(id="x")])
        found = ["x", "y"] if i == "q6" else ["a", "b", "x", "y"]
        kept = ["a", "x"] if i == "q7" else ["x", "y"]
        stages = [stage("search", "vector", found), stage("rerank", "rerank", kept, ["search"])]
        return PipelineResult(hits=[Hit(id=d) for d in kept], stages=stages)


def broken_run(component_tags=None):
    return execute_pipeline(rows(component_tags), BrokenReranker(), PipelineOptions(k=2, warmup=0))


def by_component(board):
    return {c["component"]: c for c in board["components"]}


def test_a_broken_reranker_is_where_the_loss_is_and_untraced_questions_are_not_observed():
    board = component_scoreboard(broken_run())
    assert board["format"] == "vectraudit.component_scoreboard"
    assert board["declared"] == ["retrieve", "rerank"]
    table = by_component(board)
    assert table["rerank"]["questions"] == 6
    assert table["retrieve"]["questions"] == 1
    assert board["kept"]["questions"] == 1
    assert board["not_observed"] == 2  # never assigned to a component
    assert board["not_applicable"] == {"unlabelled": 1, "no_answer": 1}
    assert board["observed"] == {"questions": 8, "families": 7}
    # Family-weighted: q0 and q1 are one family, so rerank holds 5 of 7 families.
    assert table["rerank"]["rate"] == round(5 / 7, 4)
    assert table["rerank"]["ci_low"] < table["rerank"]["rate"] < table["rerank"]["ci_high"]
    assert board["by_stage_kind"] == {"rerank": 6}
    assert max(board["components"], key=lambda c: c["rate"])["component"] == "rerank"


def test_each_planted_outcome_lands_on_its_component_after_answers_attach(tmp_path):
    run = planted_run()
    outcomes = {q["query_id"]: attribute(q)[0] for q in run["queries"]}
    # Before answers: nothing is judged, so delivered evidence is kept as far as recorded.
    assert outcomes == {
        "packed": "context",
        "wrong": "kept",
        "right": "kept",
        "dropped": "custom",
        "unseen": "retrieve",
        "unknown": "kept",
    }
    answered = attach_file(run, write(tmp_path, judged({**ALL_WRONG, "right": "correct"})))
    after = {q["query_id"]: attribute(q)[0] for q in answered["queries"]}
    assert after["wrong"] == "answer"  # judged wrong though the evidence reached the model
    assert after["right"] == "kept"
    assert after["unknown"] == "not_observed"  # wrong, but the context was not recorded
    board = component_scoreboard(answered)
    assert board["declared"] == ["retrieve", "custom", "context", "answer"]
    assert {c["component"]: c["questions"] for c in board["components"]} == {
        "retrieve": 1,
        "custom": 1,
        "context": 1,
        "answer": 1,
    }


def test_component_tags_give_a_cohort_its_own_rows():
    board = component_scoreboard(broken_run({"multi_hop": ["retrieve", "rerank"]}))
    (cohort,) = board["cohorts"]
    assert cohort["tag"] == "multi_hop"
    assert cohort["targets"] == ["retrieve", "rerank"]
    assert cohort["observed"]["questions"] == 2
    assert by_component(cohort)["retrieve"]["questions"] == 1  # q6, never retrieved
    assert cohort["lost_at_targets"] == 1
    assert cohort["kept"]["questions"] == 1  # q7


def test_component_tags_are_validated_in_the_header():
    with pytest.raises(DatasetError, match="component_tags"):
        rows({"multi_hop": ["rewrite"]})
    with pytest.raises(DatasetError, match="component_tags"):
        rows({"multi_hop": []})
    with pytest.raises(DatasetError, match="component_tags"):
        rows(["retrieve"])
    assert rows({"multi_hop": ["rerank"]}).meta["component_tags"] == {"multi_hop": ["rerank"]}


def test_a_run_without_stages_declares_nothing_and_holds_no_text():
    class Plain:
        def __call__(self, request):
            return PipelineResult(hits=[Hit(id="x")])

    board = component_scoreboard(execute_pipeline(rows(), Plain(), PipelineOptions(k=2, warmup=0)))
    assert board["declared"] == []
    assert board["not_observed"] == 10
    assert "question 1" not in json.dumps(board)
    with pytest.raises(ValueError, match=r"vectraudit\.run"):
        component_scoreboard({"format": "vectraudit.benchmark"})


def test_compare_shows_both_runs_as_information_only():
    from vectraudit.evaluation.compare import compare_runs
    from vectraudit.evaluation.policy import DEFAULT_POLICY

    class Fixed:
        def __call__(self, request):
            stages = [
                stage("search", "vector", ["a", "b"]),
                stage("rerank", "rerank", ["a", "b"], ["search"]),
            ]
            return PipelineResult(hits=[Hit(id="a"), Hit(id="b")], stages=stages)

    base = broken_run()
    cand = execute_pipeline(rows(), Fixed(), PipelineOptions(k=2, warmup=0))
    block = compare_components(base, cand)
    assert block["components"] == ["retrieve", "rerank"]
    assert block["baseline"]["rerank"]["questions"] == 6
    assert block["candidate"]["rerank"]["questions"] == 0
    result = compare_runs(base, cand, DEFAULT_POLICY)
    assert result["components"] == block
    assert all("component" not in r.get("rule", "") for r in result["reasons"])


def test_the_release_report_shows_the_components_as_information(tmp_path):
    from vectraudit.evaluation.compare import compare_runs
    from vectraudit.evaluation.policy import DEFAULT_POLICY
    from vectraudit.evaluation.report import render_html, render_markdown

    base = broken_run()
    result = compare_runs(base, broken_run(), DEFAULT_POLICY)
    markdown = render_markdown(result)
    assert "| rerank | 71.4% (" in markdown
    assert "| not observed | 2 | 2 |" in markdown
    assert "Information only" in markdown
    page = render_html(result)
    assert "<h2>Where questions first lost their evidence</h2>" in page


def test_components_command_prints_and_writes_the_scoreboard(tmp_path):
    from typer.testing import CliRunner

    from vectraudit.cli import app

    path = tmp_path / "broken.run.json"
    path.write_text(json.dumps(broken_run({"multi_hop": ["retrieve", "rerank"]})))
    out = tmp_path / "components.json"
    result = CliRunner().invoke(app, ["workbench", "components", str(path), "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "Component scoreboard (8 observed questions in 7 families" in result.output
    assert "  rerank     71.4% (95%:" in result.output
    assert (
        "Not observed: 2; not applicable: 1 unlabelled, 1 no-answer; errored: 0." in result.output
    )
    assert "Cohort multi_hop (targets retrieve, rerank): 1 of 2 observed questions" in result.output
    assert json.loads(out.read_text())["format"] == "vectraudit.component_scoreboard"
    bad = CliRunner().invoke(app, ["workbench", "components", str(tmp_path / "none.json")])
    assert bad.exit_code == 2


# --- the run view's TypeScript port is held to the same results --------------------------------

FIXTURE = Path(__file__).parents[1] / "frontend" / "src" / "lib" / "fixtures" / "components.json"
READ = ("query_id", "family_id", "tags", "error", "no_answer_correct", "metrics", "sufficiency")


def compact(run):
    """Only what the scoreboard reads, so the fixture stays small."""

    def entry(q):
        out = {k: q[k] for k in READ if k in q}
        if isinstance(q.get("evidence"), dict):
            ev = q["evidence"]
            out["evidence"] = {
                "stages": [{"name": s["name"], "kind": s["kind"]} for s in ev.get("stages") or []],
                "failures": ev.get("failures") or [],
                **({"context": ev["context"]} if "context" in ev else {}),
            }
        diagnosis = ((q.get("answer") or {}).get("diagnosis") or {}).get("class")
        if diagnosis:
            out["answer"] = {"diagnosis": {"class": diagnosis}}
        return out

    return {
        "format": run["format"],
        "run_id": "fixture",
        "label": run.get("label"),
        "dataset": {
            "meta": {k: v for k, v in run["dataset"]["meta"].items() if k == "component_tags"}
        },
        "queries": [entry(q) for q in run["queries"]],
    }


def fixture_runs(tmp_path):
    answered = attach_file(
        planted_run(), write(tmp_path, judged({**ALL_WRONG, "right": "correct"}))
    )
    return {
        "broken_reranker": broken_run({"multi_hop": ["retrieve", "rerank"]}),
        "planted_with_answers": answered,
    }


def fixture_content(tmp_path):
    content = {}
    for name, run in fixture_runs(tmp_path).items():
        small = compact(run)
        board = component_scoreboard(small)
        full = component_scoreboard(run)
        assert {k: v for k, v in board.items() if k != "run_id"} == {
            k: v for k, v in full.items() if k != "run_id"
        }, name
        content[name] = {"run": small, "scoreboard": board}
    return content


def test_the_typescript_fixture_matches_what_python_computes(tmp_path):
    assert json.loads(FIXTURE.read_text()) == fixture_content(tmp_path), (
        "Regenerate frontend/src/lib/fixtures/components.json from fixture_content()."
    )
