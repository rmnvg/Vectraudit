"""Prompt versions recorded from traces, as release evidence."""

from __future__ import annotations

import json

import pytest

from tests.test_trace_formats import docs, langfuse_page, langsmith_trace
from tests.test_trace_import import QUESTIONS, SECRET, dataset
from vectraudit.evaluation.answers import attach_answers, render_summary
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.runfile import check_run
from vectraudit.integrations.sharing import project_run
from vectraudit.pipelines.traces import Settings, Span, import_traces, span_prompts

TWO = {q: QUESTIONS[q] for q in ("q1", "q2")}  # both traced, so runs are complete
SETTINGS = Settings(id_field="doc_id")


def span(**attributes) -> Span:
    return Span(
        trace_id="t", span_id="s", parent_id=None, name="x", start_ns=0, end_ns=1,
        attributes={"openinference.span.kind": "LLM", **attributes},
    )  # fmt: skip


def langfuse_trace(trace: str, question: str, retrieved, version: int | None) -> list[dict]:
    observations = langfuse_page(question, retrieved, trace=trace)["data"]
    if version is not None:
        observations[2]["promptName"] = "support-answer"  # the generation used a managed prompt
        observations[2]["promptVersion"] = version
        observations[2]["promptId"] = "cl-prompt"
    return observations


def import_versions(tmp_path, versions: dict[str, int | None], name: str = "obs.jsonl"):
    rows = [
        *langfuse_trace("tf1", "How do I reset my password?", docs("d1", "d4"), versions["tf1"]),
        *langfuse_trace("tf2", "Where is the invoice?", docs("d2"), versions["tf2"]),
    ]
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return import_traces(dataset(TWO), path, SETTINGS)


def test_each_convention_yields_a_name_and_version_but_never_the_template():
    assert span_prompts(
        span(**{"prompt.id": "support-answer", "llm.prompt_template.version": "7"})
    ) == {("support-answer", "7")}
    assert span_prompts(span(**{"llm.prompt_template.version": 3})) == {("prompt_template", "3")}
    langsmith = {"lc_hub_owner": "acme", "lc_hub_repo": "rag-answer", "lc_hub_commit_hash": "9f1c"}
    assert span_prompts(span(metadata=json.dumps(langsmith))) == {("acme/rag-answer", "9f1c")}
    unowned = {"lc_hub_owner": "-", "lc_hub_repo": "rag-answer", "lc_hub_commit_hash": "9f1c"}
    assert span_prompts(span(metadata=json.dumps(unowned))) == {("rag-answer", "9f1c")}
    opik = {
        "opik_prompts": [
            {"name": "answer", "version": {"commit": "a1b2c3", "version_number": 4,
                                           "template": SECRET}},
            {"name": "rewrite", "version": {"version_number": 2, "template": SECRET}},
            {"name": "broken", "version": {"template": SECRET}},
            "not a prompt",
        ]
    }  # fmt: skip
    found = span_prompts(span(metadata=json.dumps(opik)))
    assert found == {("answer", "a1b2c3"), ("rewrite", "2")}
    assert SECRET not in json.dumps(sorted(found))
    for nothing in (span(), span(metadata="not json"), span(**{"prompt.id": "x"}),
                    span(**{"llm.prompt_template.version": True})):  # fmt: skip
        assert span_prompts(nothing) == set()


def test_a_langfuse_import_records_the_prompt_per_question_and_for_the_run(tmp_path):
    run, report = import_versions(tmp_path, {"tf1": 12, "tf2": 12})
    assert run["backend"]["config"]["prompts"] == {"support-answer": "12"}
    by = {q["query_id"]: q for q in run["queries"]}
    assert by["q1"]["prompts"] == [{"name": "support-answer", "version": "12"}]
    assert report["capabilities"]["prompt_versions"]["status"] == "observed"
    check_run(json.loads(json.dumps(run)))


def test_a_run_mixing_versions_says_so(tmp_path):
    run, report = import_versions(tmp_path, {"tf1": 12, "tf2": 13})
    assert run["backend"]["config"]["prompts"] == {"support-answer": ["12", "13"]}
    detail = report["capabilities"]["prompt_versions"]["detail"]
    assert "Mixed versions in this run: support-answer (12, 13)" in detail


def test_without_prompt_versions_the_config_is_unchanged_and_the_gap_is_named(tmp_path):
    run, report = import_versions(tmp_path, {"tf1": None, "tf2": None})
    assert "prompts" not in run["backend"]["config"]
    assert all("prompts" not in q for q in run["queries"])
    capability = report["capabilities"]["prompt_versions"]
    assert capability["status"] == "missing"
    assert "cannot tell whether a prompt changed" in capability["detail"]


def test_a_release_check_reports_a_prompt_change_as_a_changed_setting(tmp_path):
    baseline, _ = import_versions(tmp_path, {"tf1": 12, "tf2": 12}, "a.jsonl")
    candidate, _ = import_versions(tmp_path, {"tf1": 13, "tf2": 13}, "b.jsonl")
    result = compare_runs(baseline, candidate)
    changes = {d["path"]: (d["baseline"], d["candidate"]) for d in result["config_diff"]}
    assert changes["backend.config.prompts.support-answer"] == ("12", "13")


def test_a_check_warns_about_mixed_versions_and_one_sided_recording(tmp_path):
    mixed, _ = import_versions(tmp_path, {"tf1": 12, "tf2": 13}, "a.jsonl")
    steady, _ = import_versions(tmp_path, {"tf1": 13, "tf2": 13}, "b.jsonl")
    unrecorded, _ = import_versions(tmp_path, {"tf1": None, "tf2": None}, "c.jsonl")
    warnings = compare_runs(mixed, steady)["warnings"]
    assert any("baseline run mixes versions of prompt support-answer (12, 13)" in w
               for w in warnings)  # fmt: skip
    warnings = compare_runs(unrecorded, steady)["warnings"]
    assert any("Only the candidate run records prompt versions" in w for w in warnings)
    assert not any("prompt" in w for w in compare_runs(steady, steady)["warnings"])


def test_an_ambiguous_question_does_not_claim_a_prompt(tmp_path):
    rows = [
        *langfuse_trace("tf1", "How do I reset my password?", docs("d1"), 12),
        *[
            {
                **o,
                "startTime": o["startTime"].replace("10:00:00", "11:00:00"),
                "endTime": o["endTime"].replace("10:00:00", "11:00:00"),
            }
            for o in langfuse_trace("tf9", "How do I reset my password?", docs("d1"), 99)
        ],  # a later repeat of the same question
        *langfuse_trace("tf2", "Where is the invoice?", docs("d2"), 12),
    ]
    path = tmp_path / "dup.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    run, _ = import_traces(dataset(TWO), path, SETTINGS)
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    assert q1["error"] is not None  # two traces for one question: refused, as before
    assert "prompts" not in q1
    assert run["backend"]["config"]["prompts"] == {"support-answer": "12"}  # 99 never counted
    latest, _ = import_traces(dataset(TWO), path, Settings(id_field="doc_id", duplicates="latest"))
    q1 = next(q for q in latest["queries"] if q["query_id"] == "q1")
    assert q1["prompts"] == [{"name": "support-answer", "version": "99"}]  # tf9 is later


def test_langsmith_hub_prompts_are_recorded(tmp_path):
    runs = langsmith_trace(1, "How do I reset my password?", docs("d1"))
    runs[2]["extra"] = {"metadata": {"lc_hub_owner": "-", "lc_hub_repo": "rag-answer",
                                     "lc_hub_commit_hash": "9f1c2d"}}  # fmt: skip
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(runs))
    run, _ = import_traces(dataset(), path, SETTINGS)
    assert run["backend"]["config"]["prompts"] == {"rag-answer": "9f1c2d"}


def test_diagnoses_are_broken_down_by_prompt_version(tmp_path):
    run, _ = import_versions(tmp_path, {"tf1": 12, "tf2": 13})
    rows = [
        ("trace_id", "tf1", _judgment("correct")),
        ("trace_id", "tf2", _judgment("incorrect")),
    ]
    out = attach_answers(run, rows)
    by_prompt = out["answers"]["diagnoses_by_prompt"]
    assert set(by_prompt) == {"support-answer@12", "support-answer@13"}
    assert sum(by_prompt["support-answer@12"].values()) == 1
    text = render_summary(out)
    assert "Diagnosis by prompt version" in text
    assert "support-answer@13" in text
    single, _ = import_versions(tmp_path, {"tf1": 12, "tf2": 12}, "s.jsonl")
    assert "Diagnosis by prompt version" not in render_summary(attach_answers(single, rows))


def _judgment(verdict: str):
    from vectraudit.evaluation.answers import Judgment

    return Judgment(
        evaluator="review", kind="human", verdict=verdict, score=None, outcome="answered",
        evaluator_version=None, rubric=None,
    )  # fmt: skip


def test_hosted_projections_carry_fingerprints_that_still_show_a_change(tmp_path):
    baseline, _ = import_versions(tmp_path, {"tf1": 12, "tf2": 12}, "a.jsonl")
    candidate, _ = import_versions(tmp_path, {"tf1": 13, "tf2": 13}, "b.jsonl")
    pb, pc = project_run(baseline), project_run(candidate)
    sent = json.dumps(pc)
    assert "support-answer" not in sent
    prompts = pc["backend"]["config"]["prompts"]
    ((name, version),) = prompts.items()
    assert name.startswith("sha256:")
    assert version.startswith("sha256:")
    assert project_run(pc)["backend"]["config"]["prompts"] == prompts  # re-projection is stable
    changed = [d["path"] for d in compare_runs(pb, pc)["config_diff"]]
    assert f"backend.config.prompts.{name}" in changed
    forged = json.loads(sent)
    forged["backend"]["config"]["prompts"] = {"support-answer": "13"}
    with pytest.raises(ValueError, match="fingerprints"):
        project_run(forged)


def test_opik_prompts_on_the_trace_reach_the_run_without_their_template(tmp_path):
    from tests.test_opik_api import STAGES, project
    from vectraudit.pipelines.opik_api import records

    traces, spans = project()
    for trace in traces:
        trace["metadata"] = {
            "opik_prompts": [
                {"name": "answer", "version": {"commit": "a1b2c3", "template": SECRET}}
            ]
        }
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records(traces, spans[:4])))
    run, _ = import_traces(dataset(TWO), path, STAGES)
    assert run["backend"]["config"]["prompts"] == {"answer": "a1b2c3"}
    assert SECRET not in json.dumps({k: v for k, v in run.items() if k != "queries"})
    assert all(SECRET not in json.dumps(q.get("prompts")) for q in run["queries"])
