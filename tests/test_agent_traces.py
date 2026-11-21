"""Agent traces from real instrumentation (Phase 49.0): what a session, step and search look like.

The fixtures were produced by OpenInference, OpenLLMetry and OpenLIT instrumenting a LangGraph
ReAct agent (tests/fixtures/real_traces/generate/agent_common.py). These tests pin the shapes
recorded in docs/DECISIONS.md (49.0), and today's gap: an agent trace matches no benchmark
question, because the question is taken from a retriever's query and an agent rewrites it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.traces import Settings, import_traces, read_spans

FIXTURES = Path(__file__).parent / "fixtures" / "real_traces"
SEARCHES = {
    "t1": ["reset password", "two-factor setup"],
    "t3": ["support reply time", "billing invoices", "support"],
}


def sessions(name: str):
    spans, shape = read_spans(FIXTURES / f"agent_{name}.json")
    assert shape == "otlp-json"
    by_trace: dict[str, list] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)
    return [sorted(v, key=lambda s: s.start_ns) for v in by_trace.values()]


def task_of(spans) -> str:
    """Each library records the task differently (or not at all: OpenLIT)."""
    for span in spans:
        a = span.attributes
        if "traceloop.association.properties.task_id" in a:
            return a["traceloop.association.properties.task_id"]
        if isinstance(a.get("metadata"), str) and "task_id" in a["metadata"]:
            return json.loads(a["metadata"])["task_id"]
    question = next(
        s.attributes["gen_ai.workflow.input"]
        for s in spans
        if "gen_ai.workflow.input" in s.attributes
    )
    return "t1" if "password" in question else "t3"


@pytest.mark.parametrize("name", ["openinference", "openllmetry", "openlit"])
def test_one_trace_per_task_with_one_retriever_span_per_search(name):
    found = {}
    for spans in sessions(name):
        retrievers = [s for s in spans if s.kind == "RETRIEVER"]
        if name == "openlit":
            # OpenLIT records each search twice: the retriever and the database call.
            queries = [s.attributes.get("input.value") for s in retrievers if "GET" not in s.name]
        elif name == "openllmetry":
            # The documents are on chroma.query; the query text is on its parent span.
            parents = {s.span_id: s for s in spans}
            queries = [
                json.loads(parents[s.parent_id].attributes["traceloop.entity.input"])["query"]
                for s in retrievers
            ]
        else:
            queries = [s.attributes.get("input.value") for s in retrievers]
        found[task_of(spans)] = queries
    assert found == SEARCHES


def test_openinference_and_openllmetry_number_the_steps():
    for name, key in (
        ("openinference", None),
        ("openllmetry", "traceloop.association.properties.langgraph_step"),
    ):
        for spans in sessions(name):
            parents = {s.span_id: s for s in spans}
            steps = []
            for s in spans:
                if s.kind != "RETRIEVER":
                    continue
                if key is None:
                    steps.append(json.loads(s.attributes["metadata"])["langgraph_step"])
                else:  # chroma.query carries no step; its retrieval parent does
                    steps.append(int(parents[s.parent_id].attributes[key]))
            assert steps == sorted(steps)
            assert steps[0] == 2  # step 1 is the model deciding to search


def test_today_an_agent_trace_matches_no_question(tmp_path):
    dataset = parse_dataset((FIXTURES / "agent_tasks.jsonl").read_text())
    run, _ = import_traces(
        dataset, FIXTURES / "agent_openinference.json", Settings(id_field="doc_id")
    )
    assert all(q["error"] for q in run["queries"])
