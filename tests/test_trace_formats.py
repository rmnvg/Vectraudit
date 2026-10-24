"""LangSmith runs and Langfuse observations import like OpenInference traces."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.test_trace_import import SECRET, dataset
from vectraudit.cli import app
from vectraudit.evaluation.runfile import check_run
from vectraudit.pipelines.intake import intake
from vectraudit.pipelines.traces import Settings, TraceImportError, import_traces


def docs(*ids, text=True):
    return [
        {
            "page_content": f"{SECRET} passage {d}" if text else "",
            "metadata": {"doc_id": d, "source": f"kb/{d}.md"},
            "type": "Document",
        }
        for d in ids
    ]


def langsmith_trace(n, question, retrieved, *, wrap="documents", llm=True, start=0):
    """Root chain → retriever (→ llm), as LangSmith exports runs."""
    t = f"2026-10-20T10:00:{start:02d}.000000"
    end = f"2026-10-20T10:00:{start:02d}.900000"
    root = f"00000000-0000-0000-0000-{n:012d}"
    runs = [
        {
            "id": root,
            "trace_id": root,
            "parent_run_id": None,
            "name": "RetrievalQA",
            "run_type": "chain",
            "inputs": {"question": question},
            "outputs": {"answer": "..."},
            "start_time": t,
            "end_time": end,
            "extra": {"metadata": {"tenant": "acme", "session_id": "s1"}},
        },
        {
            "id": f"{root[:-1]}r",
            "trace_id": root,
            "parent_run_id": root,
            "name": "VectorStoreRetriever",
            "run_type": "retriever",
            "inputs": {"query": question},
            "outputs": {wrap: retrieved} if wrap else retrieved,
            "start_time": f"2026-10-20T10:00:{start:02d}.010000",
            "end_time": f"2026-10-20T10:00:{start:02d}.050000",
        },
    ]
    if llm and retrieved:
        context = "\n".join(d["page_content"] for d in retrieved[:1])
        runs.append(
            {
                "id": f"{root[:-1]}l",
                "trace_id": root,
                "parent_run_id": root,
                "name": "ChatModel",
                "run_type": "llm",
                # LangChain serialises chat messages as constructor objects.
                "inputs": {
                    "messages": [
                        [
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": ["langchain", "schema", "messages", "HumanMessage"],
                                "kwargs": {"content": f"Context: {context}", "type": "human"},
                            }
                        ]
                    ]
                },
                "outputs": {"generations": []},
                "start_time": f"2026-10-20T10:00:{start:02d}.100000",
                "end_time": f"2026-10-20T10:00:{start:02d}.800000",
            }
        )
    return runs


def write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_langsmith_runs_become_a_scored_run_with_model_context(tmp_path):
    runs = langsmith_trace(1, "How do I reset my password?", docs("d1", "d9"))
    runs += langsmith_trace(2, "Where is the invoice?", docs("d5", "d6"), start=10)
    run, report = import_traces(
        dataset(), write(tmp_path, "runs.json", runs), Settings(id_field="doc_id")
    )
    check_run(run)
    assert report["source"]["shape"] == "langsmith"
    by_id = {q["query_id"]: q for q in run["queries"]}
    assert by_id["q1"]["ids"] == ["d1", "d9"]
    assert by_id["q1"]["metrics"]["mrr"] == 1.0
    assert by_id["q2"]["metrics"]["recall"] == 0.0
    assert by_id["q3"]["error"]["code"] == "trace_missing"
    # The prompt holds only the first document: text matching finds it, and not the second.
    assert [i["id"] for i in by_id["q1"]["model_context"]["items"]] == ["d1"]
    assert by_id["q1"]["latency_ms"] == pytest.approx(40.0)
    assert "tenant" in report["capabilities"]["filters_and_tenant_context"]["detail"]
    assert SECRET not in json.dumps(run)


def test_traceable_retrievers_and_document_ids_without_metadata(tmp_path):
    retrieved = [{"id": "d1", "page_content": "x", "metadata": {}, "type": "Document"}]
    runs = langsmith_trace(1, "How do I reset my password?", retrieved, wrap=None, llm=False)
    run, _ = import_traces(dataset(), write(tmp_path, "runs.jsonl", runs))
    assert next(q for q in run["queries"] if q["query_id"] == "q1")["ids"] == ["d1"]


def langfuse_page(question, retrieved, *, retriever_type="RETRIEVER", trace="tf1"):
    """One trace from the Langfuse Observations API, input and output as raw strings."""
    return {
        "data": [
            {
                "id": f"{trace}-root",
                "traceId": trace,
                "parentObservationId": None,
                "type": "CHAIN",
                "name": "rag",
                "startTime": "2026-10-20T10:00:00.000Z",
                "endTime": "2026-10-20T10:00:00.900Z",
                "input": json.dumps({"input": question}),
                "output": json.dumps("an answer"),
                "metadata": {"tenant": "acme"},
                "sessionId": "s-1",
            },
            {
                "id": f"{trace}-ret",
                "traceId": trace,
                "parentObservationId": f"{trace}-root",
                "type": retriever_type,
                "name": "search_kb",
                "startTime": "2026-10-20T10:00:00.010Z",
                "endTime": "2026-10-20T10:00:00.030Z",
                "input": json.dumps(question),
                "output": json.dumps(retrieved),
            },
            {
                "id": f"{trace}-gen",
                "traceId": trace,
                "parentObservationId": f"{trace}-root",
                "type": "GENERATION",
                "name": "answer",
                "startTime": "2026-10-20T10:00:00.100Z",
                "endTime": "2026-10-20T10:00:00.800Z",
                "input": json.dumps(
                    [{"role": "user", "content": "Use: " + retrieved[0]["page_content"]}]
                ),
                "output": json.dumps("an answer"),
            },
        ],
        "meta": {"cursor": None},
    }


def test_a_langfuse_api_page_imports(tmp_path):
    page = langfuse_page("How do I reset my password?", docs("d1", "d4"))
    run, report = import_traces(
        dataset(), write(tmp_path, "obs.json", page), Settings(id_field="doc_id")
    )
    assert report["source"]["shape"] == "langfuse"
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    assert q1["ids"] == ["d1", "d4"]
    assert [i["id"] for i in q1["model_context"]["items"]] == ["d1"]
    assert q1["latency_ms"] == pytest.approx(20.0)


def test_a_retriever_logged_as_a_plain_span_needs_to_be_named(tmp_path):
    page = langfuse_page("How do I reset my password?", docs("d1"), retriever_type="SPAN")
    path = write(tmp_path, "obs.json", page)
    run, _ = import_traces(dataset(), path, Settings(id_field="doc_id"))
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    assert q1["error"]["code"] == "trace_no_retrieval"  # never inferred from a name or shape
    named = Settings(id_field="doc_id", stage_kinds={"search_kb": "vector"})
    run, _ = import_traces(dataset(), path, named)
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    assert q1["ids"] == ["d1"]
    assert q1["evidence"]["stages"][0]["kind"] == "vector"


def test_a_retriever_without_documents_is_a_gap_not_an_empty_result(tmp_path):
    runs = langsmith_trace(1, "How do I reset my password?", None, wrap="documents", llm=False)
    run, _ = import_traces(dataset(), write(tmp_path, "runs.json", runs))
    q1 = next(q for q in run["queries"] if q["query_id"] == "q1")
    assert q1["error"]["code"] == "trace_documents_unrecorded"
    assert run["status"] == "partial"


def test_unusable_exports_are_refused(tmp_path):
    orphan = langsmith_trace(1, "q", docs("d1"))
    orphan[1]["trace_id"] = None
    with pytest.raises(TraceImportError, match="no trace_id"):
        import_traces(dataset(), write(tmp_path, "a.json", orphan))
    mixed = [*langsmith_trace(1, "q", docs("d1")), *langfuse_page("q", docs("d1"))["data"]]
    with pytest.raises(TraceImportError, match="mixes export formats"):
        import_traces(dataset(), write(tmp_path, "b.json", mixed))


def test_cli_and_production_intake_read_the_same_exports(tmp_path):
    path = write(
        tmp_path, "runs.json", langsmith_trace(1, "How do I reset my password?", docs("d1"))
    )
    data = tmp_path / "questions.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": q.query_id, "text": q.text, "relevant": q.relevant})
            for q in dataset().queries
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench", "import-traces", str(data), str(path),
            "--id-field", "doc_id", "--out", str(tmp_path / "traces.run.json"),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    report = intake(path, dataset(), tmp_path / "state", settings=Settings(id_field="doc_id"))
    assert (report["traces_taken"], report["matched_benchmark"]) == (1, 1)
