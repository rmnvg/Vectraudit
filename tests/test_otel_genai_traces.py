"""OpenTelemetry GenAI semantic conventions (as OpenLIT emits them) import like OpenInference."""

from __future__ import annotations

import json

import pytest

from tests.test_openllmetry_traces import PASSAGE, export, span, write
from tests.test_trace_import import dataset
from vectraudit.pipelines.openllmetry import translate
from vectraudit.pipelines.traces import Settings, import_traces

QUESTION = "How do I reset my password?"


def chain():
    return span("w1", "invoke_workflow RetrievalQA", {"gen_ai.operation.name": "invoke_workflow"})


def retriever(documents, count=None, name="retrieval VectorStoreRetriever", span_id="r1"):
    attributes = {
        "gen_ai.operation.name": "retrieval",
        "gen_ai.data_source.id": "VectorStoreRetriever",
        "gen_ai.retrieval.query.text": QUESTION,
        "gen_ai.retrieval.documents": json.dumps(documents),
    }
    if count is not None:
        attributes["gen_ai.retrieval.document_count"] = count
    return span(span_id, name, attributes, parent="w1", start=2)


def llm(content):
    messages = [{"role": "user", "parts": [{"type": "text", "content": content}]}]
    return span(
        "l1",
        "chat gpt-4o",
        {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai",
         "gen_ai.input.messages": json.dumps(messages)},
        parent="w1",
        start=10,
    )  # fmt: skip


OPENLIT_DOCS = [  # OpenLIT's LangChain shape: id from metadata, truncated content, no score
    {"content": PASSAGE, "id": "d1"},
    {"content": "Unrelated billing text.", "id": "d4"},
]


def q1(run):
    return next(q for q in run["queries"] if q["query_id"] == "q1")


def test_an_openlit_langchain_trace_imports_with_question_ids_and_model_context(tmp_path):
    trace = export(chain(), retriever(OPENLIT_DOCS, count=2), llm(f"Use: {PASSAGE}"))
    run, _ = import_traces(dataset(), write(tmp_path, trace), Settings())
    q = q1(run)
    assert q["error"] is None
    assert q["ids"] == ["d1", "d4"]
    assert [i["id"] for i in q["model_context"]["items"]] == ["d1"]  # d4 never reached the model


def test_a_list_shorter_than_the_recorded_count_is_a_gap_not_a_shorter_ranking(tmp_path):
    # OpenLIT records only the first three documents; with five returned, three are not the
    # ranking, and recall or losses computed from them would be wrong.
    docs = [{"content": f"text {i}", "id": f"d{i}"} for i in range(3)]
    run, _ = import_traces(
        dataset(), write(tmp_path, export(chain(), retriever(docs, 5))), Settings()
    )
    assert q1(run)["error"]["code"] == "trace_documents_unrecorded"
    full, _ = import_traces(
        dataset(), write(tmp_path, export(chain(), retriever(docs, 3)), "b.json"), Settings()
    )
    assert q1(full)["ids"] == ["d0", "d1", "d2"]


def test_the_specification_shape_keeps_scores():
    out = translate(
        {"gen_ai.operation.name": "retrieval",
         "gen_ai.retrieval.documents": json.dumps(
             [{"id": "doc_123", "score": 0.95}, {"id": "doc_456", "score": 0.87}])},
        [],
    )  # fmt: skip
    assert out["openinference.span.kind"] == "RETRIEVER"
    first = "retrieval.documents.0.document"
    assert (out[f"{first}.id"], out[f"{first}.score"]) == ("doc_123", 0.95)
    assert "retrieval.documents.1.document.content" not in out


@pytest.mark.parametrize(
    "raw",
    ["not json", json.dumps({"id": "d1"}), json.dumps(["d1", "d2"]), json.dumps([{"other": 1}])],
)
def test_unusable_document_lists_are_gaps(raw):
    out = translate({"gen_ai.operation.name": "retrieval", "gen_ai.retrieval.documents": raw}, [])
    assert out["openinference.span.kind"] == "RETRIEVER"
    assert not any(k.startswith("retrieval.documents") for k in out)


def test_a_vector_database_span_without_ids_does_not_hide_the_retrievers_ranking(tmp_path):
    vector_db = span(
        "v1", "chroma.query",
        {"db.system.name": "chroma", "db.vector.query.top_k": 2, "db.response.returned_rows": 2},
        parent="r1", start=3,
    )  # fmt: skip
    trace = export(chain(), retriever(OPENLIT_DOCS, count=2), vector_db, llm(f"Use: {PASSAGE}"))
    run, _ = import_traces(dataset(), write(tmp_path, trace), Settings())
    q = q1(run)
    named, _ = import_traces(
        dataset(),
        write(tmp_path, trace, "named.json"),
        Settings(final_span="retrieval VectorStoreRetriever"),
    )
    # Two retrievals and nothing says which one the application delivered: the importer
    # says so instead of choosing. Naming the final span gives the ranking.
    assert q["error"]["code"] == "trace_final_ambiguous"
    assert q1(named)["ids"] == ["d1", "d4"]


def test_chat_spans_are_llm_spans_and_openinference_is_untouched():
    out = translate({"gen_ai.operation.name": "chat", "gen_ai.input.messages": "[]"}, [])
    assert out["openinference.span.kind"] == "LLM"
    assert out["input.value"] == "[]"
    kept = {"openinference.span.kind": "LLM", "gen_ai.operation.name": "retrieval"}
    assert translate(kept, [])["openinference.span.kind"] == "LLM"
