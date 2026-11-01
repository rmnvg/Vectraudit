"""OpenLLMetry (Traceloop) OTLP exports import like OpenInference ones."""

from __future__ import annotations

import json

import pytest

from tests.test_trace_import import SECRET, dataset
from vectraudit.pipelines.drift import production_questions
from vectraudit.pipelines.openllmetry import _metadata, _question, is_openllmetry, translate
from vectraudit.pipelines.traces import Settings, import_traces

QUESTION = "How do I reset my password?"
PASSAGE = f"{SECRET} To reset your password, open Settings and choose Security."


def value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": v}


def attrs(d):
    return [{"key": k, "value": value(v)} for k, v in d.items()]


def span(span_id, name, attributes, *, parent=None, events=(), start=0):
    return {
        "traceId": "a" * 32,
        "spanId": span_id,
        "parentSpanId": parent or "",
        "name": name,
        "startTimeUnixNano": str(1_700_000_000_000_000_000 + start * 1_000_000),
        "endTimeUnixNano": str(1_700_000_000_000_000_000 + (start + 5) * 1_000_000),
        "attributes": attrs(attributes),
        "events": [{"name": n, "attributes": attrs(a)} for n, a in events],
    }


def workflow(question=QUESTION):
    return span(
        "w1",
        "rag.workflow",
        {
            "traceloop.span.kind": "workflow",
            "traceloop.entity.name": "rag",
            "traceloop.entity.input": json.dumps({"args": [], "kwargs": {"question": question}}),
            "traceloop.association.properties.tenant": "acme",
            "traceloop.association.properties.session_id": "s-1",
        },
    )


def llm(content):
    return span(
        "l1",
        "openai.chat",
        {
            "gen_ai.system": "openai",
            "llm.request.type": "chat",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": content,
            "traceloop.prompt.managed": "true",
            "traceloop.prompt.key": "support-answer",
            "traceloop.prompt.version": 7,
        },
        parent="w1",
        start=10,
    )


def export(*spans):
    return {"resourceSpans": [{"scopeSpans": [{"spans": list(spans)}]}]}


def write(tmp_path, payload, name="otlp.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def pinecone_query():
    results = [
        ("db.query.result", {
            "db.query.result.id": f"vec-{d}", "db.query.result.score": s,
            "db.query.result.metadata": str({"doc_id": d, "source": f"kb/{d}.md"}),
        })
        for d, s in (("d1", 0.91), ("d4", 0.62))
    ]  # fmt: skip
    return span(
        "p1",
        "pinecone.query",
        {"db.system": "pinecone", "db.operation": "query", "db.vector.query.top_k": 2},
        parent="w1",
        events=[("db.query.embeddings", {"db.query.embeddings.vector": "[0.1]"}), *results],
        start=2,
    )


def chroma_query(texts=1):
    results = [
        ("db.query.result", {
            "db.query.result.id": d, "db.query.result.distance": dist,
            "db.query.result.document": text,
            "db.query.result.metadata": json.dumps({"lang": "en"}),
        })
        for d, dist, text in (("d1", 0.1, PASSAGE), ("d4", 0.4, "Unrelated billing text."))
    ]  # fmt: skip
    return span(
        "c1",
        "chroma.query",
        {
            "db.system": "chroma",
            "db.chroma.query.n_results": 2,
            "db.chroma.query.texts_count": texts,
        },
        parent="w1",
        events=results,
        start=2,
    )


def q1(run):
    return next(q for q in run["queries"] if q["query_id"] == "q1")


def test_a_pinecone_trace_imports_with_ids_scores_question_and_prompt(tmp_path):
    path = write(tmp_path, export(workflow(), pinecone_query(), llm("Answer briefly.")))
    run, report = import_traces(dataset(), path, Settings(id_field="doc_id"))
    q = q1(run)
    assert q["ids"] == ["d1", "d4"]  # from Pinecone's Python-repr metadata
    assert q["error"] is None
    assert q["prompts"] == [{"name": "support-answer", "version": "7"}]
    assert run["backend"]["config"]["prompts"] == {"support-answer": "7"}
    assert report["source"]["shape"] == "otlp-json"
    stage = q["evidence"]["stages"][0]
    assert stage["name"] == "pinecone.query"


def test_a_chroma_trace_locates_model_context_and_never_uses_distance_as_score(tmp_path):
    path = write(tmp_path, export(workflow(), chroma_query(), llm(f"Use: {PASSAGE}")))
    run, _ = import_traces(dataset(), path, Settings())
    q = q1(run)
    assert q["ids"] == ["d1", "d4"]
    assert [i["id"] for i in q["model_context"]["items"]] == ["d1"]  # d4 never reached the model
    spans_scores = translate(*_raw(chroma_query()))
    assert not any(k.endswith(".score") for k in spans_scores)


def _raw(otlp_span):
    from vectraudit.pipelines.traces import _otlp_attributes

    events = [
        {"name": e["name"], "attributes": _otlp_attributes(e["attributes"])}
        for e in otlp_span["events"]
    ]
    return _otlp_attributes(otlp_span["attributes"]), events


def test_several_queries_in_one_span_are_a_gap_not_a_merged_ranking(tmp_path):
    path = write(tmp_path, export(workflow(), chroma_query(texts=2)))
    run, _ = import_traces(dataset(), path, Settings())
    assert q1(run)["error"]["code"] == "trace_documents_unrecorded"
    milvus = translate(
        {"db.system": "milvus"},
        [
            {"name": "db.search.result",
             "attributes": {"db.search.query.id": i, "db.search.result.id": f"x{i}"}}
            for i in (0, 1)
        ],
    )  # fmt: skip
    assert milvus["openinference.span.kind"] == "RETRIEVER"
    assert not any(k.startswith("retrieval.documents") for k in milvus)


def test_a_vector_query_without_result_events_is_a_gap(tmp_path):
    qdrant = span(
        "q1", "qdrant.search", {"db.system": "qdrant", "db.vector.query.top_k": 5},
        parent="w1", start=2,
    )  # fmt: skip
    run, _ = import_traces(dataset(), write(tmp_path, export(workflow(), qdrant)), Settings())
    assert q1(run)["error"]["code"] == "trace_documents_unrecorded"


def test_a_retriever_is_never_inferred_from_a_name(tmp_path):
    task = span(
        "t1", "retriever", {"traceloop.span.kind": "task", "traceloop.entity.name": "retriever"},
        parent="w1", start=2,
    )  # fmt: skip
    run, _ = import_traces(dataset(), write(tmp_path, export(workflow(), task)), Settings())
    assert q1(run)["error"]["code"] == "trace_no_retrieval"


def test_openinference_spans_are_left_exactly_as_recorded():
    recorded = {"openinference.span.kind": "LLM", "gen_ai.system": "openai", "input.value": "x"}
    assert not is_openllmetry(recorded, [])
    assert is_openllmetry({"gen_ai.system": "openai"}, [])
    assert not is_openllmetry({"http.method": "GET"}, [])
    recorded_input = {
        "input.value": "kept",
        "traceloop.span.kind": "workflow",
        "traceloop.entity.input": json.dumps({"args": ["other"]}),
    }
    translated = translate(recorded_input, [])
    assert translated["input.value"] == "kept"  # recorded attributes are never overwritten


def test_association_properties_become_context_and_session(tmp_path):
    path = write(tmp_path, export(workflow(), pinecone_query()))
    found, unreadable = production_questions(path, Settings(), ["tenant"])
    assert found == [(QUESTION, {"tenant": "acme"})]
    assert unreadable == 0
    out = translate(*_raw(workflow()))
    assert out["session.id"] == "s-1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (json.dumps({"args": [], "kwargs": {"query": "q", "question": "Q"}}), "Q"),
        (json.dumps({"args": [3, "first", "second"], "kwargs": {}}), "first"),
        (json.dumps({"args": [], "kwargs": {"limit": 3}}), None),
        (json.dumps(["not", "the", "sdk", "shape"]), None),
        ("not json", None),
        (None, None),
    ],
)
def test_the_question_is_read_from_the_sdks_entity_input(raw, expected):
    assert _question(raw) == expected


def test_metadata_reprs_are_read_as_literals_only():
    assert _metadata(str({"doc_id": "d1", "n": 2})) == {"doc_id": "d1", "n": 2}
    assert _metadata('{"doc_id": "d1"}') == {"doc_id": "d1"}
    assert _metadata("__import__('os').system('echo hacked')") is None  # never executed
    assert _metadata("[1, 2]") is None
    assert _metadata("x" * 200_000) is None


def test_prompt_versions_fall_back_to_the_version_hash():
    out = translate({"gen_ai.system": "openai", "traceloop.prompt.key": "a",
                     "traceloop.prompt.version_hash": "9f1c"}, [])  # fmt: skip
    assert (out["prompt.id"], out["llm.prompt_template.version"]) == ("a", "9f1c")
    assert "prompt.id" not in translate(
        {"gen_ai.system": "openai", "traceloop.prompt.key": "a"}, []
    )
