"""OpenLLMetry and OpenTelemetry GenAI spans in an OTLP export, as OpenInference attributes.

Also covers the OpenTelemetry GenAI semantic conventions for retrieval and model calls
(``gen_ai.operation.name`` ``retrieval``/``chat``, ``gen_ai.retrieval.documents``,
``gen_ai.retrieval.query.text``, ``gen_ai.input.messages``), which OpenLIT and other
instrumentations emit; see :func:`_genai_documents` for the document list's rules.


OpenLLMetry instruments LLM applications with OpenTelemetry using its own conventions
(``opentelemetry-semantic-conventions-ai``) instead of OpenInference's. Its OTLP exports
import through the same reader once each span's attributes and events are translated, so
matching, stages, gaps, model context and the capability report stay identical across
formats. Only spans that do not already carry OpenInference attributes are translated, and
translated keys never overwrite recorded ones.

Translated, as the instrumentation records them (checked against the OpenLLMetry source):

- **Vector database queries.** Each result is a span *event* ``db.query.result`` (or
  ``db.search.result``) with ``.id``, ``.score``, ``.distance``, ``.document`` and
  ``.metadata``. A span with such events, or marked as a vector query
  (``db.vector.query.top_k``, ``db.chroma.query.n_results``), is a retriever: it is never
  inferred from a name. Without result events (Qdrant records none), the retriever recorded
  no documents and the question is a gap. Only ``score`` becomes a score; ``distance`` runs
  the other way and is not substituted. Pinecone writes metadata as a Python ``repr``;
  it is read with ``ast.literal_eval`` (no code runs), as is JSON.
- **Several queries in one span** (Chroma with several texts, Milvus with several query
  IDs) put every query's results in one list with nothing saying which query each belongs
  to. They are not merged into one ranking: the span counts as recording no documents.
- **LLM calls** (``gen_ai.system`` or ``llm.request.type``): ``gen_ai.prompt.N.content``
  and ``.role`` become input messages, so model context can be located; newer
  ``gen_ai.input.messages`` is used as the input value.
- **Workflows and tasks** (``traceloop.span.kind``): ``traceloop.entity.input``, written as
  ``{"args": [...], "kwargs": {...}}``, gives the question (a ``question``, ``query``,
  ``input`` or ``text`` keyword, else the first string argument).
- **Association properties** (``traceloop.association.properties.*``) become metadata, so
  ``--context-field tenant`` and query identity attributes work; ``session_id`` and
  ``user_id`` also become the session and user.
- **Managed prompts** (``traceloop.prompt.key`` with ``traceloop.prompt.version`` or
  ``version_hash``) become the prompt name and version recorded as release evidence. The
  template is never read.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from typing import Any

RESULT_EVENTS = {"db.query.result": "db.query.result", "db.search.result": "db.search.result"}
VECTOR_QUERY_MARKERS = ("db.vector.query.top_k", "db.chroma.query.n_results")
MULTI_QUERY_COUNTS = (
    "db.vector.query.embeddings_count",
    "db.chroma.query.embeddings_count",
    "db.chroma.query.texts_count",
)
LLM_OPERATIONS = {"chat", "text_completion", "generate_content"}
SPAN_KINDS = {"workflow": "CHAIN", "task": "CHAIN", "agent": "AGENT", "tool": "TOOL"}
_QUESTION_KEYS = ("question", "query", "input", "text")
_METADATA_LIMIT = 100_000


def is_openllmetry(attributes: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a span follows OpenLLMetry's conventions and not OpenInference's."""
    if "openinference.span.kind" in attributes:
        return False
    return (
        any(e.get("name") in RESULT_EVENTS for e in events)
        or any(k in attributes for k in VECTOR_QUERY_MARKERS)
        or any(
            k.startswith(("traceloop.", "gen_ai.")) or k == "llm.request.type" for k in attributes
        )
    )


def translate(attributes: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """OpenInference attributes for one OpenLLMetry span (recorded attributes kept as they are)."""
    out: dict[str, Any] = {}
    results = [e for e in events if e.get("name") in RESULT_EVENTS]
    operation = str(attributes.get("gen_ai.operation.name", "")).lower()
    if results or any(k in attributes for k in VECTOR_QUERY_MARKERS):
        out["openinference.span.kind"] = "RETRIEVER"
        if results and not _several_queries(attributes, results):
            out.update(_documents(results))
    elif operation == "retrieval":  # OpenTelemetry GenAI semantic conventions (OpenLIT)
        out["openinference.span.kind"] = "RETRIEVER"
        out.update(_genai_documents(attributes))
        query = attributes.get("gen_ai.retrieval.query.text")
        if isinstance(query, str) and query.strip():
            out["input.value"] = query
    elif (
        "gen_ai.system" in attributes
        or "llm.request.type" in attributes
        or operation in LLM_OPERATIONS
    ):
        out["openinference.span.kind"] = "LLM"
        out.update(_messages(attributes))
    elif str(attributes.get("traceloop.span.kind", "")).lower() in SPAN_KINDS:
        out["openinference.span.kind"] = SPAN_KINDS[str(attributes["traceloop.span.kind"]).lower()]
    question = _question(attributes.get("traceloop.entity.input"))
    if question is not None:
        out["input.value"] = question
    metadata = _association(attributes)
    if metadata:
        out["metadata"] = json.dumps(metadata, default=str)
        for source, target in (("session_id", "session.id"), ("user_id", "user.id")):
            if isinstance(metadata.get(source), str) and metadata[source]:
                out[target] = metadata[source]
    key = attributes.get("traceloop.prompt.key")
    version = attributes.get("traceloop.prompt.version") or attributes.get(
        "traceloop.prompt.version_hash"
    )
    if isinstance(key, str) and key and version not in (None, "") and not isinstance(version, bool):
        out["prompt.vendor"] = "traceloop"
        out["prompt.id"] = key
        out["llm.prompt_template.version"] = str(version)
    return {**out, **attributes}


def _several_queries(attributes: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> bool:
    for key in MULTI_QUERY_COUNTS:
        value = attributes.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1:
            return True
    query_ids = {
        _event_attributes(e).get("db.search.query.id")
        for e in results
        if "db.search.query.id" in _event_attributes(e)
    }
    return len(query_ids) > 1


def _event_attributes(event: Mapping[str, Any]) -> dict[str, Any]:
    raw = event.get("attributes")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _documents(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, event in enumerate(results):
        prefix = RESULT_EVENTS[str(event.get("name"))]
        found = _event_attributes(event)
        base = f"retrieval.documents.{i}.document"
        doc_id = found.get(f"{prefix}.id")
        if isinstance(doc_id, (str, int)) and not isinstance(doc_id, bool) and str(doc_id):
            out[f"{base}.id"] = str(doc_id)
        score = found.get(f"{prefix}.score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            out[f"{base}.score"] = float(score)
        text = found.get(f"{prefix}.document")
        if isinstance(text, str) and text:
            out[f"{base}.content"] = text
        metadata = _metadata(found.get(f"{prefix}.metadata"))
        if metadata is not None:
            out[f"{base}.metadata"] = json.dumps(metadata, default=str)
    return out


def _genai_documents(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Documents from ``gen_ai.retrieval.documents`` (OpenTelemetry GenAI conventions).

    The specification's schema requires ``id`` and ``score`` and allows more (OpenLIT adds
    ``content``). A list shorter than the recorded ``gen_ai.retrieval.document_count``
    (OpenLIT keeps only the first three) is not the ranking: no documents are taken, so the
    question is a gap instead of a silently shorter top k.
    """
    raw = attributes.get("gen_ai.retrieval.documents")
    try:
        documents = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    if not isinstance(documents, list) or not all(isinstance(d, dict) for d in documents):
        return {}
    count = attributes.get("gen_ai.retrieval.document_count")
    if isinstance(count, int) and not isinstance(count, bool) and count > len(documents):
        return {}
    out: dict[str, Any] = {}
    for i, doc in enumerate(documents):
        base = f"retrieval.documents.{i}.document"
        doc_id = doc.get("id")
        if isinstance(doc_id, (str, int)) and not isinstance(doc_id, bool) and str(doc_id):
            out[f"{base}.id"] = str(doc_id)
        score = doc.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            out[f"{base}.score"] = float(score)
        content = doc.get("content")
        if isinstance(content, str) and content:
            out[f"{base}.content"] = content
        metadata = doc.get("metadata")
        if isinstance(metadata, dict):
            out[f"{base}.metadata"] = json.dumps(metadata, default=str)
    if not out and documents:
        return {}
    return out


def _metadata(raw: Any) -> dict[str, Any] | None:
    """Result metadata written as JSON, or as a Python ``repr`` (Pinecone)."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip() or len(raw) > _METADATA_LIMIT:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        try:
            value = ast.literal_eval(raw)  # literals only: never executes code
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    return dict(value) if isinstance(value, dict) else None


def _messages(attributes: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    prefix = "gen_ai.prompt."
    for key, value in attributes.items():
        if not key.startswith(prefix):
            continue
        index, _, field = key[len(prefix) :].partition(".")
        if index.isdigit() and field in ("content", "role") and isinstance(value, str):
            out[f"llm.input_messages.{index}.message.{field}"] = value
    messages = attributes.get("gen_ai.input.messages")
    if not out and isinstance(messages, str) and messages:
        out["input.value"] = messages
    return out


def _question(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(value, dict) or not ({"args", "kwargs"} & set(value)):
        return None
    kwargs = value.get("kwargs") if isinstance(value.get("kwargs"), dict) else {}
    assert isinstance(kwargs, dict)
    for key in _QUESTION_KEYS:
        if isinstance(kwargs.get(key), str) and kwargs[key].strip():
            return str(kwargs[key])
    args = value.get("args") if isinstance(value.get("args"), list) else []
    assert isinstance(args, list)
    return next((a for a in args if isinstance(a, str) and a.strip()), None)


def _association(attributes: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "traceloop.association.properties."
    return {
        key[len(prefix) :]: value
        for key, value in attributes.items()
        if key.startswith(prefix) and isinstance(value, (str, int, float, bool))
    }
