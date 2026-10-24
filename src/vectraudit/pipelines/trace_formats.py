"""LangSmith runs and Langfuse observations as spans the OpenInference reader understands.

Teams that trace with LangSmith or Langfuse can import their exports directly. Each run or
observation becomes one :class:`~vectraudit.pipelines.traces.Span` with OpenInference
attributes, so matching, stage mapping, gaps, model-context inference and the capability
report are exactly those of an OpenInference import.

- **LangSmith** run records (``run_type``, ``inputs``, ``outputs``, ``trace_id``,
  ``parent_run_id``, ``start_time``, ``end_time``, ``extra.metadata``), as a JSON array, one
  per line, or ``{"runs": [...]}``. A ``retriever`` run's documents are its outputs, either
  ``{"documents": [...]}`` (LangChain) or a list (``@traceable``), each with
  ``page_content`` and ``metadata``. ``llm`` runs carry the prompt.
- **Langfuse** observations from the Observations API (``id``, ``traceId``,
  ``parentObservationId``, ``type``, ``startTime``, ``endTime``, ``input``, ``output``,
  ``metadata``), as a JSON array, one per line, or the API's ``{"data": [...]}`` page.
  ``RETRIEVER`` observations are retrievers and ``GENERATION`` observations are model calls.
  Input and output arrive as raw strings and are parsed here.

Nothing is inferred from names or nesting: a span is a retriever only when the export says
so (``run_type: retriever``, ``type: RETRIEVER``), or when its name is one the caller lists
as a stage (``--stage-kind``). A retriever whose output has no documents is a gap, as in
any import.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import Any

from vectraudit.pipelines.traces import Span, TraceImportError, _epoch, _span

LANGSMITH_KINDS = {"retriever": "RETRIEVER", "llm": "LLM", "chain": "CHAIN", "tool": "TOOL"}
LANGFUSE_KINDS = {"RETRIEVER": "RETRIEVER", "GENERATION": "LLM", "CHAIN": "CHAIN", "TOOL": "TOOL"}


def shape_of(obj: Any) -> str | None:
    """``langsmith``, ``langfuse`` or ``None`` (not one of these records)."""
    if not isinstance(obj, Mapping):
        return None
    if "run_type" in obj and ("inputs" in obj or "outputs" in obj):
        return "langsmith"
    if "traceId" in obj and "type" in obj and "startTime" in obj:
        return "langfuse"
    return None


def unwrap(obj: Any) -> list[Any] | None:
    """The records of an export envelope (``{"runs": [...]}`` or ``{"data": [...]}``)."""
    if isinstance(obj, Mapping):
        for key in ("runs", "data"):
            records = obj.get(key)
            if isinstance(records, list) and records and shape_of(records[0]):
                return records
    return None


def _parsed(value: Any) -> Any:
    """Langfuse sends input, output and sometimes metadata as raw JSON strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _documents(output: Any) -> list[Mapping[str, Any]] | None:
    """Retrieved documents from a retriever's output, or ``None`` when none are recorded."""
    output = _parsed(output)
    if isinstance(output, Mapping):
        for key in ("documents", "output", "result"):
            if isinstance(output.get(key), list):
                output = output[key]
                break
    if not isinstance(output, list):
        return None
    docs = [d for d in output if isinstance(d, Mapping) and ("page_content" in d or "id" in d)]
    return docs if len(docs) == len(output) else None


def _attributes(
    kind: str | None,
    inputs: Any,
    outputs: Any,
    metadata: Any,
    *,
    session: Any = None,
    user: Any = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if kind:
        attributes["openinference.span.kind"] = kind
    inputs = _parsed(inputs)
    if inputs is not None:
        attributes["input.value"] = inputs if isinstance(inputs, str) else json.dumps(inputs)
    if kind == "RETRIEVER":
        for i, doc in enumerate(_documents(outputs) or []):
            base = f"retrieval.documents.{i}.document"
            if doc.get("id") is not None:
                attributes[f"{base}.id"] = str(doc["id"])
            if isinstance(doc.get("page_content"), str):
                attributes[f"{base}.content"] = doc["page_content"]
            meta = doc.get("metadata")
            if isinstance(meta, Mapping):
                attributes[f"{base}.metadata"] = json.dumps(meta, default=str)
            score = doc.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                attributes[f"{base}.score"] = score
    metadata = _parsed(metadata)
    if isinstance(metadata, Mapping) and metadata:
        attributes["metadata"] = json.dumps(metadata, default=str)
    if isinstance(session, str) and session:
        attributes["session.id"] = session
    if isinstance(user, str) and user:
        attributes["user.id"] = user
    return attributes


def langsmith_span(run: Mapping[str, Any], stage_names: Collection[str] = ()) -> Span:
    """One LangSmith run as a span."""
    run_type = str(run.get("run_type") or "")
    name = run.get("name")
    kind = "RETRIEVER" if name in stage_names else LANGSMITH_KINDS.get(run_type)
    extra = run.get("extra") if isinstance(run.get("extra"), Mapping) else {}
    metadata = extra.get("metadata") if isinstance(extra, Mapping) else None
    meta = metadata if isinstance(metadata, Mapping) else {}
    run_id = run.get("id")
    trace_id = run.get("trace_id") or (run_id if not run.get("parent_run_id") else None)
    if trace_id is None:
        raise TraceImportError(
            f"LangSmith run {run_id} has a parent but no trace_id; export runs with trace_id."
        )
    return _span(
        str(trace_id),
        str(run_id),
        str(run["parent_run_id"]) if run.get("parent_run_id") else None,
        name,
        _epoch(run.get("start_time")),
        _epoch(run.get("end_time")),
        _attributes(
            kind,
            run.get("inputs"),
            run.get("outputs"),
            metadata,
            session=meta.get("session_id") or meta.get("thread_id") or meta.get("conversation_id"),
            user=meta.get("user_id"),
        ),
    )


def langfuse_span(observation: Mapping[str, Any], stage_names: Collection[str] = ()) -> Span:
    """One Langfuse observation as a span."""
    name = observation.get("name")
    kind = "RETRIEVER" if name in stage_names else LANGFUSE_KINDS.get(str(observation.get("type")))
    if not observation.get("traceId"):
        raise TraceImportError(
            f"Langfuse observation {observation.get('id')} has no traceId; export observations "
            "that belong to traces."
        )
    return _span(
        str(observation["traceId"]),
        str(observation.get("id")),
        str(observation["parentObservationId"]) if observation.get("parentObservationId") else None,
        name,
        _epoch(observation.get("startTime")),
        _epoch(observation.get("endTime") or observation.get("startTime")),
        _attributes(
            kind,
            observation.get("input"),
            observation.get("output"),
            observation.get("metadata"),
            session=observation.get("sessionId"),
            user=observation.get("userId"),
        ),
    )


def convert(obj: Mapping[str, Any], shape: str, stage_names: Collection[str] = ()) -> Span:
    """A LangSmith run or Langfuse observation as a span."""
    if shape == "langsmith":
        return langsmith_span(obj, stage_names)
    return langfuse_span(obj, stage_names)
