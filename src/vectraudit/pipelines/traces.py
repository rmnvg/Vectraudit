"""Import OpenInference traces as a run, with a report of what the traces can and cannot show.

Applications instrumented with OpenInference (Phoenix, Arize, OpenLLMetry-compatible
exporters) record retriever and reranker spans with ranked document IDs. This module reads
an exported trace file, matches traces to benchmark questions and turns each into the
``PipelineResult`` the runner already scores, so an imported run is judged by the same
metrics, policy and release check as a live one. Two exports are accepted:

- **OTLP JSON** as written by the OpenTelemetry file exporter: one ``{"resourceSpans": …}``
  object, or one per line;
- **span records**, one JSON object per line or a JSON array, with ``context.trace_id``,
  ``context.span_id``, ``parent_id``, ``name``, ``start_time``, ``end_time`` and
  ``attributes`` (flat dotted keys or nested objects), as Phoenix exports them.

What an import does not do, by design:

- It never invents stages. Retriever spans become stages of kind ``custom`` unless the
  caller names their kind; rerankers are ``rerank`` because OpenInference says so. An edge
  between stages is recorded only when a reranker's recorded input documents are exactly
  the outputs of earlier stages; span nesting alone is not evidence of document flow.
- A span with no recorded documents is a gap, not an empty result. A question whose final
  ranking cannot be determined fails with a stated cause, so the run cannot pass.
- It cannot rerun anything: replay needs a registered pipeline. Latency comes from the
  application's span timestamps, not from a controlled measurement, and is never compared
  with measured latency.
- Document text, prompts and model context stay in the trace file. The run keeps IDs,
  scores and timings; the report keeps counts and attribute names.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.runner import summarize
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.models import (
    EvidenceCoverage,
    Hit,
    PipelineRequest,
    PipelineResult,
    StageTrace,
)
from vectraudit.pipelines.runner import NO_RETRY, PipelineOptions, execute_pipeline

FORMAT = "openinference"
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_SPANS = 2_000_000
STAGE_KINDS = ("keyword", "vector", "filter", "fusion", "rerank", "truncate", "custom")
_QUERY_KEYS = ("query", "question", "input", "text")
Duplicates = Literal["error", "latest"]


class TraceImportError(ValueError):
    """The trace file or the import settings cannot be used; the message says why."""


@dataclass(frozen=True)
class Span:
    """One span with flattened attributes (``retrieval.documents.0.document.id`` …)."""

    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_ns: int
    end_ns: int
    attributes: Mapping[str, Any]

    @property
    def kind(self) -> str | None:
        value = self.attributes.get("openinference.span.kind")
        return value.upper() if isinstance(value, str) else None


@dataclass(frozen=True)
class Settings:
    """How to read the traces. Every field is a mapping the traces cannot supply alone."""

    query_attribute: str | None = None
    """Attribute holding the benchmark question ID (``metadata.query_id``); else match text."""
    id_field: str | None = None
    """Document metadata field to use as the ID instead of ``document.id`` (``doc_id``)."""
    final_span: str | None = None
    """Span name whose documents are the delivered ranking, when the traces have several."""
    stage_kinds: Mapping[str, str] = field(default_factory=dict)
    """Stage kind per retriever span name (``{"bm25": "keyword"}``); default ``custom``."""
    duplicates: Duplicates = "error"
    """Several traces for one question: fail that question, or use the latest trace."""

    def __post_init__(self) -> None:
        for name, kind in self.stage_kinds.items():
            if kind not in STAGE_KINDS:
                raise TraceImportError(
                    f"Unknown stage kind {kind!r} for {name!r}; "
                    f"use one of {', '.join(STAGE_KINDS)}."
                )
        if self.duplicates not in ("error", "latest"):
            raise TraceImportError("duplicates must be 'error' or 'latest'")


# --- reading -----------------------------------------------------------------------------


def read_spans(path: str | Path) -> tuple[list[Span], str]:
    """Every span in the file and which export shape it was (``otlp-json`` or ``spans``)."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise TraceImportError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    if size > MAX_FILE_BYTES:
        raise TraceImportError(f"{path} is larger than {MAX_FILE_BYTES // 2**20} MiB; split it.")
    text = path.read_text(encoding="utf-8-sig")
    objects = list(_json_objects(text, path))
    spans: list[Span] = []
    shapes = set()
    for obj in objects:
        if isinstance(obj, dict) and "resourceSpans" in obj:
            shapes.add("otlp-json")
            spans.extend(_otlp_spans(obj))
        elif isinstance(obj, dict):
            shapes.add("spans")
            spans.append(_record_span(obj))
        else:
            raise TraceImportError(f"{path}: expected JSON objects, found {type(obj).__name__}")
        if len(spans) > MAX_SPANS:
            raise TraceImportError(f"{path} has more than {MAX_SPANS} spans; split it.")
    if not spans:
        raise TraceImportError(f"{path} contains no spans.")
    if len(shapes) > 1:
        raise TraceImportError(f"{path} mixes OTLP JSON and span records; export one format.")
    return spans, shapes.pop()


def _json_objects(text: str, path: Path) -> Iterator[Any]:
    stripped = text.strip()
    try:
        whole = json.loads(stripped)
    except ValueError:
        whole = None
    else:
        yield from whole if isinstance(whole, list) else [whole]
        return
    for number, line in enumerate(stripped.splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceImportError(f"{path}:{number}: not JSON ({exc.msg})") from exc


def _otlp_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_otlp_value(v) for v in (value["arrayValue"] or {}).get("values") or []]
    if "kvlistValue" in value:
        return {
            kv["key"]: _otlp_value(kv.get("value"))
            for kv in (value["kvlistValue"] or {}).get("values") or []
        }
    return None


def _otlp_spans(obj: Mapping[str, Any]) -> Iterator[Span]:
    for resource in obj.get("resourceSpans") or []:
        for scope in (resource or {}).get("scopeSpans") or []:
            for raw in (scope or {}).get("spans") or []:
                attributes = {
                    a["key"]: _otlp_value(a.get("value"))
                    for a in raw.get("attributes") or []
                    if isinstance(a, dict) and isinstance(a.get("key"), str)
                }
                yield _span(
                    raw.get("traceId"),
                    raw.get("spanId"),
                    raw.get("parentSpanId") or None,
                    raw.get("name"),
                    _nanos(raw.get("startTimeUnixNano")),
                    _nanos(raw.get("endTimeUnixNano")),
                    attributes,
                )


def _record_span(obj: Mapping[str, Any]) -> Span:
    found = obj.get("context")
    context: Mapping[str, Any] = found if isinstance(found, dict) else {}
    attributes: dict[str, Any] = {}
    if isinstance(obj.get("attributes"), dict):
        attributes.update(obj["attributes"])
    for key, value in obj.items():  # dataframe exports flatten columns as ``attributes.x``
        if isinstance(key, str) and key.startswith("attributes."):
            attributes[key.removeprefix("attributes.")] = value
    if "span_kind" in obj and "openinference.span.kind" not in attributes:
        attributes["openinference.span.kind"] = obj["span_kind"]
    return _span(
        context.get("trace_id") or obj.get("trace_id") or obj.get("context.trace_id"),
        context.get("span_id") or obj.get("span_id") or obj.get("context.span_id"),
        obj.get("parent_id") or obj.get("parent_span_id") or None,
        obj.get("name"),
        _epoch(obj.get("start_time")),
        _epoch(obj.get("end_time")),
        attributes,
    )


def _span(
    trace_id: Any,
    span_id: Any,
    parent_id: Any,
    name: Any,
    start: int | None,
    end: int | None,
    attributes: Mapping[str, Any],
) -> Span:
    if not isinstance(trace_id, str) or not trace_id or not isinstance(span_id, str):
        raise TraceImportError("A span has no trace or span ID; export complete span records.")
    if start is None or end is None or end < start:
        raise TraceImportError(f"Span {span_id} has missing or inverted timestamps.")
    flat: dict[str, Any] = {}
    _flatten(attributes, "", flat)
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        name=name if isinstance(name, str) and name else "span",
        start_ns=start,
        end_ns=end,
        attributes=flat,
    )


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        for index, item in enumerate(value):
            _flatten(item, f"{prefix}.{index}", out)
    else:
        out[prefix] = value


def _epoch(value: Any) -> int | None:
    """Unix nanoseconds from a span record, whose integer timestamps carry no unit.

    Dataframe exports write epoch milliseconds; others write seconds, microseconds or
    nanoseconds. Present-day epochs differ by three orders of magnitude per unit, so the
    size of the number decides it.
    """
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        value = int(value) if value >= 1e14 else value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for limit, scale in ((1e11, 10**9), (1e14, 10**6), (1e17, 10**3)):
            if value < limit:
                return int(value * scale)
        return int(value)
    return _nanos(value)


def _nanos(value: Any) -> int | None:
    """Unix nanoseconds from an OTLP integer string or an ISO 8601 timestamp."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str) or not value:
        return None
    if value.isdigit():
        return int(value)
    text = value.strip().replace("Z", "+00:00").replace(" ", "T", 1)
    fraction = 0
    if "." in text:  # Python 3.10 parses at most microseconds; keep the nanoseconds here
        head, rest = text.split(".", 1)
        digits = "".join(itertools.takewhile(str.isdigit, rest))
        zone = rest[len(digits) :]
        fraction = int((digits + "000000000")[:9])
        text = head + zone
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp()) * 1_000_000_000 + fraction


# --- one trace ---------------------------------------------------------------------------


@dataclass
class _Stage:
    span: Span
    name: str
    kind: str
    documents: list[str | None] | None
    """``None`` when the span recorded no documents at all (a gap, not an empty result)."""
    scores: list[float | None]
    inputs: list[str | None] | None = None
    input_scores: list[float | None] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    derived: bool = False
    """A reranker's recorded input, shown as its own stage; it has no timing of its own."""
    linked: bool = False
    """For rerankers: whether the recorded input was traced to earlier stage outputs."""


@dataclass
class TraceView:
    """What one trace recorded, before it is matched to a question."""

    trace_id: str
    start_ns: int
    query_id: str | None
    query_texts: list[str]
    stages: list[_Stage]
    llm_context: bool
    llm_spans: int
    metadata_keys: set[str]
    session: bool
    user: bool
    duplicates_dropped: int = 0


def _documents(
    attributes: Mapping[str, Any], prefix: str, id_field: str | None
) -> tuple[list[str | None], list[float | None]] | None:
    indices = sorted(
        {
            int(rest.split(".", 1)[0])
            for key in attributes
            if key.startswith(prefix + ".")
            and (rest := key[len(prefix) + 1 :]).split(".", 1)[0].isdigit()
        }
    )
    if not indices:
        return None
    ids: list[str | None] = []
    scores: list[float | None] = []
    for i in indices:
        base = f"{prefix}.{i}.document"
        if id_field is None:
            raw = attributes.get(f"{base}.id")
        else:
            raw = attributes.get(f"{base}.metadata.{id_field}")
            if raw is None and isinstance(attributes.get(f"{base}.metadata"), str):
                raw = _json_field(attributes[f"{base}.metadata"], id_field)
        ids.append(str(raw) if isinstance(raw, (str, int)) and str(raw) else None)
        score = attributes.get(f"{base}.score")
        ok = isinstance(score, (int, float)) and not isinstance(score, bool)
        scores.append(float(score) if ok and math.isfinite(score) else None)  # type: ignore[arg-type]
    return ids, scores


def _json_field(text: str, dotted: str) -> Any:
    try:
        value: Any = json.loads(text)
    except ValueError:
        return None
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _metadata(span: Span) -> dict[str, Any]:
    raw = span.attributes.get("metadata")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        flat: dict[str, Any] = {}
        _flatten(parsed, "", flat)
        return flat
    prefix = "metadata."
    return {k[len(prefix) :]: v for k, v in span.attributes.items() if k.startswith(prefix)}


def _query_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in _QUERY_KEYS:
            if isinstance(parsed.get(key), str):
                return str(parsed[key])
    return None


def view_trace(spans: Sequence[Span], settings: Settings) -> TraceView:
    """Read one trace's retrieval stages, query identity and recorded capabilities."""
    ordered = sorted(spans, key=lambda s: (s.start_ns, s.end_ns, s.span_id))
    ids = {s.span_id for s in ordered}
    roots = [s for s in ordered if s.parent_id is None or s.parent_id not in ids]
    stages: list[_Stage] = []
    names: Counter[str] = Counter()
    metadata_keys: set[str] = set()
    query_id = None
    texts: list[str] = []
    llm_spans = 0
    llm_context = False
    for span in ordered:
        metadata = _metadata(span)
        metadata_keys.update(metadata)
        if settings.query_attribute is not None and query_id is None:
            key = settings.query_attribute
            value = (
                metadata.get(key.removeprefix("metadata."))
                if key.startswith("metadata.")
                else span.attributes.get(key)
            )
            if isinstance(value, (str, int)) and str(value):
                query_id = str(value)
        if span.kind == "LLM":
            llm_spans += 1
            llm_context = llm_context or any(
                k.startswith("llm.input_messages.") for k in span.attributes
            )
        if span.kind not in ("RETRIEVER", "RERANKER"):
            continue
        names[span.name] += 1
        name = span.name if names[span.name] == 1 else f"{span.name} #{names[span.name]}"
        if span.kind == "RETRIEVER":
            found = _documents(span.attributes, "retrieval.documents", settings.id_field)
            kind = settings.stage_kinds.get(span.name, "custom")
            inputs, input_scores = None, []
            text = _query_text(span.attributes.get("input.value"))
        else:
            found = _documents(span.attributes, "reranker.output_documents", settings.id_field)
            got = _documents(span.attributes, "reranker.input_documents", settings.id_field)
            inputs = got[0] if got else None
            input_scores = got[1] if got else []
            kind = settings.stage_kinds.get(span.name, "rerank")
            text = _query_text(span.attributes.get("reranker.query"))
        if text:
            texts.append(text)
        stages.append(
            _Stage(
                span=span,
                name=name,
                kind=kind,
                documents=found[0] if found else None,
                scores=found[1] if found else [],
                inputs=inputs,
                input_scores=input_scores,
            )
        )
    for root in roots:
        text = _query_text(root.attributes.get("input.value"))
        if text:
            texts.append(text)
    stages = _link(stages)
    dropped = sum(_dedupe(stage) for stage in stages)
    return TraceView(
        trace_id=ordered[0].trace_id,
        start_ns=ordered[0].start_ns,
        query_id=query_id,
        query_texts=texts,
        stages=stages,
        llm_context=llm_context,
        llm_spans=llm_spans,
        metadata_keys=metadata_keys,
        session=any("session.id" in s.attributes for s in ordered),
        user=any("user.id" in s.attributes for s in ordered),
        duplicates_dropped=dropped,
    )


def _link(stages: list[_Stage]) -> list[_Stage]:
    """Connect rerankers to earlier stages using only their recorded input documents.

    When the input is exactly the union of earlier outputs, those stages are the parents.
    Otherwise the input becomes its own stage: drawn from earlier outputs when every input
    document appears in them (a drop between them is then observed), of unknown origin
    when it does not. Span nesting and timing are never used as evidence of an edge.
    """
    out: list[_Stage] = []
    for stage in stages:
        if stage.inputs is None or None in stage.inputs:
            out.append(stage)
            continue
        wanted = set(stage.inputs)
        usable = [
            s for s in out if s.documents is not None and None not in s.documents and s.documents
        ]
        exact = [s for s in usable if set(s.documents or []) <= wanted]
        if exact and set().union(*(set(s.documents or []) for s in exact)) == wanted:
            stage.parents, stage.linked = [s.name for s in exact], True
            out.append(stage)
            continue
        sources = [s for s in usable if wanted & set(s.documents or [])]
        drawn = bool(sources) and wanted <= set().union(*(set(s.documents or []) for s in sources))
        holder = _Stage(
            span=stage.span,
            name=f"{stage.name} input",
            kind="custom",
            documents=list(stage.inputs),
            scores=list(stage.input_scores),
            parents=[s.name for s in sources] if drawn else [],
            derived=True,
        )
        stage.parents, stage.linked = [holder.name], drawn
        out += [holder, stage]
    return out


def _dedupe(stage: _Stage) -> int:
    """Chunks mapped to one document ID keep their best (first) rank; returns drops."""
    if stage.documents is None or None in stage.documents:
        return 0
    seen: set[str | None] = set()
    docs, scores = [], []
    for doc, score in zip(stage.documents, stage.scores, strict=True):
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
            scores.append(score)
    dropped = len(stage.documents) - len(docs)
    stage.documents, stage.scores = docs, scores
    return dropped


def _final(view: TraceView, settings: Settings) -> _Stage:
    """The stage whose documents were delivered; raises a typed cause when unknowable."""
    if not view.stages:
        raise PipelineCallError("trace_no_retrieval", "no retriever or reranker span")
    if settings.final_span is not None:
        named = [s for s in view.stages if s.span.name == settings.final_span and not s.derived]
        if not named:
            raise PipelineCallError("trace_final_ambiguous", "the final span is absent")
        return named[-1]
    parents = {p for s in view.stages for p in s.parents}
    sinks = [s for s in view.stages if s.name not in parents]
    if len(sinks) != 1:
        raise PipelineCallError("trace_final_ambiguous", "several unconnected outputs")
    return sinks[0]


def trace_result(view: TraceView, settings: Settings, k: int) -> tuple[PipelineResult, float]:
    """The run's view of one trace and its retrieval latency in milliseconds."""
    final = _final(view, settings)
    if final.documents is None:
        raise PipelineCallError("trace_documents_unrecorded", "final span has no documents")
    if None in final.documents:
        raise PipelineCallError("document_id_missing", "a delivered document has no ID")
    if len(final.documents) > k:
        raise PipelineCallError("too_many_hits", "the trace delivered more than k documents")
    usable = [
        s
        for s in view.stages
        if s is not final and s.documents is not None and None not in s.documents
    ]
    kept = {s.name for s in usable} | {final.name}
    # Complete: nothing dropped, and every stage but a retriever has a recorded origin.
    complete = len(usable) + 1 == len(view.stages) and all(
        s.parents or (s.span.kind == "RETRIEVER" and not s.derived) for s in [*usable, final]
    )
    stages: list[StageTrace] = []
    for stage in [*usable, final]:
        # The final stage moves last; a parent must still precede its child.
        earlier = {s.name for s in stages}
        stages.append(
            StageTrace(
                name=stage.name,
                kind=stage.kind,  # type: ignore[arg-type]
                parents=[p for p in stage.parents if p in kept and p in earlier],
                hits=[
                    Hit(id=doc or "", score=score)
                    for doc, score in zip(stage.documents or [], stage.scores, strict=True)
                ],
                latency_ms=0.0
                if stage.derived
                else (stage.span.end_ns - stage.span.start_ns) / 1e6,
                source="reported",
            )
        )
    hits = [h.model_copy() for h in stages[-1].hits]
    coverage = EvidenceCoverage(
        ranked_output="observed",
        stage_trace="observed" if complete else "partial",
        source_mapping="unrecorded",
    )
    start = min(s.span.start_ns for s in view.stages)
    latency = (final.span.end_ns - start) / 1e6
    return PipelineResult(hits=hits, stages=stages, coverage=coverage), latency


# --- the import --------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


@dataclass
class _Match:
    by_question: dict[str, list[TraceView]]
    unmatched: list[str]
    ambiguous_texts: int


def _match(views: list[TraceView], dataset: Dataset, settings: Settings) -> _Match:
    known = {q.query_id for q in dataset.queries}
    texts: dict[str, list[str]] = defaultdict(list)
    for q in dataset.queries:
        if q.text:
            texts[_norm(q.text)].append(q.query_id)
    by_question: dict[str, list[TraceView]] = defaultdict(list)
    unmatched = []
    for view in views:
        found = None
        if settings.query_attribute is not None:
            found = view.query_id if view.query_id in known else None
        else:
            for text in view.query_texts:
                candidates = texts.get(_norm(text)) or []
                if len(candidates) == 1:
                    found = candidates[0]
                    break
        if found is None:
            unmatched.append(view.trace_id)
        else:
            by_question[found].append(view)
    ambiguous = sum(len(ids) for ids in texts.values() if len(ids) > 1)
    return _Match(dict(by_question), unmatched, ambiguous)


class TracePipeline:
    """Answers each benchmark question from its imported trace; it cannot call anything."""

    thread_safe = True

    def __init__(
        self, match: _Match, settings: Settings, k: int, kept: dict[str, TraceView]
    ) -> None:
        self.match, self.settings, self.k, self.kept = match, settings, k, kept
        self.latency: dict[str, float] = {}

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        views = self.match.by_question.get(request.query_id) or []
        if not views:
            raise PipelineCallError("trace_missing", "no trace for this question")
        if len(views) > 1 and self.settings.duplicates == "error":
            raise PipelineCallError("trace_ambiguous", "several traces for this question")
        view = self.kept[request.query_id]
        result, latency = trace_result(view, self.settings, self.k)
        self.latency[request.query_id] = latency
        return result


def import_traces(
    dataset: Dataset,
    path: str | Path,
    settings: Settings | None = None,
    *,
    k: int | None = None,
    label: str | None = None,
    evidence: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(run, report)``: a run file scored like a live run, and a capability report.

    ``k`` defaults to the longest delivered ranking among matched traces. Questions with no
    usable trace fail with a stated cause, so the run is ``partial`` and cannot pass.
    """
    settings = settings or Settings()
    path = Path(path)
    spans, shape = read_spans(path)
    grouped: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        grouped[span.trace_id].append(span)
    views = [view_trace(group, settings) for group in grouped.values()]
    match = _match(views, dataset, settings)
    kept = {qid: max(found, key=lambda v: v.start_ns) for qid, found in match.by_question.items()}
    if k is None:
        lengths = []
        for view in kept.values():
            try:
                final = _final(view, settings)
            except PipelineCallError:
                continue
            lengths.append(len(final.documents or []))
        k = max([*lengths, 1])
    if not 1 <= k <= 10000:
        raise TraceImportError("k must be between 1 and 10000")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pipeline = TracePipeline(match, settings, k, kept)
    options = PipelineOptions(
        k=k, warmup=0, repeats=1, evidence=evidence, label=label, retry=NO_RETRY
    )
    config = {
        "adapter": "trace_import",
        "name": FORMAT,
        "sha256": digest,
        "pipeline_version": f"traces:{digest[:16]}",
        "settings": {
            "query_attribute": settings.query_attribute,
            "id_field": settings.id_field,
            "final_span": settings.final_span,
            "stage_kinds": dict(settings.stage_kinds),
            "duplicates": settings.duplicates,
        },
    }
    run = execute_pipeline(dataset, pipeline, options, config=config)
    _use_trace_timing(run, pipeline.latency)
    report = capability_report(views, match, dataset, settings, run, shape=shape, k=k)
    return run, report


def _use_trace_timing(run: dict[str, Any], latency: Mapping[str, float]) -> None:
    """Replace replay timings (meaningless) with what the application's spans recorded."""
    for q in run["queries"]:
        if q.get("error") is None and q["query_id"] in latency:
            q["latency_ms"] = latency[q["query_id"]]
            q["latency_samples_ms"] = [latency[q["query_id"]]]
        q.pop("elapsed_ms", None)
    run["timing"].update(
        scope="imported from application traces: first retrieval span start to final "
        "ranking span end, as the application recorded it; not measured by Vectraudit",
        source="trace",
        concurrency=1,
    )
    run["timing"].pop("retry", None)
    run["summary"] = summarize(run["queries"], run["dataset"]["n_queries"])


def _capability(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def capability_report(
    views: list[TraceView],
    match: _Match,
    dataset: Dataset,
    settings: Settings,
    run: Mapping[str, Any],
    *,
    shape: str,
    k: int,
) -> dict[str, Any]:
    """What the traces recorded, what they could not show, and what was inferred.

    Holds counts, span names and attribute names only, never document or question text.
    """
    errors = Counter(q["error"]["code"] for q in run["queries"] if q.get("error"))
    measured = sum(1 for q in run["queries"] if q.get("error") is None)
    kept = [
        match.by_question[q.query_id][-1]
        for q in dataset.queries
        if q.query_id in match.by_question
    ]
    stage_spans = [s for v in views for s in v.stages if not s.derived]
    unrecorded = sum(1 for s in stage_spans if s.documents is None)
    missing_ids = sum(1 for s in stage_spans if s.documents is not None and None in s.documents)
    rerankers = [s for s in stage_spans if s.span.kind == "RERANKER"]
    linked = sum(1 for s in rerankers if s.linked)
    stage_names: Counter[str] = Counter()
    for v in views:
        for s in v.stages:
            stage_names[f"{s.span.name} ({s.kind})"] += 1
    llm = sum(1 for v in views if v.llm_spans)
    llm_context = sum(1 for v in views if v.llm_context)
    metadata_keys = sorted(set().union(*(v.metadata_keys for v in views)) if views else set())
    sessions = sum(1 for v in views if v.session or v.user)
    dropped = sum(v.duplicates_dropped for v in views)

    def share(n: int, total: int) -> str:
        return f"{n} of {total}"

    total_stages = len(stage_spans)
    ids_ok = total_stages - unrecorded - missing_ids
    capabilities = {
        "stable_document_ids": _capability(
            "observed"
            if total_stages and ids_ok == total_stages
            else ("partial" if ids_ok else "missing"),
            f"{share(ids_ok, total_stages)} retrieval/reranker spans recorded an ID for every "
            f"document; {unrecorded} recorded no documents; {missing_ids} had documents "
            "without IDs"
            + ("" if settings.id_field is None else f" (IDs from metadata.{settings.id_field})"),
        ),
        "final_ranking": _capability(
            "observed" if measured == len(dataset) else ("partial" if measured else "missing"),
            f"{share(measured, len(dataset))} questions have a usable delivered ranking (k={k})",
        ),
        "stages": _capability(
            "observed" if stage_names else "missing",
            ", ".join(f"{name}: {n} traces" for name, n in sorted(stage_names.items()))
            or "no retriever or reranker spans",
        ),
        "document_flow": _capability(
            "not_applicable"
            if not rerankers
            else ("observed" if linked == len(rerankers) else "partial"),
            "no reranker spans; nothing to connect"
            if not rerankers
            else f"{share(linked, len(rerankers))} reranker spans recorded input documents "
            "drawn from earlier stage outputs, so documents lost before or inside the reranker "
            "are attributed; the others have no established origin and their losses are not "
            "attributed",
        ),
        "filters_and_tenant_context": _capability(
            "partial" if metadata_keys or sessions else "missing",
            "OpenInference has no standard attribute for retrieval filters. "
            + (
                f"Metadata keys seen: {', '.join(metadata_keys[:20])}"
                + (" …" if len(metadata_keys) > 20 else "")
                + ". "
                if metadata_keys
                else "No span metadata. "
            )
            + f"Session or user IDs in {share(sessions, len(views))} traces. "
            "The benchmark's own filters and context were not checked against them.",
        ),
        "model_context": _capability(
            "recorded_not_imported" if llm_context else "missing",
            f"LLM spans in {share(llm, len(views))} traces; input messages recorded in "
            f"{llm_context}. Context, answer and citation evidence is not imported yet.",
        ),
        "replay": _capability(
            "not_available",
            "A trace file cannot rerun the application. Register a Python or HTTP pipeline to "
            "retest a change.",
        ),
        "latency": _capability(
            "reported",
            "From span timestamps the application recorded, under whatever load it had; "
            "never compared with latency Vectraudit measured.",
        ),
    }
    gaps = []
    if match.unmatched:
        gaps.append(
            f"{len(match.unmatched)} traces matched no benchmark question"
            + (
                f" (no {settings.query_attribute} attribute with a known question ID)"
                if settings.query_attribute
                else " (their query text is not a benchmark question)"
            )
        )
    if match.ambiguous_texts and settings.query_attribute is None:
        gaps.append(
            f"{match.ambiguous_texts} benchmark questions share their text with another and "
            "cannot be matched by text; record a question ID attribute"
        )
    for code, n in sorted(errors.items()):
        gaps.append(f"{n} questions failed: {code}")
    if dropped:
        gaps.append(
            f"{dropped} repeated document IDs within a span were dropped, keeping the best rank"
        )
    return {
        "format": "vectraudit.trace_import_report",
        "format_version": 1,
        "source": {"format": FORMAT, "shape": shape, "sha256": run["backend"]["config"]["sha256"]},
        "traces": len(views),
        "retrieval_spans": len(stage_spans),
        "questions": {
            "benchmark": len(dataset),
            "matched": len(kept),
            "measured": measured,
            "by": settings.query_attribute or "query text",
            "several_traces": sum(1 for found in match.by_question.values() if len(found) > 1),
        },
        "capabilities": capabilities,
        "gaps": gaps,
        "run_id": run["run_id"],
    }


def render_report(report: Mapping[str, Any]) -> str:
    """The capability report as plain text for the terminal."""
    q = report["questions"]
    lines = [
        f"Imported {report['traces']} traces ({report['source']['shape']}); "
        f"{q['matched']} of {q['benchmark']} questions matched by {q['by']}, "
        f"{q['measured']} measured.",
        "",
        "What the traces show:",
    ]
    for name, item in report["capabilities"].items():
        lines.append(f"  {name.replace('_', ' ')}: {item['status'].replace('_', ' ')}")
        lines.append(f"    {item['detail']}")
    if report["gaps"]:
        lines += ["", "Gaps:", *(f"  - {gap}" for gap in report["gaps"])]
    return "\n".join(lines)
