"""Evaluate the LangChain or LlamaIndex retriever a team already runs.

Each function wraps a retriever as a workbench pipeline (``PipelineRequest`` ->
``PipelineResult``), so ``vectraudit workbench run`` measures it and its run files feed
``vectraudit check``. Neither framework is imported: the adapters call the documented
public methods (LangChain ``invoke`` / ``compress_documents``, LlamaIndex ``retrieve`` /
``postprocess_nodes``) on the objects they are given.

Stages are recorded from real calls, never replayed. A LangChain contextual-compression
retriever is run as its two steps (base retrieval, then the compressor on those
documents), exactly as it runs itself, so a document dropped by a reranker is attributed
to that stage. LlamaIndex postprocessors passed explicitly become stages the same way.
When the last stage returns more than ``k`` documents, an explicit truncation stage
records the cut.

Documents are identified by the id the dataset labels. Chunks that map to the same
document id count once, at their best rank (document-level evaluation).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

from vectraudit.pipelines.models import Hit, PipelineRequest, PipelineResult, StageTrace

InvokeKwargs = Callable[[PipelineRequest], Mapping[str, Any]]


def _timed(call: Callable[[], list[Any]]) -> tuple[list[Any], float]:
    start = time.perf_counter()
    out = call()
    return out, (time.perf_counter() - start) * 1000


class _Recorder:
    """Times stages and keeps their de-duplicated document ids."""

    def __init__(self) -> None:
        self.stages: list[StageTrace] = []

    def add(self, name: str, kind: Any, hits: list[Hit], latency_ms: float) -> None:
        parents = [self.stages[-1].name] if self.stages else []
        self.stages.append(
            StageTrace(
                name=name,
                kind=kind,
                parents=parents,
                hits=hits,
                latency_ms=latency_ms,
                source="measured",
            )
        )

    def result(self, k: int) -> PipelineResult:
        last = self.stages[-1]
        if len(last.hits) > k:
            start = time.perf_counter()
            kept = [h.model_copy() for h in last.hits[:k]]
            self.add("truncate", "truncate", kept, (time.perf_counter() - start) * 1000)
        return PipelineResult(hits=list(self.stages[-1].hits), stages=self.stages)


def _unique_hits(
    items: Sequence[Any], id_of: Callable[[Any], Any], score_of: Callable[[Any], Any]
) -> list[Hit]:
    hits: list[Hit] = []
    seen: set[str] = set()
    for item in items:
        raw = id_of(item)
        if raw is None or raw == "":
            raise ValueError("a retrieved document has no id")
        doc_id = str(raw)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        score = score_of(item)
        finite = isinstance(score, int | float) and math.isfinite(score)
        hits.append(Hit(id=doc_id, score=float(score) if finite else None))
    return hits


def _require_text(request: PipelineRequest) -> str:
    if not request.text:
        raise ValueError(
            f"query {request.query_id} has no text; framework retrievers take text queries"
        )
    return request.text


def _reject_unforwarded(request: PipelineRequest, how: str) -> None:
    """Fail rather than silently ignore filters or tenant context the retriever never saw."""
    if request.filters or request.context:
        raise ValueError(
            f"query {request.query_id} has filters or tenant context; {how} to apply them"
        )


def _label(obj: Any) -> str:
    return type(obj).__name__[:100]


# --- LangChain -----------------------------------------------------------------------------


def _langchain_id(id_key: str | None) -> Callable[[Any], Any]:
    if id_key is None:

        def doc_id(doc: Any) -> Any:
            value = getattr(doc, "id", None)
            if value is None:
                raise ValueError(
                    "a LangChain Document has no id; pass id_key with the metadata field "
                    "holding your document id"
                )
            return value

        return doc_id
    return lambda doc: (getattr(doc, "metadata", None) or {}).get(id_key)


def langchain_pipeline(
    retriever: Any,
    *,
    id_key: str | None = None,
    invoke_kwargs: InvokeKwargs | None = None,
    decompose: bool = True,
) -> Callable[[PipelineRequest], PipelineResult]:
    """Wrap a LangChain retriever (anything with ``invoke(query, **kwargs)``).

    ``id_key`` names the ``Document.metadata`` field holding the id your dataset labels;
    by default ``Document.id`` is used. ``invoke_kwargs`` maps each request to keyword
    arguments for ``invoke``, e.g. ``lambda r: {"k": r.k, "filter": r.filters}``; without
    it, a query with filters or tenant context fails instead of running unfiltered.
    With ``decompose`` (the default), a retriever exposing ``base_retriever`` and
    ``base_compressor`` is recorded as two stages.
    """
    if not callable(getattr(retriever, "invoke", None)):
        raise TypeError("retriever must have an invoke(query) method (a LangChain retriever)")
    doc_id = _langchain_id(id_key)
    base: Any = getattr(retriever, "base_retriever", None) if decompose else None
    compressor: Any = getattr(retriever, "base_compressor", None) if decompose else None
    two_step = callable(getattr(base, "invoke", None)) and callable(
        getattr(compressor, "compress_documents", None)
    )

    def no_score(_: Any) -> None:
        return None

    def pipeline(request: PipelineRequest) -> PipelineResult:
        text = _require_text(request)
        if invoke_kwargs is None:
            _reject_unforwarded(request, "pass invoke_kwargs")
            kwargs: Mapping[str, Any] = {}
        else:
            kwargs = invoke_kwargs(request)
        rec = _Recorder()
        if two_step:
            docs, ms = _timed(lambda: list(base.invoke(text, **kwargs)))
            rec.add(f"retrieve:{_label(base)}", "custom", _unique_hits(docs, doc_id, no_score), ms)
            kept, ms = _timed(lambda: list(compressor.compress_documents(docs, text)))
            rec.add(
                f"compress:{_label(compressor)}",
                "rerank",
                _unique_hits(kept, doc_id, no_score),
                ms,
            )
        else:
            docs, ms = _timed(lambda: list(retriever.invoke(text, **kwargs)))
            rec.add(
                f"retrieve:{_label(retriever)}", "custom", _unique_hits(docs, doc_id, no_score), ms
            )
        return rec.result(request.k)

    return pipeline


# --- LlamaIndex ----------------------------------------------------------------------------


def _llamaindex_id(id_key: str | None) -> Callable[[Any], Any]:
    if id_key is None:
        return lambda item: item.node.node_id
    if id_key == "ref_doc_id":
        return lambda item: item.node.ref_doc_id
    return lambda item: (item.node.metadata or {}).get(id_key)


def _postprocess(post: Any, nodes: list[Any], text: str) -> list[Any]:
    return list(post.postprocess_nodes(nodes, query_str=text))


def llamaindex_pipeline(
    retriever: Any,
    *,
    id_key: str | None = None,
    postprocessors: Sequence[Any] = (),
) -> Callable[[PipelineRequest], PipelineResult]:
    """Wrap a LlamaIndex retriever (anything with ``retrieve(query)``).

    ``retriever`` may also be a function ``request -> retriever`` for per-query settings,
    e.g. ``lambda r: index.as_retriever(similarity_top_k=r.k, filters=...)``; a query with
    filters or tenant context fails unless such a function is given. ``id_key`` is
    ``None`` for ``node_id``, ``"ref_doc_id"`` for the source document id, or a metadata
    field. Each of ``postprocessors`` (rerankers, similarity cut-offs) runs in order on
    the previous stage's nodes and is recorded as its own stage.
    """
    factory: Callable[[PipelineRequest], Any] | None = None
    if not callable(getattr(retriever, "retrieve", None)):
        if not callable(retriever):
            raise TypeError("retriever must have a retrieve(query) method or be a function")
        factory = retriever
    for post in postprocessors:
        if not callable(getattr(post, "postprocess_nodes", None)):
            raise TypeError(f"{_label(post)} has no postprocess_nodes method")
    node_id = _llamaindex_id(id_key)

    def score(item: Any) -> Any:
        return getattr(item, "score", None)

    def pipeline(request: PipelineRequest) -> PipelineResult:
        text = _require_text(request)
        if factory is None:
            _reject_unforwarded(request, "pass a function that builds a filtered retriever")
            active = retriever
        else:
            active = factory(request)
        rec = _Recorder()
        nodes, ms = _timed(lambda: list(active.retrieve(text)))
        rec.add(f"retrieve:{_label(active)}", "custom", _unique_hits(nodes, node_id, score), ms)
        for i, post in enumerate(postprocessors, 1):
            nodes, ms = _timed(partial(_postprocess, post, nodes, text))
            # Numbered: two postprocessors of the same class still get unique stage names.
            name = f"postprocess{i}:{_label(post)}"
            rec.add(name, "rerank", _unique_hits(nodes, node_id, score), ms)
        return rec.result(request.k)

    return pipeline
