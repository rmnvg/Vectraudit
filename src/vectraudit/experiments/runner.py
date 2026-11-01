"""Run every candidate of an experiment and produce comparable run records.

Each candidate's run record has document ids as results (see
:mod:`vectraudit.experiments.labels`), the shared dataset hash and metric contract, so
:func:`vectraudit.evaluation.compare.compare_runs` compares any two candidates, and the
winner's run file works with ``vectraudit check`` and the hosted service like any other run.
Search latency covers the index search only; document and query embedding are timed
separately.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vectraudit.core._faiss import import_faiss
from vectraudit.evaluation.dataset import Dataset, load_dataset
from vectraudit.evaluation.families import family_mean
from vectraudit.evaluation.metrics import (
    LABEL_METRICS,
    metric_contract,
    repeat_quality,
    semantic_metrics,
)
from vectraudit.evaluation.runner import (
    RUN_FORMAT,
    RUN_FORMAT_VERSION,
    provenance,
    summarize,
)
from vectraudit.experiments.chunking import Chunk, chunk_corpus
from vectraudit.experiments.corpus import Corpus, CorpusError, load_corpus
from vectraudit.experiments.embedders import Embedder, EmbeddingCache, FloatArray, make_embedder
from vectraudit.experiments.labels import chunk_labels, document_labels, documents_in_order
from vectraudit.experiments.manifest import CandidateSpec, ExperimentError, Manifest

EXPERIMENT_FORMAT = "vectraudit.experiment"
Progress = Callable[[str], None]
_FAISS_PARAMS = {"nprobe": "nprobe", "ef_search": "efSearch"}


@dataclass
class CandidateResult:
    spec: CandidateSpec
    run: dict[str, Any]
    stats: dict[str, Any]
    span_summary: dict[str, Any] | None


def _load_inputs(manifest: Manifest) -> tuple[Corpus, Dataset]:
    try:
        corpus = load_corpus(
            manifest.corpus_path, id_field=manifest.id_field, text_field=manifest.text_field
        )
        dataset = load_dataset(manifest.queries_path)
    except (CorpusError, ValueError) as e:
        raise ExperimentError(str(e)) from e
    missing = [q.query_id for q in dataset.queries if not q.text]
    if missing:
        raise ExperimentError(
            f"{len(missing)} queries have no text (e.g. {missing[0]!r}); each candidate "
            "embeds query text with its own model."
        )
    unknown = sorted(
        {
            doc
            for q in dataset.queries
            for doc in [*q.relevant, *(s.doc_id for s in q.relevant_spans)]
            if corpus.get(doc) is None
        }
    )
    if unknown:
        raise ExperimentError(
            f"Labels refer to {len(unknown)} documents missing from the corpus "
            f"(e.g. {unknown[0]!r}); label source document ids, not chunk ids."
        )
    return corpus, dataset


def _build_index(spec: CandidateSpec, vectors: FloatArray) -> tuple[Any, Any, float]:
    """The candidate's index, an exact index when it is approximate, and build seconds."""
    faiss = import_faiss()
    metric = faiss.METRIC_INNER_PRODUCT if spec.index.metric == "ip" else faiss.METRIC_L2
    started = time.perf_counter()
    try:
        index = faiss.index_factory(vectors.shape[1], spec.index.factory, metric)
    except RuntimeError as e:
        raise ExperimentError(
            f"{spec.name}: invalid FAISS factory {spec.index.factory!r}: {e}"
        ) from e
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    ps = faiss.ParameterSpace()
    for key, value in spec.index.params.items():
        ps.set_index_parameter(index, _FAISS_PARAMS[key], value)
    seconds = time.perf_counter() - started
    exact = None
    if not spec.index.exact:
        exact = (
            faiss.IndexFlatIP(vectors.shape[1])
            if spec.index.metric == "ip"
            else faiss.IndexFlatL2(vectors.shape[1])
        )
        exact.add(vectors)
    return index, exact, seconds


def _ranked_documents(
    index: Any, query: FloatArray, k: int, chunks: list[Chunk], doc_of: Mapping[str, str]
) -> tuple[list[str], list[str], float]:
    """Top-k documents (searching deeper until k distinct ones), the chunk ranking, and ms."""
    depth = min(len(chunks), k * 3)
    total_ms = 0.0
    while True:
        started = time.perf_counter()
        _, rows = index.search(query.reshape(1, -1), depth)
        total_ms += (time.perf_counter() - started) * 1000
        chunk_ids = [chunks[r].chunk_id for r in rows[0] if r >= 0]
        docs = documents_in_order(chunk_ids, doc_of)
        if len(docs) >= k or depth >= len(chunks):
            return docs[:k], chunk_ids, total_ms
        depth = min(len(chunks), depth * 2)


def run_candidate(
    spec: CandidateSpec,
    corpus: Corpus,
    dataset: Dataset,
    k: int,
    *,
    experiment: str,
    embedders: Mapping[str, Embedder] | None = None,
    cache: EmbeddingCache | None = None,
    embedder: Embedder | None = None,
) -> CandidateResult:
    if any(q.context or q.history for q in dataset.queries):
        raise ExperimentError(
            "Permission contexts and conversation history require workbench pipeline adapters."
        )
    chunks = chunk_corpus(corpus, spec.chunking)
    if not chunks:
        raise ExperimentError(f"{spec.name}: chunking produced no chunks.")
    doc_of = {c.chunk_id: c.doc_id for c in chunks}
    embedder = embedder or make_embedder(spec.embedder, embedders)
    cache = cache or EmbeddingCache(None)
    started = time.perf_counter()
    vectors, cached = cache.get_or_compute(
        spec.embedder,
        corpus.sha256,
        spec.chunking.label,
        len(chunks),
        lambda: embedder.embed_documents([c.text for c in chunks]),
    )
    embed_docs_s = None if cached else time.perf_counter() - started
    if vectors.ndim != 2 or len(vectors) != len(chunks):
        raise ExperimentError(f"{spec.name}: the embedder returned the wrong shape.")
    index, exact_index, build_s = _build_index(spec, vectors)
    started = time.perf_counter()
    query_vectors = np.ascontiguousarray(
        embedder.embed_queries([q.text or "" for q in dataset.queries]), dtype=np.float32
    )
    embed_queries_s = time.perf_counter() - started

    queries = []
    span_rows = []
    for q, qv in zip(dataset.queries, query_vectors, strict=True):
        docs, chunk_ranking, latency_ms = _ranked_documents(index, qv, k, chunks, doc_of)
        exact_docs = None
        if exact_index is not None:
            exact_docs, _, _ = _ranked_documents(exact_index, qv, k, chunks, doc_of)
        labels = document_labels(q)
        queries.append(
            {
                "query_id": q.query_id,
                **({"family_id": q.family_id} if q.family_id is not None else {}),
                "tags": list(q.cohort_tags),
                "filtered": False,
                "ids": docs,
                "repeat_ids": [docs],
                "no_answer_correct": not docs if q.no_answer else None,
                "forbidden_ids": sorted(set(docs) & set(q.forbidden)),
                "permission_checked": bool(q.forbidden),
                "exact_ids": exact_docs,
                "latency_ms": latency_ms,
                **repeat_quality([docs], labels, k, exact_docs),
                "nondeterministic": False,
                "evidence": None,
                "error": None,
            }
        )
        if q.relevant_spans:
            span_labels = chunk_labels(q.relevant_spans, chunks)
            if span_labels:
                span_rows.append(
                    (q.family_id or q.query_id, semantic_metrics(chunk_ranking[:k], span_labels, k))
                )

    describe = spec.describe()
    run = {
        "format": RUN_FORMAT,
        "format_version": RUN_FORMAT_VERSION,
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": spec.name,
        "status": "complete",
        "error": None,
        "dataset": dataset.summary(),
        "connection": {"backend": "experiment", "experiment": experiment, **describe},
        "backend": {
            "backend": "experiment",
            "distance": spec.index.metric,
            "dim": int(vectors.shape[1]),
            "count": len(chunks),
            "target": {"experiment": experiment, "candidate": spec.name},
            "backend_version": None,
            "client_version": None,
            "config": {k_: v for k_, v in describe.items() if k_ != "name"},
            "snapshot": f"corpus:{corpus.sha256}",
        },
        "capabilities": {"search": True, "exact_reference": True},
        "search": {
            "k": k,
            "params": dict(spec.index.params),
            "embedder": spec.embedder.label,
            "evidence": False,
            "exact_reference": exact_index is not None,
            "exact_reference_note": None,
        },
        "timing": {
            "scope": "index search only (query embedding timed separately)",
            "warmup": 0,
            "repeats": 1,
            "statistic": "single search",
            "concurrency": 1,
            "embedding_s": embed_queries_s,
        },
        "metric_contract": metric_contract(k),
        "provenance": provenance(
            {
                "embedding_model": spec.embedder.label,
                "chunking": spec.chunking.label,
                "index": spec.index.factory,
            }
        ),
        "queries": queries,
    }
    run["summary"] = summarize(queries, len(dataset))
    faiss = import_faiss()
    stats = {
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "index_bytes": int(faiss.serialize_index(index).size),
        "embed_documents_s": embed_docs_s,
        "embedding_cached": cached,
        "embed_queries_s": embed_queries_s,
        "build_index_s": build_s,
    }
    span_summary = None
    if span_rows:
        span_summary = {
            "n": len(span_rows),
            "n_families": len({key for key, _ in span_rows}),
            "metrics": {
                m: family_mean((key, row[m]) for key, row in span_rows) for m in LABEL_METRICS
            },
        }
    return CandidateResult(spec, run, stats, span_summary)


def run_experiment(
    manifest: Manifest,
    *,
    embedders: Mapping[str, Embedder] | None = None,
    cache_root: Path | None = None,
    progress: Progress | None = None,
) -> tuple[dict[str, Any], list[CandidateResult]]:
    """Run all candidates; returns the experiment summary and per-candidate results."""
    corpus, dataset = _load_inputs(manifest)
    cache = EmbeddingCache(cache_root)
    # Load every embedder (and so torch, for model candidates) before FAISS builds any
    # index: on macOS both ship OpenMP, and torch initialising second aborts the process.
    if progress is not None:
        progress("loading embedders")
    loaded = {spec.name: make_embedder(spec.embedder, embedders) for spec in manifest.candidates}
    results = []
    for spec in manifest.candidates:
        if progress is not None:
            progress(f"{spec.name}: chunking, embedding and searching")
        results.append(
            run_candidate(
                spec,
                corpus,
                dataset,
                manifest.k,
                experiment=manifest.name,
                embedders=embedders,
                cache=cache,
                embedder=loaded[spec.name],
            )
        )
    summary = {
        "format": EXPERIMENT_FORMAT,
        "format_version": 1,
        "name": manifest.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": manifest.k,
        "corpus": {"sha256": corpus.sha256, "documents": len(corpus)},
        "dataset": dataset.summary(),
        "candidates": [
            {
                **r.spec.describe(),
                "run_id": r.run["run_id"],
                "stats": r.stats,
                "summary": r.run["summary"],
                "span_summary": r.span_summary,
            }
            for r in results
        ],
    }
    return summary, results
