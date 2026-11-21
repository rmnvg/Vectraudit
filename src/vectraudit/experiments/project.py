"""``experiment project``: what will each candidate's recall be on the whole corpus?

Re-embedding a large corpus takes days, so candidates are compared on a sample, where recall is
inflated. This draws nested, seeded samples of the corpus that always hold every labelled
document, embeds only the largest one per candidate, measures exact recall@k at each size and
projects it to the full size with an interval (:mod:`vectraudit.experiments.scale`). It ranks
the candidates at full size and gives the chance that the sample's leader is still the leader.
It refuses to project beyond ``max_multiple`` times the largest measured size (invariant 26).

Search is exact here, whatever index a candidate declares: the projection is of the
embedding and chunking, not of an approximate index's own loss.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.families import family_id
from vectraudit.experiments.chunking import chunk_corpus
from vectraudit.experiments.corpus import Corpus, make_corpus
from vectraudit.experiments.embedders import Embedder, EmbeddingCache, make_embedder
from vectraudit.experiments.labels import document_labels
from vectraudit.experiments.manifest import CandidateSpec, ExperimentError, Manifest
from vectraudit.experiments.runner import _load_inputs
from vectraudit.experiments.scale import Floats, Ints, Thinned, measured, project

FORMAT = "vectraudit.scale_projection"
DEFAULT_FRACTIONS = (0.04, 0.06, 0.08, 0.1)
"""The sizes the 48.0 spike validated."""
DEFAULT_MAX_MULTIPLE = 10.0
"""The spike validated a tenfold extrapolation (from 10% of the corpus); see DECISIONS 48.0."""
BLOCK = 64
ASSUMPTIONS = (
    "The corpus file (or the documents sampled from it) is a uniform random share of the full "
    "corpus apart from the labelled documents, which are always included.",
    "Recall is of exact search over each candidate's embeddings and chunks; an approximate "
    "index loses more, which is not projected.",
    "The projection is for these questions: their full-corpus recall, not that of questions "
    "not asked.",
    "Documents added to the full corpus later, or unlike the ones sampled, are not modelled.",
)
Progress = Callable[[str], None]


def plan_sizes(
    corpus_size: int,
    labelled: int,
    fractions: Sequence[float],
    full_size: int,
    max_multiple: float,
) -> list[int]:
    """Nested sample sizes in documents; refuses an extrapolation beyond the allowed multiple."""
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ExperimentError("fractions must be between 0 and 1")
    if full_size < corpus_size:
        raise ExperimentError(
            f"the full size ({full_size}) is smaller than the corpus file ({corpus_size})"
        )
    if max_multiple < 1:
        raise ExperimentError("max_multiple must be at least 1")
    sizes = sorted({min(corpus_size, round(f * corpus_size)) for f in fractions})
    if sizes[0] <= labelled:
        raise ExperimentError(
            f"the smallest sample ({sizes[0]} documents) holds no more than the "
            f"{labelled} labelled documents; use larger fractions"
        )
    largest = sizes[-1]
    if full_size > max_multiple * largest:
        need = math.ceil(full_size / max_multiple)
        raise ExperimentError(
            f"projecting to {full_size} documents from {largest} is a "
            f"{full_size / largest:.1f}-fold extrapolation, beyond the allowed "
            f"{max_multiple:g}-fold; measure at least {need} documents (a larger fraction or "
            "corpus file), or pass a larger max multiple"
        )
    return sizes


def _pairs(dataset: Dataset) -> tuple[list[tuple[int, str]], Ints, list[float]]:
    """(question index, relevant document) pairs, their family units and weights."""
    rows: list[tuple[int, str]] = []
    questions: dict[str, list[int]] = {}
    for i, q in enumerate(dataset.queries):
        relevant = sorted(d for d, grade in document_labels(q).items() if grade > 0)
        if not relevant:
            continue
        fam = family_id({"query_id": q.query_id, "family_id": q.family_id})
        questions.setdefault(fam, []).append(i)
        rows += [(i, d) for d in relevant]
    if not rows:
        raise ExperimentError("no question has a relevant document to project recall for")
    families = list(questions)
    unit_of = {i: n for n, fam in enumerate(families) for i in questions[fam]}
    per_question = {i: sum(1 for j, _ in rows if j == i) for i in unit_of}
    weights = [
        1.0 / len(families) / len(questions[families[unit_of[i]]]) / per_question[i]
        for i, _ in rows
    ]
    units = np.array([unit_of[i] for i, _ in rows], dtype=np.int64)
    return rows, units, weights


def _doc_scores(
    queries: Floats, vectors: Floats, doc_index: Ints, n_docs: int, metric: str
) -> Floats:
    """Best chunk score per document: (questions, documents); higher is closer."""
    order = np.argsort(doc_index, kind="stable")
    vectors, doc_index = vectors[order], doc_index[order]
    starts = np.flatnonzero(np.r_[True, doc_index[1:] != doc_index[:-1]])
    present = doc_index[starts]
    out = np.full((len(queries), n_docs), -np.inf)
    for start in range(0, len(queries), BLOCK):
        q = queries[start : start + BLOCK]
        if metric == "ip":
            s = q @ vectors.T
        else:
            s = -((q * q).sum(1)[:, None] - 2 * q @ vectors.T + (vectors * vectors).sum(1)[None, :])
        out[start : start + BLOCK][:, present] = np.maximum.reduceat(s, starts, axis=1)
    return out


def project_experiment(
    manifest: Manifest,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    full_size: int | None = None,
    max_multiple: float = DEFAULT_MAX_MULTIPLE,
    seed: int = 0,
    draws: int = 2000,
    embedders: Mapping[str, Embedder] | None = None,
    cache_root: Path | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Measure each candidate on nested samples and project recall@k to the full size."""
    corpus, dataset = _load_inputs(manifest)
    rows, units, weights = _pairs(dataset)
    labelled_ids = sorted(
        {d for q in dataset.queries for d in document_labels(q)} & set(corpus.by_id)
    )
    full = len(corpus) if full_size is None else full_size
    sizes = plan_sizes(len(corpus), len(labelled_ids), fractions, full, max_multiple)
    labelled_set = set(labelled_ids)
    others = [d.doc_id for d in corpus.documents if d.doc_id not in labelled_set]
    order = [others[i] for i in np.random.default_rng(seed).permutation(len(others))]
    takes = [size - len(labelled_ids) for size in sizes]
    sample_ids = labelled_ids + order[: takes[-1]]
    position = {d: i for i, d in enumerate(sample_ids)}
    sample = make_corpus([corpus.by_id[d] for d in sample_ids])
    unlabelled_full = full - len(labelled_ids)
    n_lab = len(labelled_ids)
    weight = np.array(weights)
    cache = EmbeddingCache(cache_root)
    if progress is not None:
        progress("loading embedders")
    loaded = {spec.name: make_embedder(spec.embedder, embedders) for spec in manifest.candidates}
    candidates: list[dict[str, Any]] = []
    thinned: list[list[Thinned]] = []
    for spec in manifest.candidates:
        if progress is not None:
            progress(f"{spec.name}: embedding {len(sample)} sampled documents")
        counts, stats = _counts(
            spec, loaded[spec.name], sample, dataset, rows, position, n_lab, cache
        )
        a, beaters = counts
        per_size = []
        for take in takes:
            u = np.array([int(np.searchsorted(b, n_lab + take)) for b in beaters], dtype=np.int64)
            per_size.append(Thinned(a, u, take / unlabelled_full, units, weight))
        thinned.append(per_size)
        cost = project_cost(stats, full, n_lab, seed=seed)
        candidates.append({"name": spec.name, "candidate": spec.describe(), "cost": cost})
    if progress is not None:
        progress("projecting")
    largest = [t[-1] for t in thinned]
    results = project(largest, manifest.k, unlabelled_full, seed=seed, draws=draws)
    stacked = np.stack([r["draws"] for r in results])
    best = stacked == stacked.max(axis=0, keepdims=True)
    for entry, per_size, result, wins in zip(candidates, thinned, results, best, strict=True):
        entry["measured"] = [
            {"documents": size, "recall": measured(t, manifest.k)}
            for size, t in zip(sizes, per_size, strict=True)
        ]
        entry["projected"] = {"recall": result["recall"], "interval": result["interval"]}
        entry["sample_recall"] = entry["measured"][-1]["recall"]
        entry["inflation"] = entry["sample_recall"] - result["recall"]
        entry["p_best_at_full_size"] = float(wins.mean())
    leader = max(candidates, key=lambda c: (c["sample_recall"], c["name"]))
    ranking = sorted(candidates, key=lambda c: -c["projected"]["recall"])
    report: dict[str, Any] = {
        "format": FORMAT,
        "format_version": 1,
        "status": "projection",
        "metric": "recall",
        "k": manifest.k,
        "experiment": manifest.name,
        "corpus_documents": len(corpus),
        "full_size": full,
        "measured_sizes": sizes,
        "max_multiple": max_multiple,
        "extrapolation": full / sizes[-1],
        "labelled_documents": n_lab,
        "questions": len({i for i, _ in rows}),
        "pairs": len(rows),
        "seed": seed,
        "sample_sha256": sample.sha256,
        "candidates": candidates,
        "ranking": [c["name"] for c in ranking],
        "sample_leader": leader["name"],
        "p_leader_stays": leader["p_best_at_full_size"],
        "assumptions": list(ASSUMPTIONS),
    }
    report["message"] = summary(report)
    return report


def _counts(
    spec: CandidateSpec,
    embedder: Embedder,
    sample: Corpus,
    dataset: Dataset,
    rows: Sequence[tuple[int, str]],
    position: Mapping[str, int],
    labelled_count: int,
    cache: EmbeddingCache,
) -> tuple[tuple[Ints, list[Ints]], dict[str, Any]]:
    """Per pair: labelled documents above the relevant one, and the sorted sample positions
    of the unlabelled ones above it; plus the embedding cost measured on the sample.

    Labelled documents hold the first ``labelled_count`` positions of the sample."""
    chunks = chunk_corpus(sample, spec.chunking)
    if not chunks:
        raise ExperimentError(f"{spec.name}: chunking produced no chunks.")
    started = time.perf_counter()
    vectors, cached = cache.get_or_compute(
        spec.embedder,
        sample.sha256,
        spec.chunking.label,
        len(chunks),
        lambda: embedder.embed_documents([c.text for c in chunks]),
    )
    seconds = None if cached else time.perf_counter() - started
    queries = np.asarray(
        embedder.embed_queries([q.text or "" for q in dataset.queries]), dtype=np.float64
    )
    doc_index = np.array([position[c.doc_id] for c in chunks], dtype=np.int64)
    scores = _doc_scores(
        queries, np.asarray(vectors, dtype=np.float64), doc_index, len(position), spec.index.metric
    )
    a_list, beaters = [], []
    for i, doc in rows:
        row = scores[i]
        above = np.flatnonzero(row > row[position[doc]])
        a_list.append(int((above < labelled_count).sum()))
        beaters.append(np.sort(above[above >= labelled_count]))
    per_doc = np.bincount(doc_index, minlength=len(position))
    stats = {
        "chunks": len(chunks),
        "dimension": int(vectors.shape[1]),
        "embedded_documents": len(sample),
        "embed_seconds": seconds,
        "vector_bytes": int(vectors.shape[0] * vectors.shape[1] * 4),
        "labelled_chunks": int(per_doc[:labelled_count].sum()),
        "unlabelled_chunks_per_document": per_doc[labelled_count:].astype(np.int64),
    }
    return (np.array(a_list, dtype=np.int64), beaters), stats


def project_cost(
    stats: Mapping[str, Any], full_size: int, labelled: int, *, seed: int = 0, draws: int = 2000
) -> dict[str, Any]:
    """Chunks, embedding time and vector storage at the full size, scaled from the sample.

    Chunks per document come from the sampled unlabelled documents, with a 95% bootstrap
    interval over them; labelled documents are counted as they are. Embedding time scales the
    measured seconds per chunk (this machine, this batch size); it is unknown when the sample's
    vectors came from the cache. Storage is float32 vectors only: an index adds its own
    overhead, and a compressed one stores less.
    """
    per_doc = np.asarray(stats["unlabelled_chunks_per_document"], dtype=np.float64)
    if len(per_doc) == 0:
        raise ExperimentError("the sample holds no unlabelled documents to scale costs from")
    rng = np.random.default_rng(seed)
    means = per_doc[rng.integers(0, len(per_doc), (draws, len(per_doc)))].mean(axis=1)
    others = full_size - labelled

    def chunks_at(mean: float) -> float:
        return float(stats["labelled_chunks"] + others * mean)

    chunks = chunks_at(float(per_doc.mean()))
    chunk_range = [chunks_at(float(np.quantile(means, q))) for q in (0.025, 0.975)]
    bytes_per_chunk = stats["dimension"] * 4
    rate = None if stats["embed_seconds"] is None else stats["embed_seconds"] / stats["chunks"]
    return {
        "sample": {k: v for k, v in stats.items() if k != "unlabelled_chunks_per_document"},
        "chunks": chunks,
        "chunks_interval": chunk_range,
        "embed_seconds": None if rate is None else chunks * rate,
        "embed_seconds_interval": None if rate is None else [c * rate for c in chunk_range],
        "vector_bytes": chunks * bytes_per_chunk,
        "vector_bytes_interval": [c * bytes_per_chunk for c in chunk_range],
        "embed_note": (
            "Not measured: the sample's vectors came from the cache; run with --no-cache."
            if rate is None
            else "Seconds per chunk measured on this machine for the sample, scaled."
        ),
    }


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def _bytes(n: float) -> str:
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n / 1e3:.0f} kB"


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def summary(report: Mapping[str, Any]) -> str:
    """One paragraph: each candidate's projection and whether the sample's leader holds."""
    k = report["k"]
    parts = [
        f"Measured on up to {report['measured_sizes'][-1]} of {report['full_size']} documents "
        f"({report['extrapolation']:.1f}-fold extrapolation), {report['questions']} questions."
    ]
    for c in sorted(report["candidates"], key=lambda c: -c["projected"]["recall"]):
        lo, hi = c["projected"]["interval"]
        cost = c["cost"]
        when = (
            f"about {_duration(cost['embed_seconds'])} to embed on this machine"
            if cost["embed_seconds"] is not None
            else "embedding time unknown (vectors came from the cache)"
        )
        parts.append(
            f"{c['name']}: projected recall@{k} {_fmt(lo)} to {_fmt(hi)} (point "
            f"{_fmt(c['projected']['recall'])}; {_fmt(c['sample_recall'])} on the sample) at "
            f"{report['full_size']} documents; {when}; "
            f"{_bytes(cost['vector_bytes'])} of vectors."
        )
    if len(report["candidates"]) > 1:
        parts.append(
            f"The sample's leader, {report['sample_leader']}, is still the leader at full size "
            f"with probability {report['p_leader_stays']:.0%}."
        )
    return " ".join(parts)
