"""Search with per-call parameters, exact ground truth, recall and IVF miss explanations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit.core import ivf as ivf_mod
from vectraudit.core._faiss import import_faiss
from vectraudit.core.embed import Embedder
from vectraudit.core.filtering import AllowedSet
from vectraudit.core.ivf import Assignments
from vectraudit.core.types import IndexKind, LoadedIndex, Metric
from vectraudit.core.vectors import VectorSource

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class SearchResult:
    """Top-k results for one query, best first. ``ids`` are user-facing; ``-1`` = empty slot."""

    ids: IntArray
    distances: FloatArray
    metric: Metric
    latency_ms: float = 0.0
    params: dict[str, int] = field(default_factory=dict)
    """Search parameters actually used (e.g. ``{"nprobe": 8}``)."""

    @property
    def valid_ids(self) -> IntArray:
        return self.ids[self.ids >= 0]


def _as_query(query: npt.ArrayLike, d: int) -> FloatArray:
    q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
    if q.shape[1] != d:
        raise ValueError(f"Query has dimension {q.shape[1]}, index expects {d}.")
    if not np.isfinite(q).all():
        raise ValueError("Query contains NaN or infinite values.")
    return q


def resolve_search_params(
    li: LoadedIndex,
    nprobe: int | None = None,
    ef_search: int | None = None,
    sel: Any = None,
) -> tuple[Any, dict[str, int]]:
    """Build a FAISS ``SearchParameters`` object without mutating the shared index.

    Unset values fall back to what the index currently has. Returns ``(params, used)``
    where ``used`` records the effective values for display. ``sel`` is a FAISS
    ``IDSelector`` (see :meth:`AllowedSet.selector`) restricting which ids may be returned.
    """
    faiss = import_faiss()
    used: dict[str, int] = {}
    params: Any = None
    if li.kind.is_ivf:
        if ef_search is not None:
            raise ValueError("efSearch only applies to HNSW indexes.")
        nlist = int(li.ivf.nlist)
        nprobe = int(li.ivf.nprobe) if nprobe is None else int(nprobe)
        if not 1 <= nprobe <= nlist:
            raise ValueError(f"nprobe must be between 1 and nlist={nlist}, got {nprobe}.")
        params = faiss.SearchParametersIVF(nprobe=nprobe, max_codes=int(li.ivf.max_codes), sel=sel)
        used["nprobe"] = nprobe
    elif li.kind.is_hnsw:
        if nprobe is not None:
            raise ValueError("nprobe only applies to IVF indexes.")
        ef_search = int(li.core.hnsw.efSearch) if ef_search is None else int(ef_search)
        if ef_search < 1:
            raise ValueError(f"efSearch must be >= 1, got {ef_search}.")
        params = faiss.SearchParametersHNSW(efSearch=ef_search, sel=sel)
        used["efSearch"] = ef_search
    elif nprobe is not None or ef_search is not None:
        raise ValueError(f"{li.kind.value} indexes take no search parameters.")
    elif sel is not None:
        params = faiss.SearchParameters(sel=sel)

    if params is not None and li.has_refine:
        # IndexRefine rejects plain params, and its own params default k_factor to 1,
        # so pass the index's k_factor through explicitly to keep results unchanged. The
        # selector must sit in ``base_index_params``: set on the outer object it is ignored.
        params = faiss.IndexRefineSearchParameters(
            k_factor=li.refine_k_factor, base_index_params=params
        )
    return params, used


def search(
    li: LoadedIndex,
    query: npt.ArrayLike,
    k: int,
    *,
    nprobe: int | None = None,
    ef_search: int | None = None,
    exclude_id: int | None = None,
    allowed: AllowedSet | None = None,
) -> SearchResult:
    """Search the index for one query vector (input space, dimension ``li.d``).

    ``nprobe``/``ef_search`` apply to this call only. ``exclude_id`` drops that id from the
    results (used when the query is itself a stored vector) while still returning ``k``.
    ``allowed`` restricts the search to those ids *inside* FAISS; fewer than ``k`` results
    come back (padded with ``-1``) when the probed cells or visited nodes hold too few of
    them, which is the failure this exists to measure.
    """
    if not li.is_supported:
        raise ValueError(f"Cannot search: {li.unsupported_reason}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    q = _as_query(query, li.d)
    selector = None if allowed is None else allowed.selector()
    params, used = resolve_search_params(li, nprobe, ef_search, selector)
    k_search = min(k + (exclude_id is not None), max(li.ntotal, 1))

    t0 = time.perf_counter()
    distances, ids = li.index.search(q, k_search, params=params)
    latency_ms = (time.perf_counter() - t0) * 1000

    ids, distances = _drop_excluded(ids[0], distances[0], exclude_id, k)
    return SearchResult(ids, distances, li.metric, latency_ms, used)


def _drop_excluded(
    ids: IntArray, distances: FloatArray, exclude_id: int | None, k: int
) -> tuple[IntArray, FloatArray]:
    if exclude_id is not None:
        keep = ids != exclude_id
        ids, distances = ids[keep], distances[keep]
    return ids[:k].astype(np.int64), distances[:k].astype(np.float32)


class GroundTruth:
    """Exact k-NN over a :class:`VectorSource`, using the same metric as the index.

    Brute force with ``faiss.knn`` directly over the source's array: unlike an
    ``IndexFlat``, it doesn't copy the vectors (1.5 GB saved at 1M x 384).
    """

    def __init__(self, source: VectorSource, metric: Metric) -> None:
        faiss = import_faiss()
        if metric is Metric.OTHER:
            raise ValueError("Ground truth needs an L2 or inner-product metric.")
        self.source = source
        self.metric = metric
        self.d = int(source.vectors.shape[1])
        self._faiss_metric = faiss.METRIC_L2 if metric is Metric.L2 else faiss.METRIC_INNER_PRODUCT
        self._subsets: dict[str, tuple[IntArray, FloatArray]] = {}
        """Source rows and vectors of recent allowed sets, keyed by fingerprint."""

    _SUBSET_CACHE = 2

    def _subset(self, allowed: AllowedSet) -> tuple[IntArray, FloatArray]:
        cached = self._subsets.get(allowed.fingerprint)
        if cached is None:
            rows = np.flatnonzero(np.isin(self.source.ids, allowed.ids)).astype(np.int64)
            cached = (rows, np.ascontiguousarray(self.source.vectors[rows], dtype=np.float32))
            while len(self._subsets) >= self._SUBSET_CACHE:
                self._subsets.pop(next(iter(self._subsets)))
            self._subsets[allowed.fingerprint] = cached
        return cached

    def _knn(
        self, q: FloatArray, k: int, allowed: AllowedSet | None = None
    ) -> tuple[FloatArray, IntArray]:
        """Exact k-NN as ``(distances, source rows)``; rows are ``-1`` for missing slots."""
        faiss = import_faiss()
        if allowed is None:
            distances, rows = faiss.knn(q, self.source.vectors, k, metric=self._faiss_metric)
            return distances, rows
        rows, vectors = self._subset(allowed)
        if len(rows) == 0:
            return (
                np.zeros((len(q), k), dtype=np.float32),
                np.full((len(q), k), -1, dtype=np.int64),
            )
        distances, local = faiss.knn(q, vectors, min(k, len(rows)), metric=self._faiss_metric)
        if local.shape[1] < k:
            pad = k - local.shape[1]
            distances = np.pad(distances, ((0, 0), (0, pad)))
            local = np.pad(local, ((0, 0), (0, pad)), constant_values=-1)
        return distances, np.where(local >= 0, rows[np.maximum(local, 0)], -1)

    @property
    def reconstructed(self) -> bool:
        """True when computed on vectors decoded from the index, not the raw vectors."""
        return self.source.reconstructed

    def search(
        self,
        query: npt.ArrayLike,
        k: int,
        exclude_id: int | None = None,
        allowed: AllowedSet | None = None,
    ) -> SearchResult:
        """Exact top-k (user-facing ids), optionally excluding one id.

        With ``allowed``, the exact neighbours *among the allowed ids*: the truth a filtered
        search is measured against.
        """
        q = _as_query(query, self.d)
        if len(self.source) == 0:
            # Nothing stored: no neighbours (faiss.knn would report row -1, which has no id).
            empty = np.empty(0, dtype=np.int64)
            return SearchResult(empty, np.empty(0, dtype=np.float32), self.metric, 0.0)
        k_search = min(k + (exclude_id is not None), max(len(self.source), 1))
        t0 = time.perf_counter()
        distances, rows = self._knn(q, k_search, allowed)
        latency_ms = (time.perf_counter() - t0) * 1000
        rows = rows[0]
        ids = np.where(rows >= 0, self.source.ids[np.maximum(rows, 0)], -1)
        ids, dist = _drop_excluded(ids, distances[0], exclude_id, k)
        return SearchResult(ids, dist, self.metric, latency_ms)

    def search_batch(
        self,
        queries: npt.ArrayLike,
        k: int,
        exclude_ids: npt.ArrayLike | None = None,
        allowed: AllowedSet | None = None,
    ) -> IntArray:
        """Exact top-k user ids for many queries, shape ``(n, k)`` (``-1`` pads short rows).

        ``exclude_ids[i]`` is dropped from row ``i`` (queries that are stored vectors).
        ``allowed`` restricts the truth to those ids (the set is cached by fingerprint).
        """
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.d:
            raise ValueError(f"Queries must have shape (n, {self.d}), got {q.shape}.")
        if not np.isfinite(q).all():
            raise ValueError("Queries contain NaN or infinite values.")
        excl = None if exclude_ids is None else np.asarray(exclude_ids, dtype=np.int64)
        out = np.full((len(q), k), -1, dtype=np.int64)
        if len(self.source) == 0 or len(q) == 0:
            return out
        k_search = min(k + (excl is not None), max(len(self.source), 1))
        _, rows = self._knn(q, k_search, allowed)
        ids = np.where(rows >= 0, self.source.ids[np.maximum(rows, 0)], -1)
        for i in range(len(q)):
            row = ids[i]
            if excl is not None:
                row = row[row != excl[i]]
            out[i, : min(k, len(row))] = row[:k]
        return out


def recall_at_k(found: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """Fraction of the true neighbours (ignoring ``-1`` slots) present in ``found``.

    Returns 1.0 when there are no true neighbours to find.
    """
    truth_ids = np.asarray(truth, dtype=np.int64)
    truth_ids = truth_ids[truth_ids >= 0]
    if len(truth_ids) == 0:
        return 1.0
    return float(np.isin(truth_ids, np.asarray(found, dtype=np.int64)).mean())


class QueryError(ValueError):
    """The query can't be resolved to a vector."""


class QueryKind(str, Enum):
    ID = "id"
    VECTOR = "vector"
    TEXT = "text"
    ROW = "row"
    """A row of the evaluation query set (e.g. ``--queries``)."""


@dataclass(frozen=True)
class ResolvedQuery:
    """A query turned into an input-space vector.

    ``exclude_id`` is set for queries by stored id, so the vector doesn't find itself.
    """

    vector: FloatArray
    kind: QueryKind
    exclude_id: int | None = None
    text: str | None = None
    row: int | None = None
    """Row in the evaluation query set, for ``QueryKind.ROW``."""


def resolve_query(
    li: LoadedIndex,
    *,
    id: int | None = None,
    vector: npt.ArrayLike | None = None,
    text: str | None = None,
    source: VectorSource | None = None,
    embedder: Embedder | None = None,
    row: int | None = None,
    queries: npt.ArrayLike | None = None,
) -> ResolvedQuery:
    """Resolve exactly one of ``id`` (needs ``source``), ``vector``, ``text`` (needs
    ``embedder``) or ``row`` of ``queries`` (an evaluation set, such as held-out queries)
    to a query vector of dimension ``li.d``.

    Rows are the numbering sweeps and comparisons report for given queries, so a failing
    evaluation query can be explained without a stored id.
    """
    named = (("id", id), ("vector", vector), ("text", text), ("row", row))
    given = [name for name, v in named if v is not None]
    if len(given) != 1:
        raise QueryError(f"Give exactly one of id, vector, text or row (got {given or 'none'}).")

    if row is not None:
        if queries is None:
            raise QueryError(
                "Querying by row needs an evaluation query set (start with --queries)."
            )
        q = np.asarray(queries)
        if not 0 <= row < len(q):
            raise QueryError(f"Row {row} is outside the {len(q)} evaluation queries.")
        vec = _check_dim(q[row], li.d, f"Evaluation query {row}")
        return ResolvedQuery(vec, QueryKind.ROW, row=int(row))

    if id is not None:
        if source is None:
            raise QueryError("Querying by id needs stored vectors (raw or reconstructed).")
        try:
            vec = source.get([id])[0]
        except KeyError:
            raise QueryError(f"Id {id} is not in the index.") from None
        return ResolvedQuery(vec, QueryKind.ID, exclude_id=int(id))

    if vector is not None:
        return ResolvedQuery(_check_dim(vector, li.d, "Query vector"), QueryKind.VECTOR)

    assert text is not None
    if embedder is None:
        raise QueryError("Text queries need an embedder (e.g. --embedder all-MiniLM-L6-v2).")
    vec = _check_dim(embedder(text), li.d, "Embedder output")
    return ResolvedQuery(vec, QueryKind.TEXT, text=text)


def _check_dim(v: npt.ArrayLike, d: int, what: str) -> FloatArray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] != d:
        raise QueryError(f"{what} has shape {arr.shape}, the index expects ({d},).")
    if not np.isfinite(arr).all():
        raise QueryError(f"{what} contains NaN or infinite values.")
    return arr


class MissReason(str, Enum):
    """Why a true nearest neighbour did or didn't make the approximate top-k."""

    FOUND = "FOUND"
    CELL_NOT_PROBED = "CELL_NOT_PROBED"
    """Its inverted list ranked beyond ``nprobe`` in the probe order."""
    QUANTIZATION = "QUANTIZATION"
    """Its list was probed, but PQ/SQ approximate distances pushed it out of the top-k."""
    TRANSFORM = "TRANSFORM"
    """Its list was probed, but a dimension-reducing PreTransform (e.g. PCA) changed the
    ranking: the index compares vectors in the reduced space."""
    RANKED_OUT = "RANKED_OUT"
    """Its list was probed, codes are exact and no transform loses information, yet it
    missed the top-k: ties, or ground truth on vectors that differ from what is stored."""


@dataclass(frozen=True)
class NeighbourTrace:
    """One exact nearest neighbour and what happened to it."""

    id: int
    truth_rank: int
    distance: float
    """Exact distance (or similarity, for IP) to the query."""
    list_no: int
    probe_rank: int
    """Position of its list in the probe order; it is probed iff ``probe_rank < nprobe``."""
    reason: MissReason
    found_rank: int | None
    """Its rank in the approximate results, if found."""


@dataclass(frozen=True)
class IvfTrace:
    """How an IVF search visited cells, and where the true neighbours were."""

    nprobe: int
    probe_order: IntArray
    """All list numbers in probe order: the ``nprobe`` lists searched come first. With a
    flat coarse quantizer this is every list by centroid distance; with an approximate one
    (HNSW), the probed lists in FAISS's order, then the rest by exact centroid distance."""
    centroid_distances: FloatArray
    """Query-to-centroid distance (similarity for IP), aligned with ``probe_order``."""
    neighbours: list[NeighbourTrace]
    min_nprobe_for_all: int
    """Smallest nprobe whose probed lists contain every true neighbour."""
    result_list_nos: IntArray
    """List number of each approximate result (``-1`` for empty slots)."""
    allowed_in_probed: int | None = None
    """Filtered search only: how many allowed vectors sit in the probed lists. Fewer than
    ``k`` means the probed cells cannot fill the result, however good the ranking."""
    allowed_total: int | None = None
    """Filtered search only: how many vectors the filter allows in the whole index."""
    n_probed: int | None = None
    """How many lists were actually probed, when fewer than ``nprobe``: an HNSW coarse
    quantizer can return fewer lists than asked for. ``None`` means ``nprobe``."""

    @property
    def probed_count(self) -> int:
        return self.nprobe if self.n_probed is None else self.n_probed

    @property
    def probed_lists(self) -> IntArray:
        return self.probe_order[: self.probed_count]

    def reason_counts(self) -> dict[MissReason, int]:
        counts = dict.fromkeys(MissReason, 0)
        for n in self.neighbours:
            counts[n.reason] += 1
        return counts


def probed_miss_reason(li: LoadedIndex) -> MissReason:
    """Why this index misses a true neighbour whose inverted list *was* probed.

    When both apply (e.g. PCA + PQ) this is QUANTIZATION; separating them would need
    exact distances in the transformed space.
    """
    if li.kind in (IndexKind.IVF_PQ, IndexKind.IVF_SQ):
        return MissReason.QUANTIZATION
    if any(t.d_out < t.d_in for t in li.transforms):
        return MissReason.TRANSFORM
    return MissReason.RANKED_OUT


def _approximate_probe_order(
    li: LoadedIndex, q_core: npt.NDArray[np.float32], nprobe: int
) -> tuple[IntArray, FloatArray, int]:
    """Probe order for a quantizer that is not flat: what FAISS probes, then the rest.

    The first entries are the quantizer's own answer at ``nprobe`` (the lists
    ``IndexIVF::search`` visits, possibly fewer than ``nprobe``); the remaining lists follow
    by exact centroid distance. Also returns how many lists were probed.
    """
    assert li.ivf is not None
    nlist = int(li.ivf.nlist)
    probed = ivf_mod.probed_lists(li, q_core, nprobe)[0]
    probed = probed[probed >= 0]
    try:
        cents = ivf_mod.centroids(li)
    except RuntimeError:  # a quantizer that cannot reconstruct its centroids
        rest = np.setdiff1d(np.arange(nlist, dtype=np.int64), probed)
        dist = np.full(nlist, np.nan, dtype=np.float32)
        return np.concatenate([probed, rest]), dist, len(probed)
    q = q_core.reshape(-1)
    if li.metric is Metric.IP:
        all_d = cents @ q
        by_closeness = np.argsort(-all_d, kind="stable")
    else:
        all_d = ((cents - q) ** 2).sum(axis=1)
        by_closeness = np.argsort(all_d, kind="stable")
    rest = by_closeness[~np.isin(by_closeness, probed)]
    order = np.concatenate([probed, rest]).astype(np.int64)
    return order, all_d[order].astype(np.float32), len(probed)


def trace_ivf(
    li: LoadedIndex,
    query: npt.ArrayLike,
    result: SearchResult,
    truth: SearchResult,
    assignments: Assignments,
    allowed: AllowedSet | None = None,
) -> IvfTrace:
    """Explain an IVF search: the full probe order and the fate of each true neighbour.

    ``query`` is the input-space vector passed to :func:`search`; it goes through any
    PreTransform before the coarse quantizer, exactly as FAISS does.
    """
    if li.ivf is None:
        raise ValueError("IVF trace needs an IVF index.")
    nprobe = int(result.params.get("nprobe", li.ivf.nprobe))
    q_core = li.to_core_space(_as_query(query, li.d))
    nlist = int(li.ivf.nlist)
    truth_ids = truth.ids
    truth_lists = assignments.lookup(truth_ids)
    if ivf_mod.probe_order_is_exact(li):
        centroid_dist, order = li.ivf.quantizer.search(q_core, nlist)
        order = order[0].astype(np.int64)
        distances = centroid_dist[0].astype(np.float32)
        probe_rank = np.full(nlist + 1, nlist, dtype=np.int64)
        probe_rank[order[order >= 0]] = np.arange(nlist)[order >= 0]
        truth_ranks = probe_rank[np.where(truth_lists >= 0, truth_lists, nlist)]
        n_probed = None
    else:
        order, distances, probed_n = _approximate_probe_order(li, q_core, nprobe)
        n_probed = probed_n if probed_n < nprobe else None
        truth_ranks = ivf_mod.list_probe_ranks(li, q_core, truth_lists, nprobe)

    found_rank = {int(i): r for r, i in enumerate(result.ids) if i >= 0}
    probed_miss = probed_miss_reason(li)
    neighbours: list[NeighbourTrace] = []
    for rank, (nid, dist, list_no, pr) in enumerate(
        zip(truth_ids, truth.distances, truth_lists, truth_ranks.tolist(), strict=True)
    ):
        if nid < 0:
            continue
        if int(nid) in found_rank:
            reason = MissReason.FOUND
        elif pr >= nprobe:
            reason = MissReason.CELL_NOT_PROBED
        else:
            reason = probed_miss
        neighbours.append(
            NeighbourTrace(
                int(nid), rank, float(dist), int(list_no), pr, reason, found_rank.get(int(nid))
            )
        )

    allowed_in_probed = None
    if allowed is not None:
        lists = assignments.lookup(allowed.ids)
        probed_upto = nprobe if n_probed is None else n_probed
        allowed_in_probed = int(np.isin(lists, order[:probed_upto]).sum())

    return IvfTrace(
        nprobe=nprobe,
        probe_order=order,
        centroid_distances=distances,
        neighbours=neighbours,
        min_nprobe_for_all=min(nlist, max((n.probe_rank for n in neighbours), default=0) + 1),
        result_list_nos=assignments.lookup(result.ids),
        allowed_in_probed=allowed_in_probed,
        allowed_total=None if allowed is None else len(allowed),
        n_probed=n_probed,
    )


@dataclass(frozen=True)
class QueryReport:
    """Everything the query explorer shows for one query."""

    query: ResolvedQuery
    result: SearchResult
    truth: SearchResult | None = None
    recall: float | None = None
    truth_reconstructed: bool = False
    """Ground truth was computed on reconstructed vectors, so PQ/SQ error isn't measured."""
    ivf_trace: IvfTrace | None = None
    reranked: SearchResult | None = None
    reranked_recall: float | None = None
    candidate_count: int | None = None


def rerank(
    source: VectorSource, query: npt.ArrayLike, candidates: SearchResult, k: int
) -> SearchResult:
    """Rank candidate ids by exact input-space distances, without searching the corpus.

    Latency includes candidate retrieval plus raw-vector lookup and reranking.
    """
    if source.reconstructed:
        raise ValueError("Exact reranking needs raw vectors.")
    if k < 1 or candidates.metric is Metric.OTHER:
        raise ValueError("Reranking needs k >= 1 and an L2 or inner-product metric.")
    q = _as_query(query, source.vectors.shape[1])[0].astype(np.float64)
    start = time.perf_counter()
    ids = candidates.valid_ids
    x = source.get(ids).astype(np.float64)
    if candidates.metric is Metric.IP:
        distances = x @ q
        order = np.argsort(-distances, kind="stable")[:k]
    else:
        x -= q
        distances = np.einsum("ij,ij->i", x, x)
        order = np.argsort(distances, kind="stable")[:k]
    return SearchResult(
        ids[order],
        distances[order].astype(np.float32),
        candidates.metric,
        candidates.latency_ms + (time.perf_counter() - start) * 1000,
        candidates.params,
    )


def explain_query(
    li: LoadedIndex,
    query: ResolvedQuery,
    k: int,
    *,
    nprobe: int | None = None,
    ef_search: int | None = None,
    ground_truth: GroundTruth | None = None,
    assignments: Assignments | None = None,
    rerank_source: VectorSource | None = None,
    candidates: int | None = None,
    allowed: AllowedSet | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> QueryReport:
    """Search, compare with exact ground truth (if given) and trace IVF probing (if IVF).

    With ``allowed``, both the search and the ground truth are restricted to those ids.
    """
    report = progress or (lambda _f, _m: None)
    report(0.3, "Searching index")
    result = search(
        li,
        query.vector,
        k,
        nprobe=nprobe,
        ef_search=ef_search,
        exclude_id=query.exclude_id,
        allowed=allowed,
    )
    reranked = None
    candidate_count = None
    if candidates is not None:
        if candidates < k or rerank_source is None or rerank_source.reconstructed:
            raise ValueError("Reranking needs raw vectors and candidates >= k.")
        report(0.4, "Retrieving rerank candidates")
        pool = search(
            li,
            query.vector,
            candidates,
            nprobe=nprobe,
            ef_search=ef_search,
            exclude_id=query.exclude_id,
            allowed=allowed,
        )
        report(0.5, "Reranking candidates")
        reranked = rerank(rerank_source, query.vector, pool, k)
        candidate_count = len(pool.valid_ids)
    if ground_truth is None:
        return QueryReport(query, result, reranked=reranked, candidate_count=candidate_count)
    report(0.6, "Computing exact neighbours")
    truth = ground_truth.search(query.vector, k, exclude_id=query.exclude_id, allowed=allowed)
    report(0.8, "Explaining missed neighbours")
    trace = None
    if li.kind.is_ivf and assignments is not None:
        trace = trace_ivf(li, query.vector, result, truth, assignments, allowed)
    return QueryReport(
        query=query,
        result=result,
        truth=truth,
        recall=recall_at_k(result.ids, truth.ids),
        truth_reconstructed=ground_truth.reconstructed,
        ivf_trace=trace,
        reranked=reranked,
        reranked_recall=recall_at_k(reranked.ids, truth.ids) if reranked else None,
        candidate_count=candidate_count,
    )
