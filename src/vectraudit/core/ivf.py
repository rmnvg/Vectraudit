"""IVF inspection: inverted-list sizes, balance, members, centroids and assignments.

For ``IndexPreTransform``-wrapped indexes the centroids live in the *transformed* space
(dimension ``LoadedIndex.core_d``), not the input space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit.core._faiss import import_faiss
from vectraudit.core.types import LoadedIndex

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]


class NotAnIVFIndexError(ValueError):
    """The loaded index has no IVF core."""


@dataclass(frozen=True)
class ListStats:
    """Summary of how vectors are spread over the inverted lists."""

    sizes: IntArray
    n_empty: int
    imbalance_factor: float
    min: int
    median: float
    max: int
    top_lists: list[tuple[int, int]]
    """``(list_no, size)`` for the largest lists, biggest first."""
    top_5pct_share: float
    """Fraction of all vectors held by the largest 5% of lists (at least one list)."""


@dataclass(frozen=True)
class Assignments:
    """Which inverted list each stored vector lives in, keyed by user-facing id.

    ``list_nos[i]`` is the list holding ``ids[i]``. Sorted by id.
    """

    ids: IntArray
    list_nos: IntArray

    def lookup(self, ids: npt.ArrayLike) -> IntArray:
        """List number for each id, ``-1`` for ids that are not stored."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(self.ids) == 0:
            return np.full(ids.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.ids, ids)
        pos_clipped = np.minimum(pos, len(self.ids) - 1)
        found = (pos < len(self.ids)) & (self.ids[pos_clipped] == ids)
        return np.where(found, self.list_nos[pos_clipped], -1).astype(np.int64)


def _ivf(li: LoadedIndex) -> Any:
    if li.ivf is None:
        raise NotAnIVFIndexError(f"{' → '.join(li.class_chain)} is not an IVF index.")
    return li.ivf


def list_sizes(li: LoadedIndex) -> IntArray:
    """Number of vectors in each inverted list, shape ``(nlist,)``."""
    # Same as faiss.contrib.inspect_tools.get_invlist_sizes, which mypy sees as untyped.
    invlists = _ivf(li).invlists
    return np.array([invlists.list_size(i) for i in range(invlists.nlist)], dtype=np.int64)


def imbalance_factor(sizes: npt.ArrayLike) -> float:
    """FAISS imbalance factor ``nlist * Σ size² / (Σ size)²``.

    1.0 means perfectly balanced; larger is worse. Returns 0.0 for an empty index
    (FAISS itself returns NaN there, which is not JSON-friendly).
    """
    s = np.asarray(sizes, dtype=np.float64)
    total = s.sum()
    if total == 0:
        return 0.0
    return float(len(s) * np.square(s).sum() / total**2)


def list_stats(li: LoadedIndex, top: int = 20) -> ListStats:
    """Sizes plus the summary numbers the overview needs."""
    sizes = list_sizes(li)
    order = np.argsort(-sizes, kind="stable")
    n_top5 = max(1, int(np.ceil(0.05 * len(sizes))))
    total = int(sizes.sum())
    return ListStats(
        sizes=sizes,
        n_empty=int((sizes == 0).sum()),
        imbalance_factor=imbalance_factor(sizes),
        min=int(sizes.min()),
        median=float(np.median(sizes)),
        max=int(sizes.max()),
        top_lists=[(int(i), int(sizes[i])) for i in order[:top]],
        top_5pct_share=float(sizes[order[:n_top5]].sum() / total) if total else 0.0,
    )


def stored_ids(ivf: Any, list_no: int) -> IntArray:
    """Ids stored in one inverted list (without copying its codes).

    These are internal offsets when an IDMap wraps the IVF, and user ids when vectors
    were added to the IVF directly with ``add_with_ids``; ``LoadedIndex.user_ids``
    handles both.
    """
    faiss = import_faiss()
    invlists = faiss.downcast_InvertedLists(ivf.invlists)
    n = int(invlists.list_size(list_no))
    if n == 0:
        return np.empty(0, dtype=np.int64)
    ptr = invlists.get_ids(list_no)
    try:
        return np.array(faiss.rev_swig_ptr(ptr, n), dtype=np.int64, copy=True)
    finally:
        invlists.release_ids(list_no, ptr)


def list_members(li: LoadedIndex, list_no: int) -> IntArray:
    """User-facing ids of the vectors in inverted list ``list_no``."""
    ivf = _ivf(li)
    if not 0 <= list_no < ivf.nlist:
        raise IndexError(f"list_no {list_no} out of range [0, {ivf.nlist}).")
    return li.user_ids(stored_ids(ivf, list_no))


def centroids(li: LoadedIndex) -> FloatArray:
    """Coarse centroids, shape ``(nlist, core_d)``."""
    faiss = import_faiss()
    ivf = _ivf(li)
    quantizer = faiss.downcast_index(ivf.quantizer)
    c: FloatArray = quantizer.reconstruct_n(0, ivf.nlist).astype(np.float32, copy=False)
    return c


def assignments(li: LoadedIndex) -> Assignments:
    """Map every user-facing id to its inverted list. O(ntotal); cache the result."""
    ivf = _ivf(li)
    per_list = [li.user_ids(stored_ids(ivf, i)) for i in range(ivf.nlist)]
    ids = np.concatenate(per_list) if per_list else np.empty(0, dtype=np.int64)
    list_nos = np.repeat(np.arange(ivf.nlist, dtype=np.int64), [len(p) for p in per_list])
    order = np.argsort(ids, kind="stable")
    return Assignments(ids=ids[order], list_nos=list_nos[order])


# Cells of the (queries, nlist) probe-order matrices built at once: ~2M (~40 MB in total).
_PROBE_ORDER_CELLS = 1 << 21


def truth_probe_ranks(
    li: LoadedIndex, queries: npt.ArrayLike, truth: npt.ArrayLike, assign: Assignments
) -> IntArray:
    """Where each true neighbour's list falls in its query's probe order, shape ``(n, k)``.

    A neighbour is in a probed list iff its rank is below ``nprobe``, so these ranks say
    what probing alone lets a search find at any nprobe. ``queries`` are input-space
    vectors (any PreTransform is applied, as FAISS does); ``truth`` holds user ids with
    ``-1`` padding, which maps to rank ``-1``. Ids that aren't stored get rank ``nlist``.
    """
    ivf = _ivf(li)
    nlist = int(ivf.nlist)
    q = li.to_core_space(np.ascontiguousarray(queries, dtype=np.float32))
    t = np.asarray(truth, dtype=np.int64)
    if t.ndim != 2 or len(t) != len(q):
        raise ValueError(f"truth must have shape ({len(q)}, k), got {t.shape}.")
    lists = assign.lookup(t)
    ranks = np.full(t.shape, -1, dtype=np.int64)
    chunk = max(1, _PROBE_ORDER_CELLS // max(nlist, 1))
    rows = np.arange(chunk)[:, None]
    for start in range(0, len(q), chunk):
        # The coarse quantizer's own order: exactly the lists a search at nprobe visits.
        _, order = ivf.quantizer.search(q[start : start + chunk], nlist)
        m = len(order)
        rank_of = np.full((m, nlist + 1), nlist, dtype=np.int64)
        valid = order >= 0
        rank_of[np.broadcast_to(rows[:m], order.shape)[valid], order[valid]] = np.nonzero(valid)[1]
        block = lists[start : start + m]
        # Column nlist stands for "not stored" (list -1).
        block_ranks = rank_of[rows[:m], np.where(block >= 0, block, nlist)]
        ranks[start : start + m] = np.where(t[start : start + m] >= 0, block_ranks, -1)
    return ranks


def probe_coverage(ranks: npt.ArrayLike, nlist: int) -> npt.NDArray[np.float64]:
    """Mean share of each query's true neighbours in probed lists, for nprobe = 0..nlist.

    ``coverage[v]`` caps recall@k at nprobe ``v``: a search can only return vectors from
    the lists it probes. ``ranks`` come from :func:`truth_probe_ranks`; queries with no
    true neighbours count as fully covered, matching how recall scores them.
    """
    r = np.asarray(ranks, dtype=np.int64)
    if r.ndim != 2 or len(r) == 0:
        raise ValueError("ranks must have shape (n, k) with n >= 1.")
    n_valid = (r >= 0).sum(axis=1)
    has_truth = n_valid > 0
    weights = np.where(r >= 0, 1.0 / np.maximum(n_valid, 1)[:, None], 0.0) / len(r)
    # A neighbour at rank j is covered from nprobe j + 1 on; unstored ones (rank nlist) never.
    hist = np.bincount(np.minimum(r[r >= 0], nlist) + 1, weights[r >= 0], minlength=nlist + 2)
    coverage = np.cumsum(hist)[: nlist + 1] + (~has_truth).sum() / len(r)
    out: npt.NDArray[np.float64] = np.minimum(coverage, 1.0)
    return out


def query_probe_coverage(ranks: npt.ArrayLike, nprobe: int) -> npt.NDArray[np.float64]:
    """Per query, the share of its true neighbours in the first ``nprobe`` lists (1.0 if none)."""
    r = np.asarray(ranks, dtype=np.int64)
    n_valid = (r >= 0).sum(axis=1)
    covered = ((r >= 0) & (r < nprobe)).sum(axis=1)
    out: npt.NDArray[np.float64] = np.where(n_valid > 0, covered / np.maximum(n_valid, 1), 1.0)
    return out


# Coarse quantizers that are not flat --------------------------------------------------
#
# With a flat coarse quantizer, one search over all nlist centroids ranks every list, and a
# search at nprobe visits exactly the first nprobe of that ranking. An HNSW quantizer (as in
# ``IVF65536_HNSW32,Flat``) answers ``search(q, nprobe)`` approximately, with a beam of
# ``max(efSearch, nprobe)``: a request for all nlist lists returns only the lists its beam
# reached, and probe sets at different nprobe are not prefixes of one ranking. For those
# indexes every "was this list probed" question is answered by the quantizer's own call at
# that nprobe, exactly as ``IndexIVF::search`` makes it.


def probe_order_is_exact(li: LoadedIndex) -> bool:
    """Whether one full coarse search gives the lists any nprobe probes (a flat quantizer)."""
    faiss = import_faiss()
    return isinstance(faiss.downcast_index(_ivf(li).quantizer), faiss.IndexFlat)


def probed_lists(li: LoadedIndex, queries_core: npt.ArrayLike, nprobe: int) -> IntArray:
    """The lists a search at ``nprobe`` probes, in probe order, shape ``(n, nprobe)``.

    ``queries_core`` are already in the core space (after any PreTransform). ``-1`` pads a
    row when the quantizer returns fewer lists, as an HNSW quantizer can.
    """
    ivf = _ivf(li)
    nprobe = max(1, min(int(nprobe), int(ivf.nlist)))
    q = np.ascontiguousarray(queries_core, dtype=np.float32)
    _, order = ivf.quantizer.search(q, nprobe)
    out: IntArray = order.astype(np.int64, copy=False)
    return out


def list_probe_ranks(
    li: LoadedIndex, query_core: npt.ArrayLike, lists: npt.ArrayLike, nprobe: int
) -> IntArray:
    """For one query, the smallest ``nprobe - 1`` at which each of ``lists`` is probed.

    So a list is probed at ``nprobe`` iff its rank is below ``nprobe``, for any quantizer.
    With a flat quantizer this is the list's position in the full probe order. Otherwise
    lists probed at ``nprobe`` get their position there, and the others the smallest
    larger nprobe at which FAISS probes them (found by doubling, then bisection, which
    assumes larger probe sets contain smaller ones, as they nearly always do). A list the
    quantizer never returns, even at nprobe = nlist, gets ``nlist``; so does list ``-1``.
    """
    ivf = _ivf(li)
    nlist = int(ivf.nlist)
    q = np.ascontiguousarray(query_core, dtype=np.float32).reshape(1, -1)
    wanted = np.asarray(lists, dtype=np.int64)
    ranks = np.full(wanted.shape, nlist, dtype=np.int64)
    if probe_order_is_exact(li):
        order = probed_lists(li, q, nlist)[0]
        pos = np.full(nlist + 1, nlist, dtype=np.int64)
        valid = order >= 0
        pos[order[valid]] = np.nonzero(valid)[0]
        return np.where(wanted >= 0, pos[np.where(wanted >= 0, wanted, nlist)], nlist)

    nprobe = max(1, min(int(nprobe), nlist))
    probed = probed_lists(li, q, nprobe)[0]
    at = {int(list_no): i for i, list_no in enumerate(probed) if list_no >= 0}

    def probed_at(depth: int) -> set[int]:
        return {int(x) for x in probed_lists(li, q, depth)[0] if x >= 0}

    for j, list_no in enumerate(wanted.tolist()):
        if list_no < 0:
            continue
        if list_no in at:
            ranks[j] = at[list_no]
            continue
        low, high = nprobe, nprobe
        while high < nlist:
            high = min(nlist, high * 2)
            if list_no in probed_at(high):
                break
            low = high
        else:
            if list_no not in probed_at(nlist):
                continue  # never probed: keep nlist
        while high - low > 1:  # smallest depth in (low, high] that probes it
            mid = (low + high) // 2
            low, high = (low, mid) if list_no in probed_at(mid) else (mid, high)
        ranks[j] = high - 1
    return ranks


def probe_coverage_at(
    li: LoadedIndex,
    queries: npt.ArrayLike,
    truth: npt.ArrayLike,
    assign: Assignments,
    nprobe: int,
) -> npt.NDArray[np.float64]:
    """Per query, the share of its true neighbours in the lists FAISS probes at ``nprobe``.

    Asks the coarse quantizer directly, so it is exact for any quantizer (1.0 for a query
    with no true neighbours, as recall scores it). ``queries`` are input-space vectors.
    """
    q = li.to_core_space(np.ascontiguousarray(queries, dtype=np.float32))
    t = np.asarray(truth, dtype=np.int64)
    lists = assign.lookup(t)
    out = np.ones(len(q), dtype=np.float64)
    chunk = max(1, _PROBE_ORDER_CELLS // max(int(nprobe), 1))
    for start in range(0, len(q), chunk):
        probed = probed_lists(li, q[start : start + chunk], nprobe)
        for i, row in enumerate(probed):
            valid = t[start + i] >= 0
            if valid.any():
                hit = np.isin(lists[start + i][valid], row[row >= 0])
                out[start + i] = float(hit.mean())
    return out
