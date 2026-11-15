"""Filtered search: the selector is applied inside FAISS, and the truth is over the same set."""

import faiss
import numpy as np
import pytest

from tests.conftest import SMALL
from vectraudit.core import filtering as F
from vectraudit.core import search as S
from vectraudit.core import vectors as V
from vectraudit.core.ivf import assignments
from vectraudit.core.loader import load_index
from vectraudit.core.metadata import Metadata
from vectraudit.core.types import Metric

N, NLIST = SMALL["n"], SMALL["nlist"]


@pytest.fixture(scope="module")
def data(synthetic):
    return {
        "x": np.load(synthetic["vectors"]),
        "queries": np.load(synthetic["queries"]),
        "ids_idmap": np.load(synthetic["ids_idmap"]),
    }


def _allowed(n: int, every: int = 4, ids=None) -> F.AllowedSet:
    ids = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    return F.allowed_from_ids(ids[::every], ids)


# --- the allowed set --------------------------------------------------------


def test_allowed_set_pass_rate_is_the_fraction_that_passes() -> None:
    allowed = _allowed(100, every=4)
    assert len(allowed) == 25
    assert allowed.pass_rate == 0.25
    np.testing.assert_array_equal(allowed.mask([0, 1, 4, -1]), [True, False, True, False])


def test_allowed_from_metadata_matches_portable_filters() -> None:
    meta = Metadata(
        {"id": i, "lang": "en" if i % 2 else "fr", "year": 2000 + i, "tag": ["a"]}
        for i in range(10)
    )
    ids = np.arange(12)  # ids 10 and 11 have no metadata row: they never pass a filter
    en = F.allowed_from_metadata(meta, {"lang": "en"}, ids)
    assert en.ids.tolist() == [1, 3, 5, 7, 9]
    assert en.pass_rate == 5 / 12
    both = F.allowed_from_metadata(meta, {"lang": "en", "year": {"gte": 2005}}, ids)
    assert both.ids.tolist() == [5, 7, 9]
    anyof = F.allowed_from_metadata(meta, {"lang": ["fr"], "missing": 1}, ids)
    assert len(anyof) == 0  # a missing field never satisfies a condition
    assert F.allowed_from_metadata(meta, {}, ids).pass_rate == 1.0


def test_fingerprint_follows_the_set_not_its_description() -> None:
    a, b = _allowed(100, 4), _allowed(100, 4)
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != _allowed(100, 5).fingerprint
    assert a.fingerprint != _allowed(200, 4).fingerprint
    assert F.filter_fingerprint({"a": 1, "b": 2}) == F.filter_fingerprint({"b": 2, "a": 1})


def test_allowed_set_rejects_more_ids_than_the_corpus() -> None:
    with pytest.raises(F.FilterError):
        F.AllowedSet(np.arange(5, dtype=np.int64), 3)


# --- filtered search --------------------------------------------------------


@pytest.mark.parametrize("name", ["flat_l2", "ivf_flat", "ivf_pq", "hnsw_flat", "pca_ivf_flat"])
def test_filtered_search_never_returns_a_disallowed_id(synthetic, data, name) -> None:
    li = load_index(synthetic[name])
    allowed = _allowed(N)
    kwargs = {"nprobe": NLIST} if li.kind.is_ivf else {}
    for q in data["queries"][:10]:
        r = S.search(li, q, 10, allowed=allowed, **kwargs)
        assert r.valid_ids.size > 0
        assert allowed.mask(r.valid_ids).all()


def test_flat_recall_is_one_under_any_filter(synthetic, data) -> None:
    li = load_index(synthetic["flat_l2"])
    source = V.reconstruct_all(li)
    gt = S.GroundTruth(source, li.metric)
    for every in (2, 7, 50):
        allowed = _allowed(N, every)
        for q in data["queries"][:10]:
            r = S.search(li, q, 10, allowed=allowed)
            truth = gt.search(q, 10, allowed=allowed)
            assert S.recall_at_k(r.ids, truth.ids) == 1.0
            np.testing.assert_array_equal(r.ids, truth.ids)


def test_filtered_search_equals_brute_force_post_filter(synthetic, data) -> None:
    # With every cell probed an IVF-Flat index is exact, so its filtered result must equal
    # the unfiltered ranking restricted to the allowed ids.
    li = load_index(synthetic["ivf_flat"])
    allowed = _allowed(N, 3)
    for q in data["queries"][:10]:
        full = S.search(li, q, N, nprobe=NLIST)
        expected = full.ids[allowed.mask(full.ids)][:10]
        got = S.search(li, q, 10, nprobe=NLIST, allowed=allowed)
        np.testing.assert_array_equal(got.ids, expected)


def test_filtered_search_with_idmap_uses_user_facing_ids(synthetic, data) -> None:
    li = load_index(synthetic["idmap_flat"])
    ids = data["ids_idmap"]
    allowed = _allowed(N, 5, ids=ids)
    gt = S.GroundTruth(V.from_arrays(li, data["x"], ids), li.metric)
    r = S.search(li, data["queries"][0], 10, allowed=allowed)
    assert allowed.mask(r.ids).all()
    np.testing.assert_array_equal(r.ids, gt.search(data["queries"][0], 10, allowed=allowed).ids)


def test_filtered_search_with_a_refine_stage(tmp_path) -> None:
    # The selector must be passed through IndexRefine's base params to take effect.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1500, 16)).astype("float32")
    index = faiss.index_factory(16, "IVF8,PQ4x8,RFlat")
    index.train(x)
    index.add(x)
    li = load_index(index)
    assert li.has_refine
    allowed = _allowed(1500, 6)
    r = S.search(li, x[1] + 0.01, 10, nprobe=8, allowed=allowed)
    assert r.valid_ids.size == 10
    assert allowed.mask(r.valid_ids).all()


def test_a_selective_filter_can_leave_the_result_short(synthetic, data) -> None:
    # One probed cell cannot hold 10 of a 2%-pass-rate filter: FAISS returns -1 slots.
    li = load_index(synthetic["ivf_flat"])
    allowed = _allowed(N, 50)
    r = S.search(li, data["queries"][0], 10, nprobe=1, allowed=allowed)
    assert (r.ids == -1).any()
    assert allowed.mask(r.valid_ids).all()


def test_unfiltered_search_is_unchanged_by_the_filter_support(synthetic, data) -> None:
    li = load_index(synthetic["ivf_flat"])
    q = data["queries"][0]
    li.ivf.nprobe = NLIST
    _, expected = li.index.search(q.reshape(1, -1), 10)
    np.testing.assert_array_equal(S.search(li, q, 10, nprobe=NLIST).ids, expected[0])


# --- filtered ground truth --------------------------------------------------


def test_filtered_truth_is_the_exact_neighbours_among_allowed_ids(synthetic, data) -> None:
    li = load_index(synthetic["flat_l2"])
    source = V.reconstruct_all(li)
    gt = S.GroundTruth(source, li.metric)
    allowed = _allowed(N, 4)
    q = data["queries"][0]
    d2 = ((data["x"] - q) ** 2).sum(axis=1)
    d2[~allowed.mask(np.arange(N))] = np.inf
    expected = np.argsort(d2)[:10]
    np.testing.assert_array_equal(gt.search(q, 10, allowed=allowed).ids, expected)
    batch = gt.search_batch(data["queries"][:3], 10, allowed=allowed)
    np.testing.assert_array_equal(batch[0], expected)


def test_filtered_truth_pads_when_fewer_than_k_are_allowed(synthetic, data) -> None:
    li = load_index(synthetic["flat_l2"])
    gt = S.GroundTruth(V.reconstruct_all(li), li.metric)
    few = F.allowed_from_ids([5, 9, 11], np.arange(N))
    assert gt.search(data["queries"][0], 10, allowed=few).valid_ids.size == 3
    none = F.allowed_from_ids([], np.arange(N))
    assert (gt.search_batch(data["queries"][:2], 5, allowed=none) == -1).all()


def test_filtered_truth_subsets_are_cached_per_fingerprint(synthetic, data) -> None:
    li = load_index(synthetic["flat_l2"])
    gt = S.GroundTruth(V.reconstruct_all(li), li.metric)
    a = _allowed(N, 4)
    gt.search(data["queries"][0], 5, allowed=a)
    first = gt._subsets[a.fingerprint]
    gt.search(data["queries"][1], 5, allowed=_allowed(N, 4))
    assert gt._subsets[a.fingerprint] is first
    for every in (5, 6, 7):
        gt.search(data["queries"][0], 5, allowed=_allowed(N, every))
    assert len(gt._subsets) <= 2


def test_filtered_truth_inner_product(synthetic, data) -> None:
    li = load_index(synthetic["ivf_flat_ip"])
    assert li.metric is Metric.IP
    gt = S.GroundTruth(V.reconstruct_all(li), li.metric)
    allowed = _allowed(N, 4)
    q = data["queries"][0]
    got = gt.search(q, 10, allowed=allowed)
    assert allowed.mask(got.ids).all()
    assert (np.diff(got.distances) <= 0).all()  # IP: larger is closer, best first


# --- explaining a filtered query -------------------------------------------


def test_explain_query_reports_allowed_vectors_in_probed_cells(synthetic, data) -> None:
    li = load_index(synthetic["ivf_flat"])
    gt = S.GroundTruth(V.reconstruct_all(li), li.metric)
    allowed = _allowed(N, 4)
    q = S.ResolvedQuery(data["queries"][0].astype(np.float32), None, S.QueryKind.VECTOR)
    report = S.explain_query(
        li, q, 10, nprobe=2, ground_truth=gt, assignments=assignments(li), allowed=allowed
    )
    trace = report.ivf_trace
    assert trace is not None
    assert trace.allowed_total == len(allowed)
    probed = assignments(li).lookup(allowed.ids)
    assert trace.allowed_in_probed == int(np.isin(probed, trace.probed_lists).sum())
    assert all(allowed.mask([n.id])[0] for n in trace.neighbours)
    unfiltered = S.explain_query(
        li, q, 10, nprobe=2, ground_truth=gt, assignments=assignments(li)
    ).ivf_trace
    assert unfiltered is not None
    assert unfiltered.allowed_in_probed is None
