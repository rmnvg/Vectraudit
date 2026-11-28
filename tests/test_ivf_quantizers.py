"""IVF PQ FastScan indexes, and IVF indexes whose coarse quantizer is HNSW, not flat.

FastScan stores 4-bit PQ codes in interleaved blocks; it must decode exactly like the
IndexIVFPQ it can be built from. An HNSW coarse quantizer answers ``search(q, nprobe)``
approximately, so every "was this list probed" answer must match what FAISS itself probes.
"""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from vectraudit.connectors.base import SearchRequest  # noqa: E402
from vectraudit.connectors.faiss_index import FaissConnector  # noqa: E402
from vectraudit.core import ivf  # noqa: E402
from vectraudit.core.loader import load_index  # noqa: E402
from vectraudit.core.search import GroundTruth, MissReason, search, trace_ivf  # noqa: E402
from vectraudit.core.sweep import given_queries, sweep  # noqa: E402
from vectraudit.core.types import IndexKind  # noqa: E402
from vectraudit.core.vectors import from_arrays, reconstruct_all  # noqa: E402

D, N = 16, 4000


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    return rng.standard_normal((N, D)).astype(np.float32), rng.standard_normal((40, D)).astype(
        np.float32
    )


@pytest.fixture(scope="module")
def ivfpq(data):
    x, _ = data
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, 32, 8, 4)
    index.train(x)
    index.add(x)
    return index


@pytest.fixture(scope="module")
def ivf_hnsw(data):
    x, _ = data
    index = faiss.index_factory(D, "IVF256_HNSW8,Flat")
    index.train(x)
    index.add(x)
    return index


def test_fastscan_loads_as_ivf_pq_with_its_parameters(ivfpq):
    li = load_index(faiss.IndexIVFPQFastScan(ivfpq))
    assert li.kind is IndexKind.IVF_PQ
    assert li.is_supported
    params = li.params.as_dict()
    assert params["fast_scan"] is True
    assert (params["pq_m"], params["pq_nbits"]) == (8, 4)
    assert "coarse_quantizer" not in params  # flat quantizer: nothing new to say
    assert ivf.probe_order_is_exact(li)


def test_fastscan_decodes_exactly_like_the_ivfpq_it_was_built_from(ivfpq):
    for by_residual in (True, False):
        ivfpq.by_residual = by_residual
        expected = reconstruct_all(load_index(ivfpq))
        got = reconstruct_all(load_index(faiss.IndexIVFPQFastScan(ivfpq)))
        np.testing.assert_array_equal(got.ids, expected.ids)
        np.testing.assert_allclose(got.vectors, expected.vectors, rtol=1e-5, atol=1e-5)
    ivfpq.by_residual = True


def test_additive_quantizer_fastscan_stays_unsupported_with_a_reason():
    li = load_index(faiss.index_factory(D, "IVF16,RQ2x4fs_Nrq2x4"))
    assert li.kind is IndexKind.UNSUPPORTED
    assert "only PQ FastScan" in (li.unsupported_reason or "")


def test_hnsw_quantizer_is_named_and_not_treated_as_an_exact_order(ivf_hnsw):
    li = load_index(ivf_hnsw)
    assert li.kind is IndexKind.IVF_FLAT
    params = li.params.as_dict()
    assert params["coarse_quantizer"] == "IndexHNSWFlat"
    assert params["quantizer_ef_search"] == 16
    assert not ivf.probe_order_is_exact(li)


def faiss_probes(li, q, nprobe):
    _, lists = li.ivf.quantizer.search(q.reshape(1, -1), nprobe)
    return {int(i) for i in lists[0] if i >= 0}


@pytest.mark.parametrize("nprobe", [2, 16, 40, 256])  # below, at and above efSearch (16)
def test_trace_reasons_match_the_lists_faiss_probes(data, ivf_hnsw, nprobe):
    x, queries = data
    li = load_index(ivf_hnsw)
    truth_index = GroundTruth(from_arrays(li, x, None), li.metric)
    assign = ivf.assignments(li)
    for q in queries[:15]:
        result = search(li, q, 10, nprobe=nprobe)
        truth = truth_index.search(q, 10)
        trace = trace_ivf(li, q, result, truth, assign)
        probed = faiss_probes(li, q, nprobe)
        assert set(trace.probed_lists.tolist()) == probed
        for n in trace.neighbours:
            assert (n.list_no in probed) == (n.probe_rank < nprobe)
            if n.reason is not MissReason.FOUND:
                assert (n.reason is MissReason.CELL_NOT_PROBED) == (n.list_no not in probed)
        # The smallest nprobe said to reach every neighbour does reach them all.
        if trace.min_nprobe_for_all < li.ivf.nlist:
            reach = faiss_probes(li, q, trace.min_nprobe_for_all)
            assert {n.list_no for n in trace.neighbours} <= reach


def test_flat_quantizer_ranks_are_unchanged(data, ivfpq):
    x, queries = data
    li = load_index(ivfpq)
    truth = GroundTruth(from_arrays(li, x, None), li.metric).search_batch(queries, 10)
    assign = ivf.assignments(li)
    ranks = ivf.truth_probe_ranks(li, queries, truth, assign)
    for i, q in enumerate(queries[:10]):
        lists = assign.lookup(truth[i])
        np.testing.assert_array_equal(ivf.list_probe_ranks(li, q, lists, 4), ranks[i])
    np.testing.assert_allclose(
        ivf.probe_coverage_at(li, queries, truth, assign, 4),
        ivf.query_probe_coverage(ranks, 4),
    )


def test_sweep_coverage_on_an_hnsw_quantizer_is_what_faiss_probes(data, ivf_hnsw):
    x, queries = data
    li = load_index(ivf_hnsw)
    truth = GroundTruth(from_arrays(li, x, None), li.metric).search_batch(queries, 10)
    assign = ivf.assignments(li)
    result = sweep(
        li, given_queries(li, queries), truth, values=[4, 64], k=10, repeats=1, assignments=assign
    )
    assert result.coverage_curve is None  # no single ranking gives every nprobe
    for point in result.points:
        expected = []
        for q, row in zip(queries, truth, strict=True):
            probed = faiss_probes(li, q, point.value)
            expected.append(np.isin(assign.lookup(row), list(probed)).mean())
        assert point.probe_coverage == pytest.approx(float(np.mean(expected)))
        # IVF-Flat ranks exactly inside probed lists, so recall is capped by coverage.
        assert point.recall <= point.probe_coverage + 1e-9


def test_release_check_evidence_follows_faiss_probing(tmp_path, data, ivf_hnsw):
    x, queries = data
    path = tmp_path / "ivf_hnsw.index"
    faiss.write_index(ivf_hnsw, str(path))
    np.save(tmp_path / "x.npy", x)
    connector = FaissConnector(path, vectors=tmp_path / "x.npy")
    li = connector.li
    for q in queries[:10]:
        request = SearchRequest(vector=q, k=10, params={"nprobe": 8})
        found = connector._search(request)
        exact = connector._search(SearchRequest(vector=q, k=10, params={"nprobe": 8}, exact=True))
        evidence = connector.evidence(request, found, exact)
        assert evidence is not None
        probed = faiss_probes(li, q, 8)
        lists = dict(zip(exact, ivf.assignments(li).lookup([int(i) for i in exact]), strict=True))
        for nid, reason in evidence["missed"].items():
            assert (reason == MissReason.CELL_NOT_PROBED.value) == (int(lists[nid]) not in probed)
