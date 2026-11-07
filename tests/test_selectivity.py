"""Selectivity sweeps: recall against the pass rate of a filter, with measured advice."""

import faiss
import numpy as np
import pytest

from vectraudit.core import advice as A
from vectraudit.core import filtering as F
from vectraudit.core import ivf as I
from vectraudit.core import search as S
from vectraudit.core import selectivity as L
from vectraudit.core import sweep as W
from vectraudit.core import vectors as V
from vectraudit.core.loader import load_index
from vectraudit.core.metadata import Metadata

N, D = 6000, 24


@pytest.fixture(scope="module")
def x():
    return np.random.default_rng(3).standard_normal((N, D)).astype("float32")


@pytest.fixture(scope="module")
def hnsw(x):
    index = faiss.IndexHNSWFlat(D, 16)
    index.hnsw.efConstruction = 40
    index.add(x)
    li = load_index(index)
    gt = S.GroundTruth(V.from_arrays(li, x), li.metric)
    return li, gt


def _recalls(result: L.SelectivityResult) -> dict[str, list[float]]:
    return {r.spec.label: [p.recall for p in r.sweep.points] for r in result.runs}


# --- filters ----------------------------------------------------------------


def test_synthetic_filter_hits_the_pass_rate_on_sparse_ids() -> None:
    ids = np.arange(1000, 1000 + 7 * 2000, 7)  # sparse, offset ids
    for rate in (0.001, 0.01, 0.5, 1.0):
        allowed = L.synthetic_allowed(ids, rate)
        assert len(allowed) == max(1, round(len(ids) * rate))
        # The highest ids pass: VectorDBBench's `id >= cut`.
        assert allowed.ids.min() == np.sort(ids)[-len(allowed)]
    assert L.synthetic_allowed(ids, 0.0001).pass_rate == 1 / len(ids)


def test_pass_rate_is_not_the_filter_rate() -> None:
    for bad in (0, -0.1, 1.5):
        with pytest.raises(F.FilterError, match="pass rate") as e:
            L.synthetic_spec(bad)
        assert "not the fraction removed" in e.value.hint


def test_a_filter_allowing_fewer_than_k_is_refused() -> None:
    ids = np.arange(100)
    with pytest.raises(F.FilterError, match="fewer than k=10"):
        L.build_allowed(L.synthetic_spec(0.05), ids, None, 10)
    meta = Metadata({"id": i, "lang": "en" if i < 3 else "fr"} for i in range(100))
    with pytest.raises(F.FilterError, match="fewer than k=10"):
        L.build_allowed(L.metadata_spec({"lang": "en"}), ids, meta, 10)
    with pytest.raises(F.FilterError, match="needs metadata"):
        L.build_allowed(L.metadata_spec({"lang": "en"}), ids, None, 10)
    assert L.build_allowed(L.ALL, ids, None, 10) is None


# --- the sweep --------------------------------------------------------------


def test_hnsw_recall_falls_as_the_pass_rate_falls(hnsw) -> None:
    li, gt = hnsw
    qs = W.sample_queries(gt.source, 60, seed=0)
    specs = [L.synthetic_spec(p) for p in (0.5, 0.05, 0.01)]
    result = L.selectivity_sweep(li, qs, gt, specs, values=[16, 64], k=10, repeats=1)
    recalls = _recalls(result)
    assert result.runs[0].spec.label == "all"
    at16 = [recalls[r.spec.label][0] for r in result.runs]
    assert at16 == sorted(at16, reverse=True)
    assert at16[0] - at16[-1] > 0.2
    assert [round(r.pass_rate, 3) for r in result.runs] == [1.0, 0.5, 0.05, 0.01]
    assert result.runs[-1].sweep.pass_rate == pytest.approx(0.01)
    assert result.baseline.sweep.pass_rate is None


def test_filtered_sweep_matches_a_direct_filtered_search(hnsw) -> None:
    li, gt = hnsw
    qs = W.sample_queries(gt.source, 20, seed=1)
    allowed = L.synthetic_allowed(gt.source.ids, 0.1)
    truth = W.ground_truth_ids(gt, qs, 10, allowed=allowed)
    swept = W.sweep(li, qs, truth, values=[32], k=10, repeats=1, allowed=allowed)
    direct = []
    for q, own, row in zip(qs.vectors, qs.exclude_ids, truth, strict=True):
        found = S.search(li, q, 10, ef_search=32, exclude_id=int(own), allowed=allowed)
        direct.append(S.recall_at_k(found.ids, row))
    assert swept.points[0].recall == pytest.approx(float(np.mean(direct)))


def test_ivf_filtered_sweep_reports_probe_coverage(x) -> None:
    index = faiss.index_factory(D, "IVF32,Flat")
    index.train(x)
    index.add(x)
    li = load_index(index)
    gt = S.GroundTruth(V.from_arrays(li, x), li.metric)
    qs = W.sample_queries(gt.source, 40, seed=0)
    result = L.selectivity_sweep(
        li,
        qs,
        gt,
        [L.synthetic_spec(0.05)],
        values=[1, 8, 32],
        k=10,
        repeats=1,
        assignments=I.assignments(li),
    )
    for run in result.runs:
        assert run.sweep.points[0].probe_coverage is not None
        assert run.sweep.points[-1].recall == pytest.approx(1.0)  # every cell probed: exact


def test_duplicate_filters_run_once_and_metadata_filters_work(hnsw) -> None:
    li, gt = hnsw
    meta = Metadata({"id": i, "lang": "de" if i % 5 == 0 else "en"} for i in range(N))
    qs = W.sample_queries(gt.source, 20, seed=0)
    spec = L.metadata_spec({"lang": "de"})
    result = L.selectivity_sweep(
        li, qs, gt, [spec, spec], metadata=meta, values=[32], k=10, repeats=1
    )
    assert [r.spec.label for r in result.runs] == ["all", "lang=de"]
    assert result.runs[1].pass_rate == pytest.approx(0.2)


def test_progress_covers_every_filter(hnsw) -> None:
    li, gt = hnsw
    seen: list[float] = []
    L.selectivity_sweep(
        li,
        W.sample_queries(gt.source, 10, seed=0),
        gt,
        [L.synthetic_spec(0.1)],
        values=[16],
        k=10,
        repeats=1,
        progress=lambda f, _m: seen.append(f),
    )
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert max(seen) == 1.0
    assert any(f > 0.5 for f in seen[:-1])


# --- advice -----------------------------------------------------------------


def test_advice_quotes_measurements_and_lists_what_was_not_measured(hnsw) -> None:
    li, gt = hnsw
    qs = W.sample_queries(gt.source, 60, seed=0)
    result = L.selectivity_sweep(
        li, qs, gt, [L.synthetic_spec(0.005)], values=[16, 32], k=10, repeats=1
    )
    tight = result.runs[1]
    best = max(p.recall for p in tight.sweep.points)
    assert best < 0.95
    out = A.advise_selectivity(result, 0.95)
    kinds = [s.kind for s in out]
    assert A.SuggestionKind.FILTER_RECALL_SHORT in kinds
    short = next(s for s in out if s.kind is A.SuggestionKind.FILTER_RECALL_SHORT)
    assert f"{best:.2f}" in short.detail
    assert "efSearch 32" in short.detail
    assert "was not reached in the sweep range" in short.detail
    unmeasured = next(s for s in out if s.kind is A.SuggestionKind.FILTER_UNMEASURED)
    assert "brute-force" in unmeasured.detail
    assert "partitioning" in unmeasured.detail


def test_advice_is_silent_when_every_pass_rate_meets_the_target(hnsw) -> None:
    li, gt = hnsw
    qs = W.sample_queries(gt.source, 30, seed=0)
    result = L.selectivity_sweep(li, qs, gt, [L.synthetic_spec(0.9)], values=[512], k=10, repeats=1)
    assert A.advise_selectivity(result, 0.5) == []


# --- the sweep command ------------------------------------------------------


def _write_cli_files(tmp_path, x):
    index = faiss.IndexHNSWFlat(D, 16)
    index.add(x)
    faiss.write_index(index, str(tmp_path / "h.index"))
    np.save(tmp_path / "x.npy", x)
    rows = (f'{{"id": {i}, "lang": "{"de" if i % 20 == 0 else "en"}"}}' for i in range(N))
    (tmp_path / "meta.jsonl").write_text("\n".join(rows))


def _sweep(tmp_path, *extra):
    from typer.testing import CliRunner

    from vectraudit.cli import app

    args = ["sweep", str(tmp_path / "h.index"), "--vectors", str(tmp_path / "x.npy")]
    args += ["--values", "16,128", "--n-queries", "30", "--repeats", "1", *extra]
    return CliRunner().invoke(app, args)


def test_sweep_command_filters_and_json(tmp_path, x) -> None:
    import json

    _write_cli_files(tmp_path, x)
    meta = str(tmp_path / "meta.jsonl")
    out = _sweep(tmp_path, "--meta", meta, "--filter", "lang=de", "--pass-rates", "0.01", "--json")
    assert out.exit_code in (0, 2), out.output
    doc = json.loads(out.stdout)
    assert doc["format"] == "vectraudit.selectivity-sweep"
    assert doc["format_version"] == 1
    labels = [r["filter"]["label"] for r in doc["runs"]]
    assert labels == ["all", "lang=de", "synthetic@0.01"]
    assert [round(r["pass_rate"], 2) for r in doc["runs"]] == [1.0, 0.05, 0.01]
    assert "text" not in json.dumps(doc)  # no document text is ever in the result


def test_sweep_command_exit_code_2_when_a_filter_misses_the_target(tmp_path, x) -> None:
    _write_cli_files(tmp_path, x)
    out = _sweep(tmp_path, "--pass-rates", "0.005", "--target", "0.999")
    assert out.exit_code == 2
    assert "not reached" in out.output


def test_sweep_command_refuses_bad_filter_combinations(tmp_path, x) -> None:
    _write_cli_files(tmp_path, x)
    assert "needs --meta" in _sweep(tmp_path, "--filter", "lang=de").output
    assert _sweep(tmp_path, "--filter", "lang=de").exit_code == 1
    out = _sweep(tmp_path, "--pass-rates", "0.01", "--save", str(tmp_path / "r.json"))
    assert out.exit_code == 1
    assert "do not apply" in out.output
    out = _sweep(tmp_path, "--pass-rates", "99")
    assert out.exit_code == 1
    assert "not the fraction removed" in " ".join(out.output.split())
    assert _sweep(tmp_path, "--meta", str(tmp_path / "meta.jsonl")).exit_code == 1
