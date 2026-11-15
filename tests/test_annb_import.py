"""ann-benchmarks HDF5 files import as vectors, queries and an external exact reference."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from typer.testing import CliRunner

h5py = pytest.importorskip("h5py")
faiss = pytest.importorskip("faiss")

from vectraudit.cli import app  # noqa: E402
from vectraudit.connectors.faiss_index import FaissConnector  # noqa: E402
from vectraudit.evaluation import external_reference as X  # noqa: E402
from vectraudit.evaluation.annb_import import AnnbImportError, import_annb  # noqa: E402
from vectraudit.evaluation.dataset import load_dataset  # noqa: E402
from vectraudit.evaluation.runner import RunOptions, execute_run  # noqa: E402

N, D, NQ, COUNT = 300, 16, 20, 100


def brute_force(train, test, metric):
    """Neighbours and distances as ann-benchmarks computes them (distance.py)."""
    if metric == "euclidean":
        dist = np.sqrt(((test[:, None, :] - train[None, :, :]) ** 2).sum(-1))
    else:
        cos = (test @ train.T) / np.outer(
            np.linalg.norm(test, axis=1), np.linalg.norm(train, axis=1)
        )
        dist = 1 - cos
    order = np.argsort(dist, axis=1, kind="stable")[:, :COUNT]
    return order, np.take_along_axis(dist, order, axis=1)


def write_annb(path, metric="euclidean", *, seed=0, neighbors=None, distances=None, **attrs):
    """A file in ann-benchmarks' documented layout (datasets.write_output)."""
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(N, D)).astype(np.float32) * 3
    test = rng.normal(size=(NQ, D)).astype(np.float32) * 3
    nb, dist = brute_force(train.astype(np.float64), test.astype(np.float64), metric)
    with h5py.File(path, "w") as f:
        f.attrs["type"] = attrs.get("type", "dense")
        f.attrs["distance"] = attrs.get("distance", metric)
        f.attrs["dimension"] = D
        f.attrs["point_type"] = "float"
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset("neighbors", data=nb if neighbors is None else neighbors)
        f.create_dataset("distances", data=dist if distances is None else distances)
    return path, train, test, nb


def record(out, index, k=10):
    ds = load_dataset(out / "dataset.jsonl")
    ref = X.load_reference(out / "reference.json")
    with FaissConnector(out / index, vectors=out / "vectors.npy", ids=out / "ids.npy") as c:
        return execute_run(ds, c, RunOptions(k=k, warmup=0, reference=ref))


def test_a_euclidean_file_reproduces_brute_force_and_a_flat_index_scores_one(tmp_path):
    path, train, _, nb = write_annb(tmp_path / "toy-16-euclidean.hdf5")
    out = tmp_path / "out"
    result = import_annb(path, out)
    assert (result.n_vectors, result.dim, result.n_queries, result.k) == (N, D, NQ, COUNT)
    assert (result.distance, result.source_distance) == ("l2", "euclidean")
    ref = X.load_reference(out / "reference.json")
    assert ref.source == "external:annb"
    assert ref.distance == "l2"
    assert ref.neighbours("q3", COUNT) == [str(i) for i in nb[3]]
    np.testing.assert_array_equal(np.load(out / "vectors.npy"), train)
    np.testing.assert_array_equal(np.load(out / "ids.npy"), np.arange(N))
    flat = faiss.IndexFlatL2(D)
    flat.add(np.load(out / "vectors.npy"))
    faiss.write_index(flat, str(out / "flat.index"))
    run = record(out, "flat.index")
    assert run["status"] == "complete", run["error"]
    assert run["search"]["reference"]["source"] == "external:annb"
    assert all(q["ann_recall"] == 1.0 for q in run["queries"])


def test_an_angular_file_needs_normalised_vectors_for_an_inner_product_index(tmp_path):
    path, _, _, nb = write_annb(tmp_path / "toy-16-angular.hdf5", "angular")
    raw = tmp_path / "raw"
    import_annb(path, raw)
    assert X.load_reference(raw / "reference.json").distance == "cosine"
    ip = faiss.IndexFlatIP(D)
    ip.add(np.load(raw / "vectors.npy"))
    faiss.write_index(ip, str(raw / "ip.index"))
    refused = record(raw, "ip.index")
    assert refused["status"] == "failed"
    assert "computed under cosine distance, the backend ranks by ip" in refused["error"]["message"]

    unit = tmp_path / "unit"
    result = import_annb(path, unit, normalize=True)
    assert result.normalized is True
    vectors = np.load(unit / "vectors.npy")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
    assert load_dataset(unit / "dataset.jsonl").meta["normalized"] is True
    ip = faiss.IndexFlatIP(D)
    ip.add(vectors)
    faiss.write_index(ip, str(unit / "ip.index"))
    run = record(unit, "ip.index")
    assert run["status"] == "complete", run["error"]
    assert all(q["ann_recall"] == 1.0 for q in run["queries"])
    assert X.load_reference(unit / "reference.json").neighbours("q0", COUNT) == [
        str(i) for i in nb[0]
    ]


def test_the_cosine_and_inner_product_rule_needs_normalised_vectors():
    ref = X.ExternalReference(
        source="external:annb", distance="cosine", k=1, query_ids=("q0",),
        ids=np.zeros((1, 1), dtype=np.int64), filter=None, files={},
        corpus={"normalized": True}, descriptor_sha256="0",
    )  # fmt: skip
    assert ref.ranks_as("cosine")
    assert ref.ranks_as("ip")
    assert not ref.ranks_as("l2")
    raw = X.ExternalReference(**{**ref.__dict__, "corpus": {}})
    assert not raw.ranks_as("ip")


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({"distance": "jaccard"}, "sparse sets under Jaccard distance"),
        ({"distance": "hamming"}, "bit vectors under Hamming distance"),
        ({"distance": "dot"}, "expected euclidean or angular"),
        ({"type": "sparse"}, "sparse dataset"),
    ],
)
def test_unsupported_files_are_refused(tmp_path, attrs, message):
    path, *_ = write_annb(tmp_path / "x.hdf5", **attrs)
    with pytest.raises(AnnbImportError, match=message):
        import_annb(path, tmp_path / "out")


def test_the_recorded_distance_is_checked_against_the_data(tmp_path):
    # The file says angular, but its neighbours and distances are Euclidean.
    rng = np.random.default_rng(0)
    train = rng.normal(size=(N, D)) * 3
    test = rng.normal(size=(NQ, D)) * 3
    nb, dist = brute_force(train, test, "euclidean")
    path, *_ = write_annb(tmp_path / "x.hdf5", distance="angular", neighbors=nb, distances=dist)
    with pytest.raises(AnnbImportError, match="do not match angular distances"):
        import_annb(path, tmp_path / "out")


def test_options_and_contents_are_validated(tmp_path):
    path, *_ = write_annb(tmp_path / "e.hdf5")
    with pytest.raises(AnnbImportError, match="contradicts the file"):
        import_annb(path, tmp_path / "o1", distance="cosine")
    assert import_annb(path, tmp_path / "o2", distance="l2").distance == "l2"
    with pytest.raises(AnnbImportError, match="angular files only"):
        import_annb(path, tmp_path / "o3", normalize=True)
    bad = np.zeros((NQ, COUNT), dtype=np.int64) + N
    out_of_range, *_ = write_annb(tmp_path / "r.hdf5", neighbors=bad)
    with pytest.raises(AnnbImportError, match="do not exist"):
        import_annb(out_of_range, tmp_path / "o4")
    rng = np.random.default_rng(0)
    unsorted, *_ = write_annb(tmp_path / "u.hdf5", distances=rng.random((NQ, COUNT)))
    with pytest.raises(AnnbImportError, match="not sorted nearest first"):
        import_annb(unsorted, tmp_path / "o5")
    with h5py.File(tmp_path / "m.hdf5", "w") as f:
        f.attrs["distance"] = "euclidean"
        f.create_dataset("train", data=np.zeros((3, 2)))
    with pytest.raises(AnnbImportError, match="no 'test' dataset"):
        import_annb(tmp_path / "m.hdf5", tmp_path / "o6")
    (tmp_path / "not.hdf5").write_text("plain text")
    with pytest.raises(AnnbImportError, match="not an HDF5 file"):
        import_annb(tmp_path / "not.hdf5", tmp_path / "o7")
    with pytest.raises(AnnbImportError, match="is not a file"):
        import_annb(tmp_path / "missing.hdf5", tmp_path / "o8")


def test_without_h5py_the_error_says_how_to_install_or_convert(tmp_path, monkeypatch):
    path, *_ = write_annb(tmp_path / "e.hdf5")
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(AnnbImportError) as caught:
        import_annb(path, tmp_path / "out")
    assert "vectraudit[hdf5]" in caught.value.hint
    assert "convert the file yourself" in caught.value.hint


def test_the_command_prints_the_conversion(tmp_path):
    path, *_ = write_annb(tmp_path / "toy-16-angular.hdf5", "angular")
    result = CliRunner().invoke(
        app, ["dataset", "import-annb", str(path), "--out", str(tmp_path / "out"), "--normalize"]
    )
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())  # the console wraps long lines
    assert "Imported 300 vectors (d=16), 20 queries, 100 published neighbours each" in text
    assert "angular → cosine (normalised)" in text
    assert "Checked the published distances of 20 queries" in text
    refused = CliRunner().invoke(
        app,
        ["dataset", "import-annb", str(path), "--out", str(tmp_path / "o2"), "--distance", "l2"],
    )
    assert refused.exit_code != 0
    assert "contradicts the file" in " ".join(refused.output.split())
