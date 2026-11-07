"""Importing a VectorDBBench directory: corpus, queries, published neighbours, filter variants.

The directory is built here in the documented layout from random vectors, with true brute-force
neighbours, because the real datasets are large downloads Vectraudit never fetches. It was not
checked against a downloaded dataset.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation import external_reference as X
from vectraudit.evaluation.vdbbench_import import (
    VdbbenchImportError,
    import_vdbbench,
)

N, D, NQ, K = 300, 16, 12, 10
runner = CliRunner()


def _neighbours(x: np.ndarray, q: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    d2 = ((q[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    if allowed is not None:
        d2[:, ~allowed] = np.inf
    return np.argsort(d2, axis=1)[:, :K]


def make(tmp_path: Path, *, filters=(90,), labels=True, shuffle=True) -> tuple[Path, dict]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, D)).astype("float32")
    q = rng.standard_normal((NQ, D)).astype("float32")
    src = tmp_path / "vdb"
    src.mkdir()
    order = rng.permutation(N) if shuffle else np.arange(N)
    for part, rows in enumerate(np.array_split(order, 2)):
        pq.write_table(
            pa.table({"id": rows.astype("int64"), "emb": [x[i].tolist() for i in rows]}),
            src / f"train-0{part}-of-02.parquet",
        )
    pq.write_table(
        pa.table({"id": np.arange(NQ, dtype="int64"), "emb": [v.tolist() for v in q]}),
        src / "test.parquet",
    )

    def write_neighbours(name: str, found: np.ndarray) -> None:
        pq.write_table(
            pa.table({"id": np.arange(NQ, dtype="int64"), "neighbors_id": found.tolist()}),
            src / name,
        )

    write_neighbours("neighbors.parquet", _neighbours(x, q))
    for percent in filters:
        allowed = np.arange(N) >= int(N * percent / 100)
        write_neighbours(f"neighbors_int_{percent}p.parquet", _neighbours(x, q, allowed))
    if labels:
        pq.write_table(
            pa.table(
                {"id": np.arange(N, dtype="int64"), "labels": [f"l{i % 4}" for i in range(N)]}
            ),
            src / "scalar_labels.parquet",
        )
    return src, {"x": x, "q": q}


@pytest.fixture
def imported(tmp_path):
    src, data = make(tmp_path)
    out = tmp_path / "out"
    return src, out, data, import_vdbbench(src, out, distance="l2")


def test_the_corpus_is_written_sorted_by_id(imported) -> None:
    _, out, data, result = imported
    ids = np.load(out / "ids.npy")
    vectors = np.load(out / "vectors.npy")
    assert ids.tolist() == list(range(N))
    np.testing.assert_array_equal(vectors, data["x"])  # shuffled in the files, sorted here
    assert (result.n_vectors, result.dim, result.n_queries, result.k) == (N, D, NQ, K)
    np.testing.assert_array_equal(np.load(out / "queries.npy"), data["q"])


def test_the_published_neighbours_become_an_external_reference(imported) -> None:
    _, out, data, _ = imported
    ref = X.load_reference(out / "reference.json")
    assert (ref.source, ref.distance, ref.k, ref.filter) == ("external:vdbbench", "l2", K, None)
    assert ref.query_ids == tuple(f"q{i}" for i in range(NQ))
    assert ref.neighbours("q3", 5) == [str(i) for i in _neighbours(data["x"], data["q"])[3][:5]]
    assert set(ref.files) >= {"test.parquet", "neighbors.parquet", "train-00-of-02.parquet"}
    assert all(len(h) == 64 for h in ref.files.values())
    assert ref.corpus["n"] == N
    with pytest.raises(X.ExternalReferenceError, match="larger than"):
        ref.neighbours("q0", K + 1)


def test_the_dataset_carries_provenance_and_no_labels(imported) -> None:
    _, out, _, _ = imported
    lines = (out / "dataset.jsonl").read_text().splitlines()
    header = json.loads(lines[0])["dataset"]
    assert header["provenance"]["reference_source"] == "external:vdbbench"
    assert header["provenance"]["distance"] == "l2"
    assert "neighbors.parquet" in header["provenance"]["files"]
    rows = [json.loads(line) for line in lines[1:]]
    assert len(rows) == NQ
    # The neighbours are an ANN reference, not relevance labels.
    assert all("relevant" not in r and "filter" not in r for r in rows)


def test_a_filter_rate_is_converted_to_a_pass_rate(imported) -> None:
    _, out, data, result = imported
    assert result.filters == [
        {"field": "vdb_int_90p", "removed": 0.9, "pass_rate": pytest.approx(0.1), "cut": 270}
    ]
    ref = X.load_reference(out / "reference-int-90p.json")
    assert ref.filter == {"vdb_int_90p": True}
    descriptor = json.loads((out / "reference-int-90p.json").read_text())
    assert descriptor["pass_rate"] == pytest.approx(0.1)
    assert descriptor["filter_rate_removed"] == 0.9
    assert descriptor["filter_rule"] == "id >= int(n * 90/100) = 270"
    assert all(int(i) >= 270 for i in ref.neighbours("q0", K))
    rows = [json.loads(line) for line in (out / "dataset-int-90p.jsonl").read_text().splitlines()]
    assert all(r["filter"] == {"vdb_int_90p": True} for r in rows[1:])
    meta = {m["id"]: m for m in map(json.loads, (out / "metadata.jsonl").read_text().splitlines())}
    assert len(meta) == N
    assert sum(m["vdb_int_90p"] for m in meta.values()) == 30
    assert all(m["vdb_int_90p"] == (i >= 270) for i, m in meta.items())
    assert meta[5]["label"] == "l1"
    # The filtered reference is the exact top-k among the passing ids.
    allowed = np.arange(N) >= 270
    assert ref.neighbours("q1", K) == [
        str(i) for i in _neighbours(data["x"], data["q"], allowed)[1]
    ]


def test_the_source_directory_is_never_written(tmp_path) -> None:
    src, _ = make(tmp_path)
    before = sorted(p.name for p in src.iterdir())
    import_vdbbench(src, tmp_path / "out", distance="l2")
    assert sorted(p.name for p in src.iterdir()) == before


def test_no_filter_files_means_no_metadata_unless_labels_exist(tmp_path) -> None:
    src, _ = make(tmp_path, filters=(), labels=False)
    result = import_vdbbench(src, tmp_path / "out", distance="cosine")
    assert result.filters == []
    assert result.datasets == ["dataset.jsonl"]
    assert not (tmp_path / "out" / "metadata.jsonl").exists()
    assert X.load_reference(tmp_path / "out" / "reference.json").distance == "cosine"


def test_the_distance_is_required_and_checked(tmp_path) -> None:
    src, _ = make(tmp_path)
    with pytest.raises(VdbbenchImportError, match="Unknown distance"):
        import_vdbbench(src, tmp_path / "o", distance="euclid")
    out = runner.invoke(app, ["dataset", "import-vdbbench", str(src), "--out", str(tmp_path / "o")])
    assert out.exit_code != 0  # --distance has no default


@pytest.mark.parametrize(
    ("remove", "message"),
    [("test.parquet", "test.parquet is missing"), ("neighbors.parquet", "neighbors.parquet")],
)
def test_missing_files_are_named(tmp_path, remove, message) -> None:
    src, _ = make(tmp_path)
    (src / remove).unlink()
    with pytest.raises(VdbbenchImportError, match=message) as e:
        import_vdbbench(src, tmp_path / "o", distance="l2")
    assert "never downloads" in e.value.hint or "Fetch" in e.value.hint


def test_no_train_files_is_refused_with_the_hint(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(VdbbenchImportError, match="No train"):
        import_vdbbench(tmp_path / "empty", tmp_path / "o", distance="l2")


def test_neighbours_outside_the_corpus_mean_a_partial_download(tmp_path) -> None:
    src, _ = make(tmp_path)
    (src / "train-01-of-02.parquet").unlink()
    with pytest.raises(VdbbenchImportError, match="not in the train files") as e:
        import_vdbbench(src, tmp_path / "o", distance="l2")
    assert "partial" in e.value.hint


def test_a_filter_file_that_contradicts_its_rule_is_refused(tmp_path) -> None:
    src, data = make(tmp_path, filters=())
    # Neighbours of the UNFILTERED search under a name that claims a 90% filter.
    pq.write_table(
        pa.table(
            {
                "id": np.arange(NQ, dtype="int64"),
                "neighbors_id": _neighbours(data["x"], data["q"]).tolist(),
            }
        ),
        src / "neighbors_int_90p.parquet",
    )
    with pytest.raises(VdbbenchImportError, match="below 270") as e:
        import_vdbbench(src, tmp_path / "o", distance="l2")
    assert "refusing to guess" in e.value.hint


def test_missing_pyarrow_gives_the_install_hint(tmp_path, monkeypatch) -> None:
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    src, _ = make(tmp_path)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(VdbbenchImportError, match="pyarrow") as e:
        import_vdbbench(src, tmp_path / "o", distance="l2")
    assert "vectraudit[parquet]" in e.value.hint


def test_a_changed_neighbour_array_is_detected(imported) -> None:
    _, out, _, _ = imported
    ids = np.load(out / "reference.npy")
    ids[0, 0] = 0
    np.save(out / "reference.npy", ids)
    with pytest.raises(X.ExternalReferenceError, match="does not match the hash"):
        X.load_reference(out / "reference.json")


def test_the_command_prints_the_conversion(tmp_path) -> None:
    src, _ = make(tmp_path)
    out = runner.invoke(
        app,
        ["dataset", "import-vdbbench", str(src), "--distance", "l2", "--out", str(tmp_path / "o")],
    )
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "Imported 300 vectors" in text
    assert "removes 90%, pass rate 10%" in text
    bad = runner.invoke(
        app,
        [
            "dataset",
            "import-vdbbench",
            str(tmp_path),
            "--distance",
            "l2",
            "--out",
            str(tmp_path / "p"),
        ],
    )
    assert bad.exit_code == 1
    assert "Fetch" in " ".join(bad.output.split())
