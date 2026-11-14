"""Import an ann-benchmarks HDF5 dataset: vectors, queries and the published neighbours.

ann-benchmarks publishes each dataset as one HDF5 file (read in its source, ``datasets.py``
``write_output`` and ``distance.py``, commit 2e081ad32c1e):

- attributes ``type`` (``dense`` or ``sparse``), ``distance`` (``euclidean``, ``angular``,
  ``hamming`` or ``jaccard``), ``dimension`` and ``point_type`` (``float`` or ``bit``);
- datasets ``train`` (the corpus), ``test`` (the queries), ``neighbors`` (each query's 100
  nearest train rows, nearest first) and ``distances`` (their distances).

``euclidean`` distances are the true Euclidean distance; ``angular`` ones are ``1 - cosine``
on vectors that are **not** normalised in the file. They become Vectraudit's ``l2`` and
``cosine``. ``hamming`` (bit vectors) and ``jaccard`` (sparse sets) have no Vectraudit distance
and are refused, as is a ``--distance`` that contradicts the file. The metric is also
checked against the data: for a seeded sample of queries, the published distances must equal
the distances recomputed from the vectors, or the import is refused rather than guessed.

This writes the same files as ``dataset import-vdbbench``: ``vectors.npy`` and ``ids.npy``
(train row ``i`` has id ``i``, the ids ann-benchmarks' neighbours use), ``queries.npy``,
``dataset.jsonl`` and the neighbours as an external exact reference (``external:annb``) with
the hash of the source file. With ``normalize`` (angular files only), vectors and queries are
written at unit length: the angular neighbours are unchanged, and an inner-product index (for
example FAISS ``IndexFlatIP``) then ranks exactly as cosine does, so it can be scored against
the reference.

ann-benchmarks counts a returned neighbour as found when its distance is within the 100th true
distance plus 1e-3, so near-ties count; Vectraudit's ANN recall matches neighbour ids. On data
with many exact ties the two can differ slightly.

Needs the ``[hdf5]`` extra (h5py). Nothing is downloaded: fetch the file from ann-benchmarks.

Credit: ann-benchmarks (Aumüller, Bernhardsson and Faithfull, Information Systems 2020; MIT),
whose dataset layout and distance conventions these follow. See docs/credits.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit.evaluation.external_reference import sha256_file, write_reference

SOURCE = "external:annb"
HDF5_HINT = (
    'Reading ann-benchmarks files needs h5py: pip install "vectraudit[hdf5]". Without it, '
    "convert the file yourself to train.npy, test.npy and neighbors.npy with any HDF5 tool."
)
DISTANCES = {"euclidean": "l2", "angular": "cosine"}
UNSUPPORTED = {
    "hamming": "bit vectors under Hamming distance",
    "jaccard": "sparse sets under Jaccard distance",
}
CHECKED_QUERIES = 20
_ROWS = 100_000


class AnnbImportError(ValueError):
    """The file cannot be imported; ``hint`` says what to change."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class AnnbImport:
    """What was written."""

    out: Path
    n_vectors: int
    dim: int
    n_queries: int
    k: int
    distance: str
    source_distance: str
    normalized: bool
    checked_queries: int


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as e:
        raise AnnbImportError("h5py is not installed.", HDF5_HINT) from e
    return h5py


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _unit(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise AnnbImportError("A vector has length zero, so it has no direction to normalise.")
    out: npt.NDArray[np.float32] = (x / norms).astype(np.float32)
    return out


def _distances(metric: str, query: npt.NDArray[Any], rows: npt.NDArray[Any]) -> npt.NDArray[Any]:
    q = query.astype(np.float64)
    r = rows.astype(np.float64)
    if metric == "euclidean":
        out: npt.NDArray[Any] = np.sqrt(((r - q) ** 2).sum(axis=1))
        return out
    cos = (r @ q) / (np.linalg.norm(r, axis=1) * np.linalg.norm(q))
    out = 1 - cos
    return out


def import_annb(
    source: Path,
    out: Path,
    *,
    distance: str | None = None,
    normalize: bool = False,
    name: str | None = None,
    seed: int = 0,
) -> AnnbImport:
    """Convert the ann-benchmarks HDF5 file ``source`` into files under ``out``."""
    h5py = _h5py()
    if not source.is_file():
        raise AnnbImportError(
            f"{source} is not a file.",
            "Fetch the dataset from ann-benchmarks; Vectraudit never downloads datasets.",
        )
    try:
        handle = h5py.File(source, "r")
    except OSError as e:
        raise AnnbImportError(f"{source.name} is not an HDF5 file.") from e
    with handle as f:
        metric = _text(f.attrs.get("distance", ""))
        if metric in UNSUPPORTED:
            raise AnnbImportError(
                f"{source.name} holds {UNSUPPORTED[metric]}, which Vectraudit does not support.",
                "Use a euclidean or angular ann-benchmarks dataset.",
            )
        if metric not in DISTANCES:
            raise AnnbImportError(
                f"{source.name} records the distance {metric!r}; expected euclidean or angular.",
                "This does not look like an ann-benchmarks dataset file.",
            )
        mapped = DISTANCES[metric]
        if distance is not None and distance != mapped:
            raise AnnbImportError(
                f"--distance {distance} contradicts the file, which is {metric} ({mapped}).",
                "Leave --distance out: it is read from the file.",
            )
        if _text(f.attrs.get("type", "dense")) != "dense":
            raise AnnbImportError(f"{source.name} is a sparse dataset; only dense ones import.")
        if normalize and metric != "angular":
            raise AnnbImportError(
                "--normalize changes Euclidean neighbours; it applies to angular files only."
            )
        for key in ("train", "test", "neighbors", "distances"):
            if key not in f:
                raise AnnbImportError(
                    f"{source.name} has no {key!r} dataset.",
                    "An ann-benchmarks file holds train, test, neighbors and distances.",
                )
        train, test = f["train"], f["test"]
        if len(train.shape) != 2 or len(test.shape) != 2 or train.shape[1] != test.shape[1]:
            raise AnnbImportError("train and test must be matrices of the same width.")
        n, dim = int(train.shape[0]), int(train.shape[1])
        neighbors = np.asarray(f["neighbors"], dtype=np.int64)
        published = np.asarray(f["distances"], dtype=np.float64)
        queries = np.ascontiguousarray(test[:], dtype=np.float32)
        if neighbors.shape != published.shape or neighbors.shape[0] != len(queries):
            raise AnnbImportError("neighbors and distances must have one row per test query.")
        if neighbors.min() < 0 or neighbors.max() >= n:
            raise AnnbImportError("neighbors names train rows that do not exist.")
        if (np.diff(published, axis=1) < -1e-6).any():
            raise AnnbImportError("The published neighbours are not sorted nearest first.")
        rng = np.random.default_rng(seed)
        sample = np.sort(
            rng.choice(len(queries), min(CHECKED_QUERIES, len(queries)), replace=False)
        )
        for i in sample:
            if len(set(neighbors[i].tolist())) != len(neighbors[i]):
                raise AnnbImportError(f"Test query {int(i)} lists a train row twice.")
            rows = train[np.sort(neighbors[i])]
            order = np.argsort(np.argsort(neighbors[i]))
            recomputed = _distances(metric, queries[i], rows[order])
            if not np.allclose(recomputed, published[i], rtol=1e-3, atol=1e-4):
                raise AnnbImportError(
                    f"The published distances of test query {int(i)} do not match {metric} "
                    "distances recomputed from the vectors.",
                    "The file's recorded distance does not describe its neighbours; refusing "
                    "to guess.",
                )
        out.mkdir(parents=True, exist_ok=True)
        # numpy 2.2's stubs leave open_memmap untyped; newer ones type it.
        vectors = np.lib.format.open_memmap(  # type: ignore[no-untyped-call,unused-ignore]
            out / "vectors.npy", mode="w+", dtype=np.float32, shape=(n, dim)
        )
        for start in range(0, n, _ROWS):
            block = np.asarray(train[start : start + _ROWS], dtype=np.float32)
            vectors[start : start + len(block)] = _unit(block) if normalize else block
        vectors.flush()
        del vectors
    if normalize:
        queries = _unit(queries)
    np.save(out / "ids.npy", np.arange(n, dtype=np.int64), allow_pickle=False)
    np.save(out / "queries.npy", queries, allow_pickle=False)
    query_ids = [f"q{i}" for i in range(len(queries))]
    label = name or source.stem
    files = {source.name: sha256_file(source)}
    provenance = {
        "source": "ann-benchmarks",
        "reference_source": SOURCE,
        "file": source.name,
        "distance": mapped,
        "source_distance": metric,
        "normalized": normalize,
        "files": files,
    }
    with (out / "dataset.jsonl").open("w", encoding="utf-8") as stream:
        header: dict[str, Any] = {
            "name": label,
            "description": "Test queries of an ann-benchmarks dataset.",
            "provenance": provenance,
        }
        if normalize:
            header["normalized"] = True
        stream.write(json.dumps({"dataset": header}) + "\n")
        for qid, vec in zip(query_ids, queries, strict=True):
            stream.write(json.dumps({"query_id": qid, "vector": vec.tolist()}) + "\n")
    corpus = {
        "n": n,
        "dim": dim,
        "normalized": normalize,
        "vectors_file": "vectors.npy",
        "vectors_sha256": sha256_file(out / "vectors.npy"),
        "ids_file": "ids.npy",
        "ids_sha256": sha256_file(out / "ids.npy"),
        "queries_file": "queries.npy",
        "queries_sha256": sha256_file(out / "queries.npy"),
        "queries_array_sha256": hashlib.sha256(
            np.ascontiguousarray(queries, dtype="<f4").tobytes()
        ).hexdigest(),
    }
    write_reference(
        out, "reference", source=SOURCE, distance=mapped, query_ids=query_ids, ids=neighbors,
        filter=None, files=files, corpus=corpus,
        extra={"source_distance": metric, "recall_note": (
            "ann-benchmarks counts a neighbour found within the k-th true distance plus 1e-3; "
            "Vectraudit's ANN recall matches neighbour ids."
        )},
    )  # fmt: skip
    return AnnbImport(
        out, n, dim, len(queries), int(neighbors.shape[1]), mapped, metric, normalize, len(sample)
    )
