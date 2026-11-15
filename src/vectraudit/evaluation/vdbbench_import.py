"""Import a VectorDBBench dataset directory: vectors, queries and the published neighbours.

VectorDBBench ships each dataset as Parquet files:

- ``train*.parquet`` (or ``shuffle_train*.parquet``): ``id`` and ``emb``, the corpus;
- ``test.parquet``: the query embeddings;
- ``neighbors.parquet``: ``id`` (a test query) and ``neighbors_id`` (its exact nearest
  neighbours in the corpus);
- optionally ``neighbors_int_<p>p.parquet``: the neighbours when the integer filter
  ``id >= int(n * p / 100)`` removes ``p`` percent of the corpus, and ``scalar_labels.parquet``
  (``id``, ``labels``) for label filters.

This writes what the rest of Vectraudit reads: ``vectors.npy`` and ``ids.npy`` (rows sorted by
id), ``queries.npy`` and ``dataset.jsonl`` (the test queries), and the published neighbours
as an **external exact reference** (:mod:`vectraudit.evaluation.external_reference`) with the
hash of every source file. Each filtered variant gets its own dataset and reference, and the
filter becomes a boolean metadata field (``vdb_int_99p``) written to ``metadata.jsonl``, so
any backend that can filter on a field can be checked against the published neighbours.

VectorDBBench's filter rate is the fraction **removed**; Vectraudit's pass rate is the fraction
kept. The importer converts (99% removed is a 1% pass rate) and checks that a variant's
neighbours all satisfy its filter, refusing the import otherwise instead of guessing.

The distance is required and recorded, never inferred: the same vectors are cosine neighbours
or L2 neighbours depending on how the benchmark was built.

Nothing is downloaded. Fetch a dataset with VectorDBBench's own tooling and point this at
the directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit.core.metadata import PARQUET_HINT
from vectraudit.evaluation.external_reference import (
    DISTANCES,
    sha256_file,
    write_reference,
)

SOURCE = "external:vdbbench"
_INT_FILTER = re.compile(r"^neighbors_int_(\d+)p\.parquet$")
_BATCH = 20_000


class VdbbenchImportError(ValueError):
    """The directory cannot be imported; ``hint`` says what to change."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class VdbbenchImport:
    """What was written."""

    out: Path
    n_vectors: int
    dim: int
    n_queries: int
    k: int
    distance: str
    datasets: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)


def _parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise VdbbenchImportError("pyarrow is not installed.", PARQUET_HINT) from e
    return pq


def _one(directory: Path, pattern: str, what: str) -> Path:
    path = directory / pattern
    if not path.is_file():
        raise VdbbenchImportError(
            f"{pattern} is missing from {directory}.",
            f"A VectorDBBench dataset directory holds {what}. Fetch it with VectorDBBench; "
            "Vectraudit never downloads datasets.",
        )
    return path


def _embeddings(column: Any) -> npt.NDArray[np.float32]:
    """A list column of equal-length float lists as an ``(n, d)`` float32 array."""
    import pyarrow.compute as pc

    if hasattr(column, "combine_chunks"):  # a table column; a record batch's is already an array
        column = column.combine_chunks()
    if column.null_count:
        raise VdbbenchImportError("An embedding column contains nulls.")
    lengths = pc.list_value_length(column)
    lo, hi = pc.min_max(lengths).as_py().values() if len(column) else (0, 0)
    if lo != hi or not lo:
        raise VdbbenchImportError("Embeddings in the file do not all have the same dimension.")
    flat = column.flatten().to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(flat.reshape(len(column), lo), dtype=np.float32)


def _batches(pq: Any, path: Path) -> Iterator[Any]:
    yield from pq.ParquetFile(path).iter_batches(batch_size=_BATCH, columns=["id", "emb"])


def import_vdbbench(
    source: Path, out: Path, *, distance: str, name: str | None = None
) -> VdbbenchImport:
    """Convert the VectorDBBench directory ``source`` into files under ``out``."""
    if distance not in DISTANCES:
        raise VdbbenchImportError(
            f"Unknown distance {distance!r}.", f"Use one of: {', '.join(DISTANCES)}."
        )
    pq = _parquet()
    if not source.is_dir():
        raise VdbbenchImportError(f"{source} is not a directory.")
    train = sorted({*source.glob("train*.parquet"), *source.glob("shuffle_train*.parquet")})
    if not train:
        raise VdbbenchImportError(
            f"No train*.parquet in {source}.", "Fetch the dataset with VectorDBBench first."
        )
    test_path = _one(source, "test.parquet", "test.parquet (the queries)")
    neighbors_path = _one(source, "neighbors.parquet", "neighbors.parquet (the exact neighbours)")
    variants = sorted(
        (int(m.group(1)), p)
        for p in source.glob("neighbors_int_*p.parquet")
        if (m := _INT_FILTER.match(p.name))
    )
    labels_path = source / "scalar_labels.parquet"
    out.mkdir(parents=True, exist_ok=True)
    label = name or source.name

    # --- the corpus: ids first, then vectors written straight to their sorted positions -----
    ids = np.concatenate([pq.read_table(p, columns=["id"]).column("id").to_numpy() for p in train])
    ids = ids.astype(np.int64)
    n = len(ids)
    sorted_ids = np.sort(ids)
    if n == 0 or (np.diff(sorted_ids) == 0).any():
        raise VdbbenchImportError("The train files hold no vectors or repeat an id.")
    vectors: Any = None
    for path in train:
        for batch in _batches(pq, path):
            block = _embeddings(batch.column("emb"))
            if vectors is None:
                # numpy 2.2's stubs leave open_memmap untyped; newer ones type it.
                vectors = np.lib.format.open_memmap(  # type: ignore[no-untyped-call,unused-ignore]
                    out / "vectors.npy", mode="w+", dtype=np.float32, shape=(n, block.shape[1])
                )
            elif block.shape[1] != vectors.shape[1]:
                raise VdbbenchImportError("Train files disagree on the embedding dimension.")
            rows = np.searchsorted(sorted_ids, batch.column("id").to_numpy())
            vectors[rows] = block
    assert vectors is not None
    dim = int(vectors.shape[1])
    vectors.flush()
    del vectors
    np.save(out / "ids.npy", sorted_ids, allow_pickle=False)

    # --- the queries and the published neighbours --------------------------------------
    test = pq.read_table(test_path)
    queries = _embeddings(test.column("emb"))
    if queries.shape[1] != dim:
        raise VdbbenchImportError(
            f"The queries have dimension {queries.shape[1]}, the corpus {dim}."
        )
    test_ids = (
        test.column("id").to_numpy().astype(np.int64)
        if "id" in test.column_names
        else np.arange(len(queries), dtype=np.int64)
    )
    query_ids = [f"q{i}" for i in test_ids]
    np.save(out / "queries.npy", queries, allow_pickle=False)

    def neighbours(path: Path, least_id: int = 0) -> npt.NDArray[np.int64]:
        table = pq.read_table(path, columns=["id", "neighbors_id"])
        rows = {
            int(i): np.asarray(v, dtype=np.int64)
            for i, v in zip(
                table.column("id").to_pylist(),
                table.column("neighbors_id").to_pylist(),
                strict=True,
            )
        }
        missing = [int(i) for i in test_ids if int(i) not in rows]
        if missing:
            raise VdbbenchImportError(
                f"{path.name} has no neighbours for {len(missing)} test queries "
                f"(for example id {missing[0]}).",
                "Use the neighbours file that belongs to this test.parquet.",
            )
        lengths = {len(rows[int(i)]) for i in test_ids}
        if len(lengths) != 1 or 0 in lengths:
            raise VdbbenchImportError(f"{path.name} has neighbour lists of different lengths.")
        found = np.stack([rows[int(i)] for i in test_ids])
        position = np.searchsorted(sorted_ids, found)
        position = np.minimum(position, n - 1)
        if (sorted_ids[position] != found).any():
            raise VdbbenchImportError(
                f"{path.name} names neighbours that are not in the train files.",
                "A partial download? Fetch every train file of the dataset.",
            )
        if least_id and (found < least_id).any():
            raise VdbbenchImportError(
                f"{path.name} contains ids below {least_id}, which its filter removes.",
                "The file does not match the documented integer filter; refusing to guess.",
            )
        return found

    files = {p.name: sha256_file(p) for p in (*train, test_path, neighbors_path)}
    files.update({p.name: sha256_file(p) for _, p in variants})
    if labels_path.is_file():
        files[labels_path.name] = sha256_file(labels_path)
    corpus = {
        "n": n,
        "dim": dim,
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
    base_ids = neighbours(neighbors_path)
    k = int(base_ids.shape[1])
    provenance = {
        "source": "VectorDBBench",
        "reference_source": SOURCE,
        "directory": source.name,
        "distance": distance,
        "files": files,
    }

    def dataset(filename: str, flt: dict[str, Any] | None, description: str) -> None:
        header = {"dataset": {"name": f"{label}{filename[7:-6]}", "description": description,
                              "provenance": provenance}}  # fmt: skip
        with (out / filename).open("w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for qid, vec in zip(query_ids, queries, strict=True):
                row: dict[str, Any] = {"query_id": qid, "vector": vec.tolist()}
                if flt:
                    row["filter"] = flt
                f.write(json.dumps(row) + "\n")

    result = VdbbenchImport(out, n, dim, len(queries), k, distance, files=files)
    dataset("dataset.jsonl", None, "Test queries of a VectorDBBench dataset (unfiltered).")
    result.datasets.append("dataset.jsonl")
    ref = write_reference(
        out, "reference", source=SOURCE, distance=distance, query_ids=query_ids, ids=base_ids,
        filter=None, files=files, corpus=corpus,
    )  # fmt: skip
    result.references.append(ref.name)

    metadata: dict[int, dict[str, Any]] = (
        {int(i): {"id": int(i)} for i in sorted_ids} if (variants or labels_path.is_file()) else {}
    )
    for percent, path in variants:
        removed = percent / 100
        cut = int(n * removed)
        field_name = f"vdb_int_{percent}p"
        found = neighbours(path, least_id=cut)
        for i, row in metadata.items():
            row[field_name] = i >= cut
        flt = {field_name: True}
        stem = f"int-{percent}p"
        dataset(
            f"dataset-{stem}.jsonl",
            flt,
            f"Test queries with VectorDBBench's integer filter that removes {percent}% of the "
            f"corpus (a {100 - percent}% pass rate).",
        )
        write_reference(
            out, f"reference-{stem}", source=SOURCE, distance=distance, query_ids=query_ids,
            ids=found, filter=flt, files=files, corpus=corpus,
            extra={
                "filter_rule": f"id >= int(n * {percent}/100) = {cut}",
                "filter_rate_removed": removed,
                "pass_rate": 1 - removed,
                "metadata_field": field_name,
            },
        )  # fmt: skip
        result.datasets.append(f"dataset-{stem}.jsonl")
        result.references.append(f"reference-{stem}.json")
        result.filters.append(
            {"field": field_name, "removed": removed, "pass_rate": 1 - removed, "cut": cut}
        )
    if labels_path.is_file():
        table = pq.read_table(labels_path, columns=["id", "labels"])
        for i, lab in zip(
            table.column("id").to_pylist(), table.column("labels").to_pylist(), strict=True
        ):
            if int(i) in metadata:
                metadata[int(i)]["label"] = lab
    if metadata:
        with (out / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for row in metadata.values():
                f.write(json.dumps(row) + "\n")
    return result
