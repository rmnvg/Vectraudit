"""An exact reference computed elsewhere, for ANN recall.

Benchmark datasets such as VectorDBBench publish the exact nearest neighbours of their test
queries. Using them as the reference means ANN recall is measured against the benchmark's
own ground truth, and a backend with no exact search can still be scored. The reference is
**not** relevance labelling: it says which documents are nearest, not which are right.

A reference is a small descriptor (``reference.json``) plus an ``.npy`` of neighbour ids, one
row per query. The descriptor records where the neighbours came from (``reference_source``,
for example ``external:vdbbench``), the metric they were computed under, the hash of every
source file and of the neighbour array, and the filter they assume, if any. Nothing is
guessed: a run is refused when its dataset, filter or metric does not match the reference.

Two runs measured against different references have different ANN recall definitions and are
never compared on it (see :func:`reference_identity`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT = "vectraudit.exact_reference"
FORMAT_VERSION = 1
DISTANCES = ("cosine", "l2", "ip")


class ExternalReferenceError(ValueError):
    """The reference cannot be read or does not fit the run; the message says why."""


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ExternalReference:
    """Exact top-``k`` neighbour ids for a fixed list of queries."""

    source: str
    """Where the neighbours came from, e.g. ``external:vdbbench``."""
    distance: str
    k: int
    query_ids: tuple[str, ...]
    ids: npt.NDArray[np.int64] = field(repr=False, compare=False)
    """Shape ``(len(query_ids), k)``; row ``i`` answers ``query_ids[i]``, nearest first."""
    filter: Mapping[str, Any] | None
    """The filter the neighbours assume (queries must carry exactly this), or ``None``."""
    files: Mapping[str, str]
    """SHA-256 of every file the reference was built from, by name."""
    corpus: Mapping[str, Any]
    descriptor_sha256: str

    def __post_init__(self) -> None:
        if self.ids.shape != (len(self.query_ids), self.k):
            raise ExternalReferenceError(
                f"The neighbour array has shape {self.ids.shape}, expected "
                f"({len(self.query_ids)}, {self.k})."
            )

    def neighbours(self, query_id: str, k: int) -> list[str]:
        """The top-``k`` neighbour ids of one query, as the strings connectors return."""
        if k > self.k:
            raise ExternalReferenceError(
                f"k={k} is larger than the {self.k} neighbours the reference holds."
            )
        row = self._row.get(query_id)
        if row is None:
            raise ExternalReferenceError(f"The reference has no query {query_id!r}.")
        return [str(i) for i in self.ids[row, :k]]

    @property
    def _row(self) -> dict[str, int]:
        # Built once per instance; the dataclass is frozen, so cache through the instance dict.
        cached: dict[str, int] | None = self.__dict__.get("_row_cache")
        if cached is None:
            cached = {q: i for i, q in enumerate(self.query_ids)}
            object.__setattr__(self, "_row_cache", cached)
        return cached

    def ranks_as(self, distance: str) -> bool:
        """Whether a backend ranking by ``distance`` ranks exactly as this reference does.

        The same distance does; so does inner product for a cosine reference whose vectors
        and queries were written at unit length (``corpus.normalized``), since the dot product
        of unit vectors is their cosine.
        """
        if distance == self.distance:
            return True
        return (
            distance == "ip" and self.distance == "cosine" and self.corpus.get("normalized") is True
        )

    def identity(self) -> dict[str, Any]:
        """What a run records about its reference, and what comparisons check."""
        return {
            "source": self.source,
            "sha256": self.descriptor_sha256,
            "distance": self.distance,
            "k": self.k,
            "filter": dict(self.filter) if self.filter is not None else None,
            "files": dict(self.files),
        }

    def check_dataset(self, query_ids: list[str], filters: list[Mapping[str, Any]]) -> None:
        """Refuse a dataset this reference does not describe."""
        missing = [q for q in query_ids if q not in self._row]
        if missing:
            raise ExternalReferenceError(
                f"{len(missing)} dataset queries are not in the reference (for example "
                f"{missing[0]!r}). Use the dataset written beside this reference."
            )
        want = dict(self.filter) if self.filter is not None else {}
        bad = [q for q, f in zip(query_ids, filters, strict=True) if dict(f) != want]
        if bad:
            raise ExternalReferenceError(
                f"The reference assumes the filter {want or 'none'}, but {len(bad)} dataset "
                f"queries carry a different one (for example {bad[0]!r}). Neighbours computed "
                "for one filter say nothing about another."
            )


def reference_identity(run: Mapping[str, Any]) -> tuple[str, str] | None:
    """``(source, sha256)`` of the external reference a run used, or ``None`` when the
    backend's own exact search was the reference. ANN recall compares only on equal values."""
    ref = (run.get("search") or {}).get("reference")
    if not isinstance(ref, Mapping):
        return None
    return (str(ref.get("source")), str(ref.get("sha256")))


def load_reference(path: str | Path) -> ExternalReference:
    """Read ``reference.json`` and verify the neighbour array against its recorded hash."""
    p = Path(path)
    try:
        raw = p.read_bytes()
        descriptor = json.loads(raw)
    except OSError as e:
        raise ExternalReferenceError(f"Cannot read {p}: {e.strerror or e}") from e
    except ValueError as e:
        raise ExternalReferenceError(f"{p} is not JSON.") from e
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("format") != FORMAT
        or descriptor.get("format_version") != FORMAT_VERSION
    ):
        raise ExternalReferenceError(f"{p} is not a {FORMAT} v{FORMAT_VERSION} file.")
    try:
        ids_name = str(descriptor["ids_file"])
        ids_path = p.parent / ids_name
        if not ids_path.is_file():
            raise ExternalReferenceError(f"{ids_name} is missing beside {p.name}.")
        if sha256_file(ids_path) != descriptor["ids_sha256"]:
            raise ExternalReferenceError(
                f"{ids_name} does not match the hash recorded in {p.name}: it changed after the "
                "reference was written."
            )
        ids = np.load(ids_path, allow_pickle=False)
        if ids.dtype.kind not in "iu" or ids.ndim != 2:
            raise ExternalReferenceError(f"{ids_name} must be a 2-D array of integer ids.")
        distance = str(descriptor["distance"])
        if distance not in DISTANCES:
            raise ExternalReferenceError(f"Unknown distance {distance!r} in {p.name}.")
        return ExternalReference(
            source=str(descriptor["reference_source"]),
            distance=distance,
            k=int(descriptor["k"]),
            query_ids=tuple(str(q) for q in descriptor["query_ids"]),
            ids=ids.astype(np.int64),
            filter=descriptor.get("filter"),
            files={str(k): str(v) for k, v in (descriptor.get("files") or {}).items()},
            corpus=dict(descriptor.get("corpus") or {}),
            descriptor_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except KeyError as e:
        raise ExternalReferenceError(f"{p.name} lacks the field {e.args[0]!r}.") from e


def write_reference(
    directory: Path,
    name: str,
    *,
    source: str,
    distance: str,
    query_ids: list[str],
    ids: npt.NDArray[np.int64],
    filter: Mapping[str, Any] | None,
    files: Mapping[str, str],
    corpus: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``<name>.json`` and ``<name>.npy`` in ``directory``; returns the descriptor path."""
    ids_path = directory / f"{name}.npy"
    np.save(ids_path, np.ascontiguousarray(ids, dtype=np.int64), allow_pickle=False)
    descriptor = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "reference_source": source,
        "distance": distance,
        "k": int(ids.shape[1]),
        "n_queries": len(query_ids),
        "query_ids": query_ids,
        "ids_file": ids_path.name,
        "ids_sha256": sha256_file(ids_path),
        "filter": dict(filter) if filter is not None else None,
        "files": dict(files),
        "corpus": dict(corpus),
        "meaning": (
            "Exact nearest neighbours computed by the benchmark, used as ground truth for ANN "
            "recall. This is not relevance labelling."
        ),
        **dict(extra or {}),
    }
    path = directory / f"{name}.json"
    path.write_text(json.dumps(descriptor, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
