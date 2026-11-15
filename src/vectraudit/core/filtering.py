"""Metadata filters as an allowed set of ids, applied inside the FAISS search itself.

A filter is evaluated once against the metadata and becomes an :class:`AllowedSet`: the
sorted user-facing ids that satisfy it. The search passes it to FAISS as an ``IDSelector``,
so the index never returns a disallowed id, and the exact ground truth is computed over the
same set. Nothing is post-filtered, which is the point: post-filtering a top-k loses
results, and this measures what an index that filters *during* search does.

``pass_rate`` is the fraction of the corpus that **satisfies** the filter. Benchmarks that
quote a "filter rate" usually mean the fraction removed; convert before comparing.

Credit: the filter-rate framing of filtered search follows VectorDBBench (Zilliz); its filter
rate is the fraction removed, Vectraudit's pass rate the fraction kept. Our own implementation.
See docs/credits.md.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit.contracts import matches_filter
from vectraudit.core._faiss import import_faiss
from vectraudit.core.metadata import Metadata

IntArray = npt.NDArray[np.int64]


class FilterError(ValueError):
    """A filter cannot be applied; ``hint`` says what to change."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class AllowedSet:
    """The user-facing ids that satisfy a filter, out of a corpus of ``universe`` ids."""

    ids: IntArray
    """Sorted, unique."""
    universe: int
    filters: Mapping[str, Any] | None = None
    """The filter this set was built from, for display; ``None`` for a hand-built set."""

    def __post_init__(self) -> None:
        if self.universe < 0 or len(self.ids) > self.universe:
            raise FilterError(
                "The allowed set is larger than the corpus.",
                "Build it from ids that exist in the index.",
            )

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def pass_rate(self) -> float:
        """Fraction of the corpus that satisfies the filter (1.0 = everything passes)."""
        return len(self.ids) / self.universe if self.universe else 1.0

    @property
    def fingerprint(self) -> str:
        """Identifies the set itself (not how it was described), for caching ground truth."""
        h = hashlib.sha1(usedforsecurity=False)
        h.update(str(self.universe).encode())
        h.update(self.ids.tobytes())
        return h.hexdigest()[:16]

    def mask(self, ids: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Which of ``ids`` are allowed (``-1`` empty slots are never allowed)."""
        arr = np.asarray(ids, dtype=np.int64)
        return np.isin(arr, self.ids) & (arr >= 0)

    def selector(self) -> Any:
        """A FAISS ``IDSelectorBatch`` over the allowed ids; keep it alive while searching.

        Batch rather than bitmap: user ids need not be dense offsets, and with an ``IDMap``
        FAISS translates the selector so it is matched against user ids.
        """
        faiss = import_faiss()
        return faiss.IDSelectorBatch(np.ascontiguousarray(self.ids, dtype=np.int64))


def filter_fingerprint(filters: Mapping[str, Any]) -> str:
    """Stable hash of a filter description, independent of key order."""
    blob = json.dumps(filters, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode(), usedforsecurity=False).hexdigest()[:16]


def row_matches(row: Mapping[str, Any] | None, filters: Mapping[str, Any]) -> bool:
    """Whether a metadata row satisfies every condition. A missing row or field never does."""
    if row is None:
        return False
    return all(key in row and matches_filter(row[key], cond) for key, cond in filters.items())


def allowed_from_metadata(
    metadata: Metadata, filters: Mapping[str, Any], ids: npt.ArrayLike
) -> AllowedSet:
    """The ids (among ``ids``, the corpus) whose metadata satisfies ``filters``.

    Documents with no metadata row do not satisfy a filter. An empty filter allows all.
    """
    corpus = np.unique(np.asarray(ids, dtype=np.int64))
    if not filters:
        return AllowedSet(corpus, len(corpus), dict(filters))
    keep = [int(i) for i in corpus if row_matches(metadata.get(int(i)), filters)]
    return AllowedSet(np.asarray(keep, dtype=np.int64), len(corpus), dict(filters))


def allowed_from_ids(allowed: npt.ArrayLike, ids: npt.ArrayLike) -> AllowedSet:
    """An allowed set from explicit ids, restricted to those that exist in ``ids``."""
    corpus = np.unique(np.asarray(ids, dtype=np.int64))
    wanted = np.unique(np.asarray(allowed, dtype=np.int64))
    return AllowedSet(wanted[np.isin(wanted, corpus)], len(corpus))
