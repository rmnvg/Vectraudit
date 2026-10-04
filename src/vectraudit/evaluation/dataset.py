"""Evaluation datasets: labelled queries in JSONL, identified by a content hash.

One JSON object per line::

    {"query_id": "q1", "vector": [0.1, ...], "text": "how do I ...",
     "relevant": {"doc-42": 2, "doc-7": 1}, "filter": {"lang": "en"}, "tags": ["faq"]}

``query_id`` is required and unique. A query needs a ``vector``, a ``text`` (embedded at
run time), or both. ``relevant`` maps document ids to non-negative grades; a query without
it is unlabelled and only contributes ANN recall and latency. ``filter`` is a portable
filter: field -> scalar, all conditions AND-ed. ``tags`` name the cohorts a query belongs to.

The dataset version is the SHA-256 of a canonical form that ignores line order and JSON
formatting, so any change to the content is a new version and nothing else is.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
FilterValue = str | int | float | bool

MAX_QUERIES = 100_000
MAX_ID_LENGTH = 512
FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_KEYS = {"query_id", "vector", "text", "relevant", "filter", "tags"}


class DatasetError(ValueError):
    """The dataset file is malformed; the message names the line and the problem."""


@dataclass(frozen=True)
class DatasetQuery:
    """One evaluation query."""

    query_id: str
    vector: FloatArray | None
    text: str | None
    relevant: dict[str, float] = field(default_factory=dict)
    filter: dict[str, FilterValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @property
    def labelled(self) -> bool:
        return bool(self.relevant)


@dataclass(frozen=True)
class Dataset:
    """A validated dataset. ``sha256`` identifies this exact content."""

    queries: tuple[DatasetQuery, ...]
    sha256: str
    dim: int | None
    """Vector dimension, or ``None`` when every query is text-only."""

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def n_labelled(self) -> int:
        return sum(q.labelled for q in self.queries)

    @property
    def needs_embedder(self) -> bool:
        """True when some query has text but no vector."""
        return any(q.vector is None for q in self.queries)

    def summary(self) -> dict[str, Any]:
        """What a run records about the dataset it used."""
        tags = sorted({t for q in self.queries for t in q.tags})
        return {
            "sha256": self.sha256,
            "n_queries": len(self.queries),
            "n_labelled": self.n_labelled,
            "n_filtered": sum(bool(q.filter) for q in self.queries),
            "dim": self.dim,
            "tags": tags,
        }


def load_dataset(path: str | Path) -> Dataset:
    """Read and validate a JSONL dataset file."""
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"Dataset file not found: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise DatasetError(f"{p} is not UTF-8 text.") from e
    return parse_dataset(text)


def parse_dataset(text: str) -> Dataset:
    """Validate JSONL dataset text (see the module docstring for the format)."""
    queries: list[DatasetQuery] = []
    seen: set[str] = set()
    dim: int | None = None
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            q = _parse_query(json.loads(line, parse_constant=_reject_constant))
            if q.query_id in seen:
                raise ValueError(f"duplicate query_id {q.query_id!r}")
            if q.vector is not None:
                if dim is None:
                    dim = len(q.vector)
                elif len(q.vector) != dim:
                    raise ValueError(f"vector has dimension {len(q.vector)}, expected {dim}")
        except (ValueError, TypeError) as e:
            raise DatasetError(f"Line {line_no}: {e}") from e
        seen.add(q.query_id)
        queries.append(q)
        if len(queries) > MAX_QUERIES:
            raise DatasetError(f"More than {MAX_QUERIES:,} queries; split the dataset.")
    if not queries:
        raise DatasetError("The dataset has no queries.")
    return Dataset(tuple(queries), _fingerprint(queries), dim)


def _reject_constant(token: str) -> float:
    raise ValueError(f"{token} is not a valid number")


def _parse_query(obj: Any) -> DatasetQuery:
    if not isinstance(obj, dict):
        raise ValueError("each line must be a JSON object")
    unknown = set(obj) - _KEYS
    if unknown:
        raise ValueError(f"unknown field(s) {sorted(unknown)}; allowed: {sorted(_KEYS)}")
    query_id = _identifier(obj.get("query_id"), "query_id")
    vector = _vector(obj["vector"]) if obj.get("vector") is not None else None
    text = obj.get("text")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("text must be a non-empty string")
    if vector is None and text is None:
        raise ValueError("a query needs a vector, a text, or both")
    return DatasetQuery(
        query_id=query_id,
        vector=vector,
        text=text,
        relevant=_relevant(obj.get("relevant")),
        filter=_filter(obj.get("filter")),
        tags=_tags(obj.get("tags")),
    )


def _identifier(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    if len(value) > MAX_ID_LENGTH:
        raise ValueError(f"{what} is longer than {MAX_ID_LENGTH} characters")
    return value


def _vector(value: Any) -> FloatArray:
    if not isinstance(value, list) or not value:
        raise ValueError("vector must be a non-empty list of numbers")
    if any(type(x) not in (int, float) for x in value):
        raise ValueError("vector must contain only numbers")
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all() or (np.abs(arr) > np.finfo(np.float32).max).any():
        raise ValueError("vector values must be finite float32 numbers")
    return arr.astype(np.float32)


def _relevant(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("relevant must map document ids to grades")
    out: dict[str, float] = {}
    for doc_id, grade in value.items():
        _identifier(doc_id, "a relevant document id")
        if type(grade) not in (int, float) or not math.isfinite(grade) or grade < 0:
            raise ValueError(f"grade for {doc_id!r} must be a finite number >= 0")
        out[doc_id] = float(grade)
    if out and not any(g > 0 for g in out.values()):
        raise ValueError("relevant needs at least one positive grade (or omit it)")
    return out


def _filter(value: Any) -> dict[str, FilterValue]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("filter must be an object of field -> value")
    out: dict[str, FilterValue] = {}
    for key, v in value.items():
        if not FIELD_NAME.fullmatch(key):
            raise ValueError(f"filter field {key!r} must be a dotted identifier")
        if type(v) not in (str, int, float, bool):
            raise ValueError(
                f"filter value for {key!r} must be a string, number or boolean "
                "(portable filters are equality only)"
            )
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"filter value for {key!r} must be finite")
        out[key] = v
    return out


def _tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(t, str) or not t for t in value):
        raise ValueError("tags must be a list of non-empty strings")
    return tuple(sorted(set(value)))


def _fingerprint(queries: list[DatasetQuery]) -> str:
    canonical = [
        {
            "query_id": q.query_id,
            "vector": None if q.vector is None else [float(x) for x in q.vector],
            "text": q.text,
            "relevant": q.relevant,
            "filter": q.filter,
            "tags": list(q.tags),
        }
        for q in sorted(queries, key=lambda q: q.query_id)
    ]
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()
