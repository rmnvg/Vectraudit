"""Build and inspect evaluation datasets: import from CSV/JSONL, embed, diff and stats."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit.evaluation.dataset import Dataset, DatasetError, DatasetQuery, parse_dataset

FILTER_TYPES = ("str", "int", "float", "bool")
SMALL_LABELLED = 30


@dataclass(frozen=True)
class ColumnMap:
    """Which source columns hold what. ``filters`` maps column -> type (see FILTER_TYPES).

    In JSON sources a column may be a dotted path into nested objects, such as
    ``inputs.question`` in a LangSmith export; a key that literally contains the dot wins.
    """

    query_id: str = "query_id"
    text: str | None = "text"
    vector: str | None = None
    relevant: str | None = "relevant"
    tags: str | None = None
    filters: Mapping[str, str] = field(default_factory=dict)


def _relevant_cell(cell: Any, row: int) -> dict[str, float]:
    """``"doc-1:2; doc-7"`` (grade defaults to 1), a JSON object, or a list of ids."""
    if cell is None or cell == "":
        return {}
    if isinstance(cell, dict):
        return {str(k): float(v) for k, v in cell.items()}
    if isinstance(cell, list):
        return {str(d): 1.0 for d in cell}
    out: dict[str, float] = {}
    for item in str(cell).split(";"):
        item = item.strip()
        if not item:
            continue
        doc, sep, grade = item.rpartition(":")
        if not sep:
            doc, grade = item, "1"
        try:
            out[doc.strip()] = float(grade)
        except ValueError as e:
            raise DatasetError(f"Row {row}: grade {grade!r} for {doc!r} is not a number.") from e
    return out


def _typed(value: Any, kind: str, column: str, row: int) -> Any:
    if isinstance(value, (int, float, bool)) and not isinstance(value, str):
        return value
    text = str(value).strip()
    try:
        if kind == "int":
            return int(text)
        if kind == "float":
            return float(text)
        if kind == "bool":
            lowered = text.lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
            raise ValueError(text)
    except ValueError as e:
        raise DatasetError(f"Row {row}: {column} value {text!r} is not a {kind}.") from e
    return text


_MISSING = object()


def _lookup(rec: Mapping[str, Any], column: str) -> Any:
    """``rec[column]``, following a dotted path into nested objects; ``_MISSING`` if absent."""
    if column in rec:
        return rec[column]
    value: Any = rec
    for part in column.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _cell(rec: Mapping[str, Any], column: str | None) -> Any:
    """A column's value, or ``None`` when the column is unset, absent or empty."""
    if not column:
        return None
    value = _lookup(rec, column)
    return None if value is _MISSING or value == "" else value


def import_rows(records: Iterable[Mapping[str, Any]], columns: ColumnMap) -> list[dict[str, Any]]:
    """Map source records to dataset rows (validated later by :func:`parse_dataset`)."""
    for kind in columns.filters.values():
        if kind not in FILTER_TYPES:
            raise DatasetError(f"Filter type {kind!r} must be one of {', '.join(FILTER_TYPES)}.")
    rows = []
    for n, rec in enumerate(records, 1):
        if not isinstance(rec, Mapping):
            raise DatasetError(f"Row {n}: expected an object, got {type(rec).__name__}.")
        query_id = _lookup(rec, columns.query_id)
        if query_id is _MISSING:
            raise DatasetError(f"Row {n}: missing column {columns.query_id!r}.")
        row: dict[str, Any] = {"query_id": str(query_id).strip()}
        text = _cell(rec, columns.text)
        if text is not None:
            row["text"] = str(text)
        vec = _cell(rec, columns.vector)
        if vec is not None:
            row["vector"] = json.loads(vec) if isinstance(vec, str) else vec
        relevant = _relevant_cell(_cell(rec, columns.relevant), n)
        if relevant:
            row["relevant"] = relevant
        tags = _cell(rec, columns.tags)
        if tags is not None:
            row["tags"] = (
                [str(t) for t in tags]
                if isinstance(tags, list)
                else [t.strip() for t in str(tags).split(";") if t.strip()]
            )
        filters = {
            col: _typed(value, kind, col, n)
            for col, kind in columns.filters.items()
            if (value := _cell(rec, col)) is not None
        }
        if filters:
            row["filter"] = filters
        for key in ("no_answer", "context", "forbidden", "split", "family_id"):
            value = rec.get(key)
            if value not in (None, ""):
                if isinstance(value, str) and key not in {"split", "family_id"}:
                    value = json.loads(value)
                row[key] = value
        rows.append(row)
    return rows


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Records from a ``.csv`` file (header row), a ``.json`` array or a JSONL file.

    A ``.json`` file may also be an object holding the array under ``examples``, ``items``
    or ``data``, as dataset exports from evaluation tools often are.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except FileNotFoundError as e:
        raise DatasetError(f"File not found: {p}") from e
    except UnicodeDecodeError as e:
        raise DatasetError(f"{p} is not UTF-8 text.") from e
    if p.suffix.lower() == ".csv":
        return [dict(r) for r in csv.DictReader(io.StringIO(text))]
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except ValueError as e:
            raise DatasetError(f"{p.name} is not valid JSON: {e}") from e
        if isinstance(data, dict):
            data = next(
                (data[k] for k in ("examples", "items", "data") if isinstance(data.get(k), list)),
                data,
            )
        if not isinstance(data, list):
            raise DatasetError(
                f"{p.name} must hold a JSON array of records (or one under "
                '"examples", "items" or "data").'
            )
        return data
    records = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except ValueError as e:
                raise DatasetError(f"Line {n} of {p.name} is not JSON: {e}") from e
    return records


def dataset_rows(ds: Dataset) -> list[dict[str, Any]]:
    """A dataset back as JSON-ready rows."""
    rows = []
    for q in ds.queries:
        row: dict[str, Any] = {"query_id": q.query_id}
        if q.family_id is not None:
            row["family_id"] = q.family_id
        if q.text is not None:
            row["text"] = q.text
        if q.vector is not None:
            row["vector"] = [float(x) for x in q.vector]
        if q.relevant:
            row["relevant"] = q.relevant
        if q.relevant_spans:
            row["relevant_spans"] = [
                {"doc_id": s.doc_id, "start": s.start, "end": s.end, "grade": s.grade}
                for s in q.relevant_spans
            ]
        if q.filter:
            row["filter"] = q.filter
        if q.tags:
            row["tags"] = list(q.tags)
        if q.no_answer:
            row["no_answer"] = True
        if q.context:
            row["context"] = q.context
        if q.forbidden:
            row["forbidden"] = list(q.forbidden)
        if q.split != "explore":
            row["split"] = q.split
        rows.append(row)
    return rows


def render_dataset(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any] | None = None) -> str:
    """JSONL text for a dataset, validated: raises DatasetError if it would not load."""
    lines = [json.dumps({"dataset": dict(meta)}, ensure_ascii=False)] if meta else []
    lines += [json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows]
    text = "\n".join(lines) + "\n"
    parse_dataset(text)
    return text


def embed_dataset(
    ds: Dataset,
    embed: Callable[[Sequence[str]], npt.ArrayLike],
    *,
    model: str,
    normalized: bool,
    overwrite: bool = False,
) -> str:
    """Add vectors for text queries; records the model in the header. Returns JSONL text."""
    previous = ds.meta.get("embedding_model")
    has_vectors = any(q.vector is not None for q in ds.queries)
    if has_vectors and previous not in (None, model) and not overwrite:
        raise DatasetError(
            f"The dataset already has vectors from {previous!r}; pass --overwrite to replace "
            f"them with {model!r}."
        )
    targets = [
        i
        for i, q in enumerate(ds.queries)
        if q.text is not None and (overwrite or q.vector is None)
    ]
    rows = dataset_rows(ds)
    if targets:
        vectors = np.asarray(embed([ds.queries[i].text or "" for i in targets]), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(targets):
            raise DatasetError("The embedder returned the wrong shape.")
        for i, v in zip(targets, vectors, strict=True):
            rows[i]["vector"] = [float(x) for x in v]
    meta = {**ds.meta, "embedding_model": model, "normalized": normalized}
    return render_dataset(rows, meta)


def _query_fields(q: DatasetQuery) -> dict[str, Any]:
    return {
        "family_id": q.family_id,
        "text": q.text,
        "vector": None if q.vector is None else [float(x) for x in q.vector],
        "relevant": q.relevant,
        "relevant_spans": [
            {"doc_id": s.doc_id, "start": s.start, "end": s.end, "grade": s.grade}
            for s in q.relevant_spans
        ],
        "filter": q.filter,
        "tags": list(q.tags),
    }


def diff_datasets(a: Dataset, b: Dataset) -> dict[str, Any]:
    """Queries added, removed and changed (with the fields that changed) from a to b."""
    qa = {q.query_id: q for q in a.queries}
    qb = {q.query_id: q for q in b.queries}
    changed = {}
    for qid in sorted(set(qa) & set(qb)):
        fa, fb = _query_fields(qa[qid]), _query_fields(qb[qid])
        fields = [k for k in fa if fa[k] != fb[k]]
        if fields:
            changed[qid] = fields
    meta_changes = sorted(k for k in set(a.meta) | set(b.meta) if a.meta.get(k) != b.meta.get(k))
    return {
        "from": a.sha256,
        "to": b.sha256,
        "added": sorted(set(qb) - set(qa)),
        "removed": sorted(set(qa) - set(qb)),
        "changed": changed,
        "meta_changed": meta_changes,
        "same": a.sha256 == b.sha256,
    }


def dataset_stats(ds: Dataset, *, cohort_minimum: int | None = None) -> dict[str, Any]:
    """Label coverage, grade distribution, cohorts and readiness warnings."""
    labelled = [q for q in ds.queries if q.relevant or q.relevant_spans]
    from vectraudit.experiments.labels import document_labels

    labels = [document_labels(q) for q in labelled]
    positives = [sum(g > 0 for g in row.values()) for row in labels]
    grades = Counter(g for row in labels for g in row.values())
    tags = Counter(t for q in ds.queries for t in q.tags)
    tag_families = {
        tag: len({q.family_id or q.query_id for q in ds.queries if tag in q.tags}) for tag in tags
    }
    n_positive_families = len({q.family_id or q.query_id for q in labelled})
    texts = Counter(q.text.strip().lower() for q in ds.queries if q.text)
    warnings = []
    if n_positive_families < SMALL_LABELLED:
        warnings.append(
            f"Only {n_positive_families} labelled query families: changes smaller than about "
            f"{1 / max(n_positive_families, 1):.2f} on a metric are hard to tell from noise. "
            f"Aim for {SMALL_LABELLED}+ (100+ for cohorts)."
        )
    if ds.n_labelled and ds.n_labelled < len(ds):
        warnings.append(
            f"{len(ds) - ds.n_labelled} queries have no labels: they count for ANN recall "
            "and latency only."
        )
    if cohort_minimum is not None:
        small = sorted(t for t, n in tag_families.items() if n < cohort_minimum)
        if small:
            warnings.append(
                f"Cohorts below the policy minimum of {cohort_minimum} families (reported, not "
                f"judged): {', '.join(small)}."
            )
    dupes = sorted(t for t, n in texts.items() if n > 1)
    if dupes:
        warnings.append(f"{len(dupes)} query texts appear more than once.")
    if ds.needs_embedder:
        warnings.append(
            "Some queries have text but no vector: run `vectraudit dataset embed` or pass "
            "--embedder to record."
        )
    return {
        "sha256": ds.sha256,
        "n_queries": len(ds),
        "n_families": len({q.family_id or q.query_id for q in ds.queries}),
        "n_positive_families": n_positive_families,
        "tag_families": tag_families,
        "n_labelled": ds.n_labelled,
        "n_positive": len(labelled),
        "n_no_answer": sum(q.no_answer for q in ds.queries),
        "n_filtered": sum(bool(q.filter) for q in ds.queries),
        "relevant_per_query": {
            "min": min(positives, default=0),
            "mean": float(np.mean(positives)) if positives else 0.0,
            "max": max(positives, default=0),
        },
        "grades": {str(g): n for g, n in sorted(grades.items())},
        "tags": dict(sorted(tags.items())),
        "meta": dict(ds.meta),
        "warnings": warnings,
    }
