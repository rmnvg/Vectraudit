"""Runs as a table: one row per question and repeat, for analysis in a notebook.

``vectraudit runs table`` writes CSV, or Parquet through the ``[parquet]`` extra. Runs are
read with :func:`vectraudit.evaluation.runfile.read_run_header`, one question at a time, so a run
too large to load still exports in bounded memory (Parquet is written in batches).

The columns follow the design of Inspect's log dataframes: every row carries the run it came
from (``run_file``, ``run_id``, ``run_label``, like Inspect's ``log`` and ``eval_id``); a
question is identified by ``query_id`` and ``repeat`` (Inspect's sample ``id`` and ``epoch``);
metrics expand with a prefix (``metric_ndcg``, like ``score_*``); and lists are JSON strings.
Every value is an ID, a number or a flag. No question, document or answer text is written.

Per repeat: the metrics, ANN recall, latency (when the run recorded each execution), how many
results came back, forbidden results and the no-answer check. Per question, repeated on each of
its rows: family, tags, error, retries, the median latency, sufficiency, the answer verdict and
diagnosis, and cost. ``component`` (where the question first lost its evidence, see
:mod:`vectraudit.evaluation.components`) describes the first execution, so it is set on repeat
1 only.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vectraudit.evaluation.components import attribute
from vectraudit.evaluation.metrics import RANK_METRICS, SEMANTIC_METRICS
from vectraudit.evaluation.runfile import read_run_header

METRICS = (*SEMANTIC_METRICS, *RANK_METRICS)
COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_file", "string"),
    ("run_id", "string"),
    ("run_label", "string"),
    ("run_status", "string"),
    ("run_created_at", "string"),
    ("dataset_sha256", "string"),
    ("backend", "string"),
    ("k", "int"),
    ("timing_source", "string"),
    ("query_id", "string"),
    ("family_id", "string"),
    ("tags", "string"),
    ("repeat", "int"),
    ("repeats", "int"),
    ("error_code", "string"),
    ("retries", "int"),
    *((f"metric_{name}", "float") for name in METRICS),
    ("ann_recall", "float"),
    ("latency_ms", "float"),
    ("query_latency_ms", "float"),
    ("returned", "int"),
    ("nondeterministic", "bool"),
    ("no_answer_correct", "bool"),
    ("forbidden_returned", "int"),
    ("permission_checked", "bool"),
    ("abstained", "bool"),
    ("sufficiency_delivered", "string"),
    ("sufficiency_context", "string"),
    ("answer_verdict", "string"),
    ("answer_decided_by", "string"),
    ("answer_outcome", "string"),
    ("answer_diagnosis", "string"),
    ("component", "string"),
    ("cost_usd", "float"),
)
NAMES = tuple(name for name, _ in COLUMNS)
BATCH = 20_000


class RunTableError(ValueError):
    """The table cannot be written as asked."""


def _at(values: Any, i: int) -> Any:
    return values[i] if isinstance(values, list) and i < len(values) else None


def question_rows(
    header: Mapping[str, Any], q: Mapping[str, Any], run_file: str
) -> Iterator[dict[str, Any]]:
    """The rows of one question: one per repeat (one for a question that failed before any)."""
    rankings = q.get("repeat_ids")
    samples = q.get("repeat_metrics")
    if isinstance(samples, list):
        repeats = len(samples)
    elif isinstance(rankings, list) and rankings:
        repeats = len(rankings)
    else:
        repeats = 1
    answer: Mapping[str, Any] = q["answer"] if isinstance(q.get("answer"), Mapping) else {}
    sufficiency: Mapping[str, Any] = (
        q["sufficiency"] if isinstance(q.get("sufficiency"), Mapping) else {}
    )
    forbidden = set(q.get("forbidden_ids") or [])
    outcome = attribute(q)[0]
    common = {
        "run_file": run_file,
        "run_id": header.get("run_id"),
        "run_label": header.get("label"),
        "run_status": header.get("status"),
        "run_created_at": header.get("created_at"),
        "dataset_sha256": (header.get("dataset") or {}).get("sha256"),
        "backend": (header.get("backend") or {}).get("backend"),
        "k": (header.get("search") or {}).get("k"),
        "timing_source": (header.get("timing") or {}).get("source"),
        "query_id": q.get("query_id"),
        "family_id": q.get("family_id"),
        "tags": json.dumps(list(q.get("tags") or [])),
        "repeats": repeats,
        "error_code": (q.get("error") or {}).get("code"),
        "retries": len(q.get("retries") or []),
        "query_latency_ms": q.get("latency_ms"),
        "nondeterministic": q.get("nondeterministic"),
        "permission_checked": q.get("permission_checked"),
        "abstained": q.get("abstained"),
        "answer_verdict": answer.get("verdict"),
        "answer_decided_by": answer.get("decided_by"),
        "answer_outcome": answer.get("outcome"),
        "answer_diagnosis": (answer.get("diagnosis") or {}).get("class"),
        "cost_usd": q.get("cost_usd"),
    }
    for i in range(repeats):
        metrics = _at(samples, i) if isinstance(samples, list) else q.get("metrics")
        ranking = _at(rankings, i) if isinstance(rankings, list) else q.get("ids")
        safety = _at(q.get("repeat_safety"), i)
        repeat_sufficiency = _at(q.get("repeat_sufficiency"), i) or (sufficiency if i == 0 else {})
        no_answer = q.get("no_answer_correct")
        if isinstance(safety, Mapping):
            no_answer = safety.get("no_answer_correct")
        elif no_answer is not None and isinstance(ranking, list):
            no_answer = not ranking
        recall = q.get("repeat_ann_recall")
        yield {
            **common,
            "repeat": i + 1,
            **{
                f"metric_{name}": (metrics or {}).get(name)
                if isinstance(metrics, Mapping)
                else None
                for name in METRICS
            },
            "ann_recall": _at(recall, i) if isinstance(recall, list) else q.get("ann_recall"),
            "latency_ms": _at(q.get("latency_samples_ms"), i),
            "returned": len(ranking) if isinstance(ranking, list) else None,
            "no_answer_correct": no_answer,
            "forbidden_returned": (
                len(safety.get("forbidden_ids") or [])
                if isinstance(safety, Mapping)
                else len(forbidden & set(ranking))
                if isinstance(ranking, list)
                else None
            ),
            "sufficiency_delivered": (repeat_sufficiency or {}).get("delivered"),
            "sufficiency_context": (repeat_sufficiency or {}).get("context"),
            "component": outcome if i == 0 else None,
        }


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_table(paths: Sequence[str | Path], out: str | Path) -> dict[str, int]:
    """Write every run's rows to ``out`` (``.csv`` or ``.parquet``); counts of runs and rows."""
    target = Path(out)
    kind = target.suffix.lower()
    if kind not in (".csv", ".parquet"):
        raise RunTableError("Write to a .csv or .parquet file.")
    if not paths:
        raise RunTableError("Name at least one run file.")
    if kind == ".parquet":
        return _parquet(paths, target)
    rows = 0
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=NAMES)
        writer.writeheader()
        for path in paths:
            name = Path(path).name

            def emit(header: dict[str, Any], q: dict[str, Any], name: str = name) -> None:
                nonlocal rows
                for row in question_rows(header, q, name):
                    writer.writerow({k: _csv_value(v) for k, v in row.items()})
                    rows += 1

            read_run_header(path, on_query=emit)
    return {"runs": len(paths), "rows": rows}


def _parquet(paths: Sequence[str | Path], target: Path) -> dict[str, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RunTableError(
            'Parquet needs pyarrow: pip install "vectraudit[parquet]", or write a .csv file.'
        ) from e
    types = {"string": pa.string(), "int": pa.int64(), "float": pa.float64(), "bool": pa.bool_()}
    schema = pa.schema([(name, types[kind]) for name, kind in COLUMNS])
    batch: list[dict[str, Any]] = []
    rows = 0
    with pq.ParquetWriter(target, schema) as writer:

        def flush() -> None:
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()

        for path in paths:
            name = Path(path).name

            def emit(header: dict[str, Any], q: dict[str, Any], name: str = name) -> None:
                nonlocal rows
                for row in question_rows(header, q, name):
                    batch.append(row)
                    rows += 1
                if len(batch) >= BATCH:
                    flush()

            read_run_header(path, on_query=emit)
        flush()
    return {"runs": len(paths), "rows": rows}
