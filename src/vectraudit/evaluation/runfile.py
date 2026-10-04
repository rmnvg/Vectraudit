"""Save, load and validate run records (see :mod:`vectraudit.evaluation.runner`).

Run files travel between CI jobs and may be edited by hand, so loading validates the
structure the comparison relies on and rejects anything else with a clear message.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from vectraudit.evaluation.metrics import SEMANTIC_METRICS
from vectraudit.evaluation.runner import RUN_FORMAT, RUN_FORMAT_VERSION

RUN_STATUSES = frozenset({"complete", "partial", "failed", "cancelled"})


class RunFileError(ValueError):
    """The file is not a valid vectraudit run."""


def save_run(record: dict[str, Any], path: str | Path) -> Path:
    """Write a run record as JSON (strict: no NaN or infinity)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    return p


def load_run(path: str | Path) -> dict[str, Any]:
    """Read and validate a run file."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except FileNotFoundError as e:
        raise RunFileError(f"Run file not found: {p}") from e
    except (ValueError, UnicodeDecodeError) as e:
        raise RunFileError(f"{p} is not valid JSON: {e}") from e
    try:
        return check_run(data)
    except RunFileError as e:
        raise RunFileError(f"{p}: {e}") from e


def _reject_constant(token: str) -> float:
    raise ValueError(f"{token} is not allowed")


def check_run(data: Any) -> dict[str, Any]:
    """Validate the parts of a run record that comparisons read."""
    _expect(isinstance(data, dict), "a run must be a JSON object")
    _expect(data.get("format") == RUN_FORMAT, "not a vectraudit run file")
    version = data.get("format_version")
    _expect(
        version == RUN_FORMAT_VERSION,
        f"run format version {version!r} is not supported (expected {RUN_FORMAT_VERSION})",
    )
    _expect(data.get("status") in RUN_STATUSES, f"status must be one of {sorted(RUN_STATUSES)}")
    _expect(isinstance(data.get("run_id"), str), "run_id must be a string")
    dataset = _dict(data, "dataset")
    _expect(isinstance(dataset.get("sha256"), str), "dataset.sha256 must be a string")
    _expect(_count(dataset.get("n_queries")), "dataset.n_queries must be a count")
    search = _dict(data, "search")
    _expect(_count(search.get("k")) and search["k"] >= 1, "search.k must be >= 1")
    _expect(isinstance(search.get("params"), dict), "search.params must be an object")
    contract = _dict(data, "metric_contract")
    _expect(_count(contract.get("version")), "metric_contract.version must be an integer")
    for key in ("backend", "capabilities"):
        _expect(data.get(key) is None or isinstance(data[key], dict), f"{key} must be an object")
    _expect(data["status"] == "failed" or isinstance(data.get("backend"), dict), "missing backend")
    queries = data.get("queries")
    _expect(isinstance(queries, list), "queries must be a list")
    seen: set[str] = set()
    for n, q in enumerate(queries):
        try:
            _check_query(q, search["k"])
        except RunFileError as e:
            raise RunFileError(f"queries[{n}]: {e}") from e
        _expect(q["query_id"] not in seen, f"duplicate query_id {q['query_id']!r}")
        seen.add(q["query_id"])
    _expect(len(queries) <= dataset["n_queries"], "more queries than the dataset has")
    record: dict[str, Any] = data
    return record


def _check_query(q: Any, k: int) -> None:
    _expect(isinstance(q, dict), "must be an object")
    _expect(isinstance(q.get("query_id"), str), "query_id must be a string")
    _expect(isinstance(q.get("tags", []), list), "tags must be a list")
    error = q.get("error")
    _expect(error is None or isinstance(error, dict), "error must be an object or null")
    if error is not None:
        return
    ids = q.get("ids")
    _expect(isinstance(ids, list) and all(isinstance(i, str) for i in ids), "ids must be strings")
    _expect(len(ids) <= k, "more ids than k")
    exact = q.get("exact_ids")
    _expect(
        exact is None or (isinstance(exact, list) and all(isinstance(i, str) for i in exact)),
        "exact_ids must be strings or null",
    )
    latency = q.get("latency_ms")
    _expect(_number(latency) and latency >= 0, "latency_ms must be a non-negative number")
    metrics = q.get("metrics")
    if metrics is not None:
        _expect(isinstance(metrics, dict), "metrics must be an object")
        for name in SEMANTIC_METRICS:
            _expect(_fraction(metrics.get(name)), f"metrics.{name} must be in [0, 1]")
    evidence = q.get("evidence")
    _expect(evidence is None or isinstance(evidence, dict), "evidence must be an object or null")
    recall = q.get("ann_recall")
    _expect(recall is None or _fraction(recall), "ann_recall must be in [0, 1] or null")


def _dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    _expect(isinstance(value, dict), f"{key} must be an object")
    assert isinstance(value, dict)
    return value


def _count(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _fraction(x: Any) -> bool:
    return _number(x) and 0 <= x <= 1


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RunFileError(message)
