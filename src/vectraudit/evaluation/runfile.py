"""Save, load and validate run records (see :mod:`vectraudit.evaluation.runner`).

Run files travel between CI jobs and may be edited by hand, so loading validates the
structure the comparison relies on and rejects anything else with a clear message.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from vectraudit.evaluation.metrics import SEMANTIC_METRICS, ann_recall
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
    timing = data.get("timing")
    _expect(timing is None or isinstance(timing, dict), "timing must be an object or null")
    repeats: Any = (timing or {}).get("repeats")
    if contract["version"] == 2:
        _expect(
            contract.get("repeat_aggregation") == "mean_per_query", "unknown repeat aggregation"
        )
        _expect(_count(repeats) and repeats >= 1, "repeat-quality contract requires timing.repeats")
    for key in ("backend", "capabilities"):
        _expect(data.get(key) is None or isinstance(data[key], dict), f"{key} must be an object")
    _expect(data["status"] == "failed" or isinstance(data.get("backend"), dict), "missing backend")
    queries = data.get("queries")
    _expect(isinstance(queries, list), "queries must be a list")
    seen: set[str] = set()
    for n, q in enumerate(queries):
        try:
            _check_query(q, search["k"])
            if contract["version"] == 2 and q.get("error") is None:
                _check_repeat_quality(q, repeats)
        except RunFileError as e:
            raise RunFileError(f"queries[{n}]: {e}") from e
        _expect(q["query_id"] not in seen, f"duplicate query_id {q['query_id']!r}")
        seen.add(q["query_id"])
    _expect(len(queries) <= dataset["n_queries"], "more queries than the dataset has")
    if data["status"] == "complete":
        # "Complete" is what lets a run be judged at all, so it must mean what it says.
        _expect(
            not any(q.get("error") for q in queries),
            "a complete run cannot contain failed queries; mark it partial",
        )
        _expect(
            len(queries) == dataset["n_queries"],
            f"a complete run must measure every dataset query ({len(queries)} of "
            f"{dataset['n_queries']})",
        )
    record: dict[str, Any] = data
    return record


def _check_repeat_quality(q: dict[str, Any], repeats: int) -> None:
    rankings: Any = q.get("repeat_ids")
    _expect(
        isinstance(rankings, list) and len(rankings) == repeats,
        "repeat-quality ranking count must match timing.repeats",
    )
    samples: Any = q.get("repeat_metrics")
    _expect(
        isinstance(samples, list) and len(samples) == repeats,
        "repeat_metrics must contain one entry per repeat",
    )
    for sample in samples:
        _expect(
            sample is None
            or (
                isinstance(sample, dict)
                and all(_fraction(sample.get(name)) for name in SEMANTIC_METRICS)
            ),
            "repeat metrics must be finite fractions or null",
        )
    if q.get("metrics") is None:
        _expect(
            all(sample is None for sample in samples), "missing mean for measured repeat metrics"
        )
    else:
        _expect(
            all(sample is not None for sample in samples), "repeat quality coverage is incomplete"
        )
        for name in SEMANTIC_METRICS:
            _expect(
                math.isclose(
                    q["metrics"][name],
                    statistics.fmean(s[name] for s in samples),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"metrics.{name} must equal the mean of repeat_metrics",
            )
    recalls: Any = q.get("repeat_ann_recall")
    _expect(
        isinstance(recalls, list) and len(recalls) == repeats,
        "repeat_ann_recall must contain one entry per repeat",
    )
    expected = [
        ann_recall(ids, q["exact_ids"]) if q.get("exact_ids") is not None else None
        for ids in rankings
    ]
    for actual, expected_value in zip(recalls, expected, strict=True):
        _expect(
            actual is None
            if expected_value is None
            else _fraction(actual) and math.isclose(actual, expected_value, abs_tol=1e-12),
            "repeat ANN recall must match its ranking and exact reference",
        )
    mean = (
        statistics.fmean(v for v in expected if v is not None) if expected[0] is not None else None
    )
    _expect(
        q.get("ann_recall") is None
        if mean is None
        else _fraction(q.get("ann_recall")) and math.isclose(q["ann_recall"], mean, abs_tol=1e-12),
        "ann_recall must equal the mean across repeats",
    )
    if "repeat_evidence" in q:
        changed = q["repeat_evidence"]
        _expect(
            isinstance(changed, dict)
            and all(
                isinstance(key, str)
                and key.isdigit()
                and str(int(key)) == key
                and 2 <= int(key) <= repeats
                and isinstance(value, dict)
                for key, value in changed.items()
            ),
            "repeat_evidence must map later repeat numbers to evidence objects",
        )


def _check_query(q: Any, k: int) -> None:
    _expect(isinstance(q, dict), "must be an object")
    _expect(isinstance(q.get("query_id"), str), "query_id must be a string")
    if "family_id" in q:
        family = q["family_id"]
        _expect(
            isinstance(family, str) and bool(family.strip()) and len(family) <= 512,
            "family_id must be a non-empty string of at most 512 characters",
        )
    _expect(isinstance(q.get("tags", []), list), "tags must be a list")
    error = q.get("error")
    _expect(error is None or isinstance(error, dict), "error must be an object or null")
    if "observations" in q:
        from pydantic import ValidationError

        from vectraudit.pipelines.models import EvidenceObservation

        observations = q["observations"]
        _expect(isinstance(observations, list), "observations must be a list")
        try:
            for item in observations:
                observed = EvidenceObservation.model_validate(item)
                _expect(
                    error is not None
                    or observed.coverage is None
                    or observed.coverage.ranked_output == "observed",
                    "incomplete ranked observations require a query error",
                )
        except ValidationError as exc:
            raise RunFileError("invalid versioned evidence observation") from exc
    if "model_context" in q:
        from pydantic import ValidationError

        from vectraudit.pipelines.models import ModelContext

        try:
            ModelContext.model_validate(q["model_context"])
        except ValidationError as exc:
            raise RunFileError("invalid model context record") from exc
    if "answer" in q:
        answer = q["answer"]
        _expect(
            isinstance(answer, dict)
            and answer.get("verdict") in (None, "correct", "partial", "incorrect", "disputed")
            and answer.get("outcome") in ("answered", "abstained", "handoff")
            and isinstance(answer.get("judgments"), list),
            "answer must hold a verdict, an outcome and its judgments",
        )
    if "sufficiency" in q:
        sufficiency = q["sufficiency"]
        _expect(
            isinstance(sufficiency, dict)
            and sufficiency.get("delivered") in ("sufficient", "insufficient")
            and sufficiency.get("context")
            in ("sufficient", "insufficient", "unknown", "unrecorded"),
            "sufficiency must state delivered and context evidence",
        )
    if "repeat_sufficiency" in q:
        rows = q["repeat_sufficiency"]
        _expect(
            isinstance(rows, list)
            and all(
                isinstance(r, dict)
                and r.get("delivered") in ("sufficient", "insufficient")
                and r.get("context") in ("sufficient", "insufficient", "unknown", "unrecorded")
                for r in rows
            ),
            "repeat_sufficiency must state delivered and context evidence per execution",
        )
    if "permission_path" in q:
        _permission_path(q)
    repeat_ids = q.get("repeat_ids")
    _expect(repeat_ids is None or _rankings(repeat_ids, k), "repeat_ids must be rankings")
    attempts = q.get("attempts", [])
    _expect(isinstance(attempts, list), "attempts must be a list")
    for attempt in attempts:
        _expect(isinstance(attempt, dict), "attempts must be objects")
        result = attempt.get("result")
        _expect(result is None or isinstance(result, dict), "attempt result must be an object")
        _expect(
            result is None
            or result.get("repeat_ids") is None
            or _rankings(result["repeat_ids"], k),
            "attempt repeat_ids must be rankings",
        )
    if error is not None:
        return
    ids = q.get("ids")
    _expect(isinstance(ids, list) and all(isinstance(i, str) for i in ids), "ids must be strings")
    _expect(len(ids) <= k, "more ids than k")
    _expect(
        repeat_ids is None or (bool(repeat_ids) and repeat_ids[0] == ids),
        "repeat_ids must start with the recorded ranking",
    )
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
    no_answer = q.get("no_answer_correct")
    _expect(
        no_answer is None or type(no_answer) is bool, "no_answer_correct must be boolean or null"
    )
    forbidden = q.get("forbidden_ids", [])
    _expect(
        isinstance(forbidden, list) and all(isinstance(d, str) for d in forbidden),
        "forbidden_ids must be document ids",
    )
    if "permission_checked" in q:
        _expect(type(q["permission_checked"]) is bool, "permission_checked must be boolean")
    _expect(
        not forbidden or q.get("permission_checked") is True,
        "forbidden results require a permission check",
    )
    _expect(no_answer is not True or not ids, "successful no-answer check cannot deliver documents")
    recall = q.get("ann_recall")
    _expect(recall is None or _fraction(recall), "ann_recall must be in [0, 1] or null")


def _dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    _expect(isinstance(value, dict), f"{key} must be an object")
    assert isinstance(value, dict)
    return value


def _rankings(x: Any, k: int) -> bool:
    """A list of per-execution rankings, each at most ``k`` document ids."""
    return isinstance(x, list) and all(
        isinstance(r, list) and len(r) <= k and all(isinstance(i, str) for i in r) for r in x
    )


def _count(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _fraction(x: Any) -> bool:
    return _number(x) and 0 <= x <= 1


def _permission_path(q: dict[str, Any]) -> None:
    """Where forbidden documents travelled: compare_runs reads it, so it is checked here."""
    path = q["permission_path"]
    message = "permission_path must list each forbidden document's stages and destinations"
    _expect(q.get("permission_checked") is True, "permission_path requires a permission check")
    _expect(
        isinstance(path, dict)
        and type(path.get("stages_recorded")) is bool
        and type(path.get("context_recorded")) is bool
        and isinstance(path.get("documents"), list),
        message,
    )
    for doc in path["documents"]:
        _expect(
            isinstance(doc, dict)
            and isinstance(doc.get("doc_id"), str)
            and isinstance(doc.get("kinds"), list)
            and all(isinstance(kind, str) for kind in doc["kinds"])
            and type(doc.get("results")) is bool
            and (doc.get("context") is None or type(doc.get("context")) is bool),
            message,
        )


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RunFileError(message)
