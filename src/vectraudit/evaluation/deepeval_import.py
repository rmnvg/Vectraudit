"""DeepEval test results as answer judgments for ``attach-answers``.

Teams that already grade answers with DeepEval keep doing so; this reads the results it
saves (``.deepeval/.latest_test_run.json``, ``{"testRunData": {...}}``, or the test run
itself) and writes the judgment lines :mod:`vectraudit.evaluation.answers` reads. The
format follows DeepEval's own models (``LLMApiTestCase`` and ``MetricData``):
``testCases[]`` with ``name``, ``input``, ``metadata`` and ``metricsData[]``, each with
``name``, ``threshold``, ``success``, ``score``, ``evaluationModel`` and ``error``.

- **Which metrics are answer judgments is the user's choice** (``metrics``). DeepEval
  metrics measure different things, and a faithful answer can still be wrong, so none is
  assumed to mean "correct". The contextual metrics (Contextual Precision, Recall and
  Relevancy) judge the retrieved context, not the answer, and are refused: Vectraudit
  measures retrieval against the benchmark's labels itself.
- **Verdicts are DeepEval's own decisions:** ``success`` (the score against the metric's
  threshold) becomes ``correct`` or ``incorrect``, and the score is kept beside it. A
  metric that errored or was not evaluated is skipped and counted. The judge model is
  recorded as the evaluator version. Judgments are ``model`` judgments.
- **Matching a test case to a question:** ``metadata.query_id``, else
  ``metadata.trace_id``, else the question text when it matches exactly one benchmark
  question (case and spacing ignored). Ambiguous and unmatched cases are counted.
- **Nothing private is copied:** reasons, inputs, outputs and contexts are never written.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RETRIEVAL_METRICS = ("Contextual Precision", "Contextual Recall", "Contextual Relevancy")
MAX_BYTES = 200 * 2**20
_WORD = re.compile(r"\w+")


class DeepEvalImportError(ValueError):
    """The DeepEval file or settings cannot be used; the message says why."""


def load_test_run(path: str | Path) -> dict[str, Any]:
    """The test run in a DeepEval results file (wrapped under ``testRunData`` or not)."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_BYTES:
            raise DeepEvalImportError(f"{path} is larger than {MAX_BYTES // 2**20} MiB.")
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DeepEvalImportError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise DeepEvalImportError(f"{path} is not JSON.") from exc
    run = data.get("testRunData", data) if isinstance(data, dict) else None
    if not isinstance(run, dict) or not isinstance(run.get("testCases"), list):
        raise DeepEvalImportError(
            f"{path} is not a DeepEval test run: expected testCases (from "
            ".deepeval/.latest_test_run.json)."
        )
    return run


def metric_names(run: Mapping[str, Any]) -> Counter[str]:
    """How many test cases each metric was recorded on."""
    found: Counter[str] = Counter()
    for case in run["testCases"]:
        for metric in (case.get("metricsData") or []) if isinstance(case, dict) else []:
            if isinstance(metric, dict) and isinstance(metric.get("name"), str):
                found[metric["name"]] += 1
    return found


def _norm(text: str) -> str:
    return " ".join(m.group().lower() for m in _WORD.finditer(text))


def to_judgments(
    run: Mapping[str, Any],
    metrics: Sequence[str],
    questions: Mapping[str, str | None],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Judgment lines for the chosen metrics, and counts of what was matched or skipped.

    ``questions`` maps benchmark question IDs to their text.
    """
    if not metrics:
        raise DeepEvalImportError("Choose the metrics that judge answers with --metric.")
    refused = sorted(set(metrics) & set(RETRIEVAL_METRICS))
    if refused:
        raise DeepEvalImportError(
            f"{', '.join(refused)} judge the retrieved context, not the answer; Vectraudit "
            "measures retrieval against the benchmark's labels itself."
        )
    by_text: dict[str, list[str]] = defaultdict(list)
    for qid, text in questions.items():
        if text:
            by_text[_norm(text)].append(qid)
    wanted = set(metrics)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for case in run["testCases"]:
        if not isinstance(case, dict):
            counts["invalid_test_case"] += 1
            continue
        key = _match(case, questions, by_text, counts)
        if key is None:
            continue
        for metric in case.get("metricsData") or []:
            if not isinstance(metric, dict) or metric.get("name") not in wanted:
                continue
            if metric.get("error"):
                counts["metric_error"] += 1
                continue
            success = metric.get("success")
            if not isinstance(success, bool):
                counts["not_evaluated"] += 1
                continue
            row: dict[str, Any] = {
                key[0]: key[1],
                "evaluator": metric["name"],
                "kind": "model",
                "verdict": "correct" if success else "incorrect",
            }
            score = metric.get("score")
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(score)
            ):
                row["score"] = float(score)
            judge = metric.get("evaluationModel")
            if isinstance(judge, str) and judge.strip() and len(judge) <= 200:
                row["evaluator_version"] = judge
            rows.append(row)
            counts["judgments"] += 1
    return rows, counts


def _match(
    case: Mapping[str, Any],
    questions: Mapping[str, str | None],
    by_text: Mapping[str, list[str]],
    counts: Counter[str],
) -> tuple[str, str] | None:
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    assert isinstance(metadata, dict)
    query_id = metadata.get("query_id")
    if isinstance(query_id, str) and query_id in questions:
        counts["matched_by_query_id"] += 1
        return ("query_id", query_id)
    trace_id = metadata.get("trace_id")
    if isinstance(trace_id, str) and trace_id:
        counts["matched_by_trace_id"] += 1
        return ("trace_id", trace_id)
    text = case.get("input")
    found = by_text.get(_norm(text)) if isinstance(text, str) else None
    if found and len(found) == 1:
        counts["matched_by_text"] += 1
        return ("query_id", found[0])
    counts["ambiguous_text" if found else "unmatched"] += 1
    return None


def render(counts: Mapping[str, int], written: int, out: str) -> str:
    """What the import did, for the terminal."""
    labels = {
        "matched_by_query_id": "matched by metadata.query_id",
        "matched_by_trace_id": "matched by metadata.trace_id",
        "matched_by_text": "matched by question text",
        "ambiguous_text": "with text matching several questions",
        "unmatched": "matching no question",
        "invalid_test_case": "malformed",
        "metric_error": "metric results that errored",
        "not_evaluated": "metric results without a decision",
    }
    lines = [f"Wrote {written} judgments to {out}."]
    for key, label in labels.items():
        if not counts.get(key):
            continue
        skipped = key in ("metric_error", "not_evaluated")
        lines.append(f"  {counts[key]} {label} (skipped)" if skipped else
                     f"  {counts[key]} test cases {label}")  # fmt: skip
    return "\n".join(lines)
