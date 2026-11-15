"""Langfuse scores and LangSmith feedback read as answer judgments.

Teams that already judge answers in Langfuse or LangSmith can attach those judgments to a
run imported from the same traces, without converting them first. Each record becomes one
judgment in the format :mod:`vectraudit.evaluation.answers` reads, keyed by trace:

- **Langfuse scores** as the Scores API returns them (``name``, ``traceId``, ``source``,
  ``dataType``, ``value``, ``stringValue``), as a JSON array, one per line, or the API's
  ``{"data": [...]}`` page. Scores on a session or dataset run, not a trace, are skipped.
- **LangSmith feedback** (``key``, ``run_id``, optional ``trace_id``, ``score``, ``value``,
  ``feedback_source``), as a JSON array or one per line. Without ``trace_id`` the run is
  taken as the trace, which holds for feedback on a root run; feedback on a child run then
  matches no question and is counted as unmatched.

Nothing is inferred from names. An export holds every score a team records (retrieval
relevance, toxicity, end-user ratings), so the caller names the ones that judge answers.
Who judged comes from the record only where the vendor documents it (Langfuse
``ANNOTATION`` is a person, ``EVAL`` a model; LangSmith ``model`` is a model); for any other
source the caller declares it, and an undeclared score is skipped, never guessed. End-user
ratings are not reviews. A categorical value counts as a verdict only when it is
``correct``, ``partial`` or ``incorrect``, or a label the caller maps to one; numeric and
boolean values are scores and need a threshold, as in any judgments file. Comments,
corrections and author IDs are never read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

LANGFUSE_KINDS = {"ANNOTATION": "human", "EVAL": "model"}
LANGSMITH_KINDS = {"model": "model"}
SKIPPED = {
    "not_selected": "scores not named as answer judgments",
    "kind_undeclared": "scores whose source does not say whether a person or a model judged",
    "label_unmapped": "categorical values that are not a verdict and have no numeric value",
    "not_on_a_trace": "scores on a session or dataset run, not a trace",
    "text_score": "text scores (comments, not judgments)",
    "no_value": "records with neither a score nor a label",
}


def label_key(label: str) -> str:
    """Labels compare ignoring case and spacing."""
    return " ".join(label.split()).casefold()


@dataclass(frozen=True)
class Selection:
    """Which scores judge answers, who judged where the export does not say, and labels."""

    evaluators: Mapping[str, str | None] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)

    def verdict(self, label: Any) -> str | None:
        if not isinstance(label, str):
            return None
        key = label_key(label)
        mapped = {label_key(k): v for k, v in self.labels.items()}
        if key in mapped:
            return mapped[key]
        return key if key in ("correct", "partial", "incorrect") else None


def shape_of(obj: Any) -> str | None:
    """``langfuse``, ``langsmith`` or ``None`` (a judgments-file record or anything else)."""
    if not isinstance(obj, Mapping):
        return None
    if "name" in obj and "dataType" in obj and "source" in obj:
        return "langfuse"
    if "key" in obj and "run_id" in obj:
        return "langsmith"
    return None


def evaluator_of(obj: Mapping[str, Any], shape: str) -> str | None:
    """The score's name (Langfuse) or feedback key (LangSmith)."""
    name = obj.get("name") if shape == "langfuse" else obj.get("key")
    return name if isinstance(name, str) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def convert(obj: Mapping[str, Any], shape: str, selection: Selection) -> dict[str, Any] | str:
    """One vendor record as a judgment record, or the reason it is skipped (:data:`SKIPPED`)."""
    name = evaluator_of(obj, shape)
    if name is None or name not in selection.evaluators:
        return "not_selected"
    declared = selection.evaluators[name]
    if shape == "langfuse":
        trace = obj.get("traceId")
        kind = LANGFUSE_KINDS.get(str(obj.get("source"))) or declared
        data_type = obj.get("dataType")
        if data_type == "TEXT":
            return "text_score"
        verdict = selection.verdict(obj.get("stringValue")) if data_type == "CATEGORICAL" else None
        score = _number(obj.get("value"))
        if data_type == "CATEGORICAL" and verdict is None and score is None:
            return "label_unmapped"
    else:
        trace = obj.get("trace_id") or obj.get("run_id")
        source = obj.get("feedback_source")
        source_type = source.get("type") if isinstance(source, Mapping) else None
        kind = LANGSMITH_KINDS.get(str(source_type)) or declared
        verdict = selection.verdict(obj.get("value"))
        score = _number(obj.get("score"))
        if verdict is None and score is None and isinstance(obj.get("value"), str):
            return "label_unmapped"
    if not trace:
        return "not_on_a_trace"
    if kind is None:
        return "kind_undeclared"
    if verdict is None and score is None:
        return "no_value"
    record: dict[str, Any] = {"trace_id": str(trace), "evaluator": name, "kind": kind}
    if verdict is not None:
        record["verdict"] = verdict
    if score is not None:
        record["score"] = score
    return record
