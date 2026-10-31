"""Write Vectraudit's diagnosis back onto Langfuse traces as scores.

After ``import-traces`` (from a Langfuse pull or export) and ``attach-answers``, each judged
question has a diagnosis: the first observable loss between retrieval and answer
(``lost_in_context``, ``not_observed``, ``answered_correctly``, ...). This module turns
those into one categorical score per trace, so engineers see the diagnosis on the trace in
the tool they already use, and can filter and chart by it there.

What is sent, per trace: the trace ID, the score name (default ``vectraudit.diagnosis``),
the diagnosis class as a categorical value, and metadata naming Vectraudit, the format
version and a digest of the run. Nothing else: no question or document text, no document
IDs, no facts, no comment.

Each score has a deterministic ID derived from the trace and score name, so pushing a
newer diagnosis for the same trace is meant to replace the earlier score rather than add a
second one (Langfuse keys scores by ID; confirm on your deployment).

Only runs imported from Langfuse are accepted: their trace IDs are Langfuse's.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.pipelines.langfuse_api import LangfuseAPIError, LangfuseSource

SCORE_NAME = "vectraudit.diagnosis"
PUSH_FORMAT_VERSION = 1
BATCH = 100
_ENVIRONMENT = re.compile(r"^(?!langfuse)[a-z0-9_-]{1,40}$")
_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")


class LangfusePushError(ValueError):
    """The run cannot be pushed; the message says why."""


def score_id(trace_id: str, name: str) -> str:
    """A stable ID for this trace's score of this name, so a re-push replaces it."""
    return hashlib.sha256(f"vectraudit|{name}|{trace_id}".encode()).hexdigest()[:32]


def diagnosis_scores(
    run: Mapping[str, Any], *, name: str = SCORE_NAME, environment: str | None = None
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """One categorical score per judged trace, and why other questions were not included."""
    config = (run.get("backend") or {}).get("config") or {}
    if config.get("adapter") != "trace_import" or config.get("shape") != "langfuse":
        raise LangfusePushError(
            "Only runs imported from Langfuse can be pushed: their trace IDs are Langfuse's. "
            "Import a Langfuse pull or export with import-traces first."
        )
    if not isinstance(run.get("answers"), Mapping):
        raise LangfusePushError(
            "The run has no diagnosis: attach answer judgments with attach-answers first."
        )
    if not _NAME.match(name):
        raise LangfusePushError(
            "The score name may use letters, digits and _ . : - (at most 100 characters)."
        )
    if environment is not None and not _ENVIRONMENT.match(environment):
        raise LangfusePushError(
            "The environment must be lowercase letters, digits, - or _, not starting with "
            "'langfuse'."
        )
    digest = hashlib.sha256(json.dumps(run, sort_keys=True, default=str).encode()).hexdigest()[:16]
    scores: dict[str, dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    for query in run.get("queries") or []:
        trace = query.get("trace_id")
        answer = query.get("answer")
        diagnosis = answer.get("diagnosis") if isinstance(answer, Mapping) else None
        if not isinstance(trace, str) or not trace:
            skipped["no_trace"] += 1
            continue
        if not isinstance(diagnosis, Mapping) or not isinstance(diagnosis.get("class"), str):
            skipped["no_diagnosis"] += 1
            continue
        if trace in scores:
            skipped["duplicate_trace"] += 1
            continue
        score: dict[str, Any] = {
            "id": score_id(trace, name),
            "traceId": trace,
            "name": name,
            "value": diagnosis["class"],
            "dataType": "CATEGORICAL",
            "metadata": {
                "source": "vectraudit",
                "format_version": PUSH_FORMAT_VERSION,
                "run_digest": digest,
            },
        }
        if environment is not None:
            score["environment"] = environment
        scores[trace] = score
    return list(scores.values()), skipped


def push_scores(source: LangfuseSource, scores: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Send scores in batches; ``{"sent", "accepted", "rejected", "requests"}``.

    Langfuse answers 202 when a batch is accepted and 207 with counts when part of it is
    rejected. A 207 batch is not retried, as Langfuse asks. Rejection messages are not
    echoed, because they can repeat what was sent.
    """
    totals = {"sent": 0, "accepted": 0, "rejected": 0, "requests": 0}
    for i in range(0, len(scores), BATCH):
        batch = list(scores[i : i + BATCH])
        status, body = source.post("/api/public/scores", batch)
        totals["sent"] += len(batch)
        totals["requests"] += 1
        if status == 207:
            if not isinstance(body, Mapping) or not all(
                isinstance(body.get(k), int) for k in ("accepted", "rejected")
            ):
                raise LangfuseAPIError("Unexpected Langfuse response to a partial score batch.")
            totals["accepted"] += int(body["accepted"])
            totals["rejected"] += int(body["rejected"])
        else:  # 202 (or 200, which the generated reference lists)
            totals["accepted"] += len(batch)
    return totals


def preview(scores: Sequence[Mapping[str, Any]], skipped: Mapping[str, int]) -> str:
    """What a push would send, for the terminal."""
    counts = Counter(str(s["value"]) for s in scores)
    lines = [
        f"{len(scores)} traces would get a score"
        + (f" named {scores[0]['name']!r}" if scores else "")
        + ".",
    ]
    if counts:
        lines.append("Values: " + ", ".join(f"{n} {v}" for v, n in counts.most_common()))
    if skipped:
        lines.append(
            "Not included: "
            + ", ".join(f"{n} {why.replace('_', ' ')}" for why, n in sorted(skipped.items()))
        )
    if scores:
        lines += ["", "Example score (exactly as sent):", json.dumps(scores[0], indent=2)]
    return "\n".join(lines)
