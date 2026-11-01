"""Write Vectraudit's diagnosis back onto Phoenix traces as annotations.

The Phoenix counterpart of :mod:`vectraudit.pipelines.langfuse_push`. After
``import-traces`` (from a ``phoenix-pull``, a Phoenix span export, or OpenInference OTLP
traces that were also sent to Phoenix) and ``attach-answers``, each judged question has a
diagnosis. This writes one trace annotation per trace, so engineers see and filter by it in
Phoenix.

What is sent, per trace: the trace ID, the annotation name (default
``vectraudit.diagnosis``), ``annotator_kind`` ``CODE``, the diagnosis class as the label,
the identifier ``vectraudit``, and metadata naming Vectraudit, the format version and a
digest of the run. No explanation, question or document text, document IDs or facts.

Behaviour checked against a running Phoenix 21 server:

- Phoenix keys an annotation by trace, name and identifier, so a re-push **replaces** the
  earlier label instead of adding a second annotation.
- Requests use ``sync=true``. Without it, Phoenix answers 200 and silently drops
  annotations for traces it does not have. With it, a batch naming any unknown trace is
  refused whole (404) and nothing in it is written; that batch is then resent one trace at
  a time, so the traces Phoenix has are still annotated and the unknown ones are counted.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.pipelines.langfuse_push import judged_diagnoses
from vectraudit.pipelines.phoenix_api import PhoenixSource

ANNOTATION_NAME = "vectraudit.diagnosis"
IDENTIFIER = "vectraudit"
PUSH_FORMAT_VERSION = 1
BATCH = 100
SHAPES = ("spans", "otlp-json")
"""Import shapes whose trace IDs can be Phoenix's (OpenTelemetry trace IDs)."""
_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")
_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")


class PhoenixPushError(ValueError):
    """The run cannot be pushed; the message says why."""


def diagnosis_annotations(
    run: Mapping[str, Any], *, name: str = ANNOTATION_NAME
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """One trace annotation per judged trace, and why other questions were not included."""
    config = (run.get("backend") or {}).get("config") or {}
    if config.get("adapter") != "trace_import" or config.get("shape") not in SHAPES:
        raise PhoenixPushError(
            "Only runs imported from Phoenix or OpenInference traces can be pushed: their "
            "trace IDs must be Phoenix's. Import a phoenix-pull with import-traces first."
        )
    if not isinstance(run.get("answers"), Mapping):
        raise PhoenixPushError(
            "The run has no diagnosis: attach answer judgments with attach-answers first."
        )
    if not _NAME.match(name):
        raise PhoenixPushError(
            "The annotation name may use letters, digits and _ . : - (at most 100 characters)."
        )
    found, skipped, digest = judged_diagnoses(run)
    annotations: list[dict[str, Any]] = []
    for trace, diagnosis_class in found.items():
        if not _TRACE_ID.match(trace):
            skipped["not_an_otel_trace_id"] += 1
            continue
        annotations.append(
            {
                "trace_id": trace,
                "name": name,
                "annotator_kind": "CODE",
                "identifier": IDENTIFIER,
                "result": {"label": diagnosis_class},
                "metadata": {
                    "source": "vectraudit",
                    "format_version": PUSH_FORMAT_VERSION,
                    "run_digest": digest,
                },
            }
        )
    return annotations, skipped


def push_annotations(
    source: PhoenixSource, annotations: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Send annotations; ``{"sent", "written", "unknown_trace", "requests"}``."""
    totals = {"sent": len(annotations), "written": 0, "unknown_trace": 0, "requests": 0}
    for i in range(0, len(annotations), BATCH):
        batch = list(annotations[i : i + BATCH])
        before = source.calls
        if source.post("/v1/trace_annotations", {"sync": "true"}, {"data": batch}) == 200:
            totals["written"] += len(batch)
        else:  # some trace is unknown and nothing was written: find which, one at a time
            for one in batch:
                status = source.post("/v1/trace_annotations", {"sync": "true"}, {"data": [one]})
                totals["written" if status == 200 else "unknown_trace"] += 1
        totals["requests"] += source.calls - before
    return totals


def preview(annotations: Sequence[Mapping[str, Any]], skipped: Mapping[str, int]) -> str:
    """What a push would send, for the terminal."""
    counts = Counter(str(a["result"]["label"]) for a in annotations)
    lines = [
        f"{len(annotations)} traces would get an annotation"
        + (f" named {annotations[0]['name']!r}" if annotations else "")
        + ".",
    ]
    if counts:
        lines.append("Labels: " + ", ".join(f"{n} {v}" for v, n in counts.most_common()))
    if skipped:
        lines.append(
            "Not included: "
            + ", ".join(f"{n} {why.replace('_', ' ')}" for why, n in sorted(skipped.items()))
        )
    if annotations:
        lines += ["", "Example annotation (exactly as sent):", json.dumps(annotations[0], indent=2)]
    return "\n".join(lines)
