"""Production-to-benchmark upkeep: a scheduled intake of production traces.

Run on each new OpenInference export (see :mod:`vectraudit.pipelines.traces`). Each run:

- **Continues from a cursor.** The state folder remembers the newest trace already taken
  (and the trace IDs at that instant), so overlapping exports are never counted twice.
- **Matches traces to the benchmark** by normalised question text. A matched question whose
  production retrieval delivered none of its labelled relevant documents is a **likely
  recurrence**; with a workspace, it names the question's regression cases and flags a
  verified case for reopening. Labels decide here, never the application's own output.
- **Queues new questions for review.** Unmatched questions are de-duplicated by text and
  context into a review queue that keeps counts, first and last seen times and the latest
  trace IDs, so a reviewer turns an incident into a test without re-entering it.
- **Prioritises without promoting.** Feedback (``negative``, ``escalated``, ``positive``)
  and frequency raise a question's priority; they never become relevance labels. A seeded
  share of new questions is kept as an **independent sample**, chosen before any signal,
  so review is not limited to what users complained about.
- **Routes to reviewers** by context (``{"when": {"tenant": "acme"}, "reviewer": "ana"}``),
  first matching rule wins.

Question text stays in the local state folder and the exported review file; nothing is
uploaded. A question that later enters the benchmark is marked ``promoted`` on the next run.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.dataset_tools import render_dataset
from vectraudit.labelling.task import atomic_write
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.traces import (
    Settings,
    Span,
    TraceImportError,
    _final,
    _metadata,
    _norm,
    read_spans,
    view_trace,
)

SIGNALS = ("negative", "escalated", "positive")
WEIGHTS = {"escalated": 5.0, "negative": 3.0, "positive": -1.0}
MAX_TRACE_IDS = 20
MAX_QUEUE = 20_000


def _time(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec="seconds")


def _key(text: str, context: Mapping[str, Any]) -> str:
    blob = json.dumps({"text": _norm(text), "context": dict(context)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _sampled(key: str, rate: float, seed: int) -> bool:
    """Chosen from the question's identity alone, before any signal is known."""
    value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < rate


def read_feedback(path: str | Path) -> dict[str, dict[str, int]]:
    """Feedback per trace: ``{"trace_id": ..., "signal": "negative"}`` per line."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    text = Path(path).read_text(encoding="utf-8-sig")
    for line, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TraceImportError(f"{path} line {line}: not JSON ({exc.msg})") from exc
        if not (
            isinstance(row, dict)
            and isinstance(row.get("trace_id"), str)
            and row.get("signal") in SIGNALS
        ):
            raise TraceImportError(
                f"{path} line {line}: needs trace_id and a signal of {', '.join(SIGNALS)}"
            )
        out[row["trace_id"]][row["signal"]] += 1
    return out


def read_routes(path: str | Path) -> list[dict[str, Any]]:
    """Reviewer routing rules: a JSON list of ``{"when": {...}, "reviewer": "..."}``."""
    routes = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(routes, list) or not all(
        isinstance(r, dict)
        and isinstance(r.get("reviewer"), str)
        and r["reviewer"]
        and isinstance(r.get("when", {}), dict)
        for r in routes
    ):
        raise TraceImportError('routes are a list of {"when": {...}, "reviewer": "name"}')
    return routes


def _route(context: Mapping[str, Any], routes: Sequence[Mapping[str, Any]]) -> str | None:
    for rule in routes:
        if all(context.get(k) == v for k, v in (rule.get("when") or {}).items()):
            return str(rule["reviewer"])
    return None


def _cases(workspace: Path | None) -> dict[str, list[dict[str, Any]]]:
    """Regression cases per question, read-only from a workbench folder."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    folder = workspace / ".workbench" / "cases" if workspace else None
    if folder is None or not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get("query_id"), str):
            out[record["query_id"]].append(
                {"case": record.get("id"), "status": record.get("status")}
            )
    return out


def _context(spans: Sequence[Span], fields: Sequence[str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for span in sorted(spans, key=lambda s: s.start_ns):
        meta = _metadata(span)
        for name in fields:
            value = meta.get(name, span.attributes.get(name))
            if name not in found and isinstance(value, (str, int, float, bool)):
                found[name] = value
    return found


def _load(state: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cursor_path, queue_path = state / "cursor.json", state / "queue.jsonl"
    cursor = (
        json.loads(cursor_path.read_text(encoding="utf-8"))
        if cursor_path.is_file()
        else {"after_ns": 0, "at_cursor": [], "runs": 0}
    )
    queue: dict[str, dict[str, Any]] = {}
    if queue_path.is_file():
        for raw in queue_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                row = json.loads(raw)
                queue[row["key"]] = row
    return cursor, queue


def _priority(entry: Mapping[str, Any]) -> float:
    signals: Mapping[str, int] = entry["signals"]
    score: float = math.log2(1 + int(entry["count"])) + sum(
        WEIGHTS[s] * n for s, n in signals.items() if s in WEIGHTS
    )
    return round(score, 3)


def intake(
    traces: str | Path,
    dataset: Dataset,
    state: str | Path,
    *,
    settings: Settings | None = None,
    context_fields: Sequence[str] = (),
    feedback: Mapping[str, Mapping[str, int]] | None = None,
    routes: Sequence[Mapping[str, Any]] = (),
    sample_rate: float = 0.1,
    seed: int = 0,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Take the traces newer than the cursor; update the review queue and report."""
    if not 0 <= sample_rate <= 1:
        raise TraceImportError("sample_rate must be between 0 and 1")
    settings = settings or Settings()
    folder = Path(state)
    folder.mkdir(parents=True, exist_ok=True)
    cursor, queue = _load(folder)
    spans, _ = read_spans(traces)
    grouped: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        grouped[span.trace_id].append(span)
    after, boundary = int(cursor["after_ns"]), set(cursor["at_cursor"])
    fresh = []
    skipped = 0
    for trace_id, group in grouped.items():
        start = min(s.start_ns for s in group)
        if start < after or (start == after and trace_id in boundary):
            skipped += 1
            continue
        fresh.append((start, trace_id, group))
    fresh.sort()
    texts: dict[str, list[Any]] = defaultdict(list)
    for q in dataset.queries:
        if q.text:
            texts[_norm(q.text)].append(q)
    cases = _cases(Path(workspace) if workspace else None)
    feedback = feedback or {}
    recurrences, unreadable, unmatched_text, ambiguous = [], 0, 0, 0
    matched = 0
    touched: set[str] = set()
    for start, trace_id, group in fresh:
        view = view_trace(group, settings)
        text = next(iter(view.query_texts), None)
        if text is None:
            unmatched_text += 1
            continue
        context = _context(group, context_fields)
        candidates = texts.get(_norm(text)) or []
        if len(candidates) > 1:
            ambiguous += 1  # several benchmark questions share this text: not new, not judged
            continue
        if len(candidates) == 1:
            matched += 1
            question = candidates[0]
            relevant = {d for d, g in question.relevant.items() if g > 0}
            try:
                delivered = list(_final(view, settings).documents or [])
            except PipelineCallError:
                unreadable += 1
                continue
            if relevant and not relevant & set(delivered):
                linked = cases.get(question.query_id, [])
                recurrences.append(
                    {
                        "query_id": question.query_id,
                        "trace_id": trace_id,
                        "at": _time(start),
                        "delivered": delivered,
                        "cases": linked,
                        "reopen": [c["case"] for c in linked if c["status"] == "verified"],
                    }
                )
            continue
        key = _key(text, context)
        entry = queue.get(key)
        if entry is None:
            if len(queue) >= MAX_QUEUE:
                raise TraceImportError(f"the review queue holds {MAX_QUEUE} questions; triage it")
            entry = {
                "key": key,
                "text": text,
                "context": context,
                "first_seen": _time(start),
                "count": 0,
                "trace_ids": [],
                "signals": {},
                "sampled": _sampled(key, sample_rate, seed),
                "reviewer": _route(context, routes),
                "status": "new",
            }
            queue[key] = entry
        entry["count"] += 1
        entry["last_seen"] = _time(start)
        entry["trace_ids"] = [*entry["trace_ids"], trace_id][-MAX_TRACE_IDS:]
        for signal, n in (feedback.get(trace_id) or {}).items():
            entry["signals"][signal] = entry["signals"].get(signal, 0) + n
        entry["priority"] = _priority(entry)
        touched.add(key)
    benchmark_texts = set(texts)
    promoted = 0
    for entry in queue.values():
        if entry["status"] == "new" and _norm(entry["text"]) in benchmark_texts:
            entry["status"] = "promoted"
            promoted += 1
    if fresh:
        newest = fresh[-1][0]
        same = [tid for start, tid, _ in fresh if start == newest]
        cursor = {
            "after_ns": newest,
            "at_cursor": sorted(set(same) | (boundary if newest == after else set())),
            "runs": int(cursor.get("runs", 0)) + 1,
        }
    else:
        cursor = {**cursor, "runs": int(cursor.get("runs", 0)) + 1}
    atomic_write(folder / "cursor.json", json.dumps(cursor, indent=2) + "\n")
    rows = sorted(queue.values(), key=lambda e: (-e.get("priority", 0), e["first_seen"]))
    atomic_write(
        folder / "queue.jsonl",
        "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in rows),
    )
    pending = [e for e in rows if e["status"] == "new"]
    return {
        "format": "vectraudit.intake",
        "format_version": 1,
        "traces_taken": len(fresh),
        "traces_already_taken": skipped,
        "matched_benchmark": matched,
        "without_question_text": unmatched_text,
        "unreadable": unreadable,
        "ambiguous": ambiguous,
        "likely_recurrences": recurrences,
        "queue": {
            "pending": len(pending),
            "updated": len(touched),
            "promoted_now": promoted,
            "independent_sample": sum(1 for e in pending if e["sampled"]),
            "unassigned": sum(1 for e in pending if e["reviewer"] is None),
        },
        "cursor": {"after": _time(cursor["after_ns"]) if cursor["after_ns"] else None},
    }


def review_dataset(state: str | Path, *, reviewer: str | None = None, limit: int = 1000) -> str:
    """Pending questions as an unlabelled benchmark for review, highest priority first.

    The independent sample comes first within its share, so a capped export never drops it
    in favour of feedback-driven questions. Rows carry no labels: reviewers add them.
    """
    _, queue = _load(Path(state))
    pending = [
        e
        for e in queue.values()
        if e["status"] == "new" and (reviewer is None or e["reviewer"] == reviewer)
    ]
    sampled = sorted((e for e in pending if e["sampled"]), key=lambda e: e["first_seen"])
    signalled = sorted(
        (e for e in pending if not e["sampled"]),
        key=lambda e: (-e.get("priority", 0), e["first_seen"]),
    )
    share = math.ceil(limit * len(sampled) / len(pending)) if pending else 0
    chosen = [*sampled[:share], *signalled][:limit]
    rows = []
    for e in chosen:
        tags = ["incident-review", "independent-sample" if e["sampled"] else "signalled"]
        rows.append(
            {
                "query_id": f"incident-{e['key'][:20]}",
                "text": e["text"],
                **({"context": e["context"]} if e["context"] else {}),
                "tags": tags,
            }
        )
    meta = {"provenance": {"source": "production intake", "incident": True}}
    return render_dataset(rows, meta) if rows else ""


def render(report: Mapping[str, Any]) -> str:
    """The intake as plain text for the terminal."""
    q = report["queue"]
    lines = [
        f"Took {report['traces_taken']} new trace(s) "
        f"({report['traces_already_taken']} already taken); "
        f"{report['matched_benchmark']} matched benchmark questions.",
        f"Review queue: {q['pending']} pending ({q['independent_sample']} independent sample, "
        f"{q['unassigned']} without a reviewer); {q['updated']} updated now; "
        f"{q['promoted_now']} newly in the benchmark.",
    ]
    for r in report["likely_recurrences"][:20]:
        cases = ", ".join(c["case"] for c in r["cases"]) or "no case yet"
        reopen = f"; reopen verified case(s) {', '.join(r['reopen'])}" if r["reopen"] else ""
        lines.append(
            f"  likely recurrence: {r['query_id']} at {r['at']} delivered none of its "
            f"labelled documents ({cases}){reopen}"
        )
    if report["unreadable"] or report["without_question_text"] or report["ambiguous"]:
        lines.append(
            f"Not judged: {report['unreadable']} matched trace(s) without a readable final "
            f"ranking, {report['without_question_text']} without question text, and "
            f"{report['ambiguous']} whose text matches several benchmark questions."
        )
    return "\n".join(lines)
