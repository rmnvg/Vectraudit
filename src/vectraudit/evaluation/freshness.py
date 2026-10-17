"""Knowledge freshness: did a content change reach the index, results and model context?

After a source is updated, replaced, deleted or archived, the new version should become
retrievable within the team's propagation window and the old one should disappear. This
reads a **change log** the team's content system already has, one change per line::

    {"source_id": "refund-policy", "change": "update", "to_revision": "r9",
     "changed_at": "2026-10-18T09:00:00Z", "window_minutes": 60}
    {"source_id": "old-faq", "change": "replace", "replaced_by": "faq-2026",
     "changed_at": "2026-10-17T12:00:00Z"}
    {"source_id": "promo-2025", "change": "delete", "changed_at": "2026-10-16T08:00:00Z"}

and checks it against a lineage export (see :mod:`vectraudit.evaluation.lineage`) taken
afterwards, and optionally a run recorded afterwards. Questions tagged ``historical`` may
legitimately retrieve old content and are not flagged; the team defines that precedence.

Statuses per change: ``propagated`` (new content indexed, old content gone), ``pending``
(still inside its window: not a failure), ``overdue`` (past the window, new content not
indexed), ``lingering`` (the old revision or the removed source is still indexed). A run adds
stale content **retrieved** or **in the model context**, and **conflicting versions** of one
source delivered together. Nothing here re-indexes or reads document text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.lineage import LineageError, SourceRecord

HISTORICAL_TAG = "historical"
DEFAULT_WINDOW_MINUTES = 60


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1, max_length=512)
    change: Literal["update", "replace", "delete", "archive"]
    changed_at: str = Field(min_length=1, max_length=64)
    to_revision: str | None = Field(default=None, min_length=1, max_length=256)
    replaced_by: str | None = Field(default=None, min_length=1, max_length=512)
    window_minutes: int | None = Field(default=None, ge=0, le=525_600)

    @model_validator(mode="after")
    def complete(self) -> Change:
        if self.change == "update" and self.to_revision is None:
            raise ValueError("an update names its to_revision")
        if self.change == "replace" and self.replaced_by is None:
            raise ValueError("a replacement names the source that replaces it (replaced_by)")
        if self.change in ("delete", "archive") and (self.to_revision or self.replaced_by):
            raise ValueError("a deletion or archive names no new revision or replacement")
        _time(self.changed_at)
        return self


def _time(value: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"not an ISO 8601 time: {value!r}") from exc
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def read_changes(path: str | Path) -> list[Change]:
    """The change log, oldest first; one change per source per line."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LineageError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    changes = []
    for line, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            changes.append(Change.model_validate(json.loads(raw)))
        except json.JSONDecodeError as exc:
            raise LineageError(f"line {line}: not JSON ({exc.msg})") from exc
        except ValidationError as exc:
            problem = exc.errors()[0]
            where = ".".join(str(p) for p in problem["loc"]) or "change"
            raise LineageError(f"line {line}: {where}: {problem['msg']}") from None
    if not changes:
        raise LineageError(f"{path} lists no changes")
    return sorted(changes, key=lambda c: _time(c.changed_at))


def _stale_chunks(change: Change, source: SourceRecord | None) -> list[str]:
    """Indexed chunks that should be gone after this change."""
    if source is None:
        return []
    indexed = [c for c in source.chunks if c.indexed]
    if change.change == "update":
        return [c.chunk_id for c in indexed if c.indexed_revision not in (None, change.to_revision)]
    return [c.chunk_id for c in indexed]  # replaced, deleted or archived: all of it


def _new_indexed(change: Change, sources: Mapping[str, SourceRecord]) -> bool | None:
    """Whether the new content is indexed; ``None`` when the change has no new content."""
    if change.change == "update":
        source = sources.get(change.source_id)
        return bool(
            source
            and any(c.indexed and c.indexed_revision == change.to_revision for c in source.chunks)
        )
    if change.change == "replace":
        new = sources.get(change.replaced_by or "")
        return bool(new and any(c.indexed for c in new.chunks))
    return None


def check_freshness(
    changes: Sequence[Change],
    sources: Mapping[str, SourceRecord],
    *,
    as_of: datetime,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    run: Mapping[str, Any] | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    """Each change's propagation status at ``as_of``, and stale content a run delivered.

    Only the latest change per source is judged; earlier ones are superseded.
    """
    latest: dict[str, Change] = {}
    for change in changes:
        latest[change.source_id] = change
    rows: list[dict[str, Any]] = []
    stale_ids: dict[str, dict[str, str]] = {}
    for change in latest.values():
        window = change.window_minutes if change.window_minutes is not None else window_minutes
        deadline = _time(change.changed_at) + timedelta(minutes=window)
        within = as_of < deadline
        stale = _stale_chunks(change, sources.get(change.source_id))
        indexed = _new_indexed(change, sources)
        for chunk in stale:
            stale_ids[chunk] = {"source_id": change.source_id, "change": change.change}
        if not within and stale:
            status = "lingering"
        elif not within and indexed is False:
            status = "overdue"
        elif stale or indexed is False:
            status = "pending"
        else:
            status = "propagated"
        rows.append(
            {
                "source_id": change.source_id,
                "change": change.change,
                "changed_at": change.changed_at,
                "deadline": deadline.isoformat(timespec="seconds"),
                "status": status,
                "new_content_indexed": indexed,
                "stale_chunks": stale,
            }
        )
    served = _served(run, dataset, stale_ids, sources) if run is not None else None
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "format": "vectraudit.knowledge_freshness",
        "format_version": 1,
        "as_of": as_of.isoformat(timespec="seconds"),
        "changes": rows,
        "counts": counts,
        "served": served,
        "failing": counts.get("lingering", 0)
        + counts.get("overdue", 0)
        + (len(served["stale"]) + len(served["conflicts"]) if served else 0),
    }


def _served(
    run: Mapping[str, Any],
    dataset: Dataset | None,
    stale_ids: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, SourceRecord],
) -> dict[str, Any]:
    """Stale content a run delivered or put in context, and conflicting versions."""
    historical = (
        {q.query_id for q in dataset.queries if HISTORICAL_TAG in q.cohort_tags}
        if dataset
        else set()
    )
    revision_of = {
        c.chunk_id: (s.source_id, c.indexed_revision) for s in sources.values() for c in s.chunks
    }
    stale, conflicts = [], []
    for q in run.get("queries") or []:
        if q.get("error") or q["query_id"] in historical:
            continue
        delivered = list(q.get("ids") or [])
        context = q.get("model_context") if isinstance(q.get("model_context"), Mapping) else None
        in_context = {item["id"] for item in (context or {}).get("items") or []}
        for chunk in delivered:
            if chunk in stale_ids:
                stale.append(
                    {
                        "query_id": q["query_id"],
                        "chunk_id": chunk,
                        **stale_ids[chunk],
                        "where": "context" if chunk in in_context else "results",
                    }
                )
        versions: dict[str, set[str | None]] = {}
        for chunk in delivered:
            if chunk in revision_of:
                source, revision = revision_of[chunk]
                versions.setdefault(source, set()).add(revision)
        for source, seen in versions.items():
            if len(seen - {None}) > 1:
                conflicts.append(
                    {
                        "query_id": q["query_id"],
                        "source_id": source,
                        "revisions": sorted(r for r in seen if r is not None),
                    }
                )
    return {
        "run_id": run.get("run_id"),
        "historical_questions_exempt": sorted(historical),
        "stale": stale,
        "conflicts": conflicts,
    }


def render(report: Mapping[str, Any]) -> str:
    """The freshness report as plain text for the terminal."""
    counts = report["counts"]
    lines = [
        f"{len(report['changes'])} change(s) as of {report['as_of']}: "
        + ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    ]
    for row in report["changes"]:
        if row["status"] in ("lingering", "overdue"):
            detail = (
                f"{len(row['stale_chunks'])} stale chunk(s) still indexed"
                if row["status"] == "lingering"
                else "new content not indexed"
            )
            lines.append(
                f"  {row['source_id']} ({row['change']}): {row['status']}, {detail}; "
                f"due {row['deadline']}"
            )
    served = report["served"]
    if served:
        for hit in served["stale"][:20]:
            lines.append(
                f"  {hit['query_id']}: stale {hit['chunk_id']} ({hit['change']} of "
                f"{hit['source_id']}) reached the {hit['where']}"
            )
        for conflict in served["conflicts"][:20]:
            lines.append(
                f"  {conflict['query_id']}: {conflict['source_id']} delivered in revisions "
                + ", ".join(conflict["revisions"])
            )
        if served["historical_questions_exempt"]:
            lines.append(
                f"{len(served['historical_questions_exempt'])} question(s) tagged historical "
                "were not checked for stale content."
            )
    if not report["failing"]:
        lines.append("No stale or overdue content found.")
    return "\n".join(lines)
