"""Permission lifecycle: did access changes reach retrieval, the reranker and the model?

A question's ``forbidden`` list is a snapshot. Access changes over time: a user leaves a
group, a document moves to another space, an old version is archived. This reads an
**access change log** exported from the team's identity or content system, one change per
line::

    {"change": "revoke", "principal": {"user": "u-17"}, "doc_ids": ["payroll-2026"],
     "changed_at": "2026-10-18T09:00:00Z", "window_minutes": 15}
    {"change": "grant", "principal": {"groups": "finance"}, "doc_ids": ["budget"],
     "changed_at": "2026-10-18T09:00:00Z"}
    {"change": "move", "doc_ids": ["roadmap"], "from": {"tenant": "acme"},
     "to": {"tenant": "beta"}, "changed_at": "2026-10-18T09:00:00Z"}
    {"change": "archive", "doc_ids": ["manual-v1"], "changed_at": "2026-10-17T12:00:00Z"}

A principal matches a question when every field it names is in the question's ``context``
with that value (or, for list-valued context such as groups, contains it). ``archive``
applies to everyone. The latest change in effect for a principal and document decides;
a change still inside its propagation window is **pending**: the earlier access stands,
nothing fails yet, and the change is listed.

Expected access comes from this log, never from the application's own decisions: copying
what the application did into the expected result would test nothing.
:func:`apply_access_changes` writes it into a benchmark so every release check enforces
it (including where a forbidden document travelled); :func:`check_access` judges a run
recorded after the changes directly. Neither re-indexes, calls the application, or proves
authorization correct: a passing check is evidence about these principals and documents.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vectraudit.contracts import Context
from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.freshness import DEFAULT_WINDOW_MINUTES, _time
from vectraudit.evaluation.lineage import LineageError

ACCESS_TAG = "access-change"


class AccessChange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    change: Literal["revoke", "grant", "move", "archive"]
    doc_ids: list[str] = Field(min_length=1, max_length=1000)
    changed_at: str = Field(min_length=1, max_length=64)
    principal: Context | None = None
    source: Context | None = Field(default=None, alias="from")
    target: Context | None = Field(default=None, alias="to")
    window_minutes: int | None = Field(default=None, ge=0, le=525_600)

    @model_validator(mode="after")
    def complete(self) -> AccessChange:
        if any(not d or len(d) > 512 for d in self.doc_ids):
            raise ValueError("doc_ids are nonempty strings of at most 512 characters")
        if self.change in ("revoke", "grant") and not self.principal:
            raise ValueError(f"a {self.change} names the principal it applies to")
        if self.change == "move" and not (self.source and self.target):
            raise ValueError("a move names the principal it moves from and to")
        if self.change != "move" and (self.source or self.target):
            raise ValueError("only a move names from and to")
        if self.change in ("move", "archive") and self.principal:
            raise ValueError(f"a {self.change} names no principal")
        _time(self.changed_at)
        return self

    def effects(self) -> list[tuple[str, dict[str, Any]]]:
        """``(revoke|grant, principal)`` pairs: a move revokes one and grants the other."""
        if self.change == "move":
            return [("revoke", dict(self.source or {})), ("grant", dict(self.target or {}))]
        if self.change == "archive":
            return [("revoke", {})]
        return [(self.change, dict(self.principal or {}))]


def read_access_changes(path: str | Path) -> list[AccessChange]:
    """The access change log, oldest first."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LineageError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    changes = []
    for line, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            changes.append(AccessChange.model_validate(json.loads(raw)))
        except json.JSONDecodeError as exc:
            raise LineageError(f"line {line}: not JSON ({exc.msg})") from exc
        except ValidationError as exc:
            problem = exc.errors()[0]
            where = ".".join(str(p) for p in problem["loc"]) or "change"
            raise LineageError(f"line {line}: {where}: {problem['msg']}") from None
    if not changes:
        raise LineageError(f"{path} lists no changes")
    return sorted(changes, key=lambda c: _time(c.changed_at))


def matches(principal: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    """Whether a question asked with ``context`` is asked by ``principal``.

    Every principal field must be present: a scalar equals the context value or is one of a
    list-valued context's values; a list must be contained in it.
    """
    for key, want in principal.items():
        if key not in context:
            return False
        have = context[key]
        have_set = set(have) if isinstance(have, list) else {have}
        want_set = set(want) if isinstance(want, list) else {want}
        if not want_set <= have_set:
            return False
    return True


def _deadline(change: AccessChange, window_minutes: int) -> datetime:
    window = change.window_minutes if change.window_minutes is not None else window_minutes
    return _time(change.changed_at) + timedelta(minutes=window)


def expected_access(
    changes: Sequence[AccessChange],
    query: DatasetQuery,
    *,
    as_of: datetime,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> dict[str, Any]:
    """What the change log says ``query``'s principal may see at ``as_of``.

    ``revoked``: documents whose latest change in effect revokes access. ``granted``:
    documents whose latest change in effect grants it (they stop being forbidden).
    ``pending``: changes for this principal still inside their window.
    """
    state: dict[str, tuple[str, int]] = {}
    pending = []
    for index, change in enumerate(changes):
        deadline = _deadline(change, window_minutes)
        for effect, principal in change.effects():
            if not matches(principal, query.context):
                continue
            if as_of < deadline:
                pending.append(
                    {
                        "change": change.change,
                        "effect": effect,
                        "doc_ids": list(change.doc_ids),
                        "deadline": deadline.isoformat(timespec="seconds"),
                    }
                )
                continue
            for doc in change.doc_ids:
                state[doc] = (effect, index)
    return {
        "revoked": sorted(d for d, (e, _) in state.items() if e == "revoke"),
        "granted": sorted(d for d, (e, _) in state.items() if e == "grant"),
        "pending": pending,
    }


def apply_access_changes(
    dataset: Dataset,
    changes: Sequence[AccessChange],
    *,
    as_of: datetime,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> tuple[str, dict[str, Any]]:
    """The benchmark with each question's expected access at ``as_of``, and what changed.

    Revoked documents join ``forbidden``; granted ones leave it. A revoked document that
    was labelled relevant is removed from the labels (and from required evidence) and
    listed under ``labels_to_review``: a label is never silently carried past a revocation.
    Changed questions gain the ``access-change`` tag.
    """
    rows = dataset_rows(dataset)
    by_id = {q.query_id: q for q in dataset.queries}
    affected, review, pending = [], [], []
    for row in rows:
        query = by_id[row["query_id"]]
        access = expected_access(changes, query, as_of=as_of, window_minutes=window_minutes)
        if access["pending"]:
            pending.append({"query_id": query.query_id, "pending": access["pending"]})
        forbidden = set(query.forbidden)
        new = (forbidden | set(access["revoked"])) - set(access["granted"])
        if new == forbidden:
            continue
        relevant = dict(query.relevant)
        dropped = sorted(d for d in access["revoked"] if relevant.get(d, 0) > 0)
        dropped += sorted(
            {s.doc_id for s in query.relevant_spans if s.doc_id in access["revoked"]} - set(dropped)
        )
        for doc in dropped:
            relevant.pop(doc, None)
        row["relevant"] = relevant
        if not relevant:
            row.pop("relevant")
        spans = [s for s in row.pop("relevant_spans", []) if s["doc_id"] not in access["revoked"]]
        if spans:
            row["relevant_spans"] = spans
        if query.required_evidence:
            groups = [g for g in row["required_evidence"] if not set(g) & set(dropped)]
            row["required_evidence"] = groups
            if not groups:
                row.pop("required_evidence")
        row["forbidden"] = sorted(new)
        if not new:
            row.pop("forbidden")
        row["tags"] = sorted(set(row.get("tags", [])) | {ACCESS_TAG})
        affected.append(
            {
                "query_id": query.query_id,
                "now_forbidden": sorted(new - forbidden),
                "no_longer_forbidden": sorted(forbidden - new),
            }
        )
        if dropped:
            review.append({"query_id": query.query_id, "removed_labels": dropped})
    report = {
        "format": "vectraudit.access_changes",
        "format_version": 1,
        "as_of": as_of.isoformat(timespec="seconds"),
        "affected": affected,
        "labels_to_review": review,
        "pending": pending,
    }
    return render_dataset(rows, dataset.meta or None), report


def _is(item: Mapping[str, Any], doc: str) -> bool:
    """A recorded hit or context item is ``doc`` by its ID or its source ID."""
    source = item.get("source")
    return item.get("id") == doc or (isinstance(source, Mapping) and source.get("source_id") == doc)


def _reached(query: Mapping[str, Any], doc: str) -> list[str]:
    """Where a run delivered ``doc`` for one question: stages, results, model context."""
    where = []
    evidence = query.get("evidence")
    stages = evidence.get("stages") if isinstance(evidence, Mapping) else None
    for stage in stages or []:
        if any(_is(hit, doc) for hit in stage.get("hits") or []):
            where.append(f"stage {stage['name']} ({stage['kind']})")
    observed = [h for obs in query.get("observations") or [] for h in obs.get("hits") or []]
    if doc in (query.get("ids") or []) or any(_is(h, doc) for h in observed):
        where.append("results")
    context = query.get("model_context")
    items = context.get("items") if isinstance(context, Mapping) else None
    if any(_is(item, doc) for item in items or []):
        where.append("model context")
    return where


def check_access(
    changes: Sequence[AccessChange],
    run: Mapping[str, Any],
    dataset: Dataset,
    *,
    as_of: datetime,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> dict[str, Any]:
    """Revoked documents a run still served, judged at ``as_of`` (normally its time).

    ``violations`` are revoked documents that reached any recorded stage, the results or
    the model context after the window. ``pending`` are changes still inside it, with any
    old access the run still showed; they are not failures. ``granted_unseen`` lists
    granted documents labelled relevant that the run never delivered: possibly a grant not
    yet propagated, possibly ordinary ranking, so it is reported, not failed.
    """
    by_id = {q.query_id: q for q in dataset.queries}
    violations, pending, unseen = [], [], []
    checked = 0
    for record in run.get("queries") or []:
        query = by_id.get(record.get("query_id"))
        if query is None or record.get("error"):
            continue
        access = expected_access(changes, query, as_of=as_of, window_minutes=window_minutes)
        if not (access["revoked"] or access["granted"] or access["pending"]):
            continue
        checked += 1
        for doc in access["revoked"]:
            where = _reached(record, doc)
            if where:
                violations.append({"query_id": query.query_id, "doc_id": doc, "reached": where})
        for change in access["pending"]:
            still = {
                doc: where
                for doc in change["doc_ids"]
                if change["effect"] == "revoke" and (where := _reached(record, doc))
            }
            pending.append({"query_id": query.query_id, **change, "still_reached": still})
        for doc in access["granted"]:
            if query.relevant.get(doc, 0) > 0 and doc not in (record.get("ids") or []):
                unseen.append({"query_id": query.query_id, "doc_id": doc})
    return {
        "format": "vectraudit.access_check",
        "format_version": 1,
        "run_id": run.get("run_id"),
        "as_of": as_of.isoformat(timespec="seconds"),
        "questions_checked": checked,
        "violations": violations,
        "pending": pending,
        "granted_unseen": unseen,
        "failing": len(violations),
    }


def render(report: Mapping[str, Any]) -> str:
    """The access check as plain text for the terminal."""
    lines = [
        f"{report['questions_checked']} question(s) affected by access changes as of "
        f"{report['as_of']}: {len(report['violations'])} revoked document(s) still served"
    ]
    for v in report["violations"][:50]:
        lines.append(f"  {v['query_id']}: {v['doc_id']} reached " + ", ".join(v["reached"]))
    waiting = [p for p in report["pending"] if p["still_reached"]]
    if report["pending"]:
        lines.append(
            f"{len(report['pending'])} change(s) still inside their window"
            + (
                f"; {len(waiting)} still served the old access (not a failure yet)"
                if waiting
                else ""
            )
        )
    for g in report["granted_unseen"][:20]:
        lines.append(f"  {g['query_id']}: granted {g['doc_id']} was not delivered")
    if not report["questions_checked"]:
        lines.append("No question's context matches a principal in the change log.")
    elif not report["failing"]:
        lines.append("No revoked document was served.")
    return "\n".join(lines)
