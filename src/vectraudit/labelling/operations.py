"""Shared human decisions, revision checks and annotation readiness."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from vectraudit.labelling.agreement import resolve
from vectraudit.labelling.task import Judgment, Task


def judgment_for(task: Task, qid: str, doc: str, annotator: str) -> Judgment | None:
    """The current decision for an annotator and source-document pair."""
    return next(
        (j for j in task.judgments if (j.query_id, j.doc_id, j.annotator) == (qid, doc, annotator)),
        None,
    )


def judge(task: Task, judgment: Judgment) -> None:
    """Replace one decision; unchanged retries preserve the reviewer's resolution."""
    query = next((q for q in task.queries if q.query_id == judgment.query_id), None)
    if query is None or judgment.doc_id not in query.documents:
        raise ValueError("Judgment is outside this pool")
    assigned = task.assignments.get(judgment.annotator)
    if assigned is not None and judgment.query_id not in assigned:
        raise ValueError("Question is outside this annotator assignment")
    if judgment.end is not None and judgment.end > len(task.documents[judgment.doc_id]):
        raise ValueError("Span exceeds source document")
    previous = judgment_for(task, judgment.query_id, judgment.doc_id, judgment.annotator)
    if previous == judgment:
        return
    task.resolutions = [
        j
        for j in task.resolutions
        if (j.query_id, j.doc_id) != (judgment.query_id, judgment.doc_id)
    ]
    task.judgments = [
        j
        for j in task.judgments
        if (j.query_id, j.doc_id, j.annotator)
        != (judgment.query_id, judgment.doc_id, judgment.annotator)
    ] + [judgment]


def resolve_judgment(task: Task, judgment: Judgment) -> None:
    """Set an explicit reviewer decision without modifying the original votes."""
    query = next((q for q in task.queries if q.query_id == judgment.query_id), None)
    if query is None or judgment.doc_id not in query.documents or judgment.grade is None:
        raise ValueError("A reviewer must choose a grade for a pooled document")
    if judgment.end is not None and judgment.end > len(task.documents[judgment.doc_id]):
        raise ValueError("Span exceeds source document")
    task.resolutions = [
        j
        for j in task.resolutions
        if (j.query_id, j.doc_id) != (judgment.query_id, judgment.doc_id)
    ] + [judgment]


def revision(task: Task, qid: str, doc: str, *, annotator: str | None = None) -> str:
    """Fingerprint just the decision being edited, including its displayed source text."""
    query = next(q for q in task.queries if q.query_id == qid)
    if doc not in query.documents:
        raise ValueError("Document outside pool")
    if annotator is None:
        rows = sorted(
            (j.model_dump() for j in task.judgments if j.query_id == qid and j.doc_id == doc),
            key=lambda j: j["annotator"],
        )
        reviewed = [
            j.model_dump() for j in task.resolutions if j.query_id == qid and j.doc_id == doc
        ]
    else:
        current = judgment_for(task, qid, doc, annotator)
        rows = [current.model_dump()] if current is not None else []
        reviewed = []
    blob = json.dumps([query.text, task.documents[doc], rows, reviewed], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def task_status(task: Task) -> dict[str, Any]:
    """Aggregate coverage, readiness and conflicts without exposing corpus or question text."""
    pooled = sum(len(q.documents) for q in task.queries)
    decided = 0
    unresolved = 0
    positive_questions = 0
    for q in task.queries:
        positives = 0
        for doc in q.documents:
            final = resolve(task, q.query_id, doc)
            if final is not None and final.grade is not None:
                decided += 1
                positives += int(final.grade > 0)
            elif any(
                j.grade is not None and j.query_id == q.query_id and j.doc_id == doc
                for j in task.judgments
            ):
                unresolved += 1
        positive_questions += int(positives > 0)
    return {
        "queries": len(task.queries),
        "pooled_pairs": pooled,
        "resolved_pairs": decided,
        "unresolved_pairs": unresolved,
        "unjudged_pairs": pooled - decided - unresolved,
        "queries_with_positive_labels": positive_questions,
        "export_ready": positive_questions > 0 and unresolved == 0,
        "fully_judged": decided == pooled,
        "task_sha256": task.sha256,
    }
