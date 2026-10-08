"""Export accepted human decisions using the unchanged metric contract."""

from __future__ import annotations

from typing import Any

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import render_dataset
from vectraudit.labelling.agreement import agreement, decisions, resolve
from vectraudit.labelling.suggestions import suggestion_agreement
from vectraudit.labelling.task import Task


def export_dataset(task: Task) -> str:
    """Render a normal dataset; unresolved conflicts block export, never become zeros."""
    rows: list[dict[str, Any]] = []
    no_positive: list[str] = []
    for q in task.queries:
        relevant: dict[str, int] = {}
        spans: list[dict[str, Any]] = []
        for doc in q.documents:
            chosen = resolve(task, q.query_id, doc)
            if chosen is None:
                if decisions(task, q.query_id, doc):
                    raise ValueError(f"Resolve disagreement for {q.query_id}/{doc} before export")
                continue
            if chosen.grade is None:
                continue
            relevant[doc] = chosen.grade
            if chosen.start is not None:
                spans.append(
                    {"doc_id": doc, "start": chosen.start, "end": chosen.end, "grade": chosen.grade}
                )
        if not any(relevant.values()):
            no_positive.append(q.query_id)
            # Existing metrics require positive ground truth. Keep the question unlabelled,
            # rather than inventing a positive or changing the dataset contract.
            relevant = {}
        row: dict[str, Any] = {"query_id": q.query_id, "text": q.text, "relevant": relevant}
        if q.family_id is not None:
            row["family_id"] = q.family_id
        if spans:
            row["relevant_spans"] = spans
        rows.append(row)
    if len(no_positive) == len(rows):
        raise ValueError("No positive human labels yet; label at least one relevant document")
    meta = {
        "name": task.name,
        "provenance": {
            "task_sha256": task.digest(),
            "corpus_sha256": task.corpus_sha256,
            "pooling_sources": task.sources,
            "annotators": sorted({j.annotator for j in [*task.judgments, *task.resolutions]}),
            "agreement": agreement(task),
            "suggestion_agreement": suggestion_agreement(task),
            "accepted_suggestions": [
                j.model_dump() for j in task.judgments if j.accepted_suggestion
            ],
            "queries_without_positive_labels": no_positive,
            "resolution": "strict majority of grade and span; otherwise named reviewer",
        },
    }
    text = render_dataset(rows, meta)
    parse_dataset(text)
    return text
