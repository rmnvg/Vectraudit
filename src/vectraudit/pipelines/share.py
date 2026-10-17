"""Share one saved workbench record with a hosted project: preview first, then upload.

Used by ``vectraudit workbench share`` and the workbench's share panel, so both send exactly
the same allowlisted projection (:mod:`vectraudit.integrations.shared_records`).
"""

from __future__ import annotations

from typing import Any

from vectraudit.integrations.shared_records import Kind, preview_record, project_record
from vectraudit.pipelines.guided import GuidedWorkspace

CATEGORIES = {
    "case": "cases",
    "benchmark": "benchmarks",
    "policy": "policies",
    "investigation": "investigations",
}
GENERIC_ERROR = "Cannot share this record. Check its kind, ID, schema and workspace path."
WITHHELD = {
    "case": ["owner", "investigation notes", "passage and document text", "context and filters"],
    "benchmark": ["passage and document text", "context and filters", "draft notes"],
    "policy": ["saved notes"],
    "investigation": [
        "reviewer names",
        "dispute reasons",
        "evidence requests",
        "closing notes",
        "question and document text",
    ],
}


def projection(
    workspace: GuidedWorkspace,
    kind: Kind,
    identifier: str,
    *,
    share_query_text: bool = False,
) -> dict[str, Any]:
    """The allowlisted projection of one saved record; the local record is unchanged."""
    from vectraudit.pipelines.local_display import CaseQuestionDisplay
    from vectraudit.pipelines.workflows import LocalRecords

    if kind not in CATEGORIES:
        raise ValueError("Kind must be case, benchmark, policy or investigation")
    record = LocalRecords(workspace, CATEGORIES[kind]).read(identifier)
    if kind == "investigation":
        share_query_text = False  # investigations hold no question text to share
    if kind == "case" and share_query_text:
        record = CaseQuestionDisplay(workspace).decorate(record)
    return project_record(kind, record, share_query_text=share_query_text)


def describe(projected: dict[str, Any]) -> dict[str, Any]:
    """What a person needs to decide: what is included, what is withheld, any text."""
    kind = projected["kind"]
    data = projected["data"]
    if kind == "case":
        included = [
            f"Question ID {data['query_id']} and its status ({data['status']})",
            f"{len(data['findings'])} finding(s) with document IDs",
            f"{len(data['history'])} history event(s) and the policy identity",
        ]
        texts = [data["question_text"]] if data.get("question_text") else []
        titles = data.get("document_titles") or {}
        if titles:
            included.append(f"Titles of {len(titles)} cited document(s)")
    elif kind == "benchmark":
        included = [
            f"Benchmark identity {data['dataset_sha256'][:12]}",
            f"{len(data['judgments'])} question(s) with relevance judgments, spans and "
            "forbidden IDs",
        ]
        texts = [q["text"] for q in data["judgments"] if q.get("text")]
    elif kind == "investigation":
        included = [
            f"Question ID {data['query_id']}, its losses and status ({data['status']})",
            f"Diagnosis {data['diagnosis']}, {data['evidence_requests']} evidence request(s) "
            "counted",
            f"{len(data['trials'])} trial(s), {len(data['validations'])} validation(s), "
            f"{len(data['protections'])} protecting policy version(s)",
            "Times to an accepted diagnosis and a verified fix",
        ]
        texts = []
    else:
        included = [f"Policy version {data['version']} and its rules ({data['sha256'][:12]})"]
        texts = []
    withheld = list(WITHHELD[kind])
    if kind == "case" and not data.get("document_titles"):
        withheld.append("document titles")
    return {
        "included": included,
        "withheld": withheld,
        "question_text": texts[:5],
        "question_text_count": len(texts),
    }


def preview(projected: dict[str, Any]) -> dict[str, Any]:
    return {"transfer": preview_record(projected), "summary": describe(projected)}
