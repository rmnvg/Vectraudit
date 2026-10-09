"""Authenticated, read-only team views of uploaded workbench revisions."""

from __future__ import annotations

import json
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from vectraudit.cloud import web
from vectraudit.cloud.deps import SessionDep, SettingsDep
from vectraudit.cloud.errors import CloudError, not_found
from vectraudit.cloud.models import SharedRecord
from vectraudit.cloud.routes.history import _after, _encode_cursor
from vectraudit.integrations.shared_records import Kind

router = APIRouter(include_in_schema=False)
e = web.e


@router.get("/app/{workspace}/{project}/records")
def records_page(
    workspace: str,
    project: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    kind: Kind | None = None,
    source_id: str | None = Query(default=None, max_length=32),
    cursor: str | None = None,
) -> HTMLResponse:
    try:
        identity, access = web._project_access(request, session, settings, workspace, project)
        assert access.project is not None
        stmt = select(SharedRecord).where(SharedRecord.project_id == access.project.id)
        if kind:
            stmt = stmt.where(SharedRecord.kind == kind)
        if source_id:
            stmt = stmt.where(SharedRecord.source_id == source_id)
        after = _after(SharedRecord.uploaded_at, SharedRecord.id, cursor)
        if after is not None:
            stmt = stmt.where(after)
        rows = list(
            session.scalars(
                stmt.order_by(SharedRecord.uploaded_at.desc(), SharedRecord.id.desc()).limit(51)
            )
        )
    except CloudError as err:
        return web._error_page(err, None)
    base = f"/app/{quote(workspace)}/{quote(project)}"
    table = []
    for row in rows[:50]:
        data = row.data["data"]
        summary = (
            f"{data['query_id']}: {data['status']} (local)"
            if row.kind == "case"
            else f"{len(data['judgments'])} questions"
            if row.kind == "benchmark"
            else f"Policy version {data['version']}"
        )
        table.append(
            [
                e(row.kind),
                f'<a href="{base}/records/{row.id}">{e(row.source_id)}</a>',
                e(row.revision),
                e(summary),
                "Included" if row.query_text else "Withheld",
                e(row.uploaded_at.isoformat()),
                e(row.uploaded_by),
            ]
        )
    body = (
        f'<p><a href="{base}">Project</a></p><h1>Shared workbench records</h1>'
        "<p>Uploaded revisions, newest upload first. "
        "Upload order does not establish the latest local revision. "
        "Local verification is evidence from the workbench, not hosted release approval. "
        "Records remain after raw runs expire.</p>"
        f'<p><a href="{base}/records">All</a> · <a href="{base}/records?kind=case">Cases</a> · '
        f'<a href="{base}/records?kind=benchmark">Benchmarks</a> · '
        f'<a href="{base}/records?kind=policy">Policies</a></p>'
        + web._table(
            [
                "Kind",
                "Local ID",
                "Revision",
                "Recorded state",
                "Question text",
                "Uploaded",
                "Uploaded by",
            ],
            table,
            "No shared records. Upload a saved revision with vectraudit workbench share.",
        )
    )
    if len(rows) > 50:
        params = {"cursor": _encode_cursor(rows[49].uploaded_at, rows[49].id)}
        if kind:
            params["kind"] = kind
        if source_id:
            params["source_id"] = source_id
        body += f'<p><a href="{base}/records?{e(urlencode(params))}">Older uploads</a></p>'
    return web.page("Shared workbench records", body, identity)


@router.get("/app/{workspace}/{project}/records/{record_id}")
def record_page(
    workspace: str,
    project: str,
    record_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    offset: int = Query(default=0, ge=0, le=2000),
) -> HTMLResponse:
    try:
        identity, access = web._project_access(request, session, settings, workspace, project)
        assert access.project is not None
        row = session.scalar(
            select(SharedRecord).where(
                SharedRecord.project_id == access.project.id, SharedRecord.id == record_id
            )
        )
        if row is None:
            raise not_found("Shared record")
    except CloudError as err:
        return web._error_page(err, None)
    base = f"/app/{quote(workspace)}/{quote(project)}"
    api = f"/api/v1/workspaces/{quote(workspace)}/projects/{quote(project)}/records/{row.id}/file"
    data = row.data["data"]
    sharing = row.data["sharing"]
    text_mode = (
        "explicitly shared when available" if row.query_text else "withheld by sharing settings"
    )
    body = (
        f'<p><a href="{base}/records?kind={row.kind}&amp;source_id={row.source_id}">'
        "All uploaded revisions of this record</a></p>"
        f"<h1>Shared {e(row.kind)}</h1><p>Local ID: {e(row.source_id)} · "
        f"revision: {e(row.revision)}</p>"
        f"<p>Uploaded by {e(row.uploaded_by)} at {e(row.uploaded_at.isoformat())}.</p>"
        "<p>Immutable local evidence; no live synchronization or hosted approval is implied.</p>"
        f"<p>Question text: {text_mode}. "
        "Passages, notes, owner names, context and raw traces remain local.</p>"
        f'<p><a href="{api}">Download projected record</a></p>'
        f"<p>Source SHA-256: {e(sharing['source_sha256'])}<br>"
        f"Payload SHA-256: {e(sharing['payload_sha256'])}</p>"
    )
    if row.kind == "case":
        body += (
            f"<h2>Question {e(data['query_id'])}</h2>"
            f"<p>{e(data.get('question_text') or 'Question text unavailable.')}</p>"
            f"<p>Locally recorded status: <strong>{e(data['status'])}</strong>. "
            f"Benchmark: {e(data['dataset_sha256'])}.</p>"
            f"<p>Original baseline: {e(data['baseline'])} · "
            f"original candidate: {e(data['candidate'])}.</p>"
            f"<p>Original policy SHA-256: {e(data['policy']['sha256'])}.</p>"
            + web._table(
                ["Finding", "Severity", "Document IDs"],
                [
                    [e(f["kind"]), e(f["severity"]), e(", ".join(f["doc_ids"]))]
                    for f in data["findings"]
                ],
                "No findings.",
            )
        )
        events = list(reversed(data["history"]))
        body += "<h2>Recorded case history</h2>" + web._table(
            ["When", "Event", "Status", "Candidate", "Local verdict", "Policy SHA-256"],
            [
                [
                    e(h["at"]),
                    e(h["kind"]),
                    e(h.get("status")),
                    e(h.get("candidate")),
                    e(h.get("verdict")),
                    e((h.get("policy") or {}).get("sha256")),
                ]
                for h in events[offset : offset + 50]
            ],
            "No events.",
        )
        count = len(events)
    elif row.kind == "benchmark":
        inventory = data["inventory"]
        body += (
            f"<h2>Benchmark identity</h2><p>Dataset SHA-256: {e(data['dataset_sha256'])}.</p>"
            f"<p>Draft {e(data['draft_id'])} at revision {e(data['draft_revision'])}. "
            f"Inventory: {len(inventory) if inventory is not None else 'unknown'} document hashes; "
            "available in the download.</p>"
            "<h2>Judgment metadata</h2>"
            + web._table(
                [
                    "Question",
                    "Text",
                    "Split",
                    "Relevant grades",
                    "Irrelevant IDs",
                    "No answer",
                    "Forbidden IDs",
                    "Passage spans",
                ],
                [
                    [
                        e(q["query_id"]),
                        e(q.get("text") or "Withheld / unavailable"),
                        e(q["split"]),
                        e(json.dumps(q["relevant"])),
                        e(", ".join(q["irrelevant"])),
                        e(str(q["no_answer"])),
                        e(", ".join(q["forbidden"])),
                        e(str(len(q["relevant_spans"]))),
                    ]
                    for q in data["judgments"][offset : offset + 50]
                ],
                "No questions.",
            )
        )
        count = len(data["judgments"])
    else:
        body += (
            f"<h2>Saved policy version {data['version']}</h2><p>SHA-256: {e(data['sha256'])}. "
            "An archived policy does not change the project release policy.</p>"
            f"<pre>{e(json.dumps(data['rules'], indent=2))}</pre>"
        )
        count = 0
    if offset + 50 < count:
        body += f'<p><a href="{base}/records/{row.id}?offset={offset + 50}">Next 50</a></p>'
    return web.page(f"Shared {row.kind}", body, identity)
