"""Immutable shared workbench revisions; listing never implies last upload is newest locally."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from vectraudit.cloud import repo
from vectraudit.cloud.deps import PrincipalDep, SessionDep
from vectraudit.cloud.errors import CloudError, not_found
from vectraudit.cloud.models import SharedRecord
from vectraudit.cloud.routes.history import _after, _encode_cursor
from vectraudit.integrations.shared_records import MAX_BYTES, Kind, digest, validate_record

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace}/projects/{project}/records", tags=["shared records"]
)


class RecordOut(BaseModel):
    id: str
    kind: str
    source_id: str
    revision: str
    query_text: bool
    uploaded_at: datetime
    uploaded_by: str | None


class RecordPage(BaseModel):
    records: list[RecordOut]
    next_cursor: str | None


def record_out(row: SharedRecord) -> RecordOut:
    return RecordOut(**{key: getattr(row, key) for key in RecordOut.model_fields})


@router.post("", status_code=201)
async def upload_record(
    workspace: str,
    project: str,
    request: Request,
    response: Response,
    session: SessionDep,
    principal: PrincipalDep,
) -> RecordOut:
    access = repo.authorize(session, principal, workspace, "write_runs", project)
    assert access.project is not None
    raw = await request.body()
    if len(raw) > MAX_BYTES:
        raise CloudError(413, "payload_too_large", "Shared record exceeds 4 MiB.")
    claimed = request.headers.get("x-vectraudit-digest")
    if claimed is not None and claimed != "sha256=" + hashlib.sha256(raw).hexdigest():
        raise CloudError(400, "digest_mismatch", "Shared record transfer digest mismatch.")
    try:
        data = validate_record(json.loads(raw))
    except (ValueError, TypeError, KeyError):
        # Validation errors may include rejected private input; never echo it.
        raise CloudError(
            422,
            "invalid_shared_record",
            "Invalid workbench projection. Generate it with workbench share.",
        ) from None
    stmt = select(SharedRecord).where(
        SharedRecord.project_id == access.project.id,
        SharedRecord.kind == data["kind"],
        SharedRecord.source_id == data["id"],
        SharedRecord.revision == data["revision"],
        SharedRecord.query_text == data["sharing"]["query_text"],
    )
    content_digest = digest(data)

    def existing(row: SharedRecord) -> RecordOut:
        if row.digest != content_digest:
            raise CloudError(
                409,
                "record_conflict",
                "This local revision was already uploaded with different content.",
                "Save a new local revision; existing shared revisions are immutable.",
            )
        response.status_code = 200
        return record_out(row)

    previous = session.scalar(stmt)
    if previous is not None:
        return existing(previous)
    row = SharedRecord(
        workspace_id=access.workspace.id,
        project_id=access.project.id,
        kind=data["kind"],
        source_id=data["id"],
        revision=data["revision"],
        query_text=data["sharing"]["query_text"],
        digest=content_digest,
        uploaded_by=principal.actor,
        data=data,
    )
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        return existing(session.scalars(stmt).one())
    repo.audit(
        session,
        access.workspace.id,
        principal.actor,
        "record.upload",
        "shared_record",
        row.id,
        kind=row.kind,
        source_id=row.source_id,
        revision=row.revision,
        query_text=row.query_text,
    )
    return record_out(row)


@router.get("")
def list_records(
    workspace: str,
    project: str,
    session: SessionDep,
    principal: PrincipalDep,
    kind: Kind | None = None,
    source_id: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = None,
) -> RecordPage:
    access = repo.authorize(session, principal, workspace, "read", project)
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
            stmt.order_by(SharedRecord.uploaded_at.desc(), SharedRecord.id.desc()).limit(limit + 1)
        )
    )
    more = len(rows) > limit
    rows = rows[:limit]
    return RecordPage(
        records=[record_out(row) for row in rows],
        next_cursor=_encode_cursor(rows[-1].uploaded_at, rows[-1].id) if more else None,
    )


@router.get("/{record_id}/file")
def get_record(
    workspace: str, project: str, record_id: str, session: SessionDep, principal: PrincipalDep
) -> dict[str, Any]:
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    row = session.scalar(
        select(SharedRecord).where(
            SharedRecord.project_id == access.project.id, SharedRecord.id == record_id
        )
    )
    if row is None:
        raise not_found("Shared record")
    return row.data
