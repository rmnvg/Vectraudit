"""Run upload and retrieval.

A run file is validated with the same rules as the CLI (:func:`check_run`). Uploads are
idempotent on the client's run id: the same content again returns the stored run (200),
different content under the same id is a conflict (409), so CI retries never duplicate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from vectraudit.cloud import quotas, repo
from vectraudit.cloud.deps import PrincipalDep, SessionDep, SettingsDep
from vectraudit.cloud.errors import CloudError
from vectraudit.cloud.models import Run, Workspace
from vectraudit.cloud.service import find_run
from vectraudit.evaluation.runfile import RunFileError, check_run
from vectraudit.integrations.sharing import project_run

router = APIRouter(prefix="/api/v1/workspaces/{workspace}/projects/{project}", tags=["runs"])
DIGEST_HEADER = "x-vectraudit-digest"


class RunOut(BaseModel):
    id: str
    run_id: str
    status: str
    label: str | None
    commit_sha: str | None
    dataset_sha256: str
    backend: str | None
    n_queries: int
    created_at: datetime | None
    uploaded_at: datetime
    uploaded_by: str | None
    summary: dict[str, Any]
    url: str


def canonical_digest(record: dict[str, Any]) -> str:
    """SHA-256 of the run's canonical JSON (key order and whitespace don't matter)."""
    blob = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_out(run: Run, ws: Workspace, project_slug: str, public_url: str) -> RunOut:
    return RunOut(
        id=run.id,
        run_id=run.run_id,
        status=run.status,
        label=run.label,
        commit_sha=run.commit_sha,
        dataset_sha256=run.dataset_sha256,
        backend=run.backend,
        n_queries=run.n_queries,
        created_at=run.created_at,
        uploaded_at=run.uploaded_at,
        uploaded_by=run.uploaded_by,
        summary=run.summary,
        url=f"{public_url}/api/v1/workspaces/{ws.slug}/projects/{project_slug}/runs/{run.run_id}",
    )


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


@router.post("/runs", status_code=201)
async def upload_run(
    workspace: str,
    project: str,
    request: Request,
    response: Response,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> RunOut:
    access = repo.authorize(session, principal, workspace, "write_runs", project)
    assert access.project is not None
    raw = await request.body()
    if len(raw) > settings.max_upload_bytes:
        raise CloudError(413, "payload_too_large", "Run file too large.")
    claimed = request.headers.get(DIGEST_HEADER)
    if claimed is not None:
        expected = "sha256=" + hashlib.sha256(raw).hexdigest()
        if claimed != expected:
            raise CloudError(
                400,
                "digest_mismatch",
                "The body does not match its X-Vectraudit-Digest.",
                "The upload was corrupted or altered in transit; retry.",
            )
    try:
        record = project_run(check_run(json.loads(raw)))
    except (ValueError, RunFileError) as e:
        raise CloudError(422, "invalid_run", f"Not a valid run file: {e}") from e
    n_queries = int(record["dataset"]["n_queries"])
    if n_queries > settings.max_queries_per_run:
        raise CloudError(
            413,
            "too_many_queries",
            f"{n_queries:,} queries exceeds the limit of {settings.max_queries_per_run:,} per run.",
        )
    digest = canonical_digest(record)
    existing = session.scalars(
        select(Run).where(Run.project_id == access.project.id, Run.run_id == record["run_id"])
    ).first()
    if existing is not None:
        if existing.digest != digest:
            raise CloudError(
                409,
                "run_conflict",
                f"Run {record['run_id']} was already uploaded with different content.",
                "Run ids are unique per project; record a new run instead of editing one.",
            )
        response.status_code = 200
        return run_out(existing, access.workspace, project, settings.public_url)
    # After the idempotency check: retrying a stored run never trips a quota.
    quotas.check_upload(session, access.workspace, n_queries)
    backend = record.get("backend") or {}
    run = Run(
        workspace_id=access.workspace.id,
        project_id=access.project.id,
        run_id=record["run_id"],
        digest=digest,
        status=record["status"],
        label=record.get("label"),
        commit_sha=(record.get("provenance") or {}).get("commit_sha"),
        dataset_sha256=record["dataset"]["sha256"],
        backend=backend.get("backend"),
        n_queries=n_queries,
        created_at=_parse_time(record.get("created_at")),
        uploaded_by=principal.actor,
        summary=record.get("summary") or {},
        data=record,
    )
    try:
        with session.begin_nested():  # a concurrent identical upload may win the race
            session.add(run)
    except IntegrityError:
        winner = session.scalars(
            select(Run).where(Run.project_id == access.project.id, Run.run_id == record["run_id"])
        ).one()
        if winner.digest != digest:
            raise CloudError(
                409,
                "run_conflict",
                f"Run {record['run_id']} was uploaded concurrently with different content.",
            ) from None
        response.status_code = 200
        return run_out(winner, access.workspace, project, settings.public_url)
    return run_out(run, access.workspace, project, settings.public_url)


@router.get("/runs/{run_id}")
def get_run(
    workspace: str,
    project: str,
    run_id: str,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> RunOut:
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    run = find_run(session, access.workspace.id, access.project.id, run_id)
    return run_out(run, access.workspace, project, settings.public_url)


@router.get("/runs/{run_id}/file")
def get_run_file(
    workspace: str, project: str, run_id: str, session: SessionDep, principal: PrincipalDep
) -> dict[str, Any]:
    """The retained hosted projection, for release checks and audits."""
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    return find_run(session, access.workspace.id, access.project.id, run_id).data
