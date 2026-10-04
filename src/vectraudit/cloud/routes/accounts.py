"""Accounts: the caller, personal tokens, workspaces, members, projects, runner tokens."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel

from vectraudit.cloud import quotas, repo, service
from vectraudit.cloud.auth import Principal
from vectraudit.cloud.deps import PrincipalDep, SessionDep
from vectraudit.cloud.errors import CloudError, forbidden
from vectraudit.cloud.models import ApiToken, User, utcnow
from vectraudit.cloud.schemas import (
    AuditOut,
    MemberIn,
    MemberOut,
    MeOut,
    NewTokenOut,
    ProjectIn,
    ProjectOut,
    RoleIn,
    RunnerTokenIn,
    TokenIn,
    TokenOut,
    UserOut,
    WorkspaceIn,
    WorkspaceOut,
)

router = APIRouter(prefix="/api/v1", tags=["accounts"])


def _user(session: SessionDep, principal: Principal) -> User:
    if principal.kind != "user":
        raise forbidden("This needs a personal token, not a runner token.")
    user = session.get(User, principal.user_id)
    assert user is not None
    return user


def _token_out(row: ApiToken) -> TokenOut:
    return TokenOut(
        id=row.id,
        kind=row.kind,
        prefix=row.prefix,
        name=row.name,
        scopes=list(row.scopes or []),
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
    )


def _user_out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email, name=user.name)


@router.get("/me")
def me(session: SessionDep, principal: PrincipalDep) -> MeOut:
    user = _user(session, principal)
    return MeOut(
        user=_user_out(user),
        workspaces=[
            WorkspaceOut(slug=ws.slug, name=ws.name, plan=ws.plan, role=role)
            for ws, role in repo.workspaces_of(session, user.id)
        ],
    )


@router.get("/tokens")
def list_tokens(session: SessionDep, principal: PrincipalDep) -> list[TokenOut]:
    user = _user(session, principal)
    return [_token_out(t) for t in repo.personal_tokens(session, user.id)]


@router.post("/tokens", status_code=201)
def create_token(body: TokenIn, session: SessionDep, principal: PrincipalDep) -> NewTokenOut:
    user = _user(session, principal)
    expires = utcnow() + timedelta(days=body.expires_in_days) if body.expires_in_days else None
    row, new = repo.create_personal_token(session, user, body.name, principal.actor, expires)
    return NewTokenOut(**_token_out(row).model_dump(), token=new.token)


@router.delete("/tokens/{token_id}", status_code=204)
def revoke_token(token_id: str, session: SessionDep, principal: PrincipalDep) -> Response:
    _user(session, principal)
    repo.revoke_personal_token(session, principal, token_id)
    return Response(status_code=204)


@router.get("/workspaces")
def list_workspaces(session: SessionDep, principal: PrincipalDep) -> list[WorkspaceOut]:
    user = _user(session, principal)
    return [
        WorkspaceOut(slug=ws.slug, name=ws.name, plan=ws.plan, role=role)
        for ws, role in repo.workspaces_of(session, user.id)
    ]


@router.post("/workspaces", status_code=201)
def create_workspace(
    body: WorkspaceIn, session: SessionDep, principal: PrincipalDep
) -> WorkspaceOut:
    user = _user(session, principal)
    if not user.is_admin:
        raise forbidden("Only platform administrators create workspaces.")
    ws, _ = repo.create_workspace(session, principal.actor, body.slug, body.name, body.owner_email)
    return WorkspaceOut(slug=ws.slug, name=ws.name, plan=ws.plan, role=None)


@router.get("/workspaces/{workspace}")
def get_workspace(workspace: str, session: SessionDep, principal: PrincipalDep) -> WorkspaceOut:
    access = repo.authorize(session, principal, workspace, "read")
    ws = access.workspace
    return WorkspaceOut(slug=ws.slug, name=ws.name, plan=ws.plan, role=access.role)


@router.get("/workspaces/{workspace}/members")
def list_members(workspace: str, session: SessionDep, principal: PrincipalDep) -> list[MemberOut]:
    access = repo.authorize(session, principal, workspace, "read")
    if access.role == "runner":
        raise forbidden("Runner tokens can't list members.")
    return [
        MemberOut(user=_user_out(u), role=m.role, since=m.created_at)
        for u, m in repo.members(session, access.workspace)
    ]


@router.post("/workspaces/{workspace}/members", status_code=201)
def add_member(
    workspace: str, body: MemberIn, session: SessionDep, principal: PrincipalDep
) -> MemberOut:
    access = repo.authorize(session, principal, workspace, "manage_members")
    user, m = repo.add_member(session, access, body.email, body.role)
    session.flush()
    return MemberOut(user=_user_out(user), role=m.role, since=m.created_at)


@router.patch("/workspaces/{workspace}/members/{user_id}")
def change_role(
    workspace: str, user_id: str, body: RoleIn, session: SessionDep, principal: PrincipalDep
) -> MemberOut:
    access = repo.authorize(session, principal, workspace, "manage_members")
    m = repo.change_role(session, access, user_id, body.role)
    user = session.get(User, user_id)
    assert user is not None
    return MemberOut(user=_user_out(user), role=m.role, since=m.created_at)


@router.delete("/workspaces/{workspace}/members/{user_id}", status_code=204)
def remove_member(
    workspace: str, user_id: str, session: SessionDep, principal: PrincipalDep
) -> Response:
    access = repo.authorize(session, principal, workspace, "manage_members")
    repo.remove_member(session, access, user_id)
    return Response(status_code=204)


def _project_out(p: object) -> ProjectOut:
    return ProjectOut.model_validate(p, from_attributes=True)


@router.get("/workspaces/{workspace}/projects")
def list_projects(workspace: str, session: SessionDep, principal: PrincipalDep) -> list[ProjectOut]:
    access = repo.authorize(session, principal, workspace, "read")
    found = repo.projects(session, access.workspace)
    if access.role == "runner":
        found = [p for p in found if p.id == principal.project_id]
    return [_project_out(p) for p in found]


@router.post("/workspaces/{workspace}/projects", status_code=201)
def create_project(
    workspace: str, body: ProjectIn, session: SessionDep, principal: PrincipalDep
) -> ProjectOut:
    access = repo.authorize(session, principal, workspace, "manage_project")
    return _project_out(repo.create_project(session, access, body.slug, body.name))


@router.get("/workspaces/{workspace}/projects/{project}")
def get_project(
    workspace: str, project: str, session: SessionDep, principal: PrincipalDep
) -> ProjectOut:
    access = repo.authorize(session, principal, workspace, "read", project)
    return _project_out(access.project)


@router.get("/workspaces/{workspace}/projects/{project}/runner-tokens")
def list_runner_tokens(
    workspace: str, project: str, session: SessionDep, principal: PrincipalDep
) -> list[TokenOut]:
    access = repo.authorize(session, principal, workspace, "manage_runner_tokens", project)
    return [_token_out(t) for t in repo.runner_tokens(session, access)]


@router.post("/workspaces/{workspace}/projects/{project}/runner-tokens", status_code=201)
def create_runner_token(
    workspace: str,
    project: str,
    body: RunnerTokenIn,
    session: SessionDep,
    principal: PrincipalDep,
) -> NewTokenOut:
    access = repo.authorize(session, principal, workspace, "manage_runner_tokens", project)
    row, new = repo.create_runner_token(session, access, body.name, body.scopes)
    return NewTokenOut(**_token_out(row).model_dump(), token=new.token)


@router.delete(
    "/workspaces/{workspace}/projects/{project}/runner-tokens/{token_id}", status_code=204
)
def revoke_runner_token(
    workspace: str, project: str, token_id: str, session: SessionDep, principal: PrincipalDep
) -> Response:
    access = repo.authorize(session, principal, workspace, "manage_runner_tokens", project)
    repo.revoke_runner_token(session, access, token_id)
    return Response(status_code=204)


@router.get("/workspaces/{workspace}/audit")
def audit_log(
    workspace: str, session: SessionDep, principal: PrincipalDep, limit: int = 100
) -> list[AuditOut]:
    access = repo.authorize(session, principal, workspace, "read_audit")
    return [
        AuditOut.model_validate(e, from_attributes=True)
        for e in repo.audit_events(session, access.workspace, min(max(limit, 1), 500))
    ]


@router.get("/workspaces/{workspace}/usage")
def workspace_usage(workspace: str, session: SessionDep, principal: PrincipalDep) -> dict[str, Any]:
    """Plan limits and this month's usage."""
    access = repo.authorize(session, principal, workspace, "read")
    if access.role == "runner":
        raise forbidden("Runner tokens can't read usage.")
    return quotas.status(session, access.workspace)


@router.get("/admin/usage")
def admin_usage(session: SessionDep, principal: PrincipalDep, month: str = "") -> Response:
    """Usage per workspace for one month (YYYY-MM, default this month) as CSV."""
    user = _user(session, principal)
    if not user.is_admin:
        raise forbidden("Only platform administrators can export usage.")
    try:
        start = datetime.strptime(month, "%Y-%m") if month else quotas.month_start()
    except ValueError as e:
        raise CloudError(422, "invalid_month", "month must look like 2026-10.") from e
    rows = quotas.usage_rows(session, start.year, start.month)
    out = io.StringIO()
    fields = ["workspace", "plan", "month", "runs", "queries", "checks", "stored_runs", "members"]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="usage-{start:%Y-%m}.csv"'},
    )


class ConfirmIn(BaseModel):
    confirm: str | None = None
    """The slug of what is being deleted, typed by a person."""


@router.get("/workspaces/{workspace}/export")
def export_workspace(workspace: str, session: SessionDep, principal: PrincipalDep) -> Response:
    """Everything stored for the workspace, as a JSON download (owners)."""
    access = repo.authorize(session, principal, workspace, "export")
    data = service.export_workspace(session, access.workspace)
    filename = f"vectraudit-{access.workspace.slug}-export.json"
    repo.audit(
        session,
        access.workspace.id,
        principal.actor,
        "workspace.export",
        "workspace",
        access.workspace.id,
    )
    return Response(
        json.dumps(data, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/workspaces/{workspace}/projects/{project}")
def delete_project(
    workspace: str, project: str, body: ConfirmIn, session: SessionDep, principal: PrincipalDep
) -> dict[str, int]:
    access = repo.authorize(session, principal, workspace, "delete", project)
    return service.delete_project(session, access, body.confirm)


@router.delete("/workspaces/{workspace}")
def delete_workspace(
    workspace: str, body: ConfirmIn, session: SessionDep, principal: PrincipalDep
) -> dict[str, int]:
    access = repo.authorize(session, principal, workspace, "delete")
    return service.delete_workspace(session, access, body.confirm)
