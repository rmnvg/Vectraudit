"""Tenant-scoped data access. Every read of a tenant-owned table goes through here.

Access rules:

- A user reaches a workspace only through a membership; roles rank viewer < engineer <
  owner. Platform admins can create workspaces but get **no** implicit access to customer
  data.
- A runner token reaches exactly one project of one workspace, and only the actions its
  scopes allow.
- Objects in other workspaces are reported as *not found*, never *forbidden*, so callers
  can't probe which workspaces or projects exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vectraudit.cloud.auth import RUNNER_SCOPES, Principal
from vectraudit.cloud.errors import CloudError, forbidden, not_found
from vectraudit.cloud.models import (
    ROLES,
    ApiToken,
    AuditEvent,
    Membership,
    Project,
    User,
    Workspace,
    utcnow,
)
from vectraudit.cloud.security import NewToken, generate_token

ROLE_RANK = {"viewer": 1, "engineer": 2, "owner": 3}
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]+$")

ACTION_ROLE = {
    "read": "viewer",
    "read_baselines": "viewer",
    "write_runs": "engineer",
    "write_checks": "engineer",
    "pin_baseline": "engineer",
    "manage_project": "engineer",
    "manage_runner_tokens": "engineer",
    "manage_members": "owner",
    "read_audit": "owner",
}
RUNNER_ACTION_SCOPE = {
    "read": "runs:read",
    "write_runs": "runs:write",
    "read_baselines": "baselines:read",
    "write_checks": "checks:write",
}


@dataclass(frozen=True)
class Access:
    workspace: Workspace
    project: Project | None
    role: str
    """The caller's role, or ``runner``."""
    principal: Principal


def check_slug(slug: str, what: str) -> str:
    if not SLUG.fullmatch(slug):
        raise CloudError(
            422,
            "invalid_slug",
            f"{what} {slug!r} must be 2-63 lowercase letters, digits or dashes.",
        )
    return slug


def normalize_email(email: str) -> str:
    email = email.strip().lower()
    if len(email) > 320 or not EMAIL.fullmatch(email):
        raise CloudError(422, "invalid_email", f"{email!r} is not a valid email address.")
    return email


def authorize(
    session: Session,
    principal: Principal,
    workspace_slug: str,
    action: str,
    project_slug: str | None = None,
) -> Access:
    """The workspace (and project) the caller may perform ``action`` on, or 404/403."""
    ws = session.scalars(select(Workspace).where(Workspace.slug == workspace_slug)).first()
    if ws is None:
        raise not_found("Workspace")
    if principal.kind == "runner":
        if principal.workspace_id != ws.id:
            raise not_found("Workspace")
        project = _project(session, ws, project_slug) if project_slug else None
        if project is not None and project.id != principal.project_id:
            raise not_found("Project")
        scope = RUNNER_ACTION_SCOPE.get(action)
        if scope is None or scope not in principal.scopes:
            raise forbidden(
                f"Runner tokens can't {action.replace('_', ' ')}.",
                "Use a personal token of a workspace member for this.",
            )
        return Access(ws, project, "runner", principal)
    membership = session.scalars(
        select(Membership).where(
            Membership.workspace_id == ws.id, Membership.user_id == principal.user_id
        )
    ).first()
    if membership is None:
        raise not_found("Workspace")
    need = ACTION_ROLE.get(action, "owner")
    if ROLE_RANK[membership.role] < ROLE_RANK[need]:
        raise forbidden(f"This needs the {need} role; you are {membership.role}.")
    project = _project(session, ws, project_slug) if project_slug else None
    return Access(ws, project, membership.role, principal)


def _project(session: Session, ws: Workspace, slug: str) -> Project:
    project = session.scalars(
        select(Project).where(Project.workspace_id == ws.id, Project.slug == slug)
    ).first()
    if project is None:
        raise not_found("Project")
    return project


def audit(
    session: Session,
    workspace_id: str | None,
    actor: str,
    action: str,
    target_type: str,
    target_id: str,
    **details: Any,
) -> None:
    session.add(
        AuditEvent(
            workspace_id=workspace_id,
            actor=actor,
            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details,
        )
    )


# --- users and workspaces ----------------------------------------------------------------


def get_or_create_user(session: Session, email: str, name: str = "") -> User:
    email = normalize_email(email)
    user = session.scalars(select(User).where(User.email == email)).first()
    if user is None:
        user = User(email=email, name=name)
        session.add(user)
        session.flush()
    return user


def create_workspace(
    session: Session, actor: str, slug: str, name: str, owner_email: str
) -> tuple[Workspace, User]:
    check_slug(slug, "Workspace slug")
    if session.scalars(select(Workspace).where(Workspace.slug == slug)).first() is not None:
        raise CloudError(409, "conflict", f"Workspace {slug!r} already exists.")
    owner = get_or_create_user(session, owner_email)
    ws = Workspace(slug=slug, name=name)
    session.add(ws)
    session.flush()
    session.add(Membership(workspace_id=ws.id, user_id=owner.id, role="owner"))
    audit(session, ws.id, actor, "workspace.create", "workspace", ws.id, owner=owner.email)
    return ws, owner


def workspaces_of(session: Session, user_id: str) -> list[tuple[Workspace, str]]:
    rows = session.execute(
        select(Workspace, Membership.role)
        .join(Membership, Membership.workspace_id == Workspace.id)
        .where(Membership.user_id == user_id)
        .order_by(Workspace.slug)
    ).all()
    return [(ws, role) for ws, role in rows]


# --- members -----------------------------------------------------------------------------


def members(session: Session, ws: Workspace) -> list[tuple[User, Membership]]:
    rows = session.execute(
        select(User, Membership)
        .join(Membership, Membership.user_id == User.id)
        .where(Membership.workspace_id == ws.id)
        .order_by(User.email)
    ).all()
    return [(u, m) for u, m in rows]


def _check_role(role: str) -> str:
    if role not in ROLES:
        raise CloudError(422, "invalid_role", f"Role must be one of {', '.join(ROLES)}.")
    return role


def _membership(session: Session, ws: Workspace, user_id: str) -> Membership:
    m = session.scalars(
        select(Membership).where(Membership.workspace_id == ws.id, Membership.user_id == user_id)
    ).first()
    if m is None:
        raise not_found("Member")
    return m


def _owner_count(session: Session, ws: Workspace) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(Membership)
            .where(Membership.workspace_id == ws.id, Membership.role == "owner")
        )
        or 0
    )


def add_member(session: Session, access: Access, email: str, role: str) -> tuple[User, Membership]:
    _check_role(role)
    user = get_or_create_user(session, email)
    existing = session.scalars(
        select(Membership).where(
            Membership.workspace_id == access.workspace.id, Membership.user_id == user.id
        )
    ).first()
    if existing is not None:
        raise CloudError(409, "conflict", f"{user.email} is already a member.")
    m = Membership(workspace_id=access.workspace.id, user_id=user.id, role=role)
    session.add(m)
    audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "member.add",
        "user",
        user.id,
        email=user.email,
        role=role,
    )
    return user, m


def change_role(session: Session, access: Access, user_id: str, role: str) -> Membership:
    _check_role(role)
    m = _membership(session, access.workspace, user_id)
    if m.role == "owner" and role != "owner" and _owner_count(session, access.workspace) == 1:
        raise CloudError(409, "last_owner", "A workspace needs at least one owner.")
    old, m.role = m.role, role
    audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "member.role",
        "user",
        user_id,
        old=old,
        new=role,
    )
    return m


def remove_member(session: Session, access: Access, user_id: str) -> None:
    m = _membership(session, access.workspace, user_id)
    if m.role == "owner" and _owner_count(session, access.workspace) == 1:
        raise CloudError(409, "last_owner", "A workspace needs at least one owner.")
    session.delete(m)
    # A member's personal tokens stay valid for their other workspaces; access here ends
    # with the membership.
    audit(session, access.workspace.id, access.principal.actor, "member.remove", "user", user_id)


# --- projects ----------------------------------------------------------------------------


def projects(session: Session, ws: Workspace) -> list[Project]:
    return list(
        session.scalars(select(Project).where(Project.workspace_id == ws.id).order_by(Project.slug))
    )


def create_project(session: Session, access: Access, slug: str, name: str) -> Project:
    check_slug(slug, "Project slug")
    exists = session.scalars(
        select(Project).where(Project.workspace_id == access.workspace.id, Project.slug == slug)
    ).first()
    if exists is not None:
        raise CloudError(409, "conflict", f"Project {slug!r} already exists.")
    project = Project(workspace_id=access.workspace.id, slug=slug, name=name)
    session.add(project)
    session.flush()
    audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "project.create",
        "project",
        project.id,
        slug=slug,
    )
    return project


# --- tokens ------------------------------------------------------------------------------


def create_personal_token(
    session: Session, user: User, name: str, actor: str, expires_at: datetime | None = None
) -> tuple[ApiToken, NewToken]:
    new = generate_token("personal")
    row = ApiToken(
        kind="personal",
        prefix=new.prefix,
        token_hash=new.token_hash,
        name=name,
        user_id=user.id,
        scopes=[],
        created_by=actor,
        expires_at=expires_at,
    )
    session.add(row)
    session.flush()
    audit(session, None, actor, "token.create", "token", row.id, prefix=row.prefix, kind="personal")
    return row, new


def create_runner_token(
    session: Session, access: Access, name: str, scopes: list[str] | None = None
) -> tuple[ApiToken, NewToken]:
    assert access.project is not None
    scopes = list(scopes or RUNNER_SCOPES)
    unknown = set(scopes) - set(RUNNER_SCOPES)
    if unknown:
        raise CloudError(
            422,
            "invalid_scope",
            f"Unknown scope(s): {', '.join(sorted(unknown))}.",
            f"Runner scopes: {', '.join(RUNNER_SCOPES)}.",
        )
    new = generate_token("runner")
    row = ApiToken(
        kind="runner",
        prefix=new.prefix,
        token_hash=new.token_hash,
        name=name,
        workspace_id=access.workspace.id,
        project_id=access.project.id,
        scopes=scopes,
        created_by=access.principal.actor,
    )
    session.add(row)
    session.flush()
    audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "token.create",
        "token",
        row.id,
        prefix=row.prefix,
        kind="runner",
        project=access.project.slug,
        scopes=scopes,
    )
    return row, new


def runner_tokens(session: Session, access: Access) -> list[ApiToken]:
    assert access.project is not None
    return list(
        session.scalars(
            select(ApiToken)
            .where(
                ApiToken.workspace_id == access.workspace.id,
                ApiToken.project_id == access.project.id,
                ApiToken.kind == "runner",
            )
            .order_by(ApiToken.created_at)
        )
    )


def revoke_runner_token(session: Session, access: Access, token_id: str) -> ApiToken:
    assert access.project is not None
    row = session.scalars(
        select(ApiToken).where(
            ApiToken.id == token_id,
            ApiToken.workspace_id == access.workspace.id,
            ApiToken.project_id == access.project.id,
            ApiToken.kind == "runner",
        )
    ).first()
    if row is None:
        raise not_found("Token")
    if row.revoked_at is None:
        row.revoked_at = utcnow()
        audit(
            session,
            access.workspace.id,
            access.principal.actor,
            "token.revoke",
            "token",
            row.id,
            prefix=row.prefix,
        )
    return row


def personal_tokens(session: Session, user_id: str) -> list[ApiToken]:
    return list(
        session.scalars(
            select(ApiToken)
            .where(ApiToken.user_id == user_id, ApiToken.kind == "personal")
            .order_by(ApiToken.created_at)
        )
    )


def revoke_personal_token(session: Session, principal: Principal, token_id: str) -> ApiToken:
    row = session.scalars(
        select(ApiToken).where(
            ApiToken.id == token_id,
            ApiToken.user_id == principal.user_id,
            ApiToken.kind == "personal",
        )
    ).first()
    if row is None:
        raise not_found("Token")
    if row.revoked_at is None:
        row.revoked_at = utcnow()
        audit(session, None, principal.actor, "token.revoke", "token", row.id, prefix=row.prefix)
    return row


def audit_events(session: Session, ws: Workspace, limit: int = 100) -> list[AuditEvent]:
    return list(
        session.scalars(
            select(AuditEvent)
            .where(AuditEvent.workspace_id == ws.id)
            .order_by(AuditEvent.created_at.desc())
            .limit(limit)
        )
    )
