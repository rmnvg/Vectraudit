"""Operations shared by the JSON API and the dashboard, so both enforce the same rules."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from vectraudit.cloud import repo
from vectraudit.cloud.errors import CloudError, not_found
from vectraudit.cloud.models import (
    ApiToken,
    AuditEvent,
    Baseline,
    Check,
    Invite,
    Membership,
    Project,
    Run,
    User,
    WebSession,
    Workspace,
    utcnow,
)
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy, parse_policy

BASELINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def project_policy(project: Project) -> Policy:
    return DEFAULT_POLICY if project.policy is None else parse_policy(project.policy)


def find_run(session: Session, workspace_id: str, project_id: str, run_id: str) -> Run:
    run = session.scalars(
        select(Run).where(
            Run.workspace_id == workspace_id, Run.project_id == project_id, Run.run_id == run_id
        )
    ).first()
    if run is None:
        raise not_found("Run")
    return run


def get_baseline(session: Session, project: Project, name: str) -> tuple[Baseline, Run]:
    row = session.execute(
        select(Baseline, Run)
        .join(Run, Run.id == Baseline.run_pk)
        .where(Baseline.project_id == project.id, Baseline.name == name)
    ).first()
    if row is None:
        raise not_found(f"Baseline {name!r}")
    return row[0], row[1]


def baselines(session: Session, project: Project) -> list[tuple[Baseline, Run]]:
    rows = session.execute(
        select(Baseline, Run)
        .join(Run, Run.id == Baseline.run_pk)
        .where(Baseline.project_id == project.id)
        .order_by(Baseline.name)
    ).all()
    return [(b, r) for b, r in rows]


def pin_baseline(
    session: Session, access: repo.Access, name: str, run_id: str, note: str
) -> tuple[Baseline, Run]:
    """Point baseline ``name`` at a complete run (caller authorized ``pin_baseline``)."""
    assert access.project is not None
    if not BASELINE_NAME.fullmatch(name):
        raise CloudError(
            422,
            "invalid_name",
            "Baseline names are 1-100 letters, digits, '.', '_' or '-'.",
            "Use e.g. main or release-2026.10 (no slashes).",
        )
    run = find_run(session, access.workspace.id, access.project.id, run_id)
    if run.status != "complete":
        raise CloudError(
            409,
            "run_not_complete",
            f"Run {run.run_id} is {run.status}; only complete runs can be baselines.",
        )
    baseline = session.scalars(
        select(Baseline).where(Baseline.project_id == access.project.id, Baseline.name == name)
    ).first()
    previous = None
    if baseline is None:
        baseline = Baseline(
            workspace_id=access.workspace.id,
            project_id=access.project.id,
            name=name,
            run_pk=run.id,
            pinned_by=access.principal.actor,
            note=note,
        )
        session.add(baseline)
    else:
        previous = baseline.run_pk
        baseline.run_pk = run.id
        baseline.pinned_by = access.principal.actor
        baseline.pinned_at = utcnow()
        baseline.note = note
    session.flush()
    old_run = session.get(Run, previous) if previous else None
    repo.audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "baseline.pin",
        "baseline",
        baseline.id,
        name=name,
        run_id=run.run_id,
        previous_run_id=old_run.run_id if old_run else None,
        note=note,
    )
    return baseline, run


def run_check(
    session: Session, access: repo.Access, baseline_name: str, candidate_run_id: str
) -> tuple[Check, Run, Run, bool]:
    """Compare a run with a pinned baseline. Returns (check, base, cand, created)."""
    assert access.project is not None
    _, base = get_baseline(session, access.project, baseline_name)
    cand = find_run(session, access.workspace.id, access.project.id, candidate_run_id)
    policy = project_policy(access.project)
    policy_digest = digest(policy.as_dict())
    same = select(Check).where(
        Check.project_id == access.project.id,
        Check.baseline_run_pk == base.id,
        Check.candidate_run_pk == cand.id,
        Check.policy_digest == policy_digest,
    )
    found = session.scalars(same).first()
    if found is not None:
        return found, base, cand, False
    result = compare_runs(base.data, cand.data, policy)
    check = Check(
        workspace_id=access.workspace.id,
        project_id=access.project.id,
        baseline_run_pk=base.id,
        candidate_run_pk=cand.id,
        policy_digest=policy_digest,
        verdict=result["verdict"],
        result=result,
        created_by=access.principal.actor,
    )
    try:
        with session.begin_nested():  # a concurrent identical check may win the race
            session.add(check)
    except IntegrityError:
        return session.scalars(same).one(), base, cand, False
    return check, base, cand, True


def load_check(session: Session, access: repo.Access, check_id: str) -> tuple[Check, Run, Run]:
    assert access.project is not None
    check = session.scalars(
        select(Check).where(
            Check.workspace_id == access.workspace.id,
            Check.project_id == access.project.id,
            Check.id == check_id,
        )
    ).first()
    if check is None:
        raise not_found("Check")
    base = session.get(Run, check.baseline_run_pk)
    cand = session.get(Run, check.candidate_run_pk)
    assert base is not None
    assert cand is not None
    return check, base, cand


def recent_checks(session: Session, access: repo.Access, limit: int = 20) -> list[Check]:
    assert access.project is not None
    return list(
        session.scalars(
            select(Check)
            .where(Check.workspace_id == access.workspace.id, Check.project_id == access.project.id)
            .order_by(Check.created_at.desc())
            .limit(limit)
        )
    )


def recent_runs(session: Session, access: repo.Access, limit: int = 20) -> list[Run]:
    assert access.project is not None
    return list(
        session.scalars(
            select(Run)
            .where(Run.workspace_id == access.workspace.id, Run.project_id == access.project.id)
            .order_by(Run.uploaded_at.desc())
            .limit(limit)
        )
    )


# --- export and deletion -----------------------------------------------------------------


def export_workspace(session: Session, ws: Workspace) -> dict[str, Any]:
    """Everything stored for a workspace, as JSON. Token secrets are never stored, so
    tokens appear by prefix and name only."""
    members = [
        {"email": u.email, "role": m.role, "since": m.created_at.isoformat()}
        for u, m in repo.members(session, ws)
    ]
    projects = []
    for p in repo.projects(session, ws):
        runs = list(
            session.scalars(select(Run).where(Run.project_id == p.id).order_by(Run.uploaded_at))
        )
        by_pk = {r.id: r.run_id for r in runs}
        checks = session.scalars(
            select(Check).where(Check.project_id == p.id).order_by(Check.created_at)
        )
        tokens = session.scalars(select(ApiToken).where(ApiToken.project_id == p.id))
        projects.append(
            {
                "slug": p.slug,
                "name": p.name,
                "created_at": p.created_at.isoformat(),
                "policy": p.policy,
                "access_roles": p.access_roles,
                "baselines": [
                    {
                        "name": b.name,
                        "run_id": r.run_id,
                        "pinned_by": b.pinned_by,
                        "pinned_at": b.pinned_at.isoformat(),
                        "note": b.note,
                    }
                    for b, r in baselines(session, p)
                ],
                "runner_tokens": [
                    {
                        "prefix": t.prefix,
                        "name": t.name,
                        "scopes": t.scopes,
                        "created_at": t.created_at.isoformat(),
                        "revoked_at": t.revoked_at.isoformat() if t.revoked_at else None,
                    }
                    for t in tokens
                ],
                "runs": [
                    {
                        "uploaded_at": r.uploaded_at.isoformat(),
                        "uploaded_by": r.uploaded_by,
                        "run": r.data,
                    }
                    for r in runs
                ],
                "checks": [
                    {
                        "id": c.id,
                        "verdict": c.verdict,
                        "created_at": c.created_at.isoformat(),
                        "created_by": c.created_by,
                        "baseline_run_id": by_pk.get(c.baseline_run_pk),
                        "candidate_run_id": by_pk.get(c.candidate_run_pk),
                        "result": c.result,
                    }
                    for c in checks
                ],
            }
        )
    audit = [
        {
            "actor": a.actor,
            "action": a.action,
            "target_type": a.target_type,
            "target_id": a.target_id,
            "details": a.details,
            "created_at": a.created_at.isoformat(),
        }
        for a in session.scalars(
            select(AuditEvent)
            .where(AuditEvent.workspace_id == ws.id)
            .order_by(AuditEvent.created_at)
        )
    ]
    return {
        "format": "vectraudit.workspace-export",
        "format_version": 1,
        "exported_at": utcnow().isoformat(),
        "workspace": {
            "slug": ws.slug,
            "exports_enabled": ws.exports_enabled,
            "name": ws.name,
            "plan": ws.plan,
            "created_at": ws.created_at.isoformat(),
        },
        "members": members,
        "projects": projects,
        "audit": audit,
    }


def _delete_project_rows(session: Session, project_ids: list[str]) -> dict[str, int]:
    """Delete in dependency order (baselines block deleting their runs)."""
    counts = {}
    for name, model, column in (
        ("checks", Check, Check.project_id),
        ("baselines", Baseline, Baseline.project_id),
        ("runs", Run, Run.project_id),
        ("runner_tokens", ApiToken, ApiToken.project_id),
    ):
        result = session.execute(delete(model).where(column.in_(project_ids)))
        counts[name] = int(getattr(result, "rowcount", 0) or 0)
    session.execute(delete(Project).where(Project.id.in_(project_ids)))
    return counts


def _confirm(given: str | None, expected: str, what: str) -> None:
    if given != expected:
        raise CloudError(
            422,
            "confirmation_required",
            f"Deleting a {what} can't be undone: confirm by sending its slug.",
            f'Send {{"confirm": "{expected}"}}.',
        )


def delete_project(session: Session, access: repo.Access, confirm: str | None) -> dict[str, int]:
    assert access.project is not None
    _confirm(confirm, access.project.slug, "project")
    counts = _delete_project_rows(session, [access.project.id])
    repo.audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "project.delete",
        "project",
        access.project.id,
        slug=access.project.slug,
        **counts,
    )
    return counts


def delete_workspace(session: Session, access: repo.Access, confirm: str | None) -> dict[str, int]:
    """Delete a workspace and everything in it. The record of the deletion is kept outside
    the workspace (its own audit log goes with it)."""
    ws = access.workspace
    _confirm(confirm, ws.slug, "workspace")
    project_ids = [p.id for p in repo.projects(session, ws)]
    counts = _delete_project_rows(session, project_ids) if project_ids else {}
    counts["projects"] = len(project_ids)
    counts["members"] = int(
        getattr(
            session.execute(delete(Membership).where(Membership.workspace_id == ws.id)),
            "rowcount",
            0,
        )
        or 0
    )
    session.execute(delete(ApiToken).where(ApiToken.workspace_id == ws.id))
    session.execute(delete(Invite).where(Invite.workspace_id == ws.id))
    session.execute(delete(AuditEvent).where(AuditEvent.workspace_id == ws.id))
    session.execute(delete(Workspace).where(Workspace.id == ws.id))
    repo.audit(
        session,
        None,
        access.principal.actor,
        "workspace.delete",
        "workspace",
        ws.id,
        slug=ws.slug,
        **counts,
    )
    return counts


def delete_user(session: Session, user: User, actor: str) -> list[str]:
    """Remove a person: memberships, tokens and sessions. Refuses if they are the last
    owner of a workspace. Returns the workspaces they left."""
    left = []
    for ws, role in repo.workspaces_of(session, user.id):
        if role == "owner":
            owners = session.scalar(
                select(func.count())
                .select_from(Membership)
                .where(Membership.workspace_id == ws.id, Membership.role == "owner")
            )
            if (owners or 0) <= 1:
                raise CloudError(
                    409,
                    "last_owner",
                    f"{user.email} is the last owner of {ws.slug}.",
                    "Make someone else an owner first, or delete the workspace.",
                )
        repo.audit(
            session,
            ws.id,
            actor,
            "member.remove",
            "user",
            user.id,
            email=user.email,
            reason="user deleted",
        )
        left.append(ws.slug)
    session.execute(delete(Membership).where(Membership.user_id == user.id))
    session.execute(delete(ApiToken).where(ApiToken.user_id == user.id))
    session.execute(delete(WebSession).where(WebSession.user_id == user.id))
    session.execute(delete(Invite).where(Invite.user_id == user.id))
    repo.audit(session, None, actor, "user.delete", "user", user.id, email=user.email)
    session.delete(user)
    return left


def purge_sessions(session: Session, older_than_days: int = 30) -> int:
    """Delete sessions that expired or were revoked more than ``older_than_days`` ago."""
    cutoff = utcnow() - timedelta(days=older_than_days)
    result = session.execute(
        delete(WebSession).where(
            (WebSession.expires_at < cutoff) | (WebSession.revoked_at < cutoff)
        )
    )
    return int(getattr(result, "rowcount", 0) or 0)
