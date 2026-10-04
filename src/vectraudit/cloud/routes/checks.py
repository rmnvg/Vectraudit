"""Project policy, pinned baselines and server-side checks.

Baselines move only by an explicit, audited pin from an engineer or owner, never as a side
effect of a check, and only complete runs can be pinned. Checks reuse
:func:`compare_runs`, so the hosted verdict is the CLI's verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from vectraudit.cloud import repo
from vectraudit.cloud.deps import PrincipalDep, SessionDep, SettingsDep
from vectraudit.cloud.errors import CloudError, not_found
from vectraudit.cloud.models import Baseline, Check, Project, Run, utcnow
from vectraudit.cloud.routes.runs import find_run
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy, PolicyError, parse_policy

router = APIRouter(prefix="/api/v1/workspaces/{workspace}/projects/{project}", tags=["checks"])
BASELINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class BaselineIn(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=2000)


class BaselineOut(BaseModel):
    name: str
    run_id: str
    run_label: str | None
    commit_sha: str | None
    pinned_by: str
    pinned_at: datetime
    note: str


class CheckIn(BaseModel):
    candidate_run_id: str = Field(min_length=1, max_length=64)
    baseline: str = Field(default="main", min_length=1, max_length=100)
    """Name of the pinned baseline to compare against."""


class CheckOut(BaseModel):
    id: str
    verdict: str
    baseline_run_id: str
    candidate_run_id: str
    created_at: datetime
    created_by: str | None
    result: dict[str, Any]
    url: str


def _policy(project: Project) -> Policy:
    return DEFAULT_POLICY if project.policy is None else parse_policy(project.policy)


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _baseline_out(b: Baseline, run: Run) -> BaselineOut:
    return BaselineOut(
        name=b.name,
        run_id=run.run_id,
        run_label=run.label,
        commit_sha=run.commit_sha,
        pinned_by=b.pinned_by,
        pinned_at=b.pinned_at,
        note=b.note,
    )


def _check_out(c: Check, base: Run, cand: Run, url: str) -> CheckOut:
    return CheckOut(
        id=c.id,
        verdict=c.verdict,
        baseline_run_id=base.run_id,
        candidate_run_id=cand.run_id,
        created_at=c.created_at,
        created_by=c.created_by,
        result=c.result,
        url=url,
    )


# --- policy ------------------------------------------------------------------------------


@router.get("/policy")
def get_policy(
    workspace: str, project: str, session: SessionDep, principal: PrincipalDep
) -> dict[str, Any]:
    """The project's policy (the built-in default when none is set)."""
    access = repo.authorize(session, principal, workspace, "read_baselines", project)
    assert access.project is not None
    return {
        "policy": access.project.policy,
        "effective": _policy(access.project).as_dict(),
        "is_default": access.project.policy is None,
    }


@router.put("/policy")
def set_policy(
    workspace: str,
    project: str,
    body: dict[str, Any],
    session: SessionDep,
    principal: PrincipalDep,
) -> dict[str, Any]:
    access = repo.authorize(session, principal, workspace, "manage_project", project)
    assert access.project is not None
    try:
        parse_policy(body)
    except PolicyError as e:
        raise CloudError(422, "invalid_policy", str(e)) from e
    access.project.policy = body
    repo.audit(
        session,
        access.workspace.id,
        principal.actor,
        "policy.set",
        "project",
        access.project.id,
        digest=_digest(body),
    )
    return {"policy": body, "effective": _policy(access.project).as_dict(), "is_default": False}


# --- baselines ---------------------------------------------------------------------------


def _baseline(session: SessionDep, project: Project, name: str) -> tuple[Baseline, Run]:
    row = session.execute(
        select(Baseline, Run)
        .join(Run, Run.id == Baseline.run_pk)
        .where(Baseline.project_id == project.id, Baseline.name == name)
    ).first()
    if row is None:
        raise not_found(f"Baseline {name!r}")
    return row[0], row[1]


@router.get("/baselines")
def list_baselines(
    workspace: str, project: str, session: SessionDep, principal: PrincipalDep
) -> list[BaselineOut]:
    access = repo.authorize(session, principal, workspace, "read_baselines", project)
    assert access.project is not None
    rows = session.execute(
        select(Baseline, Run)
        .join(Run, Run.id == Baseline.run_pk)
        .where(Baseline.project_id == access.project.id)
        .order_by(Baseline.name)
    ).all()
    return [_baseline_out(b, r) for b, r in rows]


@router.get("/baselines/{name}")
def get_baseline(
    workspace: str, project: str, name: str, session: SessionDep, principal: PrincipalDep
) -> BaselineOut:
    access = repo.authorize(session, principal, workspace, "read_baselines", project)
    assert access.project is not None
    return _baseline_out(*_baseline(session, access.project, name))


@router.get("/baselines/{name}/file")
def get_baseline_file(
    workspace: str, project: str, name: str, session: SessionDep, principal: PrincipalDep
) -> dict[str, Any]:
    """The pinned run's file, for comparing locally (`vectraudit check cloud:main ...`)."""
    access = repo.authorize(session, principal, workspace, "read_baselines", project)
    assert access.project is not None
    return _baseline(session, access.project, name)[1].data


@router.put("/baselines/{name}")
def pin_baseline(
    workspace: str,
    project: str,
    name: str,
    body: BaselineIn,
    session: SessionDep,
    principal: PrincipalDep,
) -> BaselineOut:
    access = repo.authorize(session, principal, workspace, "pin_baseline", project)
    assert access.project is not None
    if not BASELINE_NAME.fullmatch(name):
        raise CloudError(
            422,
            "invalid_name",
            "Baseline names are 1-100 letters, digits, '.', '_' or '-'.",
            "Use e.g. main or release-2026.10 (no slashes).",
        )
    run = find_run(session, access.workspace.id, access.project.id, body.run_id)
    if run.status != "complete":
        raise CloudError(
            409,
            "run_not_complete",
            f"Run {run.run_id} is {run.status}; only complete runs can be baselines.",
        )
    existing = session.scalars(
        select(Baseline).where(Baseline.project_id == access.project.id, Baseline.name == name)
    ).first()
    previous = None
    if existing is None:
        existing = Baseline(
            workspace_id=access.workspace.id,
            project_id=access.project.id,
            name=name,
            run_pk=run.id,
            pinned_by=principal.actor,
            note=body.note,
        )
        session.add(existing)
    else:
        previous = existing.run_pk
        existing.run_pk = run.id
        existing.pinned_by = principal.actor
        existing.pinned_at = utcnow()
        existing.note = body.note
    session.flush()
    old_run = session.get(Run, previous) if previous else None
    repo.audit(
        session,
        access.workspace.id,
        principal.actor,
        "baseline.pin",
        "baseline",
        existing.id,
        name=name,
        run_id=run.run_id,
        previous_run_id=old_run.run_id if old_run else None,
        note=body.note,
    )
    return _baseline_out(existing, run)


# --- checks ------------------------------------------------------------------------------


def _check_url(public_url: str, workspace: str, project: str, check_id: str) -> str:
    return f"{public_url}/api/v1/workspaces/{workspace}/projects/{project}/checks/{check_id}"


@router.post("/checks", status_code=201)
def create_check(
    workspace: str,
    project: str,
    body: CheckIn,
    response: Response,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> CheckOut:
    """Compare an uploaded run with a pinned baseline under the project policy."""
    access = repo.authorize(session, principal, workspace, "write_checks", project)
    assert access.project is not None
    _, base = _baseline(session, access.project, body.baseline)
    cand = find_run(session, access.workspace.id, access.project.id, body.candidate_run_id)
    policy = _policy(access.project)
    policy_digest = _digest(policy.as_dict())

    def url(c: Check) -> str:
        return _check_url(settings.public_url, workspace, project, c.id)

    found = session.scalars(
        select(Check).where(
            Check.project_id == access.project.id,
            Check.baseline_run_pk == base.id,
            Check.candidate_run_pk == cand.id,
            Check.policy_digest == policy_digest,
        )
    ).first()
    if found is not None:
        response.status_code = 200
        return _check_out(found, base, cand, url(found))
    result = compare_runs(base.data, cand.data, policy)
    check = Check(
        workspace_id=access.workspace.id,
        project_id=access.project.id,
        baseline_run_pk=base.id,
        candidate_run_pk=cand.id,
        policy_digest=policy_digest,
        verdict=result["verdict"],
        result=result,
        created_by=principal.actor,
    )
    try:
        with session.begin_nested():
            session.add(check)
    except IntegrityError:
        winner = session.scalars(
            select(Check).where(
                Check.project_id == access.project.id,
                Check.baseline_run_pk == base.id,
                Check.candidate_run_pk == cand.id,
                Check.policy_digest == policy_digest,
            )
        ).one()
        response.status_code = 200
        return _check_out(winner, base, cand, url(winner))
    return _check_out(check, base, cand, url(check))


def _load_check(session: SessionDep, project: Project, check_id: str) -> tuple[Check, Run, Run]:
    check = session.scalars(
        select(Check).where(Check.project_id == project.id, Check.id == check_id)
    ).first()
    if check is None:
        raise not_found("Check")
    base = session.get(Run, check.baseline_run_pk)
    cand = session.get(Run, check.candidate_run_pk)
    assert base is not None
    assert cand is not None
    return check, base, cand


@router.get("/checks/{check_id}")
def get_check(
    workspace: str,
    project: str,
    check_id: str,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> CheckOut:
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    check, base, cand = _load_check(session, access.project, check_id)
    return _check_out(
        check, base, cand, _check_url(settings.public_url, workspace, project, check.id)
    )
