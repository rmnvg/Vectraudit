"""Project policy, pinned baselines and server-side checks.

Baselines move only by an explicit, audited pin from an engineer or owner, never as a side
effect of a check, and only complete runs can be pinned. Checks reuse
:func:`compare_runs`, so the hosted verdict is the CLI's verdict.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from vectraudit.cloud import repo, service
from vectraudit.cloud.deps import PrincipalDep, SessionDep, SettingsDep
from vectraudit.cloud.errors import CloudError
from vectraudit.cloud.models import Baseline, Check, Run
from vectraudit.evaluation.policy import PolicyError, parse_policy

router = APIRouter(prefix="/api/v1/workspaces/{workspace}/projects/{project}", tags=["checks"])


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
    report_url: str


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
        report_url=f"{url}/report",
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
        "effective": service.project_policy(access.project).as_dict(),
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
        digest=service.digest(body),
    )
    return {
        "policy": body,
        "effective": service.project_policy(access.project).as_dict(),
        "is_default": False,
    }


# --- baselines ---------------------------------------------------------------------------


@router.get("/baselines")
def list_baselines(
    workspace: str, project: str, session: SessionDep, principal: PrincipalDep
) -> list[BaselineOut]:
    access = repo.authorize(session, principal, workspace, "read_baselines", project)
    assert access.project is not None
    rows = service.baselines(session, access.project)
    return [_baseline_out(b, r) for b, r in rows]


@router.get("/baselines/{name}")
def get_baseline(
    workspace: str, project: str, name: str, session: SessionDep, principal: PrincipalDep
) -> BaselineOut:
    access = repo.authorize(session, principal, workspace, "read_baselines", project)
    assert access.project is not None
    return _baseline_out(*service.get_baseline(session, access.project, name))


@router.get("/baselines/{name}/file")
def get_baseline_file(
    workspace: str, project: str, name: str, session: SessionDep, principal: PrincipalDep
) -> dict[str, Any]:
    """The pinned run's file, for comparing locally (`vectraudit check cloud:main ...`)."""
    access = repo.authorize(session, principal, workspace, "read_baselines", project)
    assert access.project is not None
    return service.get_baseline(session, access.project, name)[1].data


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
    return _baseline_out(*service.pin_baseline(session, access, name, body.run_id, body.note))


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
    check, base, cand, created = service.run_check(
        session, access, body.baseline, body.candidate_run_id
    )
    if not created:
        response.status_code = 200
    return _check_out(
        check, base, cand, _check_url(settings.public_url, workspace, project, check.id)
    )


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
    check, base, cand = service.load_check(session, access, check_id)
    return _check_out(
        check, base, cand, _check_url(settings.public_url, workspace, project, check.id)
    )
