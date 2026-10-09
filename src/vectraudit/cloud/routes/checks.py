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

from vectraudit.cloud import exceptions, repo, service
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
        proposed = parse_policy(body)
    except PolicyError as e:
        raise CloudError(422, "invalid_policy", str(e)) from e
    # Who may accept regressions is governance: only an owner may change it.
    if proposed.exceptions != service.project_policy(access.project).exceptions and (
        access.role != "owner"
    ):
        raise CloudError(
            403,
            "forbidden",
            "Changing the policy's exceptions block needs the owner role.",
            "Ask a workspace owner, or keep the current exceptions block.",
        )
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


# --- audited exceptions --------------------------------------------------------------------


class ExceptionIn(BaseModel):
    rule: str = Field(min_length=1, max_length=1200)
    """A failing reason's rule key in this check, e.g. ``metric:ndcg.max_drop``."""
    reason: str = Field(min_length=1, max_length=2000)
    owner: str = Field(min_length=1, max_length=200)
    """Who follows up on the accepted regression."""
    expires_at: datetime
    max_checks: int | None = Field(default=None, ge=1, le=1000)
    """Optional: stop after covering this many checks (including this one)."""


@router.post("/checks/{check_id}/exceptions", status_code=201)
def grant_exception(
    workspace: str,
    project: str,
    check_id: str,
    body: ExceptionIn,
    session: SessionDep,
    principal: PrincipalDep,
) -> dict[str, Any]:
    """Accept one failing rule of this check, as the project policy allows, until expiry."""
    access = repo.authorize(session, principal, workspace, "grant_exception", project)
    granted = exceptions.grant(
        session,
        access,
        check_id,
        rule=body.rule,
        reason=body.reason,
        owner=body.owner,
        expires_at=body.expires_at,
        max_checks=body.max_checks,
    )
    return exceptions.payload(session, granted)


@router.get("/exceptions")
def list_exceptions(
    workspace: str,
    project: str,
    session: SessionDep,
    principal: PrincipalDep,
    active: bool = False,
) -> list[dict[str, Any]]:
    """Exceptions newest first; ``active=true`` returns only those that apply to new checks."""
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    rows = exceptions.listing(session, access.project, active_only=active)
    return [exceptions.payload(session, e) for e in rows]


@router.post("/exceptions/{exception_id}/revoke")
def revoke_exception(
    workspace: str, project: str, exception_id: str, session: SessionDep, principal: PrincipalDep
) -> dict[str, Any]:
    """Stop an exception covering future checks; earlier decisions keep their record."""
    access = repo.authorize(session, principal, workspace, "grant_exception", project)
    return exceptions.payload(session, exceptions.revoke(session, access, exception_id))
