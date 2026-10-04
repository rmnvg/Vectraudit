"""Operations shared by the JSON API and the dashboard, so both enforce the same rules."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from vectraudit.cloud import repo
from vectraudit.cloud.errors import CloudError, not_found
from vectraudit.cloud.models import Baseline, Check, Project, Run, utcnow
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
