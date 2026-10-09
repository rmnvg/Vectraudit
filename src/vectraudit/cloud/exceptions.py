"""Hosted, audited exceptions: grant, revoke, and apply them to checks.

The rules live in :mod:`vectraudit.evaluation.exceptions`; this module stores who accepted
which failing rule, why, for how long and how often, and keeps each check's stored verdict
and its decision-ledger row consistent with the exceptions that covered it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from vectraudit.cloud import ledger, repo
from vectraudit.cloud.errors import CloudError, forbidden, not_found
from vectraudit.cloud.models import Check, Decision, PolicyException, Project, User, utcnow
from vectraudit.evaluation.exceptions import (
    apply_exceptions,
    failing_rules,
    required_role,
    rule_kind,
)
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy, parse_policy

MAX_RULE = 1200
MAX_REASON = 2000
MAX_OWNER = 200
MAX_CHECKS = 1000


def _policy(project: Project) -> Policy:
    return DEFAULT_POLICY if project.policy is None else parse_policy(project.policy)


def status(exception: PolicyException, now: datetime | None = None) -> str:
    """``active``, ``revoked``, ``expired`` or ``used`` (its check allowance is spent)."""
    now = now or utcnow()
    if exception.revoked_at is not None:
        return "revoked"
    if exception.expires_at <= now:
        return "expired"
    if exception.max_checks is not None and exception.uses >= exception.max_checks:
        return "used"
    return "active"


def _actor(session: Session, actor: str) -> str:
    if actor.startswith("user:"):
        user = session.get(User, actor[5:])
        if user is not None:
            return user.email
    return actor


def payload(session: Session, exception: PolicyException) -> dict[str, Any]:
    """The form :func:`apply_exceptions` and API clients read."""
    return {
        "id": exception.id,
        "rule": exception.rule,
        "kind": rule_kind(exception.rule),
        "reason": exception.reason,
        "owner": exception.owner,
        "accepted_by": _actor(session, exception.created_by),
        "created_at": exception.created_at.isoformat(),
        "expires_at": exception.expires_at.isoformat(),
        "max_checks": exception.max_checks,
        "uses": exception.uses,
        "status": status(exception),
        "origin_check_id": exception.origin_check_id,
        "revoked_at": exception.revoked_at.isoformat() if exception.revoked_at else None,
        "revoked_by": _actor(session, exception.revoked_by) if exception.revoked_by else None,
    }


def listing(
    session: Session, project: Project, *, active_only: bool = False
) -> list[PolicyException]:
    rows = list(
        session.scalars(
            select(PolicyException)
            .where(PolicyException.project_id == project.id)
            .order_by(PolicyException.created_at.desc())
        )
    )
    return [e for e in rows if status(e) == "active"] if active_only else rows


def apply_to_check(session: Session, project: Project, check: Check) -> list[str]:
    """Re-evaluate ``check`` with active exceptions plus those that already covered it.

    Updates the stored result, verdict and ledger row; counts a use the first time an
    exception covers this check. Returns the newly applied exception IDs.
    """
    result = dict(check.result)
    before = {a["exception_id"] for a in result.get("exceptions_applied", [])}
    if "verdict_before_exceptions" in result:
        result["verdict"] = result.pop("verdict_before_exceptions")
    result["reasons"] = [
        {k: v for k, v in reason.items() if k != "accepted"} for reason in result["reasons"]
    ]
    rows = {e.id: e for e in listing(session, project, active_only=True)}
    for exception_id in before:
        earlier = session.get(PolicyException, exception_id)
        if earlier is not None:
            rows.setdefault(earlier.id, earlier)
    applied = apply_exceptions(
        result, [payload(session, e) for e in rows.values()], _policy(project)
    )
    newly = [a["exception_id"] for a in applied["exceptions_applied"]]
    newly = [i for i in newly if i not in before]
    if applied != check.result:
        check.result = applied
        check.verdict = applied["verdict"]
        decision = session.scalars(select(Decision).where(Decision.check_id == check.id)).first()
        if decision is not None:
            decision.verdict = check.verdict
            decision.summary = ledger.summarize_check(applied)
    for exception_id in newly:
        rows[exception_id].uses += 1
    return newly


def grant(
    session: Session,
    access: repo.Access,
    check_id: str,
    *,
    rule: str,
    reason: str,
    owner: str,
    expires_at: datetime,
    max_checks: int | None,
) -> PolicyException:
    """Accept ``rule`` as it failed in ``check_id`` (caller authorized ``grant_exception``)."""
    project = access.project
    assert project is not None
    check = session.scalars(
        select(Check).where(Check.project_id == project.id, Check.id == check_id)
    ).first()
    if check is None:
        raise not_found("Check")
    reason, owner = reason.strip(), owner.strip()
    if not reason or len(reason) > MAX_REASON:
        raise CloudError(422, "invalid_reason", "Give a reason of 1-2,000 characters.")
    if not owner or len(owner) > MAX_OWNER:
        raise CloudError(422, "invalid_owner", "Name who follows up (1-200 characters).")
    if max_checks is not None and not 1 <= max_checks <= MAX_CHECKS:
        raise CloudError(422, "invalid_max_checks", "max_checks must be 1-1,000.")
    if len(rule) > MAX_RULE or rule not in failing_rules(check.result):
        raise CloudError(
            422,
            "rule_not_failing",
            "That rule did not fail in this check.",
            "Use a rule key listed in the check's failing reasons.",
        )
    policy = _policy(project)
    need = required_role(policy, rule)
    if need is None:
        raise forbidden(
            f"The project policy does not allow exceptions for {rule_kind(rule)} rules.",
            "An owner can open rule kinds in the policy's exceptions block.",
        )
    if repo.ROLE_RANK.get(access.role, 0) < repo.ROLE_RANK[need]:
        raise forbidden(f"Accepting this rule needs the {need} role; you are {access.role}.")
    assert policy.exceptions is not None
    now = utcnow()
    if expires_at.tzinfo is None:
        raise CloudError(422, "invalid_expiry", "expires_at needs a time zone.")
    if not now < expires_at <= now + timedelta(days=policy.exceptions.max_days):
        raise CloudError(
            422,
            "invalid_expiry",
            f"Expiry must be in the future and within {policy.exceptions.max_days} days.",
        )
    if any(e.rule == rule for e in listing(session, project, active_only=True)):
        raise CloudError(409, "exception_exists", "An active exception already covers this rule.")
    exception = PolicyException(
        workspace_id=access.workspace.id,
        project_id=project.id,
        rule=rule,
        reason=reason,
        owner=owner,
        origin_check_id=check.id,
        created_by=access.principal.actor,
        expires_at=expires_at,
        max_checks=max_checks,
        uses=0,
    )
    session.add(exception)
    session.flush()
    repo.audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "exception.grant",
        "exception",
        exception.id,
        rule=rule,
        check_id=check.id,
        owner=owner,
        expires_at=expires_at.isoformat(),
        max_checks=max_checks,
    )
    apply_to_check(session, project, check)
    return exception


def revoke(session: Session, access: repo.Access, exception_id: str) -> PolicyException:
    """Stop an exception covering future checks; checks it covered keep their record."""
    assert access.project is not None
    exception = session.scalars(
        select(PolicyException).where(
            PolicyException.project_id == access.project.id, PolicyException.id == exception_id
        )
    ).first()
    if exception is None:
        raise not_found("Exception")
    if exception.revoked_at is None:
        exception.revoked_at = utcnow()
        exception.revoked_by = access.principal.actor
        repo.audit(
            session,
            access.workspace.id,
            access.principal.actor,
            "exception.revoke",
            "exception",
            exception.id,
            rule=exception.rule,
        )
    return exception
