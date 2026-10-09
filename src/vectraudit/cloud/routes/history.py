"""Run and check history, private HTML reports, and signed share links.

Reports are served to authenticated workspace members (and the project's runner). Share
links are off unless the operator enables them (``VECTRAUDIT_CLOUD_SHARE_LINKS``); each
is an HMAC over the check id and an expiry, so it can't be forged or extended, and
rotating the secret key revokes them all.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import ColumnElement, and_, or_, select
from sqlalchemy.orm import aliased

from vectraudit.cloud import lead, ledger, repo
from vectraudit.cloud.deps import PrincipalDep, SessionDep, SettingsDep
from vectraudit.cloud.errors import CloudError, not_found
from vectraudit.cloud.models import Check, Run, utcnow
from vectraudit.cloud.routes.runs import RunOut, run_out
from vectraudit.cloud.settings import Settings
from vectraudit.evaluation.report import render_html

router = APIRouter(tags=["history"])
PROJECT = "/api/v1/workspaces/{workspace}/projects/{project}"
MAX_SHARE_DAYS = 30
REPORT_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
    "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "private, no-store",
}


class Page(BaseModel):
    next_cursor: str | None


class RunPage(Page):
    runs: list[RunOut]


class CheckSummary(BaseModel):
    id: str
    verdict: str
    baseline_run_id: str
    candidate_run_id: str
    candidate_label: str | None
    commit_sha: str | None
    created_at: datetime
    reasons: list[str]
    report_url: str


class CheckPage(Page):
    checks: list[CheckSummary]


class TimelineEntry(BaseModel):
    kind: str
    """``decision`` (a release check, kept after retention) or ``baseline_pin``."""
    at: datetime
    verdict: str | None = None
    baseline_name: str | None = None
    baseline_run_id: str | None = None
    candidate_run_id: str | None = None
    candidate_label: str | None = None
    candidate_commit: str | None = None
    policy_digest: str | None = None
    decided_by: str | None = None
    trigger: str | None = None
    """For decisions: ``release`` or ``monitor`` (a scheduled run)."""
    check_id: str | None = None
    evidence_available: bool | None = None
    """Whether the full check and its report still exist (False once retention purged it)."""
    report_url: str | None = None
    summary: dict[str, Any] | None = None
    run_id: str | None = None
    previous_run_id: str | None = None
    note: str | None = None
    actor: str | None = None
    action: str | None = None
    """For ``exception`` entries: ``granted`` or ``revoked``."""
    rule: str | None = None
    expires_at: str | None = None
    owner: str | None = None


class TimelinePage(Page):
    entries: list[TimelineEntry]


class ShareIn(BaseModel):
    days: int = Field(default=7, ge=1, le=MAX_SHARE_DAYS)


class ShareOut(BaseModel):
    url: str
    expires_at: datetime


def _encode_cursor(at: datetime, row_id: str) -> str:
    return base64.urlsafe_b64encode(f"{at.isoformat()}|{row_id}".encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        at, row_id = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        parsed = datetime.fromisoformat(at)
    except (ValueError, UnicodeDecodeError) as e:
        raise CloudError(422, "invalid_cursor", "The cursor is not valid.") from e
    return parsed, row_id


def _after(column: Any, id_column: Any, cursor: str | None) -> ColumnElement[bool] | None:
    """Keyset pagination: rows strictly after the cursor in (time desc, id desc) order."""
    if not cursor:
        return None
    at, row_id = _decode_cursor(cursor)
    condition: ColumnElement[bool] = or_(column < at, and_(column == at, id_column < row_id))
    return condition


@router.get(f"{PROJECT}/runs")
def list_runs(
    workspace: str,
    project: str,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
    status: str | None = None,
    backend: str | None = None,
    commit: str | None = Query(default=None, max_length=64),
    dataset: str | None = Query(default=None, max_length=64),
    label: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = None,
) -> RunPage:
    """Runs newest first, filtered by status, backend, commit (prefix), dataset or label."""
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    stmt = select(Run).where(
        Run.workspace_id == access.workspace.id, Run.project_id == access.project.id
    )
    if status:
        stmt = stmt.where(Run.status == status)
    if backend:
        stmt = stmt.where(Run.backend == backend)
    if commit:
        stmt = stmt.where(Run.commit_sha.startswith(commit, autoescape=True))
    if dataset:
        stmt = stmt.where(Run.dataset_sha256.startswith(dataset, autoescape=True))
    if label:
        stmt = stmt.where(Run.label == label)
    after = _after(Run.uploaded_at, Run.id, cursor)
    if after is not None:
        stmt = stmt.where(after)
    rows = list(
        session.scalars(stmt.order_by(Run.uploaded_at.desc(), Run.id.desc()).limit(limit + 1))
    )
    more = len(rows) > limit
    rows = rows[:limit]
    return RunPage(
        runs=[run_out(r, access.workspace, project, settings.public_url) for r in rows],
        next_cursor=_encode_cursor(rows[-1].uploaded_at, rows[-1].id) if more else None,
    )


def report_url(settings: Settings, workspace: str, project: str, check_id: str) -> str:
    return (
        f"{settings.public_url}{PROJECT.format(workspace=workspace, project=project)}"
        f"/checks/{check_id}/report"
    )


@router.get(f"{PROJECT}/checks")
def list_checks(
    workspace: str,
    project: str,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
    verdict: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = None,
) -> CheckPage:
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    base, cand = aliased(Run), aliased(Run)
    stmt = (
        select(Check, base.run_id, cand.run_id, cand.label, cand.commit_sha)
        .join(base, base.id == Check.baseline_run_pk)
        .join(cand, cand.id == Check.candidate_run_pk)
        .where(Check.workspace_id == access.workspace.id, Check.project_id == access.project.id)
    )
    if verdict:
        stmt = stmt.where(Check.verdict == verdict)
    after = _after(Check.created_at, Check.id, cursor)
    if after is not None:
        stmt = stmt.where(after)
    rows = session.execute(
        stmt.order_by(Check.created_at.desc(), Check.id.desc()).limit(limit + 1)
    ).all()
    more = len(rows) > limit
    rows = rows[:limit]
    return CheckPage(
        checks=[
            CheckSummary(
                id=c.id,
                verdict=c.verdict,
                baseline_run_id=b_id,
                candidate_run_id=c_id,
                candidate_label=c_label,
                commit_sha=c_commit,
                created_at=c.created_at,
                reasons=[r["message"] for r in c.result.get("reasons", [])],
                report_url=report_url(settings, workspace, project, c.id),
            )
            for c, b_id, c_id, c_label, c_commit in rows
        ],
        next_cursor=_encode_cursor(rows[-1][0].created_at, rows[-1][0].id) if more else None,
    )


@router.get(f"{PROJECT}/decisions")
def list_decisions(
    workspace: str,
    project: str,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = None,
) -> TimelinePage:
    """Release decisions and baseline pins, newest first; kept after runs expire."""
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    before = _decode_cursor(cursor)[0] if cursor else None
    entries = ledger.timeline(session, access.project, before=before, limit=limit + 1)
    more = len(entries) > limit
    entries = entries[:limit]
    for entry in entries:
        if entry.get("evidence_available"):
            entry["report_url"] = report_url(settings, workspace, project, entry["check_id"])
    return TimelinePage(
        entries=[TimelineEntry(**entry) for entry in entries],
        next_cursor=_encode_cursor(entries[-1]["at"], entries[-1]["kind"]) if more else None,
    )


class LeadSummary(BaseModel):
    """A release decision in plain language, for an engineering lead."""

    check_id: str | None
    verdict: str
    trigger: str
    """``release`` or ``monitor`` (a scheduled run of an unchanged setup)."""
    candidate: str
    headline: str
    action: str
    changes: list[str]
    """What changed: quality metrics, p95 latency and recorded settings."""
    matters: list[str]
    """Why it matters: failing rules, accepted exceptions, critical and regressed questions."""
    uncertain: list[str]
    """What this evidence cannot show: noise-sized changes, small samples, expired reports."""
    evidence_available: bool | None
    report_url: str | None = None
    markdown: str


@router.get(f"{PROJECT}/decisions/{{check_id}}/summary")
def decision_summary(
    workspace: str,
    project: str,
    check_id: str,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> LeadSummary:
    """One release decision in plain language; available after its runs expire."""
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    entry = ledger.decision(session, access.project, check_id)
    if entry is None:
        raise not_found("Decision")
    url = None
    if entry["evidence_available"]:
        url = report_url(settings, workspace, project, check_id)
    return LeadSummary(**lead.explain(entry), report_url=url)


class TrendPoint(BaseModel):
    at: datetime
    check_id: str | None
    verdict: str
    trigger: str
    """``release`` or ``monitor``."""
    candidate: str
    policy_digest: str | None
    value: float
    baseline: float | None
    ci_low: float | None = None
    ci_high: float | None = None


class TrendMetric(BaseModel):
    label: str
    points: list[TrendPoint]
    change: float
    """Latest minus first candidate value in this series."""


class TrendLatency(BaseModel):
    points: list[TrendPoint]
    change: float | None


class TrendSeries(BaseModel):
    """Checks against one benchmark version; a new version starts a new series."""

    dataset_sha256: str
    decisions: int
    first_at: datetime
    last_at: datetime
    primary_metric: str | None
    verdicts: dict[str, int]
    metrics: dict[str, TrendMetric]
    latency_p95: TrendLatency
    policy_changes: list[datetime]


class TrendsOut(BaseModel):
    series: list[TrendSeries]
    decisions: int
    """Decisions read (the newest ``limit``)."""


@router.get(f"{PROJECT}/trends")
def trends(
    workspace: str,
    project: str,
    session: SessionDep,
    principal: PrincipalDep,
    limit: int = Query(default=200, ge=2, le=1000),
) -> TrendsOut:
    """Candidate quality and p95 latency per benchmark version, from the decision ledger."""
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    rows = ledger.history(session, access.project, limit=limit)
    return TrendsOut(series=[TrendSeries(**s) for s in lead.trends(rows)], decisions=len(rows))


class MonitorRun(BaseModel):
    at: datetime
    check_id: str | None
    verdict: str


class MonitorAlert(BaseModel):
    alert_id: str | None
    """Check ID of the run that opened the alert; stable until a run passes."""
    opened_at: datetime
    failing_runs: int
    complete: bool
    """False if the streak reaches the oldest monitor decision read, so it may be longer."""
    resolved_at: datetime | None
    resolved_by: str | None


class MonitorStatus(BaseModel):
    """Scheduled-monitor state, derived from monitor decisions in the ledger."""

    state: Literal["none", "healthy", "alerting"]
    last: MonitorRun | None
    alert: MonitorAlert | None
    """The open alert while alerting; the most recently resolved one while healthy."""


MONITOR_WINDOW = 500


@router.get(f"{PROJECT}/monitor")
def monitor_status(
    workspace: str, project: str, session: SessionDep, principal: PrincipalDep
) -> MonitorStatus:
    """Whether scheduled monitor runs currently alert, since when, and what resolved the last
    alert. Read from the decision ledger, so it outlives run retention."""
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    rows = ledger.history(session, access.project, limit=MONITOR_WINDOW, trigger="monitor")
    return MonitorStatus(**lead.monitor_status(rows))


def _check(session: SessionDep, workspace_id: str, project_id: str, check_id: str) -> Check:
    check = session.scalars(
        select(Check).where(
            Check.workspace_id == workspace_id,
            Check.project_id == project_id,
            Check.id == check_id,
        )
    ).first()
    if check is None:
        raise not_found("Check")
    return check


@router.get(f"{PROJECT}/checks/{{check_id}}/report", response_class=HTMLResponse)
def check_report(
    workspace: str, project: str, check_id: str, session: SessionDep, principal: PrincipalDep
) -> HTMLResponse:
    """The investigation report, for workspace members and the project's runner."""
    access = repo.authorize(session, principal, workspace, "read", project)
    assert access.project is not None
    check = _check(session, access.workspace.id, access.project.id, check_id)
    return HTMLResponse(render_html(check.result), headers=REPORT_HEADERS)


def _signature(secret: str, check_id: str, expires: int) -> str:
    mac = hmac.new(secret.encode(), f"share:{check_id}:{expires}".encode(), hashlib.sha256)
    return base64.urlsafe_b64encode(mac.digest()).decode().rstrip("=")


@router.post(f"{PROJECT}/checks/{{check_id}}/share", status_code=201)
def share_check(
    workspace: str,
    project: str,
    check_id: str,
    body: ShareIn,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> ShareOut:
    """A signed, expiring link that opens the report without signing in."""
    if not settings.share_links:
        raise CloudError(
            403,
            "share_links_disabled",
            "Public report links are disabled on this service.",
            "An operator can enable them with VECTRAUDIT_CLOUD_SHARE_LINKS=true.",
        )
    access = repo.authorize(session, principal, workspace, "manage_project", project)
    assert access.project is not None
    check = _check(session, access.workspace.id, access.project.id, check_id)
    expires_at = (utcnow() + timedelta(days=body.days)).replace(microsecond=0)
    expires = int(expires_at.timestamp())
    sig = _signature(settings.secret_key, check.id, expires)
    repo.audit(
        session,
        access.workspace.id,
        principal.actor,
        "report.share",
        "check",
        check.id,
        expires_at=expires_at.isoformat(),
    )
    return ShareOut(
        url=f"{settings.public_url}/r/{check.id}?e={expires}&s={sig}", expires_at=expires_at
    )


@router.get("/r/{check_id}", response_class=HTMLResponse, include_in_schema=False)
def shared_report(
    check_id: str, session: SessionDep, settings: SettingsDep, e: int = 0, s: str = ""
) -> HTMLResponse:
    """Open a shared report. Invalid, expired or disabled links all look the same."""
    valid = (
        settings.share_links
        and e > int(datetime.now(timezone.utc).timestamp())
        and hmac.compare_digest(s, _signature(settings.secret_key, check_id, e))
    )
    check = session.get(Check, check_id) if valid else None
    if check is None:
        raise not_found("Report")
    return HTMLResponse(render_html(check.result), headers=REPORT_HEADERS)
