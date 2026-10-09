"""Plans, limits, usage and retention.

Limits come from the workspace's plan, overridden per workspace for custom pilot deals.
Uploads are checked against runs per calendar month (UTC), queries per run and stored
runs. Concurrency is not limited here: evaluation runs execute on the customer's runner,
not on this service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vectraudit.cloud.errors import CloudError
from vectraudit.cloud.models import Baseline, Check, Membership, Run, Workspace, utcnow


@dataclass(frozen=True)
class Limits:
    runs_per_month: int
    queries_per_run: int
    stored_runs: int
    retention_days: int


PLANS: dict[str, Limits] = {
    "pilot": Limits(
        runs_per_month=500, queries_per_run=2_000, stored_runs=1_000, retention_days=30
    ),
    "team": Limits(
        runs_per_month=5_000, queries_per_run=10_000, stored_runs=20_000, retention_days=90
    ),
    "enterprise": Limits(
        runs_per_month=100_000, queries_per_run=100_000, stored_runs=500_000, retention_days=365
    ),
}


def limits_for(ws: Workspace) -> Limits:
    base = PLANS.get(ws.plan, PLANS["pilot"])
    overrides = {
        k: int(v)
        for k, v in (ws.limits or {}).items()
        if k in {f.name for f in fields(Limits)} and isinstance(v, int) and v > 0
    }
    return replace(base, **overrides)


def month_start(now: datetime | None = None) -> datetime:
    now = now or utcnow()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _count(session: Session, stmt: Any) -> int:
    return int(session.scalar(stmt) or 0)


def usage(session: Session, ws: Workspace, since: datetime, until: datetime) -> dict[str, int]:
    """What a workspace used in [since, until), plus what it stores now."""
    in_window = (Run.workspace_id == ws.id, Run.uploaded_at >= since, Run.uploaded_at < until)
    return {
        "runs": _count(session, select(func.count()).select_from(Run).where(*in_window)),
        "queries": _count(
            session, select(func.coalesce(func.sum(Run.n_queries), 0)).where(*in_window)
        ),
        "checks": _count(
            session,
            select(func.count())
            .select_from(Check)
            .where(
                Check.workspace_id == ws.id, Check.created_at >= since, Check.created_at < until
            ),
        ),
        "stored_runs": _count(
            session, select(func.count()).select_from(Run).where(Run.workspace_id == ws.id)
        ),
        "members": _count(
            session,
            select(func.count()).select_from(Membership).where(Membership.workspace_id == ws.id),
        ),
    }


def status(session: Session, ws: Workspace) -> dict[str, Any]:
    """Limits and this month's usage, for the workspace's owners."""
    start = month_start()
    used = usage(session, ws, start, start + timedelta(days=32))
    lim = limits_for(ws)
    return {
        "plan": ws.plan,
        "limits": asdict(lim),
        "period_start": start.isoformat(),
        "usage": used,
        "remaining": {
            "runs_this_month": max(lim.runs_per_month - used["runs"], 0),
            "stored_runs": max(lim.stored_runs - used["stored_runs"], 0),
        },
        "oldest_run_expires_after_days": lim.retention_days,
    }


def check_upload(session: Session, ws: Workspace, n_queries: int) -> None:
    """Raise if this upload would exceed the workspace's limits."""
    lim = limits_for(ws)
    if n_queries > lim.queries_per_run:
        raise CloudError(
            413,
            "too_many_queries",
            f"{n_queries:,} queries exceeds your plan's {lim.queries_per_run:,} per run.",
            "Split the dataset, or ask us about a larger plan.",
        )
    start = month_start()
    runs = _count(
        session,
        select(func.count())
        .select_from(Run)
        .where(Run.workspace_id == ws.id, Run.uploaded_at >= start),
    )
    if runs >= lim.runs_per_month:
        raise CloudError(
            429,
            "quota_exceeded",
            f"This workspace has used its {lim.runs_per_month:,} runs for this month.",
            "The quota resets on the 1st (UTC); ask us to raise it if you need more.",
        )
    stored = _count(session, select(func.count()).select_from(Run).where(Run.workspace_id == ws.id))
    if stored >= lim.stored_runs:
        raise CloudError(
            429,
            "storage_full",
            f"This workspace stores its maximum of {lim.stored_runs:,} runs.",
            "Old runs are purged after the retention period; ask us to raise the limit.",
        )


def purge_expired(session: Session, now: datetime | None = None) -> dict[str, int]:
    """Delete runs older than their workspace's retention; pinned baselines are kept.

    Checks that reference a deleted run go with it, after their decisions are written to
    the ledger, which outlives retention. Returns runs deleted per workspace slug.
    """
    from vectraudit.cloud.ledger import record_missing

    now = now or utcnow()
    pinned = select(Baseline.run_pk)
    deleted: dict[str, int] = {}
    for ws in session.scalars(select(Workspace)):
        cutoff = now - timedelta(days=limits_for(ws).retention_days)
        expired = list(
            session.scalars(
                select(Run).where(
                    Run.workspace_id == ws.id, Run.uploaded_at < cutoff, Run.id.not_in(pinned)
                )
            )
        )
        for run in expired:
            checks = list(
                session.scalars(
                    select(Check).where(
                        (Check.baseline_run_pk == run.id) | (Check.candidate_run_pk == run.id)
                    )
                )
            )
            record_missing(session, checks)
            for check in checks:
                session.delete(check)
            session.delete(run)
        if expired:
            deleted[ws.slug] = len(expired)
    return deleted


def usage_rows(session: Session, year: int, month: int) -> list[dict[str, Any]]:
    """One row per workspace for a calendar month (UTC), for manual invoicing."""
    since = datetime(year, month, 1, tzinfo=timezone.utc)
    until = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)
    rows = []
    for ws in session.scalars(select(Workspace).order_by(Workspace.slug)):
        rows.append(
            {
                "workspace": ws.slug,
                "plan": ws.plan,
                "month": f"{year}-{month:02d}",
                **usage(session, ws, since, until),
            }
        )
    return rows
