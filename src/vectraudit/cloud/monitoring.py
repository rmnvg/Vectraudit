"""Monitor schedules: notice when a scheduled monitor stops reporting.

A failing monitor run raises an alert (:func:`vectraudit.cloud.lead.monitor_status`), but a
monitor whose job stopped, lost its credentials or can no longer reach the service creates
no run at all, and its last result keeps looking current. A project can declare how often a
monitor run is expected; :func:`evaluate` opens one staleness alert when a run is overdue
and resolves it when the next run arrives (or the schedule changes). :func:`sweep` runs it
for every scheduled project; ``vectraudit-cloud deliver-webhooks`` calls it on each pass.

The alert is keyed by the last run before the silence and the schedule's start, so a sweep
every minute still announces one outage once.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from vectraudit.cloud import lead, ledger, repo, webhooks
from vectraudit.cloud.models import MonitorSchedule, Project, Workspace, utcnow

MIN_EVERY = 5
MAX_EVERY = 31 * 24 * 60
MAX_GRACE = 7 * 24 * 60
DEFAULT_GRACE = 60


def schedule(session: Session, project: Project) -> MonitorSchedule | None:
    return session.get(MonitorSchedule, project.id)


def _last_run(session: Session, project: Project) -> dict[str, Any] | None:
    rows = ledger.history(session, project, limit=1, trigger="monitor")
    return rows[0] if rows else None


def state(session: Session, project: Project, now: datetime | None = None) -> dict[str, Any] | None:
    """The schedule, when the next run is due and whether it is overdue (``None``: none set)."""
    found = schedule(session, project)
    if found is None:
        return None
    last = _last_run(session, project)
    timing = lead.monitor_overdue(
        every_minutes=found.every_minutes,
        grace_minutes=found.grace_minutes,
        expected_since=found.expected_since,
        last_at=last["at"] if last else None,
        now=now or utcnow(),
    )
    return {
        "every_minutes": found.every_minutes,
        "grace_minutes": found.grace_minutes,
        "expected_since": found.expected_since,
        "updated_at": found.updated_at,
        "updated_by": found.updated_by,
        "due_at": timing["due_at"],
        "overdue": timing["overdue"],
        "stale_alert": found.stale_alert,
    }


def evaluate(
    session: Session,
    ws: Workspace,
    project: Project,
    *,
    now: datetime | None = None,
    resolved_by: str | None = None,
    resolution: str = "reported",
) -> str | None:
    """Open or resolve the project's staleness alert; returns ``opened``, ``resolved`` or None.

    ``resolved_by`` names the monitor check that ended the silence, and ``resolution`` says
    why an alert closes (``reported``, ``schedule_changed``; removal goes through
    :func:`clear`).
    """
    moment = now or utcnow()
    # Lock the row so a sweep and a monitor upload cannot both open or both resolve.
    found = session.get(MonitorSchedule, project.id, with_for_update=True)
    if found is None:
        return None
    last = _last_run(session, project)
    timing = lead.monitor_overdue(
        every_minutes=found.every_minutes,
        grace_minutes=found.grace_minutes,
        expected_since=found.expected_since,
        last_at=last["at"] if last else None,
        now=moment,
    )
    open_alert = found.stale_alert
    if timing["overdue"] and open_alert is None:
        last_id = last["check_id"] if last else None
        # The last run names the silence; the schedule start keeps a re-declared schedule
        # from being deduplicated against an alert it already closed.
        start = int(found.expected_since.timestamp())
        alert = {
            "alert_id": f"silent:{last_id or 'never'}-{start}",
            "opened_at": moment.isoformat(),
            "last_check_id": last_id,
            "last_run_at": last["at"].isoformat() if last else None,
            "due_at": timing["due_at"].isoformat(),
            "every_minutes": found.every_minutes,
        }
        found.stale_alert = alert
        key, payload = webhooks.stale_event("monitor.stale_opened", ws, project, alert)
        webhooks.notify(session, project, "monitor", key, payload)
        return "opened"
    if not timing["overdue"] and open_alert is not None:
        _resolve(session, ws, project, found, moment, resolved_by, resolution)
        return "resolved"
    return None


def _resolve(
    session: Session,
    ws: Workspace,
    project: Project,
    found: MonitorSchedule,
    moment: datetime,
    resolved_by: str | None,
    resolution: str,
) -> None:
    assert found.stale_alert is not None
    alert = {
        **found.stale_alert,
        "every_minutes": found.every_minutes,
        "resolved_at": moment.isoformat(),
        "resolved_by": resolved_by,
        "resolution": resolution,
    }
    found.stale_alert = None
    key, payload = webhooks.stale_event("monitor.stale_resolved", ws, project, alert)
    webhooks.notify(session, project, "monitor", key, payload)


def set_schedule(
    session: Session, access: repo.Access, every_minutes: int, grace_minutes: int
) -> dict[str, Any]:
    """Declare or change the expected cadence (audited); re-evaluates staleness at once."""
    assert access.project is not None
    moment = utcnow()
    found = session.get(MonitorSchedule, access.project.id, with_for_update=True)
    if found is None:
        found = MonitorSchedule(
            project_id=access.project.id,
            workspace_id=access.workspace.id,
            expected_since=moment,
        )
        session.add(found)
    found.every_minutes = every_minutes
    found.grace_minutes = grace_minutes
    found.updated_at = moment
    found.updated_by = access.principal.actor
    session.flush()
    repo.audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "monitor.schedule_set",
        "project",
        access.project.id,
        every_minutes=every_minutes,
        grace_minutes=grace_minutes,
    )
    evaluate(session, access.workspace, access.project, now=moment, resolution="schedule_changed")
    result = state(session, access.project, moment)
    assert result is not None
    return result


def clear(session: Session, access: repo.Access) -> bool:
    """Stop expecting monitor runs (audited); an open staleness alert is closed and announced."""
    assert access.project is not None
    found = session.get(MonitorSchedule, access.project.id, with_for_update=True)
    if found is None:
        return False
    if found.stale_alert is not None:
        _resolve(
            session, access.workspace, access.project, found, utcnow(), None, "schedule_removed"
        )
    session.delete(found)
    repo.audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "monitor.schedule_clear",
        "project",
        access.project.id,
    )
    return True


def sweep(sessions: sessionmaker[Session], now: datetime | None = None) -> dict[str, int]:
    """Evaluate every scheduled project once, each in its own transaction."""
    from vectraudit.cloud.db import transaction

    moment = now or utcnow()
    with transaction(sessions) as session:
        ids = list(session.scalars(select(MonitorSchedule.project_id)))
    counts = {"opened": 0, "resolved": 0}
    for project_id in ids:
        with transaction(sessions) as session:
            project = session.get(Project, project_id)
            ws = session.get(Workspace, project.workspace_id) if project is not None else None
            if project is None or ws is None:
                continue
            outcome = evaluate(session, ws, project, now=moment)
            if outcome:
                counts[outcome] += 1
    return counts
