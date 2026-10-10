"""Monitor schedules: a scheduled monitor that stops reporting raises one alert, then resolves."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy")

from typer.testing import CliRunner

from tests.test_cloud_checks import pin, via
from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import sign_in
from tests.test_cloud_webhooks import Recorder, add_hook, deliver, public_dns  # noqa: F401
from vectraudit.cloud import lead, monitoring
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import AuditEvent, Decision, MonitorSchedule, WebhookDelivery
from vectraudit.integrations import cloud as cloud_client

P = "/api/v1/workspaces/acme/projects/search"
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def test_overdue_counts_from_the_last_run_or_the_schedule_start():
    timing = dict(every_minutes=60, grace_minutes=15, expected_since=T0)
    fresh = lead.monitor_overdue(
        **timing, last_at=T0 + timedelta(hours=2), now=T0 + timedelta(hours=3)
    )
    assert fresh["due_at"] == T0 + timedelta(hours=3, minutes=15)
    assert fresh["overdue"] is False
    late = lead.monitor_overdue(
        **timing, last_at=T0 + timedelta(hours=2), now=T0 + timedelta(hours=3, minutes=16)
    )
    assert late["overdue"] is True
    # Silence before the schedule existed does not count against it.
    before = lead.monitor_overdue(
        **timing, last_at=T0 - timedelta(days=9), now=T0 + timedelta(minutes=70)
    )
    assert (before["reference"], before["overdue"]) == (T0, False)
    never = lead.monitor_overdue(**timing, last_at=None, now=T0 + timedelta(minutes=76))
    assert never["overdue"] is True
    assert [lead.interval_label(m) for m in (1440, 2880, 360, 90)] == [
        "1 day",
        "2 days",
        "6 h",
        "90 min",
    ]


def schedule(world, every=60, grace=15, who="acme_engineer"):
    return world.client.put(
        f"{P}/monitor/schedule",
        json={"every_minutes": every, "grace_minutes": grace},
        headers=world.as_(who),
    )


def status(world):
    return world.client.get(f"{P}/monitor", headers=world.as_("acme_viewer")).json()


def backdate(cloud_app, hours: float) -> None:
    """Move the project's monitor history and schedule start into the past."""
    shift = timedelta(hours=hours)
    with transaction(cloud_app.state.sessions) as s:
        for decision in s.query(Decision):
            decision.decided_at -= shift
        for found in s.query(MonitorSchedule):
            found.expected_since -= shift


def monitor_run(world, run_id: str) -> dict:
    client = client_for(world, "acme_runner")
    client.upload_run(a_run(run_id, label="nightly"))
    return client.create_check(run_id, trigger="monitor")


@pytest.fixture
def monitored(world):
    """acme/search with a pinned baseline and one passing monitor run."""
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    return monitor_run(world, "night-1")


def test_engineers_declare_the_schedule_and_it_is_audited(world, cloud_app, monitored):
    assert status(world)["schedule"] is None
    assert schedule(world, who="acme_viewer").status_code == 403
    assert schedule(world, every=1).status_code == 422
    assert schedule(world, grace=-1).status_code == 422
    saved = schedule(world, every=1440, grace=60)
    assert saved.status_code == 200, saved.text
    body = saved.json()["schedule"]
    assert (body["every_minutes"], body["grace_minutes"], body["overdue"]) == (1440, 60, False)
    assert body["stale_alert"] is None
    assert status(world)["schedule"]["due_at"] == body["due_at"]
    assert (
        world.client.delete(f"{P}/monitor/schedule", headers=world.as_("acme_viewer")).status_code
        == 403
    )
    assert (
        world.client.delete(f"{P}/monitor/schedule", headers=world.as_("acme_engineer")).status_code
        == 204
    )
    assert (
        world.client.delete(f"{P}/monitor/schedule", headers=world.as_("acme_engineer")).status_code
        == 404
    )
    with transaction(cloud_app.state.sessions) as s:
        actions = [
            a for (a,) in s.query(AuditEvent.action).filter(AuditEvent.action.like("monitor.%"))
        ]
    assert sorted(actions) == ["monitor.schedule_clear", "monitor.schedule_set"]
    # Other tenants cannot see or change it.
    assert world.client.get(f"{P}/monitor", headers=world.as_("globex_owner")).status_code == 404
    assert schedule(world, who="globex_owner").status_code == 404


def test_a_silent_monitor_alerts_once_and_the_next_run_resolves_it(world, cloud_app, monitored):
    add_hook(world, events=["monitor"], style="slack")
    decisions_only = add_hook(world, events=["decision"]).json()
    assert schedule(world, every=60, grace=15).status_code == 200
    sender = Recorder()
    sessions = cloud_app.state.sessions
    assert monitoring.sweep(sessions) == {"opened": 0, "resolved": 0}  # on time
    backdate(cloud_app, hours=2)
    assert status(world)["schedule"]["overdue"] is True
    assert monitoring.sweep(sessions) == {"opened": 1, "resolved": 0}
    assert monitoring.sweep(sessions) == {"opened": 0, "resolved": 0}  # announced once
    deliver(cloud_app, sender)
    assert len(sender.calls) == 1
    text = json.loads(sender.calls[0][1])["text"]
    assert text.startswith("*Monitor overdue: No scheduled monitor run for acme/search since ")
    assert "expected every 1 h" in text
    assert "<https://va.example.com/app/acme/search|Open in Vectraudit>" in text
    opened_key = sender.calls[0][2]["X-Vectraudit-Event-Key"]
    assert opened_key.startswith(f"monitor.stale_opened:silent:{monitored['id']}-")
    alert = status(world)["schedule"]["stale_alert"]
    assert alert["last_check_id"] == monitored["id"]

    resumed = monitor_run(world, "night-2")
    deliver(cloud_app, sender)
    assert len(sender.calls) == 2  # the resolution, and no alert for a passing run
    resolved = json.loads(sender.calls[1][1])["text"]
    assert resolved.startswith(
        "*Monitor overdue resolved: Scheduled monitor for acme/search reported again."
    )
    assert sender.calls[1][2]["X-Vectraudit-Event-Key"] == opened_key.replace(
        "stale_opened", "stale_resolved"
    )
    now = status(world)["schedule"]
    assert (now["overdue"], now["stale_alert"]) == (False, None)
    with transaction(sessions) as s:
        assert (
            s.query(WebhookDelivery)
            .filter(WebhookDelivery.webhook_id == decisions_only["id"])
            .count()
            == 0
        )

    # A second silence is a new alert, keyed by the run before it.
    backdate(cloud_app, hours=2)
    assert monitoring.sweep(sessions) == {"opened": 1, "resolved": 0}
    deliver(cloud_app, sender)
    assert sender.calls[2][2]["X-Vectraudit-Event-Key"].startswith(
        f"monitor.stale_opened:silent:{resumed['id']}-"
    )


def test_signed_events_carry_times_and_ids_only(world, cloud_app, monitored):
    add_hook(world, events=["monitor"])
    schedule(world, every=30, grace=0)
    backdate(cloud_app, hours=1)
    monitoring.sweep(cloud_app.state.sessions)
    sender = Recorder()
    deliver(cloud_app, sender)
    payload = json.loads(sender.calls[0][1])
    assert payload["event"] == "monitor.stale_opened"
    assert payload["url"] == "https://va.example.com/app/acme/search"
    assert set(payload["alert"]) == {
        "alert_id",
        "opened_at",
        "last_check_id",
        "last_run_at",
        "due_at",
        "every_minutes",
    }
    assert "reasons" not in payload
    assert "check_id" not in payload


def test_a_schedule_with_no_run_yet_and_removal_while_overdue(world, cloud_app):
    add_hook(world, events=["monitor"])
    schedule(world, every=60, grace=0)
    backdate(cloud_app, hours=2)
    sessions = cloud_app.state.sessions
    assert monitoring.sweep(sessions)["opened"] == 1
    alert = status(world)["schedule"]["stale_alert"]
    assert alert["alert_id"].startswith("silent:never-")
    assert alert["last_check_id"] is None
    assert (
        world.client.delete(f"{P}/monitor/schedule", headers=world.as_("acme_engineer")).status_code
        == 204
    )
    sender = Recorder()
    deliver(cloud_app, sender)
    opened, removed = (json.loads(body) for _, body, _ in sender.calls)
    assert "since the schedule was declared" in opened["headline"]
    assert removed["event"] == "monitor.stale_resolved"
    assert removed["resolution"] == "schedule_removed"
    assert monitoring.sweep(sessions) == {"opened": 0, "resolved": 0}


def test_lengthening_the_schedule_closes_the_alert(world, cloud_app, monitored):
    add_hook(world, events=["monitor"])
    schedule(world, every=60, grace=0)
    backdate(cloud_app, hours=3)
    assert monitoring.sweep(cloud_app.state.sessions)["opened"] == 1
    changed = schedule(world, every=1440, grace=0).json()["schedule"]
    assert (changed["overdue"], changed["stale_alert"]) == (False, None)
    sender = Recorder()
    deliver(cloud_app, sender)
    resolved = json.loads(sender.calls[-1][1])
    assert resolved["resolution"] == "schedule_changed"
    assert "every 1 day" in resolved["headline"]
    # Shortening it again makes the same silence overdue at once.
    assert schedule(world, every=60, grace=0).json()["schedule"]["overdue"] is True


def test_dashboard_shows_overdue_and_engineers_set_the_schedule(world, cloud_app, monitored):
    engineer = sign_in(cloud_app, world.tokens["acme_engineer"])
    page = engineer.get("/app/acme/search").text
    assert "Monitor schedule" in page
    assert "No schedule: a stopped monitor is not noticed." in page
    csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
    bad = engineer.post(
        "/app/acme/search/monitor/schedule",
        data={"csrf": csrf, "every_hours": "0", "grace_minutes": "10"},
    )
    assert bad.status_code == 422
    saved = engineer.post(
        "/app/acme/search/monitor/schedule",
        data={"csrf": csrf, "every_hours": "1.5", "grace_minutes": "10"},
    )
    assert saved.status_code in (200, 303)
    assert status(world)["schedule"]["every_minutes"] == 90
    assert "Expected every 90 min (+10 min grace)" in engineer.get("/app/acme/search").text
    backdate(cloud_app, hours=3)
    viewer = sign_in(cloud_app, world.tokens["acme_viewer"]).get("/app/acme/search").text
    assert "No scheduled run</strong> since one was due" in viewer
    assert "Save schedule" not in viewer
    missing_csrf = engineer.post("/app/acme/search/monitor/schedule/clear", data={})
    assert missing_csrf.status_code == 403
    engineer.post("/app/acme/search/monitor/schedule/clear", data={"csrf": csrf})
    assert status(world)["schedule"] is None


def test_export_and_cli(world, cloud_app, monitored, monkeypatch):
    from vectraudit.cli import app

    monkeypatch.setattr(cloud_client, "urllib_send", via(world.client))
    monkeypatch.setenv("VECTRAUDIT_CLOUD_URL", "http://testserver")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_PROJECT", "acme/search")
    monkeypatch.setenv("VECTRAUDIT_CLOUD_TOKEN", world.tokens["acme_engineer"])
    runner = CliRunner()
    shown = runner.invoke(app, ["cloud", "monitor"])
    assert shown.exit_code == 0, shown.output
    assert "No schedule declared" in shown.output
    saved = runner.invoke(app, ["cloud", "monitor", "--every", "6h", "--grace", "30m"])
    assert saved.exit_code == 0, saved.output
    assert "Expected every 360 min (+30 min grace)" in saved.output
    assert runner.invoke(app, ["cloud", "monitor", "--every", "soon"]).exit_code == 1
    backdate(cloud_app, hours=7)
    assert "Overdue" in runner.invoke(app, ["cloud", "monitor"]).output
    exported = world.client.get(
        "/api/v1/workspaces/acme/export", headers=world.as_("acme_owner")
    ).json()
    project = next(p for p in exported["projects"] if p["slug"] == "search")
    assert project["monitor_schedule"]["every_minutes"] == 360
    assert runner.invoke(app, ["cloud", "monitor", "--clear"]).exit_code == 0
    assert status(world)["schedule"] is None
