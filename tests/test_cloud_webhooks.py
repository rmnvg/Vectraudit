"""Signed webhooks: who may add them, what is sent, once per event, and how failures retry."""

from __future__ import annotations

import json
import re
import socket
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

pytest.importorskip("sqlalchemy")

from tests.test_cloud_checks import pin
from tests.test_cloud_monitor import _bad
from tests.test_cloud_runs import a_run, client_for
from tests.test_cloud_web import sign_in
from vectraudit.cloud import webhooks
from vectraudit.cloud.db import transaction
from vectraudit.cloud.models import AuditEvent, Project, Webhook, WebhookDelivery, utcnow

P = "/api/v1/workspaces/acme/projects/search"
HOOKS = f"{P}/webhooks"
PUBLIC = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def resolver(address: str):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return lambda *args, **kwargs: [(family, socket.SOCK_STREAM, 6, "", (address, 443))]


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Every host resolves to a public address unless a test says otherwise."""
    monkeypatch.setattr(webhooks, "_getaddrinfo", lambda *a, **k: PUBLIC)


class Recorder:
    def __init__(self, statuses=(200,)):
        self.statuses = list(statuses)
        self.calls: list[tuple[str, bytes, dict]] = []

    def __call__(self, url, body, headers):
        self.calls.append((url, body, dict(headers)))
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, Exception):
            raise status
        return status


def deliver(cloud_app, send, **kwargs):
    return webhooks.deliver_due(
        cloud_app.state.sessions, public_url="https://va.example.com", send=send, **kwargs
    )


def add_hook(world, events=("decision", "monitor"), style="signed", who="acme_owner"):
    return world.client.post(
        HOOKS,
        json={"url": "https://hooks.example.com/T0/secret-path", "style": style, "events": events},
        headers=world.as_(who),
    )


@pytest.mark.parametrize(
    ("url", "address", "message"),
    [
        ("http://hooks.example.com/x", "93.184.216.34", "https"),
        ("https://user:pw@hooks.example.com/x", "93.184.216.34", "credentials"),
        ("https://internal.example/x", "10.0.0.5", "private"),
        ("https://metadata.example/x", "169.254.169.254", "private"),
        ("https://loop.example/x", "127.0.0.1", "private"),
        ("https://mapped.example/x", "::ffff:127.0.0.1", "private"),
        ("https://v6.example/x", "::1", "private"),
    ],
)
def test_only_https_public_endpoints_are_accepted(url, address, message):
    with pytest.raises(webhooks.WebhookError, match=message):
        webhooks.check_url(url, resolve=resolver(address))
    assert webhooks.check_url("https://hooks.example.com/x", resolve=resolver("93.184.216.34"))
    assert webhooks.check_url("http://localhost:9/x", allow_private=True)


def test_owners_manage_webhooks_and_the_secret_is_shown_once(world, cloud_app) -> None:
    for who in ("acme_engineer", "acme_viewer", "acme_runner"):
        assert add_hook(world, who=who).status_code == 403
    created = add_hook(world)
    assert created.status_code == 201
    body = created.json()
    assert body["secret"].startswith("whsec_")
    assert "secret-path" not in body["url"]  # paths often carry tokens
    listed = world.client.get(HOOKS, headers=world.as_("acme_engineer")).json()
    assert [h["id"] for h in listed] == [body["id"]]
    assert "secret" not in listed[0]
    assert world.client.get(HOOKS, headers=world.as_("acme_viewer")).status_code == 403
    rotated = world.client.post(
        f"{HOOKS}/{body['id']}/rotate-secret", headers=world.as_("acme_owner")
    ).json()
    assert rotated["secret"] != body["secret"]
    bad = world.client.post(
        HOOKS, json={"url": "http://hooks.example.com/x"}, headers=world.as_("acme_owner")
    )
    assert bad.status_code == 422
    assert bad.json()["error_code"] == "invalid_webhook_url"
    with transaction(cloud_app.state.sessions) as s:
        audit = s.query(AuditEvent).filter(AuditEvent.action.like("webhook.%")).all()
        assert {a.action for a in audit} == {"webhook.create", "webhook.rotate"}
        assert "secret-path" not in json.dumps([a.details for a in audit])
    exported = world.client.get(
        "/api/v1/workspaces/acme/export", headers=world.as_("acme_owner")
    ).json()
    search = next(p for p in exported["projects"] if p["slug"] == "search")
    assert search["webhooks"][0]["style"] == "signed"
    assert body["secret"] not in json.dumps(exported)
    assert rotated["secret"] not in json.dumps(exported)
    assert "secret-path" not in json.dumps(exported)
    gone = world.client.delete(f"{HOOKS}/{body['id']}", headers=world.as_("acme_owner"))
    assert gone.status_code == 204
    assert world.client.get(HOOKS, headers=world.as_("acme_owner")).json() == []


def test_decisions_are_sent_once_signed_and_without_evidence(world, cloud_app) -> None:
    hook = add_hook(world).json()
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    worse = _bad("worse")
    worse["queries"][0]["text"] = "PRIVATE_QUESTION_TEXT"
    client.upload_run(worse)
    check = client.create_check("worse")
    assert client.create_check("worse")["id"] == check["id"]  # an identical repeat
    sender = Recorder()
    assert deliver(cloud_app, sender) == {"delivered": 1, "retrying": 0, "failed": 0}
    assert deliver(cloud_app, sender)["delivered"] == 0  # nothing left to send
    ((url, body, headers),) = sender.calls
    assert url == "https://hooks.example.com/T0/secret-path"
    assert headers["X-Vectraudit-Event"] == "decision"
    assert headers["X-Vectraudit-Event-Key"] == f"decision:{check['id']}"
    assert webhooks.verify(
        hook["secret"], headers["X-Vectraudit-Timestamp"], body, headers["X-Vectraudit-Signature"]
    )
    stamp, sig = headers["X-Vectraudit-Timestamp"], headers["X-Vectraudit-Signature"]
    assert not webhooks.verify("whsec_other", stamp, body, sig)
    event = json.loads(body)
    assert event["verdict"] == "fail"
    assert event["headline"].startswith("nightly @ abc123 is blocked")
    assert event["url"] == f"https://va.example.com/app/acme/search/decisions/{check['id']}"
    assert set(event) == {
        "format", "format_version", "event", "event_key", "occurred_at", "workspace",
        "project", "check_id", "verdict", "trigger", "candidate", "headline", "action", "url",
        "delivery_id",
    }  # fmt: skip
    assert "PRIVATE_QUESTION_TEXT" not in body.decode()
    assert "q0" not in body.decode()  # no question IDs or reasons
    pages = world.client.get(
        f"{HOOKS}/{hook['id']}/deliveries", headers=world.as_("acme_engineer")
    ).json()
    assert pages[0]["status"] == "delivered"
    assert pages[0]["attempts"] == 1


def test_a_monitor_outage_notifies_once_and_once_more_when_resolved(world, cloud_app) -> None:
    add_hook(world, events=["monitor"], style="slack")
    decisions_only = add_hook(world, events=["decision"]).json()
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    sender = Recorder()
    sent = []
    for i, bad in enumerate([False, True, True, True, False, False]):
        run = _bad(f"n{i}") if bad else a_run(f"n{i}", label="nightly")
        client.upload_run(run)
        client.create_check(f"n{i}", trigger="monitor")
        before = len(sender.calls)
        deliver(cloud_app, sender)
        sent.append(len(sender.calls) - before)
    assert sent == [0, 1, 0, 0, 1, 0]  # opened at the first failure, resolved by the pass
    opened, resolved = (json.loads(body)["text"] for _, body, _ in sender.calls)
    assert opened.startswith("*Monitor alert: Scheduled monitor: nightly @ abc123 now breaks")
    assert resolved.startswith("*Monitor alert resolved: Scheduled monitor: nightly @ abc123")
    assert "<https://va.example.com/app/acme/search/decisions/" in resolved
    keys = [h["X-Vectraudit-Event-Key"] for _, _, h in sender.calls]
    assert keys[0].split(":", 1)[1] == keys[1].split(":", 1)[1]  # the same alert ID
    with transaction(cloud_app.state.sessions) as s:
        assert (
            s.query(WebhookDelivery)
            .filter(WebhookDelivery.webhook_id == decisions_only["id"])
            .count()
            == 0
        )


def test_failures_retry_with_backoff_then_give_up(world, cloud_app) -> None:
    add_hook(world)
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    for name in ("c1", "c2", "c3"):
        client.upload_run(a_run(name, label=name))
        client.create_check(name)

    def by_key(s):
        return {d.payload["candidate"].split(" ")[0]: d for d in s.query(WebhookDelivery)}

    flaky = Recorder([500, 404, webhooks.WebhookError("Connection failed (TimeoutError).")])
    assert deliver(cloud_app, flaky) == {"delivered": 0, "retrying": 2, "failed": 1}
    with transaction(cloud_app.state.sessions) as s:
        rows = by_key(s)
        assert rows["c1"].status == "pending"
        assert rows["c1"].last_error == "HTTP 500"
        assert rows["c2"].status == "failed"  # 404: retrying cannot help
        assert rows["c3"].last_error == "Connection failed (TimeoutError)."
        wait = rows["c1"].next_attempt_at - utcnow()
        assert timedelta(seconds=50) < wait <= timedelta(seconds=60)
    assert deliver(cloud_app, Recorder())["delivered"] == 0  # not due yet
    later = utcnow() + timedelta(days=1)
    for attempt in range(webhooks.MAX_ATTEMPTS - 1):
        counts = deliver(cloud_app, Recorder([503]), now=later + timedelta(days=attempt))
    assert counts == {"delivered": 0, "retrying": 0, "failed": 2}
    with transaction(cloud_app.state.sessions) as s:
        assert {d.attempts for d in s.query(WebhookDelivery)} == {1, webhooks.MAX_ATTEMPTS}
        assert webhooks.prune(s, now=utcnow() + timedelta(days=31)) == 3


def test_an_endpoint_that_turns_private_is_not_called(world, cloud_app, monkeypatch) -> None:
    add_hook(world)
    client = client_for(world, "acme_runner")
    client.upload_run(a_run("base"))
    pin(world, "acme_engineer", "base")
    client.upload_run(a_run("c1"))
    client.create_check("c1")
    monkeypatch.setattr(webhooks, "_getaddrinfo", resolver("10.1.2.3"))
    sender = Recorder()
    assert deliver(cloud_app, sender)["retrying"] == 1
    assert sender.calls == []


class _Receiver(BaseHTTPRequestHandler):
    received: ClassVar[list[tuple[dict, bytes]]] = []
    redirect = False

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).received.append((dict(self.headers), body))
        if type(self).redirect:
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest")
        else:
            self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_real_http_delivery_and_redirects_are_not_followed(world, cloud_app) -> None:
    server = HTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/hook"
        with transaction(cloud_app.state.sessions) as s:
            project = s.query(Project).filter(Project.slug == "search").first()
            s.add(
                Webhook(
                    workspace_id=project.workspace_id,
                    project_id=project.id,
                    url=url,
                    style="signed",
                    events=["decision"],
                    secret="whsec_local",
                )
            )
        client = client_for(world, "acme_runner")
        client.upload_run(a_run("base"))
        pin(world, "acme_engineer", "base")
        client.upload_run(a_run("c1"))
        client.create_check("c1")
        refused = deliver(cloud_app, webhooks.http_send)  # private address, setting off
        assert refused["retrying"] == 1
        assert _Receiver.received == []
        later = utcnow() + timedelta(hours=1)
        sent = deliver(cloud_app, webhooks.http_send, allow_private=True, now=later)
        assert sent["delivered"] == 1
        headers, body = _Receiver.received[0]
        stamp, sig = headers["X-Vectraudit-Timestamp"], headers["X-Vectraudit-Signature"]
        assert webhooks.verify("whsec_local", stamp, body, sig)
        _Receiver.redirect = True
        client.upload_run(a_run("c2"))
        client.create_check("c2")
        moved = deliver(cloud_app, webhooks.http_send, allow_private=True, now=later)
        assert moved["failed"] == 1  # a 302 is recorded, never followed
        assert len(_Receiver.received) == 2
    finally:
        server.shutdown()
        _Receiver.received.clear()
        _Receiver.redirect = False


def test_dashboard_lets_owners_add_test_and_remove(world, cloud_app) -> None:
    owner = sign_in(cloud_app, world.tokens["acme_owner"])
    page = owner.get("/app/acme/search").text
    assert "Notifications" in page
    assert 'action="/app/acme/search/webhooks"' in page
    csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
    created = owner.post(
        "/app/acme/search/webhooks",
        data={
            "csrf": csrf,
            "url": "https://hooks.slack.com/services/T/B/token",
            "style": "slack",
            "event_monitor": "1",
        },
    )
    assert created.status_code == 201
    assert "whsec_" in created.text
    page = owner.get("/app/acme/search").text
    assert "Slack" in page
    assert "Monitor alerts and resolutions" in page
    assert "token" not in page.split("Notifications", 1)[1].split("</table>", 1)[0]
    with transaction(cloud_app.state.sessions) as s:
        hook_id = s.query(Webhook.id).scalar()
    owner.post(f"/app/acme/search/webhooks/{hook_id}/test", data={"csrf": csrf})
    with transaction(cloud_app.state.sessions) as s:
        assert s.query(WebhookDelivery).one().event == "ping"
    engineer = sign_in(cloud_app, world.tokens["acme_engineer"])
    seen = engineer.get("/app/acme/search").text
    assert "Notifications" in seen
    assert 'action="/app/acme/search/webhooks"' not in seen
    assert (
        "Notifications"
        not in sign_in(cloud_app, world.tokens["acme_viewer"]).get("/app/acme/search").text
    )
    owner.post(f"/app/acme/search/webhooks/{hook_id}/delete", data={"csrf": csrf})
    with transaction(cloud_app.state.sessions) as s:
        assert s.query(Webhook).count() == 0
        assert s.query(WebhookDelivery).count() == 0
