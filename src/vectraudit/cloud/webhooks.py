"""Signed webhooks: release decisions and monitor alerts delivered to a customer's channel.

Events go through an outbox (``webhook_deliveries``). A check enqueues at most one event
per webhook and event key: ``decision:<check>`` for a release decision, and
``monitor.alert_opened:<alert>`` / ``monitor.alert_resolved:<alert>`` for a scheduled
monitor alert (the alert ID is stable for the whole failing streak, see
:func:`vectraudit.cloud.lead.monitor_status`), and ``monitor.stale_opened:<alert>`` /
``monitor.stale_resolved:<alert>`` when an expected monitor run does not arrive (see
:mod:`vectraudit.cloud.monitoring`). So a repeated check, a long outage or a repeated
sweep notifies once, and once more when it is resolved.

:func:`deliver_due` sends due events, outside any database transaction, and retries with
backoff. Transport is at-least-once, like every webhook system: a receiver dedupes on the
``X-Vectraudit-Event-Key`` header.

What leaves the service is deliberately small: verdict, headline, action, IDs and a
dashboard link, built from the decision ledger. No reasons, question IDs or metric details;
receivers with access follow the link. Requests go only to https URLs on public addresses
(checked at registration and before every send), redirects are not followed, and nothing
from the response is stored beyond its status code.

The ``email`` style sends the same small message as plain-text email to one ``mailto:``
address, through the operator's mail server (:class:`~vectraudit.cloud.settings.
SmtpSettings`), with the same outbox, retries and once-per-event keys. A refused recipient
fails at once; a temporary mail-server error retries on the same schedule.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import smtplib
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Any
from urllib.parse import quote, urlsplit

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from vectraudit.cloud import lead, ledger
from vectraudit.cloud.models import Check, Project, Webhook, WebhookDelivery, Workspace, utcnow
from vectraudit.cloud.settings import SmtpSettings

STYLES = ("signed", "slack", "email")
ADDRESS = re.compile(r"^[^@\s,;:<>()\[\]\\\"]+@[^@\s,;:<>()\[\]\\\"]+\.[^@\s,;:<>()\[\]\\\".]+$")
EVENTS = ("decision", "monitor")
FORMAT = "vectraudit.event"
RETRY_SECONDS = (60, 300, 1800, 7200, 21600)
"""Waits after failed attempts; after the last one the delivery is marked failed."""
MAX_ATTEMPTS = len(RETRY_SECONDS) + 1
LEASE = timedelta(minutes=5)
"""A claimed delivery is not picked up by another worker for this long."""
TIMEOUT_SECONDS = 10
KEEP_FINISHED = timedelta(days=30)

Sender = Callable[[str, bytes, Mapping[str, str]], int]
Resolver = Callable[..., Any]


class WebhookError(ValueError):
    """A webhook URL is not allowed, or a delivery could not reach it."""


def new_secret() -> str:
    return "whsec_" + secrets.token_urlsafe(32)


def _getaddrinfo(*args: Any, **kwargs: Any) -> Any:
    """The resolver used for checks (a seam for tests; connections use the system one)."""
    return socket.getaddrinfo(*args, **kwargs)


def check_url(url: str, *, allow_private: bool = False, resolve: Resolver | None = None) -> str:
    """Accept only https URLs whose host resolves to public addresses.

    ``allow_private`` (an operator setting for development) also allows http and private,
    loopback or link-local addresses. A host that later re-resolves to a private address is
    refused at delivery, which checks again.
    """
    if len(url) > 2048:
        raise WebhookError("Webhook URLs are limited to 2048 characters.")
    parts = urlsplit(url)
    if parts.scheme != "https" and not (allow_private and parts.scheme == "http"):
        raise WebhookError("Webhook URLs must use https.")
    if not parts.hostname:
        raise WebhookError("The webhook URL has no host.")
    if parts.username or parts.password:
        raise WebhookError("Put credentials in the path or a header-free URL, not user:pass@.")
    if allow_private:
        return url
    try:
        port = parts.port or 443
        infos = (resolve or _getaddrinfo)(parts.hostname, port, type=socket.SOCK_STREAM)
    except (OSError, ValueError) as exc:
        raise WebhookError(f"Cannot resolve {parts.hostname}.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global:
            raise WebhookError(
                f"{parts.hostname} resolves to a private, loopback or reserved address; "
                "webhooks only go to public endpoints."
            )
    return url


def check_email(url: str) -> str:
    """Accept ``mailto:`` with exactly one plain address (no lists, names or headers)."""
    if not url.startswith("mailto:"):
        raise WebhookError("Email destinations are written mailto:team@example.com.")
    address = url[len("mailto:") :]
    if len(address) > 254 or "?" in address or not ADDRESS.match(address):
        raise WebhookError("Give one plain email address, such as mailto:team@example.com.")
    return "mailto:" + address


def check_destination(
    url: str, style: str, *, allow_private: bool = False, resolve: Resolver | None = None
) -> str:
    """An email address for the ``email`` style, otherwise a public https URL."""
    if style == "email":
        return check_email(url)
    return check_url(url, allow_private=allow_private, resolve=resolve)


def preview_url(url: str) -> str:
    """``https://hooks.example.com/…abcd``: enough to recognize, without path secrets.

    An address previews as ``mailto:t…@example.com``.
    """
    if url.startswith("mailto:"):
        local, _, domain = url[len("mailto:") :].partition("@")
        return f"mailto:{local[:1]}…@{domain}"
    parts = urlsplit(url)
    tail = (parts.path + ("?" + parts.query if parts.query else ""))[-4:]
    return f"{parts.scheme}://{parts.netloc}/…{tail}" if tail.strip("/") else url


def signature(secret: str, timestamp: str, body: bytes) -> str:
    """``v1=<hex HMAC-SHA256 of "<timestamp>.<body>">``."""
    mac = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
    return "v1=" + mac.hexdigest()


def verify(secret: str, timestamp: str, body: bytes, header: str, *, tolerance: int = 300) -> bool:
    """Receiver-side check, also used in tests: signature matches and timestamp is fresh."""
    try:
        age = abs(time.time() - int(timestamp))
    except ValueError:
        return False
    return age <= tolerance and hmac.compare_digest(signature(secret, timestamp, body), header)


# --- events -------------------------------------------------------------------------------


def _event(
    kind: str, key: str, ws: Workspace, project: Project, entry: Mapping[str, Any] | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": FORMAT,
        "format_version": 1,
        "event": kind,
        "event_key": key,
        "occurred_at": utcnow().isoformat(),
        "workspace": ws.slug,
        "project": project.slug,
    }
    if entry is not None:
        summary = lead.explain(entry)
        payload.update(
            check_id=entry["check_id"],
            verdict=entry["verdict"],
            trigger=summary["trigger"],
            candidate=summary["candidate"],
            headline=summary["headline"],
            action=summary["action"],
            path=f"/app/{quote(ws.slug)}/{quote(project.slug)}/decisions/{quote(entry['check_id'])}",
        )
    return payload


def check_events(session: Session, ws: Workspace, project: Project, check: Check) -> list[Any]:
    """``(category, event_key, payload)`` for what ``check`` newly caused, if anything."""
    entry = ledger.decision(session, project, check.id)
    if entry is None:
        return []
    if check.trigger != "monitor":
        key = f"decision:{check.id}"
        return [("decision", key, _event("decision", key, ws, project, entry))]
    rows = ledger.history(session, project, limit=500, trigger="monitor")
    status = lead.monitor_status(rows)
    alert = status["alert"]
    if alert is None:
        return []
    alert_info = {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in alert.items()
        if k != "complete"
    }
    if status["state"] == "alerting" and alert["alert_id"] == check.id:
        kind = "monitor.alert_opened"
    elif status["state"] == "healthy" and alert["resolved_by"] == check.id:
        kind = "monitor.alert_resolved"
    else:
        return []  # a later run of an open alert, or a pass with nothing to resolve
    key = f"{kind}:{alert['alert_id']}"
    payload = _event(kind, key, ws, project, entry)
    payload["alert"] = alert_info
    return [("monitor", key, payload)]


def enqueue(
    session: Session, hook: Webhook, kind: str, key: str, payload: Mapping[str, Any]
) -> bool:
    """Add one outbox row; False if this webhook already has the event (deduplicated)."""
    exists = session.scalars(
        select(WebhookDelivery.id).where(
            WebhookDelivery.webhook_id == hook.id, WebhookDelivery.event_key == key
        )
    ).first()
    if exists is not None:
        return False
    try:
        with session.begin_nested():  # a concurrent writer may enqueue the same event
            session.add(
                WebhookDelivery(
                    webhook_id=hook.id,
                    project_id=hook.project_id,
                    event=kind,
                    event_key=key,
                    payload=dict(payload),
                    status="pending",
                    attempts=0,
                    next_attempt_at=utcnow(),
                )
            )
    except IntegrityError:
        return False
    return True


def notify(
    session: Session, project: Project, category: str, key: str, payload: Mapping[str, Any]
) -> int:
    """Enqueue one event for every webhook of ``project`` subscribed to ``category``."""
    hooks = session.scalars(select(Webhook).where(Webhook.project_id == project.id))
    return sum(
        enqueue(session, hook, payload["event"], key, payload)
        for hook in hooks
        if category in hook.events
    )


def on_check(session: Session, ws: Workspace, project: Project, check: Check) -> int:
    """Enqueue what a newly created check means for this project's webhooks."""
    if session.scalars(select(Webhook.id).where(Webhook.project_id == project.id)).first() is None:
        return 0
    return sum(
        notify(session, project, category, key, payload)
        for category, key, payload in check_events(session, ws, project, check)
    )


def stale_event(
    kind: str, ws: Workspace, project: Project, alert: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """``(event_key, payload)`` for ``monitor.stale_opened`` or ``monitor.stale_resolved``.

    Keyed by the staleness alert's ID (the last run before the silence), so repeated sweeps
    announce it once. Carries times and IDs only, like every other event.
    """
    key = f"{kind}:{alert['alert_id']}"
    payload = _event(kind, key, ws, project, None)
    every = lead.interval_label(alert["every_minutes"])
    if kind == "monitor.stale_opened":
        since = alert.get("last_run_at")
        payload["headline"] = (
            f"No scheduled monitor run for {ws.slug}/{project.slug} since "
            + (f"{since[:16].replace('T', ' ')} UTC" if since else "the schedule was declared")
            + f"; one is expected every {every}."
        )
        payload["action"] = (
            "Check that the scheduled monitor job is still running and can reach the "
            "service. Until it reports, nothing is watching this project."
        )
    elif alert.get("resolution") == "schedule_removed":
        payload["headline"] = (
            f"Monitor schedule for {ws.slug}/{project.slug} removed; the staleness alert is closed."
        )
    elif alert.get("resolution") == "schedule_changed":
        payload["headline"] = (
            f"Monitor schedule for {ws.slug}/{project.slug} changed to every {every}; the "
            "monitor is no longer overdue."
        )
    else:
        payload["headline"] = f"Scheduled monitor for {ws.slug}/{project.slug} reported again."
        if alert.get("resolved_by"):
            payload["check_id"] = alert["resolved_by"]
    payload["path"] = f"/app/{quote(ws.slug)}/{quote(project.slug)}"
    payload["alert"] = {k: v for k, v in alert.items() if k != "resolution"}
    if alert.get("resolution"):
        payload["resolution"] = alert["resolution"]
    return key, payload


def ping(session: Session, ws: Workspace, project: Project, hook: Webhook) -> None:
    """A test event, so an owner can see a delivery arrive and verify the signature."""
    key = f"ping:{secrets.token_hex(8)}"
    payload = _event("ping", key, ws, project, None)
    payload["headline"] = f"Test event from Vectraudit for {ws.slug}/{project.slug}."
    enqueue(session, hook, "ping", key, payload)


# --- delivery -----------------------------------------------------------------------------


def _slack_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


PREFIXES = {
    "monitor.alert_opened": "Monitor alert: ",
    "monitor.alert_resolved": "Monitor alert resolved: ",
    "monitor.stale_opened": "Monitor overdue: ",
    "monitor.stale_resolved": "Monitor overdue resolved: ",
}


def _email(hook: Webhook, payload: Mapping[str, Any], delivery: WebhookDelivery) -> bytes:
    """A plain-text message: the same small event a Slack message carries, no evidence."""
    message = EmailMessage()
    headline = PREFIXES.get(payload["event"], "") + str(payload.get("headline", ""))
    message["Subject"] = f"[Vectraudit] {payload['workspace']}/{payload['project']}: {headline}"[
        :200
    ]
    message["To"] = hook.url[len("mailto:") :]
    message["X-Vectraudit-Event"] = delivery.event
    message["X-Vectraudit-Event-Key"] = delivery.event_key
    message["X-Vectraudit-Delivery"] = delivery.id
    lines = [headline]
    if payload.get("action"):
        lines += ["", str(payload["action"])]
    if payload.get("url"):
        lines += ["", f"Open in Vectraudit: {payload['url']}"]
    lines += [
        "",
        f"Project {payload['workspace']}/{payload['project']}. "
        "This address was added as a notification destination by a project owner.",
    ]
    message.set_content("\n".join(lines))
    return bytes(message)


def render(
    hook: Webhook, delivery: WebhookDelivery, public_url: str, now: float
) -> tuple[bytes, dict[str, str]]:
    """The request body and headers for one attempt (signed in both styles)."""
    payload = dict(delivery.payload)
    if payload.get("path"):
        payload["url"] = public_url.rstrip("/") + payload.pop("path")
    payload["delivery_id"] = delivery.id
    if hook.style == "email":
        return _email(hook, payload, delivery), {"To": hook.url[len("mailto:") :]}
    if hook.style == "slack":
        prefix = PREFIXES.get(payload["event"], "")
        lines = [f"*{_slack_escape(prefix + payload.get('headline', ''))}*"]
        if payload.get("action"):
            lines.append(_slack_escape(payload["action"]))
        if payload.get("url"):
            lines.append(f"<{payload['url']}|Open in Vectraudit>")
        body_obj: dict[str, Any] = {"text": "\n".join(lines)}
    else:
        body_obj = payload
    body = json.dumps(body_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    stamp = str(int(now))
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Vectraudit-Webhooks/1",
        "X-Vectraudit-Event": delivery.event,
        "X-Vectraudit-Event-Key": delivery.event_key,
        "X-Vectraudit-Delivery": delivery.id,
        "X-Vectraudit-Timestamp": stamp,
        "X-Vectraudit-Signature": signature(hook.secret, stamp, body),
    }
    return body, headers


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # a redirect could point anywhere, including private addresses


def http_send(url: str, body: bytes, headers: Mapping[str, str]) -> int:
    """POST and return the HTTP status (3xx is returned, not followed)."""
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WebhookError(f"Connection failed ({type(exc).__name__}).") from exc


def smtp_send(
    settings: SmtpSettings | None, url: str, body: bytes, headers: Mapping[str, str]
) -> int:
    """Send one rendered message; returns 200, or 400 when the recipient is refused.

    A temporary server reply (4xx) returns 503 so the delivery retries. A missing mail
    server, a failed login or a lost connection raise :class:`WebhookError` and retry too,
    so an operator can fix the configuration before the delivery gives up.
    """
    if settings is None:
        raise WebhookError("Email delivery is not configured on this service (SMTP settings).")
    message = BytesParser(policy=email_policy).parsebytes(body)
    message["From"] = settings.sender
    try:
        if settings.security == "ssl":
            client: smtplib.SMTP = smtplib.SMTP_SSL(
                settings.host,
                settings.port,
                timeout=TIMEOUT_SECONDS,
                context=ssl.create_default_context(),
            )
        else:
            client = smtplib.SMTP(settings.host, settings.port, timeout=TIMEOUT_SECONDS)
        with client:
            if settings.security == "starttls":
                client.starttls(context=ssl.create_default_context())
            if settings.username:
                client.login(settings.username, settings.password or "")
            client.send_message(message, to_addrs=[headers["To"]])
    except smtplib.SMTPRecipientsRefused:
        return 400
    except smtplib.SMTPAuthenticationError as exc:
        raise WebhookError("The mail server refused the configured login.") from exc
    except smtplib.SMTPResponseException as exc:
        return 503 if 400 <= exc.smtp_code < 500 else 400
    except (smtplib.SMTPException, OSError) as exc:
        raise WebhookError(f"Mail server connection failed ({type(exc).__name__}).") from exc
    return 200


def _permanent(status: int) -> bool:
    """Redirects (never followed) and 4xx other than 408/429 (a bad URL, a revoked Slack
    hook) will not change on retry."""
    return 300 <= status < 400 or (400 <= status < 500 and status not in (408, 429))


def deliver_due(
    sessions: sessionmaker[Session],
    *,
    public_url: str,
    allow_private: bool = False,
    send: Sender = http_send,
    send_email: Sender | None = None,
    smtp: SmtpSettings | None = None,
    resolve: Resolver | None = None,
    limit: int = 100,
    now: datetime | None = None,
) -> dict[str, int]:
    """Send due deliveries once each; returns counts of delivered, retrying and failed.

    Email deliveries go through ``send_email`` (default: :func:`smtp_send` with ``smtp``).
    """
    from vectraudit.cloud.db import transaction

    moment = now or utcnow()
    with transaction(sessions) as session:
        due = list(
            session.scalars(
                select(WebhookDelivery)
                .where(
                    WebhookDelivery.status == "pending",
                    WebhookDelivery.next_attempt_at <= moment,
                )
                .order_by(WebhookDelivery.next_attempt_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )
        for claimed_row in due:
            claimed_row.next_attempt_at = moment + LEASE  # claimed by this worker
        claimed = [claimed_row.id for claimed_row in due]
    counts = {"delivered": 0, "retrying": 0, "failed": 0}
    for delivery_id in claimed:
        with transaction(sessions) as session:
            row = session.get(WebhookDelivery, delivery_id)
            hook = session.get(Webhook, row.webhook_id) if row is not None else None
            if row is None or hook is None:
                continue
            url, style = hook.url, hook.style
            body, headers = render(hook, row, public_url, time.time())
        status: int | None = None
        error: str | None = None
        try:
            if style == "email":
                check_email(url)
                mailer = send_email or (lambda u, b, h: smtp_send(smtp, u, b, h))
                status = mailer(url, body, headers)
            else:
                check_url(url, allow_private=allow_private, resolve=resolve)
                status = send(url, body, headers)
        except WebhookError as exc:
            error = str(exc)[:200]
        with transaction(sessions) as session:
            row = session.get(WebhookDelivery, delivery_id)
            if row is None:
                continue
            row.attempts += 1
            row.last_status = status
            ok = status is not None and 200 <= status < 300
            row.last_error = error or (None if ok else f"HTTP {status}")
            if ok:
                row.status = "delivered"
                row.delivered_at = utcnow()
                counts["delivered"] += 1
            elif (status is not None and _permanent(status)) or row.attempts >= MAX_ATTEMPTS:
                row.status = "failed"
                counts["failed"] += 1
            else:
                row.next_attempt_at = utcnow() + timedelta(seconds=RETRY_SECONDS[row.attempts - 1])
                counts["retrying"] += 1
    return counts


def prune(session: Session, *, now: datetime | None = None) -> int:
    """Delete delivered and failed deliveries older than 30 days; pending ones stay."""
    from sqlalchemy import delete

    cutoff = (now or utcnow()) - KEEP_FINISHED
    result = session.execute(
        delete(WebhookDelivery).where(
            WebhookDelivery.status.in_(("delivered", "failed")),
            WebhookDelivery.created_at < cutoff,
        )
    )
    return int(getattr(result, "rowcount", 0) or 0)
