"""Project webhooks: owners register endpoints; engineers can see what was delivered."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from vectraudit.cloud import repo, webhooks
from vectraudit.cloud.deps import PrincipalDep, SessionDep, SettingsDep
from vectraudit.cloud.errors import CloudError, not_found
from vectraudit.cloud.models import Webhook, WebhookDelivery
from vectraudit.cloud.settings import Settings

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace}/projects/{project}/webhooks", tags=["webhooks"]
)
MAX_PER_PROJECT = 10


Event = Literal["decision", "monitor"]


def _all_events() -> list[Event]:
    return ["decision", "monitor"]


class WebhookIn(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    style: Literal["signed", "slack"] = "signed"
    """``signed``: Vectraudit JSON with an HMAC signature; ``slack``: a Slack incoming
    webhook URL, posted a ``text`` message (signature headers are still sent)."""
    events: list[Event] = Field(default_factory=lambda: _all_events(), min_length=1, max_length=2)


class WebhookOut(BaseModel):
    id: str
    url: str
    """Scheme, host and the last characters only: paths often carry tokens."""
    style: str
    events: list[str]
    created_at: datetime
    created_by: str | None


class WebhookCreated(WebhookOut):
    secret: str
    """Signing key, shown once. Verify ``X-Vectraudit-Signature`` with it."""


class DeliveryOut(BaseModel):
    id: str
    event: str
    event_key: str
    status: str
    attempts: int
    last_status: int | None
    last_error: str | None
    created_at: datetime
    next_attempt_at: datetime
    delivered_at: datetime | None


def webhook_out(hook: Webhook) -> WebhookOut:
    return WebhookOut(
        id=hook.id,
        url=webhooks.preview_url(hook.url),
        style=hook.style,
        events=list(hook.events),
        created_at=hook.created_at,
        created_by=hook.created_by,
    )


def _hook(session: SessionDep, project_id: str, webhook_id: str) -> Webhook:
    hook = session.scalars(
        select(Webhook).where(Webhook.project_id == project_id, Webhook.id == webhook_id)
    ).first()
    if hook is None:
        raise not_found("Webhook")
    return hook


def create(
    session: SessionDep, access: repo.Access, settings: Settings, body: WebhookIn
) -> WebhookCreated:
    """Shared by the API and the dashboard."""
    assert access.project is not None
    existing = session.scalars(
        select(Webhook.id).where(Webhook.project_id == access.project.id)
    ).all()
    if len(existing) >= MAX_PER_PROJECT:
        raise CloudError(
            409, "too_many_webhooks", f"A project can have at most {MAX_PER_PROJECT} webhooks."
        )
    try:
        url = webhooks.check_url(body.url.strip(), allow_private=settings.webhook_allow_private)
    except webhooks.WebhookError as exc:
        raise CloudError(422, "invalid_webhook_url", str(exc)) from None
    hook = Webhook(
        workspace_id=access.workspace.id,
        project_id=access.project.id,
        url=url,
        style=body.style,
        events=sorted(set(body.events)),
        secret=webhooks.new_secret(),
        created_by=access.principal.actor,
    )
    session.add(hook)
    session.flush()
    repo.audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "webhook.create",
        "webhook",
        hook.id,
        url=webhooks.preview_url(url),
        style=hook.style,
        events=hook.events,
    )
    return WebhookCreated(**webhook_out(hook).model_dump(), secret=hook.secret)


def delete(session: SessionDep, access: repo.Access, webhook_id: str) -> None:
    assert access.project is not None
    hook = _hook(session, access.project.id, webhook_id)
    session.query(WebhookDelivery).filter(WebhookDelivery.webhook_id == hook.id).delete()
    repo.audit(
        session,
        access.workspace.id,
        access.principal.actor,
        "webhook.delete",
        "webhook",
        hook.id,
        url=webhooks.preview_url(hook.url),
    )
    session.delete(hook)


@router.get("")
def list_webhooks(
    workspace: str, project: str, session: SessionDep, principal: PrincipalDep
) -> list[WebhookOut]:
    access = repo.authorize(session, principal, workspace, "read_webhooks", project)
    assert access.project is not None
    rows = session.scalars(
        select(Webhook).where(Webhook.project_id == access.project.id).order_by(Webhook.created_at)
    )
    return [webhook_out(h) for h in rows]


@router.post("", status_code=201)
def create_webhook(
    workspace: str,
    project: str,
    body: WebhookIn,
    session: SessionDep,
    principal: PrincipalDep,
    settings: SettingsDep,
) -> WebhookCreated:
    """Register an endpoint (owner). The response is the only time the secret is shown."""
    access = repo.authorize(session, principal, workspace, "manage_webhooks", project)
    return create(session, access, settings, body)


@router.delete("/{webhook_id}", status_code=204)
def delete_webhook(
    workspace: str, project: str, webhook_id: str, session: SessionDep, principal: PrincipalDep
) -> Response:
    access = repo.authorize(session, principal, workspace, "manage_webhooks", project)
    delete(session, access, webhook_id)
    return Response(status_code=204)


@router.post("/{webhook_id}/rotate-secret")
def rotate_secret(
    workspace: str, project: str, webhook_id: str, session: SessionDep, principal: PrincipalDep
) -> WebhookCreated:
    """Replace the signing key; deliveries sent from now on use the new one."""
    access = repo.authorize(session, principal, workspace, "manage_webhooks", project)
    assert access.project is not None
    hook = _hook(session, access.project.id, webhook_id)
    hook.secret = webhooks.new_secret()
    repo.audit(
        session, access.workspace.id, access.principal.actor, "webhook.rotate", "webhook", hook.id
    )
    return WebhookCreated(**webhook_out(hook).model_dump(), secret=hook.secret)


@router.post("/{webhook_id}/test", status_code=202)
def test_webhook(
    workspace: str, project: str, webhook_id: str, session: SessionDep, principal: PrincipalDep
) -> dict[str, str]:
    """Queue a ``ping`` event; it is sent with the next delivery run."""
    access = repo.authorize(session, principal, workspace, "manage_webhooks", project)
    assert access.project is not None
    hook = _hook(session, access.project.id, webhook_id)
    webhooks.ping(session, access.workspace, access.project, hook)
    return {"status": "queued"}


@router.get("/{webhook_id}/deliveries")
def list_deliveries(
    workspace: str,
    project: str,
    webhook_id: str,
    session: SessionDep,
    principal: PrincipalDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[DeliveryOut]:
    """Recent deliveries, newest first; finished ones are kept for 30 days."""
    access = repo.authorize(session, principal, workspace, "read_webhooks", project)
    assert access.project is not None
    hook = _hook(session, access.project.id, webhook_id)
    rows = session.scalars(
        select(WebhookDelivery)
        .where(WebhookDelivery.webhook_id == hook.id)
        .order_by(WebhookDelivery.created_at.desc(), WebhookDelivery.id.desc())
        .limit(limit)
    )
    return [DeliveryOut(**{k: getattr(r, k) for k in DeliveryOut.model_fields}) for r in rows]
