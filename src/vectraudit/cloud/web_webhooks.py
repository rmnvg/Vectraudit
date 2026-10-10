"""Dashboard: project notifications (webhooks). Engineers see status; owners manage them."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from vectraudit.cloud import repo, web, webhooks
from vectraudit.cloud.deps import SessionDep, SettingsDep
from vectraudit.cloud.errors import CloudError
from vectraudit.cloud.models import Webhook, WebhookDelivery
from vectraudit.cloud.routes import webhooks as api

router = APIRouter(include_in_schema=False)
e = web.e
EVENT_NAMES = {"decision": "Release decisions", "monitor": "Monitor alerts and resolutions"}


def _last_delivery(session: Session, hook: Webhook) -> str:
    last = session.scalars(
        select(WebhookDelivery)
        .where(WebhookDelivery.webhook_id == hook.id)
        .order_by(WebhookDelivery.created_at.desc(), WebhookDelivery.id.desc())
        .limit(1)
    ).first()
    if last is None:
        return '<span class="muted">Nothing sent yet</span>'
    when = (last.delivered_at or last.created_at).strftime("%Y-%m-%d %H:%M")
    if last.status == "delivered":
        return f"{web._badge('pass')} {e(last.event)} delivered {e(when)}"
    if last.status == "failed":
        return f"{web._badge('fail')} {e(last.event)} gave up: {e(last.last_error or '')}"
    note = f", last {e(last.last_error)}" if last.last_error else ""
    return f'<span class="muted">{e(last.event)} pending (attempt {last.attempts}{note})</span>'


def section(session: Session, form: web.Web, access: repo.Access, base: str) -> str:
    """The project page's Notifications section (engineers and owners)."""
    if repo.ROLE_RANK.get(access.role, 0) < repo.ROLE_RANK["engineer"]:
        return ""
    assert access.project is not None
    owner = access.role == "owner"
    rows = []
    for hook in session.scalars(
        select(Webhook).where(Webhook.project_id == access.project.id).order_by(Webhook.created_at)
    ):
        actions = ""
        if owner:
            target = f"{base}/webhooks/{quote(hook.id)}"
            actions = (
                f'<form class="inline" method="post" action="{target}/test">'
                f"{web._csrf_field(form)}<button>Send test</button></form> "
                f'<form class="inline" method="post" action="{target}/delete">'
                f"{web._csrf_field(form)}<button>Remove</button></form>"
            )
        rows.append(
            [
                f'<span class="mono">{e(webhooks.preview_url(hook.url))}</span>',
                e("Slack" if hook.style == "slack" else "Signed JSON"),
                e(", ".join(EVENT_NAMES[x] for x in hook.events)),
                _last_delivery(session, hook),
                actions,
            ]
        )
    body = (
        "<h2>Notifications</h2>"
        '<p class="muted">Each release decision, and each monitor alert and its resolution, is '
        "sent once per webhook with a headline, verdict and link; no reasons or question IDs. "
        "Deliveries retry with backoff for about 8 hours.</p>"
        + web._table(["Endpoint", "Format", "Events", "Last delivery", ""], rows, "No webhooks.")
    )
    if owner:
        checks = "".join(
            f'<label><input type="checkbox" name="event_{k}" value="1" checked> {e(v)}</label> '
            for k, v in EVENT_NAMES.items()
        )
        body += (
            f'<form method="post" action="{base}/webhooks">{web._csrf_field(form)}'
            '<input name="url" type="url" required maxlength="2048" '
            'placeholder="https://hooks.slack.com/services/…" size="50"> '
            '<select name="style"><option value="signed">Signed JSON</option>'
            '<option value="slack">Slack incoming webhook</option></select> '
            f"{checks}<button>Add webhook</button></form>"
        )
    return body


def _back(workspace: str, project: str, msg: str) -> Response:
    return RedirectResponse(
        f"/app/{quote(workspace)}/{quote(project)}?msg={quote(msg)}", status_code=303
    )


@router.post("/app/{workspace}/{project}/webhooks")
async def create_action(
    workspace: str, project: str, request: Request, session: SessionDep, settings: SettingsDep
) -> Response:
    form = await web._form(request)
    try:
        w, access = web._project_access(
            request, session, settings, workspace, project, "manage_webhooks"
        )
        web._require_csrf(w, form, settings)
        events = [k for k in EVENT_NAMES if form.get(f"event_{k}")]
        try:
            body = api.WebhookIn.model_validate(
                {"url": form.get("url", ""), "style": form.get("style"), "events": events}
            )
        except ValidationError:
            raise CloudError(
                422,
                "invalid_webhook",
                "Give an https URL, a format and at least one kind of event.",
            ) from None
        created = api.create(session, access, settings, body)
    except CloudError as err:
        return web._error_page(err, None)
    back = f"/app/{quote(workspace)}/{quote(project)}"
    verify = (
        "Verify each request: HMAC-SHA256 of <code>timestamp + '.' + body</code> with this "
        "secret, compared with <code>X-Vectraudit-Signature</code> (after <code>v1=</code>); "
        "reject timestamps older than 5 minutes and dedupe on "
        "<code>X-Vectraudit-Event-Key</code>."
    )
    page = (
        "<h1>Webhook added</h1><p>Store the signing secret now; it won't be shown again.</p>"
        f'<p class="secret">{e(created.secret)}</p><p>{verify}</p>'
        f'<p class="muted">{e(created.url)} · {e(created.style)} · '
        f"{e(', '.join(created.events))}</p>"
        f'<p><a href="{back}">Back to the project</a></p>'
    )
    return web.page("Webhook added", page, w, status=201)


@router.post("/app/{workspace}/{project}/webhooks/{webhook_id}/delete")
async def delete_action(
    workspace: str,
    project: str,
    webhook_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> Response:
    form = await web._form(request)
    try:
        w, access = web._project_access(
            request, session, settings, workspace, project, "manage_webhooks"
        )
        web._require_csrf(w, form, settings)
        api.delete(session, access, webhook_id)
    except CloudError as err:
        return web._error_page(err, None)
    return _back(workspace, project, "Webhook removed.")


@router.post("/app/{workspace}/{project}/webhooks/{webhook_id}/test")
async def test_action(
    workspace: str,
    project: str,
    webhook_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> Response:
    form = await web._form(request)
    try:
        w, access = web._project_access(
            request, session, settings, workspace, project, "manage_webhooks"
        )
        web._require_csrf(w, form, settings)
        assert access.project is not None
        hook = api._hook(session, access.project.id, webhook_id)
        webhooks.ping(session, access.workspace, access.project, hook)
    except CloudError as err:
        return web._error_page(err, None)
    return _back(workspace, project, "Test event queued; it is sent with the next delivery run.")
