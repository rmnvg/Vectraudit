"""``vectraudit-cloud``: run and administer the hosted control plane."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from vectraudit.cloud.models import Workspace
from vectraudit.cloud.settings import Settings, SettingsError

app = typer.Typer(
    name="vectraudit-cloud", help="Vectraudit hosted control plane.", no_args_is_help=True
)
console = Console()
err_console = Console(stderr=True)


def _settings() -> Settings:
    try:
        return Settings.from_env()
    except SettingsError as e:
        err_console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1) from e


@app.command()
def migrate() -> None:
    """Apply database migrations up to the latest revision."""
    from vectraudit.cloud.migrate import head_revision, upgrade

    settings = _settings()
    upgrade(settings.database_url)
    console.print(f"Database at revision {head_revision()}.")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port.")] = 8800,
    workers: Annotated[int, typer.Option(min=1, help="Worker processes.")] = 1,
    migrate_first: Annotated[
        bool,
        typer.Option("--migrate", help="Apply migrations before serving (single instance)."),
    ] = False,
    log_format: Annotated[str, typer.Option(help="json or text.")] = "json",
) -> None:
    """Serve the API. Run `migrate` first (or pass --migrate on a single instance)."""
    import uvicorn

    from vectraudit.cloud.logs import configure

    settings = _settings()  # fail fast on bad configuration
    configure(log_format)
    if migrate_first:
        from vectraudit.cloud.migrate import upgrade

        upgrade(settings.database_url)
    uvicorn.run(
        "vectraudit.cloud.cli:asgi_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        proxy_headers=True,
        access_log=False,  # our access log omits query strings, which can carry secrets
        log_config=None,
    )


def _sessions() -> sessionmaker[Session]:
    from vectraudit.cloud.db import make_engine, session_factory

    return session_factory(make_engine(_settings().database_url))


def _print_token(token: str, who: str) -> None:
    console.print(f"Token for {escape(who)} (shown once; store it securely):")
    # Plain print so the secret can be copied without terminal markup.
    print(token)


@app.command("create-admin")
def create_admin(email: Annotated[str, typer.Argument(help="Operator email.")]) -> None:
    """Create a platform administrator (can create workspaces; no customer data access)."""
    from vectraudit.cloud import repo
    from vectraudit.cloud.db import transaction

    with transaction(_sessions()) as session:
        user = repo.get_or_create_user(session, email)
        user.is_admin = True
        _, new = repo.create_personal_token(session, user, "admin (CLI)", "cli")
    _print_token(new.token, email)


@app.command("create-workspace")
def create_workspace(
    slug: Annotated[str, typer.Argument(help="URL name, e.g. acme.")],
    name: Annotated[str, typer.Option(help="Display name.")],
    owner_email: Annotated[str, typer.Option(help="First owner; created if new.")],
) -> None:
    """Create a workspace with its first owner, and print the owner's API token."""
    from vectraudit.cloud import repo
    from vectraudit.cloud.db import transaction
    from vectraudit.cloud.errors import CloudError

    try:
        with transaction(_sessions()) as session:
            _, owner = repo.create_workspace(session, "cli", slug, name, owner_email)
            _, new = repo.create_personal_token(session, owner, f"{slug} owner (CLI)", "cli")
    except CloudError as e:
        err_console.print(f"[bold red]Error:[/] {escape(e.message)}")
        raise typer.Exit(code=1) from e
    console.print(f"Workspace {escape(slug)} created; owner {escape(owner.email)}.")
    _print_token(new.token, owner.email)


@app.command("create-token")
def create_token(
    email: Annotated[str, typer.Argument(help="Existing user's email.")],
    name: Annotated[str, typer.Option(help="Token label.")] = "CLI",
) -> None:
    """Issue a personal API token for an existing user (e.g. a lost token)."""
    from sqlalchemy import select

    from vectraudit.cloud import repo
    from vectraudit.cloud.db import transaction
    from vectraudit.cloud.models import User

    with transaction(_sessions()) as session:
        user = session.scalars(
            select(User).where(User.email == repo.normalize_email(email))
        ).first()
        if user is None:
            err_console.print(f"[bold red]Error:[/] no user {escape(email)}.")
            raise typer.Exit(code=1)
        _, new = repo.create_personal_token(session, user, name, "cli")
    _print_token(new.token, email)


@app.command()
def purge() -> None:
    """Delete runs past retention (pinned baselines are kept) and old sessions."""
    from vectraudit.cloud import quotas, repo, service, webhooks
    from vectraudit.cloud.db import transaction

    with transaction(_sessions()) as session:
        deleted = quotas.purge_expired(session)
        for slug, n in deleted.items():
            ws_id = session.scalars(select(Workspace.id).where(Workspace.slug == slug)).first()
            repo.audit(session, ws_id, "cli", "retention.purge", "workspace", ws_id or slug, runs=n)
        sessions = service.purge_sessions(session)
        deliveries = webhooks.prune(session)
    total = sum(deleted.values())
    console.print(f"Purged {sessions:,} old sessions.")
    console.print(f"Purged {deliveries:,} finished webhook deliveries older than 30 days.")
    console.print(
        f"Purged {total:,} expired runs"
        + (": " + ", ".join(f"{k} {v}" for k, v in deleted.items()) if deleted else ".")
    )


@app.command("deliver-webhooks")
def deliver_webhooks(
    watch: Annotated[
        int,
        typer.Option(min=0, help="Keep running, checking every N seconds (0: run once)."),
    ] = 0,
) -> None:
    """Check monitor schedules, then send due webhook deliveries and schedule retries.

    Run every minute, or with --watch. Each pass first opens or resolves staleness alerts
    for projects whose expected monitor run is overdue (see ``check-monitors``).
    """
    import time

    from vectraudit.cloud import monitoring, webhooks

    settings = _settings()
    sessions = _sessions()
    while True:
        stale = monitoring.sweep(sessions)
        if any(stale.values()):
            console.print(
                f"Monitor schedules: {stale['opened']} overdue, {stale['resolved']} resolved."
            )
        counts = webhooks.deliver_due(
            sessions, public_url=settings.public_url, allow_private=settings.webhook_allow_private
        )
        if any(counts.values()) or not watch:
            console.print(
                f"Delivered {counts['delivered']}, retrying {counts['retrying']}, "
                f"failed {counts['failed']}."
            )
        if not watch:
            return
        time.sleep(watch)


@app.command("check-monitors")
def check_monitors() -> None:
    """Open or resolve staleness alerts for overdue monitor schedules, once.

    ``deliver-webhooks`` already does this on every pass; use this when deliveries run
    elsewhere or to check from cron.
    """
    from vectraudit.cloud import monitoring

    counts = monitoring.sweep(_sessions())
    console.print(
        f"Monitor schedules: {counts['opened']} newly overdue, {counts['resolved']} resolved."
    )


@app.command("backfill-decisions")
def backfill_decisions() -> None:
    """Record ledger decisions for checks made before the decision ledger existed."""
    from vectraudit.cloud.db import transaction
    from vectraudit.cloud.ledger import record_missing
    from vectraudit.cloud.models import Check

    with transaction(_sessions()) as session:
        written = record_missing(session, list(session.scalars(select(Check))))
    console.print(f"Recorded {written:,} decisions; existing ones were left unchanged.")


@app.command()
def usage(
    month: Annotated[str, typer.Option(help="YYYY-MM (default: this month).")] = "",
) -> None:
    """Print usage per workspace as CSV, for invoicing."""
    import csv
    import sys
    from datetime import datetime

    from vectraudit.cloud import quotas

    start = datetime.strptime(month, "%Y-%m") if month else quotas.month_start()
    with _sessions()() as session:
        rows = quotas.usage_rows(session, start.year, start.month)
    fields = ["workspace", "plan", "month", "runs", "queries", "checks", "stored_runs", "members"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


@app.command("set-plan")
def set_plan(
    slug: Annotated[str, typer.Argument(help="Workspace slug.")],
    plan: Annotated[str, typer.Argument(help="pilot, team or enterprise.")],
    limit: Annotated[
        list[str] | None,
        typer.Option(help="Override, e.g. runs_per_month=2000. Repeatable."),
    ] = None,
) -> None:
    """Change a workspace's plan and limit overrides (audited)."""
    from dataclasses import fields as dc_fields

    from vectraudit.cloud import quotas, repo
    from vectraudit.cloud.db import transaction

    if plan not in quotas.PLANS:
        err_console.print(f"[bold red]Error:[/] plan must be one of {', '.join(quotas.PLANS)}.")
        raise typer.Exit(code=1)
    names = {f.name for f in dc_fields(quotas.Limits)}
    overrides: dict[str, int] = {}
    for item in limit or []:
        key, _, value = item.partition("=")
        if key not in names or not value.isdigit() or int(value) < 1:
            err_console.print(
                f"[bold red]Error:[/] bad limit {escape(item)!s}; keys: {', '.join(sorted(names))}."
            )
            raise typer.Exit(code=1)
        overrides[key] = int(value)
    with transaction(_sessions()) as session:
        ws = session.scalars(select(Workspace).where(Workspace.slug == slug)).first()
        if ws is None:
            err_console.print(f"[bold red]Error:[/] no workspace {escape(slug)}.")
            raise typer.Exit(code=1)
        ws.plan = plan
        ws.limits = overrides or None
        repo.audit(
            session, ws.id, "cli", "workspace.plan", "workspace", ws.id, plan=plan, limits=overrides
        )
        effective = quotas.limits_for(ws)
    console.print(f"{escape(slug)}: plan {plan}, limits {effective}")


@app.command("delete-user")
def delete_user(
    email: Annotated[str, typer.Argument(help="The person to remove.")],
    yes: Annotated[bool, typer.Option("--yes", help="Confirm: this can't be undone.")] = False,
) -> None:
    """Remove a person: memberships, tokens and sessions (workspace data stays)."""
    from vectraudit.cloud import repo, service
    from vectraudit.cloud.db import transaction
    from vectraudit.cloud.errors import CloudError
    from vectraudit.cloud.models import User

    if not yes:
        err_console.print("[bold red]Error:[/] pass --yes to confirm the deletion.")
        raise typer.Exit(code=1)
    try:
        with transaction(_sessions()) as session:
            user = session.scalars(
                select(User).where(User.email == repo.normalize_email(email))
            ).first()
            if user is None:
                err_console.print(f"[bold red]Error:[/] no user {escape(email)}.")
                raise typer.Exit(code=1)
            left = service.delete_user(session, user, "cli")
    except CloudError as e:
        err_console.print(f"[bold red]Error:[/] {escape(e.message)}")
        if e.hint:
            err_console.print(f"[dim]Hint: {escape(e.hint)}[/]")
        raise typer.Exit(code=1) from e
    console.print(f"Deleted {escape(email)}; removed from: {', '.join(left) or 'no workspaces'}.")


@app.command("revoke-access")
def revoke_access(
    workspace: Annotated[
        str | None, typer.Option(help="Only this workspace (default: everything).")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm.")] = False,
) -> None:
    """Incident containment: revoke runner tokens, personal tokens and sessions.

    With --workspace, revokes that workspace's runner tokens and the personal tokens and
    sessions of its members (which also ends their access elsewhere). Without it, revokes
    every token and session on the service.
    """
    from sqlalchemy import ColumnElement, update

    from vectraudit.cloud import repo
    from vectraudit.cloud.db import transaction
    from vectraudit.cloud.models import ApiToken, Membership, WebSession, utcnow

    if not yes:
        err_console.print("[bold red]Error:[/] pass --yes to confirm.")
        raise typer.Exit(code=1)
    now = utcnow()
    with transaction(_sessions()) as session:
        ws_id = None
        token_scope: ColumnElement[bool]
        session_scope: ColumnElement[bool]
        if workspace is not None:
            ws_id = session.scalar(select(Workspace.id).where(Workspace.slug == workspace))
            if ws_id is None:
                err_console.print(f"[bold red]Error:[/] no workspace {escape(workspace)}.")
                raise typer.Exit(code=1)
            users = select(Membership.user_id).where(Membership.workspace_id == ws_id)
            token_scope = (ApiToken.workspace_id == ws_id) | ApiToken.user_id.in_(users)
            session_scope = WebSession.user_id.in_(users)
        else:
            token_scope = ApiToken.id.is_not(None)
            session_scope = WebSession.id.is_not(None)
        tokens = session.execute(
            update(ApiToken)
            .where(token_scope, ApiToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        sessions = session.execute(
            update(WebSession)
            .where(session_scope, WebSession.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        counts = {
            "tokens": int(getattr(tokens, "rowcount", 0) or 0),
            "sessions": int(getattr(sessions, "rowcount", 0) or 0),
        }
        repo.audit(
            session, ws_id, "cli", "incident.revoke_access", "workspace", ws_id or "*", **counts
        )
    scope = f"workspace {workspace}" if workspace else "the whole service"
    console.print(
        f"Revoked {counts['tokens']} tokens and {counts['sessions']} sessions for {escape(scope)}."
    )


@app.command()
def audit(
    workspace: Annotated[str, typer.Argument(help="Workspace slug.")],
    limit: Annotated[int, typer.Option(min=1, max=100_000)] = 1000,
) -> None:
    """Print a workspace's audit log as JSON lines, newest first (investigations)."""
    import json

    from vectraudit.cloud.models import AuditEvent

    with _sessions()() as session:
        ws_id = session.scalar(select(Workspace.id).where(Workspace.slug == workspace))
        if ws_id is None:
            err_console.print(f"[bold red]Error:[/] no workspace {escape(workspace)}.")
            raise typer.Exit(code=1)
        events = session.scalars(
            select(AuditEvent)
            .where(AuditEvent.workspace_id == ws_id)
            .order_by(AuditEvent.created_at.desc())
            .limit(limit)
        )
        for e in events:
            print(
                json.dumps(
                    {
                        "at": e.created_at.isoformat(),
                        "actor": e.actor,
                        "action": e.action,
                        "target_type": e.target_type,
                        "target_id": e.target_id,
                        "details": e.details,
                    },
                    default=str,
                )
            )


def asgi_app() -> object:
    """ASGI factory for uvicorn workers."""
    from vectraudit.cloud.app import create_app
    from vectraudit.cloud.logs import configure

    configure()
    return create_app(Settings.from_env())
