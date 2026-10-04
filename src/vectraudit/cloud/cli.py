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
) -> None:
    """Serve the API (run `migrate` first)."""
    import uvicorn

    _settings()  # fail fast on bad configuration
    uvicorn.run(
        "vectraudit.cloud.cli:asgi_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        proxy_headers=True,
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
    """Delete runs past their workspace's retention (pinned baselines are kept)."""
    from vectraudit.cloud import quotas, repo
    from vectraudit.cloud.db import transaction

    with transaction(_sessions()) as session:
        deleted = quotas.purge_expired(session)
        for slug, n in deleted.items():
            ws_id = session.scalars(select(Workspace.id).where(Workspace.slug == slug)).first()
            repo.audit(session, ws_id, "cli", "retention.purge", "workspace", ws_id or slug, runs=n)
    total = sum(deleted.values())
    console.print(
        f"Purged {total:,} expired runs"
        + (": " + ", ".join(f"{k} {v}" for k, v in deleted.items()) if deleted else ".")
    )


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


def asgi_app() -> object:
    """ASGI factory for uvicorn workers."""
    from vectraudit.cloud.app import create_app

    return create_app(Settings.from_env())
