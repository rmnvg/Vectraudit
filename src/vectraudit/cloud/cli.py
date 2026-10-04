"""``vectraudit-cloud``: run and administer the hosted control plane."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

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


def asgi_app() -> object:
    """ASGI factory for uvicorn workers."""
    from vectraudit.cloud.app import create_app

    return create_app(Settings.from_env())
