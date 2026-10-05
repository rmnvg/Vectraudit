"""Offline setup diagnostics."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table


def doctor(
    connection: Path | None = None,
    dataset: Path | None = None,
    policy: Path | None = None,
    embedder: str | None = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print a support-safe JSON report.")
    ] = False,
    strict: Annotated[bool, typer.Option(help="Treat warnings as a failed preflight.")] = False,
) -> None:
    """Inspect local release setup without connecting, loading models or uploading data."""
    import json

    from vectraudit.evaluation.doctor import diagnose

    report = diagnose(connection=connection, dataset=dataset, policy=policy, embedder=embedder)
    if as_json:
        typer.echo(json.dumps(report, indent=2))
    else:
        table = Table("Status", "Check", "Next step")
        for finding in report["checks"]:
            table.add_row(finding["status"], escape(finding["message"]), escape(finding["hint"]))
        Console().print(table)
        typer.echo(f"Offline {report['scope']} checks: {report['status']}.")
        for limit in report["limits"]:
            typer.echo(limit)
    if report["status"] == "blocked" or (strict and report["status"] == "needs_attention"):
        raise typer.Exit(1)
