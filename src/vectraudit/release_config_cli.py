"""``vectraudit release``: validate, run and check an exported release configuration.

Exit codes of ``check`` match ``vectraudit check``: 0 pass, 1 error, 3 fail, 4 not comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from vectraudit.evaluation.compare import Verdict
from vectraudit.evaluation.runfile import RunFileError, load_run
from vectraudit.release_cli import EXIT_CODES, _fail, console, emit_check, err_console

release_app = typer.Typer(
    no_args_is_help=True,
    help="Run the same release check locally and in CI from an exported configuration.",
)


@release_app.command("validate")
def validate_command(
    release: Annotated[Path, typer.Argument(help="release.json from a workbench export.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Check files, hashes and prerequisites without running the pipeline. Exit 1 on errors."""
    from vectraudit.pipelines.release import validate_release

    issues = validate_release(release)
    if as_json:
        typer.echo(json.dumps({"issues": issues}, indent=2))
    else:
        for issue in issues:
            style = "bold red" if issue["level"] == "error" else "yellow"
            console.print(f"[{style}]{issue['level'].title()}:[/] {escape(issue['message'])}")
        if not issues:
            console.print("[green]Release configuration is ready.[/]")
    if any(i["level"] == "error" for i in issues):
        raise typer.Exit(code=1)


@release_app.command("run")
def run_command(
    release: Annotated[Path, typer.Argument(help="This checkout's release.json.")],
    out: Annotated[Path, typer.Option(help="Run file to write.")],
    approved: Annotated[
        Path | None,
        typer.Option(help="Approved release.json whose benchmark and k both sides use."),
    ] = None,
    label: Annotated[str | None, typer.Option(help="Run label.")] = None,
) -> None:
    """Run this checkout's pipeline on the approved benchmark. Exit 2 if not complete."""
    from vectraudit.evaluation.runfile import save_run
    from vectraudit.pipelines.release import ReleaseError, run_release

    try:
        run = run_release(release, approved_path=approved, label=label)
    except ReleaseError as e:
        raise _fail(str(e)) from e
    out.parent.mkdir(parents=True, exist_ok=True)
    save_run(run, out)
    s = run["summary"]
    console.print(f"{escape(run['label'] or '')}: {run['status']} → {escape(str(out))}")
    if s["metrics"]:
        console.print(", ".join(f"{k} {v:.4f}" for k, v in s["metrics"].items()))
    if run["status"] != "complete":
        err_console.print(f"[bold red]{s['n_errors']:,} queries failed.[/]")
        raise typer.Exit(code=2)


@release_app.command("check")
def check_command(
    approved: Annotated[Path, typer.Argument(help="Approved release.json (policy source).")],
    baseline: Annotated[Path, typer.Argument(help="Baseline run file.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate run file.")],
    proposed: Annotated[
        Path | None, typer.Option(help="This change's release.json; its edits are reported.")
    ] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write the result JSON.")] = None,
    html: Annotated[Path | None, typer.Option(help="Write an HTML report.")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Write a Markdown summary.")] = None,
    github_summary: Annotated[
        bool, typer.Option(help="Append the Markdown summary to $GITHUB_STEP_SUMMARY when set.")
    ] = True,
    report_url: Annotated[str | None, typer.Option(help="Link to the full report.")] = None,
    github_check: Annotated[bool, typer.Option(help="Publish a check run.")] = False,
    github_comment: Annotated[bool, typer.Option(help="Create or update one PR comment.")] = False,
    check_name: Annotated[str, typer.Option(help="Check run name.")] = "Retrieval release check",
) -> None:
    """Decide under the approved policy and benchmark.

    Exit codes: 0 pass, 1 error, 3 fail, 4 not comparable.
    """
    from vectraudit.pipelines.release import ReleaseError, check_release

    try:
        result = check_release(
            approved, load_run(baseline), load_run(candidate), proposed_path=proposed
        )
    except (ReleaseError, RunFileError) as e:
        raise _fail(str(e)) from e
    release = result["release"]
    console.print(
        f"{escape(release['name'])} v{release['version']} · policy "
        f"{escape(release['policy']['name'])} v{release['policy']['version']} "
        f"({release['policy']['sha256'][:12]})"
    )
    emit_check(
        result,
        json_out=json_out,
        html=html,
        markdown=markdown,
        github_summary=github_summary,
        report_url=report_url,
        github_check=github_check,
        github_comment=github_comment,
        check_name=check_name,
    )
    raise typer.Exit(code=EXIT_CODES[Verdict(result["verdict"])])


@release_app.command("update")
def update_command(
    release: Annotated[Path, typer.Argument(help="release.json to re-approve.")],
) -> None:
    """Record reviewed edits to the policy or benchmark as a new configuration version."""
    from vectraudit.pipelines.release import ReleaseError, update_release

    try:
        updated, changes = update_release(release)
    except ReleaseError as e:
        raise _fail(str(e)) from e
    if not changes:
        console.print("No changes: the benchmark and policy match the configuration.")
        return
    for change in changes:
        console.print(f"Updated {escape(change)}")
    console.print(
        f"{escape(updated.name)} is now configuration v{updated.version}. Commit it for review."
    )
