"""``vectraudit experiment run``: compare candidate retrieval setups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from vectraudit.evaluation.runfile import save_run

experiment_app = typer.Typer(
    help="Compare embedding models, chunkings and indexes on one labelled set.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


@experiment_app.command("run")
def experiment_run(
    manifest: Annotated[Path, typer.Argument(help="Experiment manifest (JSON).")],
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("experiment-results"),
    primary: Annotated[str, typer.Option(help="Ranking metric: ndcg, recall, mrr.")] = "ndcg",
    cache: Annotated[
        bool, typer.Option(help="Reuse cached document embeddings between runs.")
    ] = True,
) -> None:
    """Run every candidate and write result.json, report.html, summary.md and runs/."""
    from vectraudit.core._faiss import FaissNotInstalledError
    from vectraudit.core.embed import EmbedderUnavailableError
    from vectraudit.core.projection import default_cache_root
    from vectraudit.experiments.manifest import ExperimentError, load_manifest
    from vectraudit.experiments.report import rank, render_html, render_markdown, to_json
    from vectraudit.experiments.runner import run_experiment

    try:
        spec = load_manifest(manifest)
        with console.status("Running candidates…") as status:
            summary, results = run_experiment(
                spec,
                cache_root=default_cache_root() / "embeddings" if cache else None,
                progress=lambda msg: status.update(escape(msg)),
            )
        ranked = rank(summary, results, primary)
    except (ExperimentError, ValueError) as e:
        err_console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1) from e
    except (FaissNotInstalledError, EmbedderUnavailableError) as e:
        err_console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1) from e
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        save_run(r.run, out / "runs" / f"{r.spec.name}.json")
    (out / "result.json").write_text(to_json(ranked) + "\n")
    (out / "report.html").write_text(render_html(ranked), encoding="utf-8")
    (out / "summary.md").write_text(render_markdown(ranked), encoding="utf-8")

    table = Table("#", "Candidate", primary, "Δ vs leader [95% CI]", "Chunks", "p95 ms")
    for row in ranked["ranking"]:
        lo, hi = row["ci_vs_leader"]
        delta = (
            ""
            if row["rank"] == 1
            else (
                f"{row['delta_vs_leader']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                + (" tied" if row["tied_with_leader"] else "")
            )
        )
        p95 = (row["latency_ms"] or {}).get("p95")
        table.add_row(
            str(row["rank"]),
            row["name"],
            f"{row['metrics'][primary]:.3f}",
            delta,
            str(row["stats"]["chunks"]),
            "" if p95 is None else f"{p95:.2f}",
        )
    console.print(table)
    from vectraudit.experiments.report import verdict_sentence

    console.print(verdict_sentence(ranked))
    console.print(
        f"Report: {escape(str(out / 'report.html'))}; runs in {escape(str(out / 'runs'))}/"
    )


@experiment_app.command("export-mteb")
def experiment_export_mteb(
    result: Annotated[
        Path, typer.Argument(help="An experiment result directory (or result.json).")
    ],
    out: Annotated[Path, typer.Option(help="Directory to write <candidate>/<task>.json into.")],
    task_name: Annotated[
        str, typer.Option("--task", help="Name for the task (your corpus and labels).")
    ],
    split: Annotated[str, typer.Option(help="Split name to record (MTEB uses test).")] = "test",
) -> None:
    """Write each candidate's result in the MTEB result layout, to sit beside public scores.

    Not like for like with leaderboard tasks: it is your corpus and labels, averaged over question
    families at the experiment's k. Each file says so, and mteb_version is null.
    """
    from vectraudit.experiments.mteb_export import export_experiment

    path = result / "result.json" if result.is_dir() else result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        written = export_experiment(data, out, task_name=task_name, split=split)
    except (OSError, ValueError) as e:  # MtebExportError is a ValueError
        err_console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1) from e
    for p in written:
        console.print(escape(str(p)))
    console.print(
        "[dim]Not like for like with leaderboard scores: your corpus, your labels, "
        "family-weighted.[/]"
    )
