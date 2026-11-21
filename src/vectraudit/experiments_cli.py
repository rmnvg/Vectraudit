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


@experiment_app.command("project")
def experiment_project(
    manifest: Annotated[Path, typer.Argument(help="Experiment manifest (JSON).")],
    out: Annotated[Path, typer.Option(help="Report file (JSON).")] = Path("projection.json"),
    fractions: Annotated[
        str, typer.Option(help="Nested sample sizes as fractions of the corpus file.")
    ] = "0.04,0.06,0.08,0.1",
    full_size: Annotated[
        int | None,
        typer.Option(
            min=1, help="Documents in the full corpus (default: the corpus file's count)."
        ),
    ] = None,
    max_multiple: Annotated[
        float,
        typer.Option(min=1, help="Refuse to project beyond this multiple of the largest sample."),
    ] = 10.0,
    seed: Annotated[int, typer.Option(help="Seed for the samples and the interval.")] = 0,
    cache: Annotated[
        bool, typer.Option(help="Reuse cached document embeddings between runs.")
    ] = True,
) -> None:
    """Project each candidate's recall@k from corpus samples to the full corpus.

    Nested, seeded samples always hold every labelled document; only the largest is embedded.
    Exit 2 when the projection is refused (too far beyond the measured sizes).
    """
    from vectraudit.core._faiss import FaissNotInstalledError
    from vectraudit.core.embed import EmbedderUnavailableError
    from vectraudit.core.projection import default_cache_root
    from vectraudit.experiments.manifest import ExperimentError, load_manifest
    from vectraudit.experiments.project import project_experiment

    try:
        parsed = [float(f) for f in fractions.split(",") if f.strip()]
        spec = load_manifest(manifest)
        with console.status("Projecting…") as status:
            report = project_experiment(
                spec,
                fractions=parsed,
                full_size=full_size,
                max_multiple=max_multiple,
                seed=seed,
                cache_root=default_cache_root() / "embeddings" if cache else None,
                progress=lambda msg: status.update(escape(msg)),
            )
    except (ExperimentError, ValueError) as e:
        err_console.print(f"[bold red]Projection refused:[/] {escape(str(e))}")
        raise typer.Exit(code=2) from e
    except (FaissNotInstalledError, EmbedderUnavailableError) as e:
        err_console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1) from e
    k = report["k"]
    table = Table("Candidate", *(f"recall@{k} at {n}" for n in report["measured_sizes"]))
    table.add_column(f"projected at {report['full_size']} [95%]")
    table.add_column("P(best)")
    for c in sorted(report["candidates"], key=lambda c: -c["projected"]["recall"]):
        lo, hi = c["projected"]["interval"]
        table.add_row(
            c["name"],
            *(f"{m['recall']:.3f}" for m in c["measured"]),
            f"{c['projected']['recall']:.3f} [{lo:.3f}, {hi:.3f}]",
            f"{c['p_best_at_full_size']:.0%}",
        )
    console.print(table)
    console.print(escape(report["message"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(f"Wrote {escape(str(out))}")


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
