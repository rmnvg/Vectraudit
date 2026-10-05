"""Local labelling commands."""

from pathlib import Path
from typing import Annotated

import typer

from vectraudit.experiments.corpus import load_corpus
from vectraudit.labelling.pooling import load_questions, pool
from vectraudit.labelling.task import save_task

label_app = typer.Typer(help="Pool and label source documents locally.", no_args_is_help=True)


@label_app.command("pool")
def pool_command(
    queries: Path,
    corpus: Path,
    out: Annotated[Path, typer.Option()],
    k: int = 10,
    run: Annotated[list[Path] | None, typer.Option()] = None,
    experiment: Annotated[Path | None, typer.Option(help="Experiment result directory.")] = None,
    manifest: Path | None = None,
    lexical: bool = True,
    name: str = "labels",
) -> None:
    """Create a private task from CSV/JSONL questions and a JSONL corpus."""
    try:
        runs = list(run or [])
        if experiment is not None:
            candidates = sorted((experiment / "runs").glob("*.json"))
            if not candidates:
                raise ValueError("Experiment directory has no runs")
            runs.extend(candidates)
        task = pool(
            load_questions(queries),
            load_corpus(corpus),
            k=k,
            runs=runs,
            manifest=manifest,
            lexical=lexical,
            name=name,
        )
        if out.exists():
            raise ValueError("Output already exists; choose a new task path")
        save_task(task, out)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Saved {len(task.queries)} questions to {out}; content hash {task.sha256}")
