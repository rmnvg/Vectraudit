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


@label_app.command("serve")
def serve_command(task: Path, port: int = 8766) -> None:
    """Label a task in a local browser; binds only to 127.0.0.1."""
    import uvicorn

    from vectraudit.labelling.web import create_app

    try:
        application = create_app(task)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Local labelling: http://127.0.0.1:{port}")
    uvicorn.run(application, host="127.0.0.1", port=port, access_log=False)


@label_app.command("agreement")
def agreement_command(task: Path) -> None:
    """Print pairwise kappa and disagreements as JSON."""
    import json

    from vectraudit.labelling.agreement import agreement
    from vectraudit.labelling.task import load_task

    typer.echo(json.dumps(agreement(load_task(task)), indent=2))


@label_app.command("sample")
def sample_command(task: Path, annotator: str, fraction: float = 0.2, seed: int = 0) -> None:
    """Assign an annotator a blind second pass; stop the server before modifying."""
    from vectraudit.labelling.agreement import assign_sample
    from vectraudit.labelling.task import load_task

    value = load_task(task)
    ids = assign_sample(value, annotator, fraction, seed=seed)
    save_task(value, task)
    typer.echo(f"Assigned {len(ids)} questions to {annotator}")


@label_app.command("resolve")
def resolve_command(
    task: Path,
    query_id: str,
    doc_id: str,
    reviewer: str,
    grade: int,
    start: int | None = None,
    end: int | None = None,
) -> None:
    """Record the reviewer's final decision; stop the server before modifying."""
    from vectraudit.labelling.task import Judgment, load_task

    value = load_task(task)
    judgment = Judgment(
        query_id=query_id, doc_id=doc_id, annotator=reviewer, grade=grade, start=start, end=end
    )
    value.resolutions = [
        j for j in value.resolutions if (j.query_id, j.doc_id) != (query_id, doc_id)
    ] + [judgment]
    save_task(value, task)
