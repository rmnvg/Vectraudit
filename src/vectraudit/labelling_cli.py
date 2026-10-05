"""Local labelling commands."""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, ParamSpec

import typer

from vectraudit.experiments.corpus import load_corpus
from vectraudit.labelling.pooling import load_questions, pool
from vectraudit.labelling.task import save_task

label_app = typer.Typer(help="Pool and label source documents locally.", no_args_is_help=True)


P = ParamSpec("P")


def user_errors(command: Callable[P, None]) -> Callable[P, None]:
    """Show actionable task/file errors without a Python traceback."""

    @wraps(command)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except (ValueError, OSError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

    return wrapped


@label_app.command("pool")
@user_errors
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
@user_errors
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
@user_errors
def agreement_command(task: Path) -> None:
    """Print pairwise kappa and disagreements as JSON."""
    import json

    from vectraudit.labelling.agreement import agreement
    from vectraudit.labelling.task import load_task

    typer.echo(json.dumps(agreement(load_task(task)), indent=2))


@label_app.command("sample")
@user_errors
def sample_command(task: Path, annotator: str, fraction: float = 0.2, seed: int = 0) -> None:
    """Assign an annotator a blind second pass with an atomic task transaction."""
    from vectraudit.labelling.agreement import assign_sample
    from vectraudit.labelling.storage import update_task

    ids = update_task(task, lambda value: assign_sample(value, annotator, fraction, seed=seed))
    typer.echo(f"Assigned {len(ids)} questions to {annotator}")


@label_app.command("resolve")
@user_errors
def resolve_command(
    task: Path,
    query_id: str,
    doc_id: str,
    reviewer: str,
    grade: int,
    start: int | None = None,
    end: int | None = None,
) -> None:
    """Record the reviewer's final decision with an atomic task transaction."""
    from vectraudit.labelling.operations import resolve_judgment
    from vectraudit.labelling.storage import update_task
    from vectraudit.labelling.task import Judgment

    judgment = Judgment(
        query_id=query_id, doc_id=doc_id, annotator=reviewer, grade=grade, start=start, end=end
    )
    update_task(task, lambda value: resolve_judgment(value, judgment))


@label_app.command("export")
@user_errors
def export_command(task: Path, out: Annotated[Path, typer.Option()], force: bool = False) -> None:
    """Export human labels to a versioned dataset without changing the task."""
    from vectraudit.evaluation.dataset import parse_dataset
    from vectraudit.labelling.export import export_dataset
    from vectraudit.labelling.task import atomic_write, load_task

    try:
        canonical = task.resolve()
        if out.resolve() in (canonical, canonical.with_name(canonical.name + ".lock")):
            raise ValueError("Dataset output must not overwrite the task")
        if out.exists() and not force:
            raise ValueError("Dataset output exists; use a new version path or --force.")
        text = export_dataset(load_task(task))
        dataset = parse_dataset(text)
        atomic_write(out, text)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Exported {dataset.n_labelled}/{len(dataset)} labelled queries; {dataset.sha256}")
    if dataset.n_labelled < len(dataset):
        typer.echo(
            "Questions without positive labels remain unlabelled and have no quality metrics."
        )


@label_app.command("suggest")
@user_errors
def suggest_command(
    task: Path,
    llm_model: Annotated[Path, typer.Option(help="Explicit opt-in: local causal model directory.")],
) -> None:
    """Generate optional suggestions locally; stop the server first. No labels are accepted."""
    from vectraudit.labelling.suggestions import LocalLLMJudge, suggest
    from vectraudit.labelling.task import load_task

    value = load_task(task)
    suggest(value, LocalLLMJudge(llm_model), name=f"local-llm:{llm_model.name}")
    save_task(value, task)
    typer.echo("Suggestions saved. Each requires a human decision in label serve.")


@label_app.command("status")
@user_errors
def status_command(task: Path, as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Show coverage, disagreements and whether a human-labelled export is ready."""
    import json

    from vectraudit.labelling.operations import task_status
    from vectraudit.labelling.task import load_task

    stats = task_status(load_task(task))
    if as_json:
        typer.echo(json.dumps(stats, indent=2))
    else:
        typer.echo(
            f"{stats['queries_with_positive_labels']}/{stats['queries']} "
            "questions with positive labels; "
            f"{stats['unresolved_pairs']} unresolved disagreements; "
            f"{stats['unjudged_pairs']} unjudged or skipped pairs."
        )
        typer.echo(
            "Export ready." if stats["export_ready"] else "Export needs human labels or review."
        )
