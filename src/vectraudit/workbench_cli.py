"""Local retrieval workbench commands: execute, optimize, inspect and guard."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer

from vectraudit.evaluation.dataset import load_dataset
from vectraudit.evaluation.policy import DEFAULT_POLICY, load_policy
from vectraudit.evaluation.runfile import load_run
from vectraudit.labelling.storage import task_lock
from vectraudit.labelling.task import atomic_write
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.checkpoint import validate_path
from vectraudit.pipelines.manifest import candidate_config, load_manifest, open_adapter
from vectraudit.pipelines.monitor import monitor_check
from vectraudit.pipelines.optimize import optimize
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline

workbench_app = typer.Typer(
    no_args_is_help=True, help="Test complete pipelines, compare fixes and guard releases locally."
)


def _write(path: Path, data: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False) + "\n")


@workbench_app.command("run")
def run_command(
    manifest: Path,
    out: Annotated[Path, typer.Option(help="Directory for run artifacts.")] = Path(
        "workbench-runs"
    ),
    resume: Annotated[
        Path | None, typer.Option(help="Resume the printed checkpoint directory.")
    ] = None,
    retry_errors: Annotated[
        bool, typer.Option(help="Explicitly retry failed queries when resuming.")
    ] = False,
) -> None:
    """Execute trusted Python, HTTP or built-in hybrid candidates from a local manifest."""
    try:
        if retry_errors and resume is None:
            raise ValueError("--retry-errors requires --resume")
        spec = load_manifest(manifest)
        if len(spec.candidates) > spec.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        ds = load_dataset(manifest.parent / spec.dataset)
        if len({c.name for c in spec.candidates}) != len(spec.candidates):
            raise ValueError("candidate names must be unique")
        checkpoints = resume or out / ".checkpoints" / uuid.uuid4().hex
        configs = [candidate_config(candidate, manifest.parent) for candidate in spec.candidates]
        batch_identity = {
            "dataset": ds.sha256,
            "candidates": configs,
            "k": spec.k,
            "repeats": spec.repeats,
            "corpus_version": spec.corpus_version,
        }
        batch_path = checkpoints / "batch.json"
        validate_path(batch_path)
        with task_lock(batch_path, timeout=0):
            if resume:
                batch = json.loads(batch_path.read_text())
                if batch["identity"] != batch_identity:
                    raise ValueError("Cannot resume: dataset, candidates or settings changed")
            else:
                batch = {"identity": batch_identity, "started": []}
                _write(batch_path, batch)
            typer.echo(f"Checkpoints: {checkpoints}")
            incomplete = False
            for candidate, config in zip(spec.candidates, configs, strict=True):
                checkpoint = checkpoints / (
                    hashlib.sha256(candidate.name.encode()).hexdigest() + ".sqlite"
                )
                if candidate.name in batch["started"] and not checkpoint.exists():
                    raise ValueError(
                        "A started checkpoint is missing; restore it or start a new batch"
                    )
                resuming = bool(resume) and checkpoint.exists()
                pipeline = open_adapter(candidate.adapter, manifest.parent)
                if candidate.name not in batch["started"]:
                    batch["started"].append(candidate.name)
                    _write(batch_path, batch)
                run = execute_pipeline(
                    ds,
                    pipeline,
                    PipelineOptions(
                        k=spec.k,
                        params=candidate.params,
                        repeats=spec.repeats,
                        warmup=0,
                        label=candidate.name,
                    ),
                    config=config,
                    corpus_version=spec.corpus_version or config.get("corpus_sha256"),
                    checkpoint=checkpoint,
                    resume=resuming,
                    retry_errors=retry_errors and resuming,
                )
                _write(out / f"{run['run_id']}.run.json", run)
                typer.echo(f"{candidate.name}: {run['status']} ({run['run_id']})")
                incomplete |= run["status"] != "complete"
                if run["status"] == "cancelled":
                    break
            if incomplete:
                raise typer.Exit(2)
    except (OSError, ValueError, ImportError, AttributeError, sqlite3.Error) as exc:
        typer.echo(f"Workbench configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc


@workbench_app.command("optimize")
def optimize_command(manifest: Path, out: Path = Path("optimization.json")) -> None:
    """Bounded candidate search; one held-out validation; recommendations require review."""
    try:
        spec = load_manifest(manifest)
        if len(spec.candidates) > spec.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        ds = load_dataset(manifest.parent / spec.dataset)
        candidates = [
            (c.name, open_adapter(c.adapter, manifest.parent), c.params) for c in spec.candidates
        ]
        result = optimize(
            ds,
            candidates,
            k=spec.k,
            max_candidates=spec.max_candidates,
            max_p95_ms=spec.max_p95_ms,
            max_cost_usd=spec.max_cost_usd,
            max_memory_bytes=spec.max_memory_bytes,
            repeats=spec.repeats,
            corpus_version=spec.corpus_version,
        )
        _write(out, result)
        typer.echo(
            f"Selected: {result['selected'] or 'none'}; held-out eligible: {result['eligible']}"
        )
        if not result["eligible"]:
            raise typer.Exit(3)
    except (OSError, ValueError, ImportError, AttributeError) as exc:
        typer.echo(f"Optimization error: {exc}", err=True)
        raise typer.Exit(2) from exc


@workbench_app.command("benchmark")
def benchmark_command(
    dataset: Path, out: Path = Path("benchmark.json"), runs: Path | None = None
) -> None:
    """Audit coverage and duplicates; prioritize exploratory judgments from run pools."""
    try:
        result = audit_benchmark(
            load_dataset(dataset),
            [load_run(p) for p in sorted(runs.glob("*.run.json"))] if runs else [],
        )
        _write(out, result)
        typer.echo(
            f"{result['n_labelled']}/{result['n_queries']} annotated; "
            f"{len(result['duplicates'])} duplicate groups"
        )
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@workbench_app.command("monitor")
def monitor_command(
    manifest: Path,
    baseline: Path,
    history: Path = Path("monitor-history"),
    policy: Path | None = None,
) -> None:
    """One scheduled check against a frozen baseline. Exit 3 alerts the scheduler."""
    try:
        spec = load_manifest(manifest)
        if len(spec.candidates) != 1:
            raise ValueError("monitor manifest must select exactly one candidate")
        ds = load_dataset(manifest.parent / spec.dataset)
        base = load_run(baseline)
        if base["status"] != "complete" or base["dataset"]["sha256"] != ds.sha256:
            raise ValueError("monitor needs a complete baseline from this frozen benchmark")
        candidate = spec.candidates[0]
        config = candidate_config(candidate, manifest.parent)
        run = execute_pipeline(
            ds,
            open_adapter(candidate.adapter, manifest.parent),
            PipelineOptions(k=spec.k, params=candidate.params, repeats=spec.repeats, warmup=0),
            config=config,
            corpus_version=spec.corpus_version or config.get("corpus_sha256"),
        )
        result = monitor_check(base, run, load_policy(policy) if policy else DEFAULT_POLICY)
        _write(history / f"{run['run_id']}.run.json", run)
        _write(history / f"{run['run_id']}.monitor.json", result)
        typer.echo(json.dumps({"verdict": result["verdict"], "alerts": result["alerts"]}))
        if result["verdict"] != "pass":
            raise typer.Exit(3)
    except (OSError, ValueError, ImportError, AttributeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@workbench_app.command("serve")
def serve_command(
    directory: Annotated[Path, typer.Argument()] = Path("workbench-runs"),
    port: int = 8766,
    manifest: Annotated[
        Path | None, typer.Option(help="Register trusted pipeline adapters for the guided UI.")
    ] = None,
    policy: Annotated[
        Path | None,
        typer.Option(help="Offer this policy file (e.g. your CI policy) for comparisons."),
    ] = None,
) -> None:
    """Open the local workbench without requiring a FAISS index. No data is uploaded."""
    import uvicorn

    from vectraudit.evaluation.policy import PolicyError, load_policy
    from vectraudit.pipelines.web import create_workbench

    if policy is not None:
        try:
            load_policy(policy)
        except PolicyError as exc:
            typer.echo(f"Policy error: {exc}", err=True)
            raise typer.Exit(2) from exc
    uvicorn.run(create_workbench(directory, manifest, policy), host="127.0.0.1", port=port)


@workbench_app.command("export")
def export_command(
    release: Annotated[str, typer.Argument(help="Release configuration id from the workbench.")],
    workspace: Annotated[Path, typer.Option(help="Workbench directory.")] = Path("workbench-runs"),
    out: Annotated[Path, typer.Option(help="Repository root to write into.")] = Path(),
    force: Annotated[bool, typer.Option(help="Overwrite files that already exist.")] = False,
) -> None:
    """Write an exported release check into a repository for review. Nothing is pushed."""
    import re

    if not re.fullmatch(r"[a-f0-9]{32}", release):
        typer.echo("Release configuration ids are 32 hexadecimal characters.", err=True)
        raise typer.Exit(2)
    path = workspace / ".workbench" / "releases" / f"{release}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        typer.echo(f"Cannot read release configuration {path}: {exc}", err=True)
        raise typer.Exit(2) from exc
    written, skipped = [], []
    for relative, content in sorted(record["files"].items()):
        target = out / relative
        if target.exists() and not force:
            skipped.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    for target in written:
        typer.echo(f"wrote {target}")
    for target in skipped:
        typer.echo(f"skipped {target} (exists; --force overwrites)")
    for item in record["issues"]["prerequisites"]:
        typer.echo(f"prerequisite: {item}")
    for item in record["issues"]["warnings"]:
        typer.echo(f"warning: {item}")
    if skipped:
        raise typer.Exit(1)
