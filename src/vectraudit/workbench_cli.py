"""Local retrieval workbench commands: execute, optimize, inspect and guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from vectraudit.evaluation.dataset import load_dataset
from vectraudit.evaluation.policy import DEFAULT_POLICY, load_policy
from vectraudit.evaluation.runfile import load_run
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.manifest import candidate_config, load_manifest, open_adapter
from vectraudit.pipelines.monitor import monitor_check
from vectraudit.pipelines.optimize import optimize
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline

workbench_app = typer.Typer(
    no_args_is_help=True, help="Test complete pipelines, compare fixes and guard releases locally."
)


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temporary name and atomic replace keep scheduled readers from seeing partial JSON.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        temporary = Path(f.name)
        json.dump(data, f, indent=2, allow_nan=False, ensure_ascii=False)
        f.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@workbench_app.command("run")
def run_command(
    manifest: Path,
    out: Annotated[Path, typer.Option(help="Directory for run artifacts.")] = Path(
        "workbench-runs"
    ),
) -> None:
    """Execute trusted Python, HTTP or built-in hybrid candidates from a local manifest."""
    try:
        spec = load_manifest(manifest)
        if len(spec.candidates) > spec.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        ds = load_dataset(manifest.parent / spec.dataset)
        if len({c.name for c in spec.candidates}) != len(spec.candidates):
            raise ValueError("candidate names must be unique")
        incomplete = False
        for candidate in spec.candidates:
            config = candidate_config(candidate, manifest.parent)
            pipeline = open_adapter(candidate.adapter, manifest.parent)
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
            )
            _write(out / f"{run['run_id']}.run.json", run)
            typer.echo(f"{candidate.name}: {run['status']} ({run['run_id']})")
            incomplete |= run["status"] != "complete"
        if incomplete:
            raise typer.Exit(2)
    except (OSError, ValueError, ImportError, AttributeError) as exc:
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
) -> None:
    """Open the local workbench without requiring a FAISS index. No data is uploaded."""
    import uvicorn

    from vectraudit.pipelines.web import create_workbench

    uvicorn.run(create_workbench(directory, manifest), host="127.0.0.1", port=port)
