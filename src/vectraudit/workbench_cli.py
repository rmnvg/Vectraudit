"""Local retrieval workbench commands: execute, optimize, inspect and guard."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from vectraudit.evaluation.dataset import load_dataset
from vectraudit.evaluation.policy import DEFAULT_POLICY, PolicyError, load_policy
from vectraudit.evaluation.runfile import load_run
from vectraudit.labelling.storage import task_lock
from vectraudit.labelling.task import atomic_write
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.budget import planned_for, planned_total, reconcile, reconcile_line
from vectraudit.pipelines.checkpoint import validate_path
from vectraudit.pipelines.guided import write_run
from vectraudit.pipelines.manifest import candidate_config, load_manifest, open_adapter
from vectraudit.pipelines.monitor import monitor_check
from vectraudit.pipelines.optimize import optimize
from vectraudit.pipelines.runner import PipelineOptions, RetryPolicy, execute_pipeline

workbench_app = typer.Typer(
    no_args_is_help=True, help="Test complete pipelines, compare fixes and guard releases locally."
)


def _write(path: Path, data: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False) + "\n")


MaxCalls = Annotated[
    int | None,
    typer.Option(min=1, help="Refuse to start if the planned pipeline calls exceed this."),
]
MaxSpend = Annotated[
    float | None,
    typer.Option(
        min=0,
        help="Refuse to start if planned spend (USD) exceeds this or any cost is unknown.",
    ),
]
Measured = Annotated[
    list[Path] | None,
    typer.Option(help="Earlier run files whose adapter-reported cost_usd prices each call."),
]


def _guard(
    spec: Any,
    ds: Any,
    mode: Any,
    *,
    max_calls: int | None,
    max_spend: float | None,
    measured: list[Path] | None,
    baseline: str | None = None,
    retry: RetryPolicy | None = None,
) -> dict[str, Any]:
    """The plan; refuses before any adapter opens when calls or spend exceed a limit."""
    from vectraudit.pipelines.budget import check_limits, plan

    retry = retry or RetryPolicy()
    budget = plan(
        spec,
        ds,
        mode=mode,
        measured=[load_run(p) for p in measured or []],
        baseline=baseline,
        retry_budget=retry.max_retries,
        max_attempts=retry.max_attempts,
    )
    if max_calls is not None or max_spend is not None:
        check_limits(budget, max_calls=max_calls, max_spend=max_spend)
    return budget


def _used(run: dict[str, Any], planned: dict[str, Any]) -> str:
    """Record the plan beside the run's actual usage; returns the one-line comparison."""
    from vectraudit.pipelines.budget import reconcile, reconcile_line

    run["usage"]["planned"] = planned
    return reconcile_line(reconcile(planned, run["usage"]))


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
    max_attempts: Annotated[
        int,
        typer.Option(
            min=1,
            max=10,
            help="Tries per call for transient failures (timeout, rate limit, 502-504, "
            "connection); 1 disables automatic retries.",
        ),
    ] = RetryPolicy().max_attempts,
    retry_budget: Annotated[
        int, typer.Option(min=0, help="Maximum automatic retries per candidate run.")
    ] = RetryPolicy().max_retries,
    max_calls: MaxCalls = None,
    max_spend: MaxSpend = None,
    measured: Measured = None,
) -> None:
    """Execute trusted Python, HTTP or built-in hybrid candidates from a local manifest.

    See ``workbench preview`` for the calls and spend a run plans; ``--max-calls`` and
    ``--max-spend`` refuse to start one that exceeds them (exit 2).
    """
    try:
        retry = RetryPolicy(max_attempts=max_attempts, max_retries=retry_budget)
        if retry_errors and resume is None:
            raise ValueError("--retry-errors requires --resume")
        spec = load_manifest(manifest)
        if len(spec.candidates) > spec.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        ds = load_dataset(manifest.parent / spec.dataset)
        if len({c.name for c in spec.candidates}) != len(spec.candidates):
            raise ValueError("candidate names must be unique")
        # A resumed batch was already approved when it started: plan it, but don't refuse.
        budget = _guard(
            spec,
            ds,
            "run",
            max_calls=None if resume else max_calls,
            max_spend=None if resume else max_spend,
            measured=measured,
            retry=retry,
        )
        checkpoints = resume or out / ".checkpoints" / uuid.uuid4().hex
        configs = [candidate_config(candidate, manifest.parent) for candidate in spec.candidates]
        batch_identity = {
            "dataset": ds.sha256,
            "candidates": configs,
            "k": spec.k,
            "repeats": spec.repeats,
            "corpus_version": spec.corpus_version,
            **({} if retry == RetryPolicy() else {"retry": asdict(retry)}),
        }
        batch_path = checkpoints / "batch.json"
        # --out (or --resume) is the user's choice; folders created inside it must not be links.
        validate_path(batch_path, resume or out)
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
                        retry=retry,
                        concurrency=candidate.concurrency,
                        max_rps=candidate.max_rps,
                    ),
                    config=config,
                    corpus_version=spec.corpus_version or config.get("corpus_sha256"),
                    checkpoint=checkpoint,
                    resume=resuming,
                    retry_errors=retry_errors and resuming,
                )
                used = _used(
                    run,
                    planned_for(
                        budget,
                        candidate.name,
                        retry_budget=retry.max_retries,
                        max_attempts=retry.max_attempts,
                    ),
                )
                write_run(out / f"{run['run_id']}.run.json", run)
                typer.echo(f"{candidate.name}: {run['status']} ({run['run_id']})")
                typer.echo(f"  {used}")
                incomplete |= run["status"] != "complete"
                if run["status"] == "cancelled":
                    break
            if incomplete:
                raise typer.Exit(2)
    except (OSError, ValueError, ImportError, AttributeError, sqlite3.Error) as exc:
        typer.echo(f"Workbench configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc


@workbench_app.command("optimize")
def optimize_command(
    manifest: Path,
    out: Path = Path("optimization.json"),
    policy: Annotated[
        Path | None, typer.Option(help="Release policy that must approve the chosen candidate.")
    ] = None,
    baseline: Annotated[
        str | None, typer.Option(help="Candidate name of the current configuration.")
    ] = None,
    max_attempts: Annotated[
        int,
        typer.Option(
            min=1,
            max=10,
            help="Tries per call for transient failures; 1 disables automatic retries.",
        ),
    ] = RetryPolicy().max_attempts,
    retry_budget: Annotated[
        int, typer.Option(min=0, help="Maximum automatic retries per candidate run.")
    ] = RetryPolicy().max_retries,
    max_calls: MaxCalls = None,
    max_spend: MaxSpend = None,
    measured: Measured = None,
) -> None:
    """Bounded candidate search; one held-out validation; recommendations require review.

    Each candidate runs with its manifest ``concurrency`` and ``max_rps``, as in
    ``workbench run``.

    With --policy and --baseline, only a candidate that passes the release policy against the
    baseline on held-out questions is reported as approved (exit 0; 3 otherwise). Without a
    policy the result is a best exploratory candidate, never a release approval.
    """
    try:
        if (policy is None) != (baseline is None):
            raise ValueError("--policy and --baseline are used together")
        rules = load_policy(policy) if policy is not None else None
        spec = load_manifest(manifest)
        if len(spec.candidates) > spec.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        ds = load_dataset(manifest.parent / spec.dataset)
        retry = RetryPolicy(max_attempts=max_attempts, max_retries=retry_budget)
        budget = _guard(
            spec,
            ds,
            "optimize",
            max_calls=max_calls,
            max_spend=max_spend,
            measured=measured,
            baseline=baseline,
            retry=retry,
        )
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
            policy=rules,
            baseline=baseline,
            execution={
                c.name: {"concurrency": c.concurrency, "max_rps": c.max_rps}
                for c in spec.candidates
            },
            retry=retry,
        )
        result["usage"] = reconcile(planned_total(budget), result["usage"])
        _write(out, result)
        typer.echo(f"{result['status_label']}: {result['selected'] or 'none'}")
        typer.echo(f"Usage: {reconcile_line(result['usage'])}")
        if result["latency_note"]:
            typer.echo(f"Note: {result['latency_note']}")
        check = (result["heldout"] or {}).get("check")
        if check is not None:
            typer.echo(f"Held-out release check: {check['verdict']}")
            for reason in check["reasons"]:
                typer.echo(f"  - {reason}")
        if rules is None:
            ok = result["eligible"]
        else:
            ok = result["status"] in ("approved_under_policy", "baseline_remains_best")
        if not ok:
            raise typer.Exit(3)
    except (OSError, ValueError, ImportError, AttributeError, PolicyError) as exc:
        typer.echo(f"Optimization error: {exc}", err=True)
        raise typer.Exit(2) from exc


@workbench_app.command("preview")
def preview_command(
    manifest: Path,
    mode: Annotated[str, typer.Option(help="What will run: run, optimize or monitor.")] = "run",
    baseline: Annotated[
        str | None, typer.Option(help="Optimize only: the baseline candidate under a policy.")
    ] = None,
    measured: Measured = None,
    retry_budget: Annotated[
        int, typer.Option(min=0, help="Retry budget the run will use.")
    ] = RetryPolicy().max_retries,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write the plan as JSON.")] = None,
) -> None:
    """Calls and spend a run, optimization or monitor would make, before running anything.

    Reads the manifest and dataset only: no adapter is opened and no code is imported.
    """
    from vectraudit.pipelines.budget import format_usd, plan

    try:
        if mode not in ("run", "optimize", "monitor"):
            raise ValueError("--mode must be run, optimize or monitor")
        spec = load_manifest(manifest)
        ds = load_dataset(manifest.parent / spec.dataset)
        budget = plan(
            spec,
            ds,
            mode=mode,  # type: ignore[arg-type]
            measured=[load_run(p) for p in measured or []],
            baseline=baseline,
            retry_budget=retry_budget,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Preview error: {exc}", err=True)
        raise typer.Exit(2) from exc
    for row in budget["candidates"]:
        price = format_usd(row["cost_per_call_usd"])
        if row["cost_per_call_usd"] is not None:
            price += f" ({row['cost_source']})"
        typer.echo(
            f"{row['name']}: {row['questions']} questions x {budget['repeats']} repeats = "
            f"{row['calls']} calls; per call {price}; spend {format_usd(row['spend_usd'])}"
        )
    held = budget["heldout"]
    if held:
        typer.echo(
            f"Held-out validation: {held['questions']} questions x {budget['repeats']} repeats "
            f"for up to {held['runs_up_to']} candidate(s)"
        )
    calls, spend = budget["calls"], budget["spend_usd"]
    span = str(calls["min"])
    if calls["max"] != calls["min"]:
        span += f" to {calls['max']}"
    typer.echo(f"Total: {span} calls, plus up to {calls['retry_allowance']} retries")
    known = len(budget["candidates"]) - len(budget["unknown_cost"])
    money = format_usd(spend["min"])
    if spend["max"] is not None and spend["max"] != spend["min"]:
        money += f" to {format_usd(spend['max'])}"
    if not known:
        money = "unknown"
    elif not spend["complete"]:
        money += f" for the {known} candidate(s) with known costs"
    typer.echo(f"Spend: {money}")
    for note in budget["notes"]:
        typer.echo(f"- {note}")
    if json_out is not None:
        _write(json_out, budget)


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
    baseline: Annotated[
        str, typer.Argument(help="Frozen baseline run file, or cloud:NAME for a pinned one.")
    ],
    history: Path = Path("monitor-history"),
    policy: Path | None = None,
    upload: Annotated[
        bool,
        typer.Option(
            help="Upload the run and record a hosted monitor check (needs a cloud:NAME baseline)."
        ),
    ] = False,
    project: Annotated[str | None, typer.Option(help="Hosted workspace/project.")] = None,
    max_calls: MaxCalls = None,
    max_spend: MaxSpend = None,
    measured: Measured = None,
) -> None:
    """One scheduled check against a frozen baseline. Exit 3 alerts the scheduler.

    Exit codes: 0 pass (or every failure accepted by exception), 3 alert, 2 invalid local
    input or failed run, 1 hosted service error. With ``--upload`` the check appears in the
    project's timeline, trends and monitor status; the local history is written first.
    """
    cloud = baseline.startswith("cloud:")
    if upload and not cloud:
        typer.echo("--upload compares with a pinned baseline: pass cloud:<name>.", err=True)
        raise typer.Exit(2)
    client = None
    if cloud:
        from vectraudit.release_cli import _cloud_baseline_and_policy, _cloud_client

        client = _cloud_client(project)
    try:
        spec = load_manifest(manifest)
        if len(spec.candidates) != 1:
            raise ValueError("monitor manifest must select exactly one candidate")
        ds = load_dataset(manifest.parent / spec.dataset)
        if cloud:
            base, rules = _cloud_baseline_and_policy(client, baseline[6:], policy)
        else:
            base = load_run(Path(baseline))
            rules = load_policy(policy) if policy else DEFAULT_POLICY
        if base["status"] != "complete" or base["dataset"]["sha256"] != ds.sha256:
            raise ValueError("monitor needs a complete baseline from this frozen benchmark")
        budget = _guard(
            spec, ds, "monitor", max_calls=max_calls, max_spend=max_spend, measured=measured
        )
        candidate = spec.candidates[0]
        config = candidate_config(candidate, manifest.parent)
        run = execute_pipeline(
            ds,
            open_adapter(candidate.adapter, manifest.parent),
            PipelineOptions(k=spec.k, params=candidate.params, repeats=spec.repeats, warmup=0),
            config=config,
            corpus_version=spec.corpus_version or config.get("corpus_sha256"),
        )
        used = _used(run, planned_for(budget, candidate.name))
        compared = run
        if cloud:
            # The hosted baseline is a sharing projection: compare like with like, as the
            # hosted check will, and apply the project's active audited exceptions.
            from vectraudit.integrations.sharing import project_run
            from vectraudit.release_cli import _apply_cloud_exceptions

            compared = project_run(run)
        result = monitor_check(base, compared, rules)
        if cloud:
            result["check"] = _apply_cloud_exceptions(client, result["check"], rules)
            result["verdict"] = result["check"]["verdict"]
            result["alerts"] = [
                r
                for r in result["check"]["reasons"]
                if r["verdict"] != "pass" and "accepted" not in r
            ]
        result["usage"] = run["usage"]
        write_run(history / f"{run['run_id']}.run.json", run)
        _write(history / f"{run['run_id']}.monitor.json", result)
        typer.echo(f"Usage: {used}")  # before the JSON verdict, which stays the last line
    except (OSError, ValueError, ImportError, AttributeError, PolicyError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if upload:
        from vectraudit.evaluation.compare import Verdict
        from vectraudit.release_cli import _record_cloud_check

        _record_cloud_check(client, compared, baseline[6:], Verdict(result["verdict"]), "monitor")
    typer.echo(json.dumps({"verdict": result["verdict"], "alerts": result["alerts"]}))
    if result["verdict"] not in ("pass", "accepted"):
        raise typer.Exit(3)


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


@workbench_app.command("share")
def share_command(
    workspace: Path,
    kind: str,
    identifier: str,
    project: Annotated[str | None, typer.Option(help="Hosted workspace/project.")] = None,
    preview: Annotated[
        bool, typer.Option(help="Show transfer metadata without connecting or uploading.")
    ] = False,
    share_query_text: Annotated[
        bool,
        typer.Option(
            help="Explicitly include available question text; never passages, notes or context."
        ),
    ] = False,
) -> None:
    """Share one saved case, benchmark snapshot or policy revision with a project."""
    from typing import cast

    from fastapi import HTTPException

    from vectraudit.integrations.cloud import CloudClient, CloudClientError, CloudTarget
    from vectraudit.integrations.shared_records import preview_record
    from vectraudit.pipelines.guided import GuidedWorkspace
    from vectraudit.pipelines.share import CATEGORIES, GENERIC_ERROR, projection

    try:
        if kind not in CATEGORIES:
            raise ValueError("Kind must be case, benchmark or policy")
        if not workspace.is_dir():
            raise ValueError("Workspace directory does not exist")
        projected = projection(
            GuidedWorkspace(workspace),
            cast(Any, kind),
            identifier,
            share_query_text=share_query_text,
        )
        transfer = preview_record(projected)
        if preview:
            typer.echo(json.dumps({"preview": transfer}, indent=2))
            return
        result = CloudClient(CloudTarget.from_env(project)).upload_record(projected)
        typer.echo(json.dumps({"uploaded": result, "transfer": transfer}, indent=2))
    except (ValueError, OSError, KeyError, TypeError, HTTPException, CloudClientError) as exc:
        # Pydantic and dataset validation can contain rejected source text.
        message = exc.message if isinstance(exc, CloudClientError) else GENERIC_ERROR
        typer.echo(message, err=True)
        raise typer.Exit(2) from None
