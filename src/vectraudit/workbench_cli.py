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


def _stage_kinds(items: list[str] | None) -> dict[str, str]:
    """``--stage-kind SPAN=KIND`` options as a mapping."""
    kinds = {}
    for item in items or []:
        name, sep, kind = item.partition("=")
        if not sep or not name:
            raise ValueError(f"--stage-kind expects SPAN=KIND, got {item!r}")
        kinds[name] = kind
    return kinds


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
    paired: Annotated[
        bool,
        typer.Option(
            "--paired",
            help="Run exactly two candidates interleaved, question by question, alternating "
            "which goes first, so endpoint drift affects both alike. Serial; no resume.",
        ),
    ] = False,
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
        if paired:
            _run_paired(spec, ds, manifest, out, retry, resume, max_calls, max_spend, measured)
            return
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


def _run_paired(
    spec: Any,
    ds: Any,
    manifest: Path,
    out: Path,
    retry: RetryPolicy,
    resume: Path | None,
    max_calls: int | None,
    max_spend: float | None,
    measured: list[Path] | None,
) -> None:
    """``workbench run --paired``: two candidates, interleaved per question."""
    from vectraudit.pipelines.runner import PairedSide, execute_paired

    if resume is not None:
        raise ValueError("--paired runs cannot be resumed; start a new paired run")
    if len(spec.candidates) != 2:
        raise ValueError("--paired needs a manifest with exactly two candidates")
    if any(c.concurrency != 1 for c in spec.candidates):
        raise ValueError("--paired runs one question at a time; set concurrency 1")
    budget = _guard(
        spec, ds, "run", max_calls=max_calls, max_spend=max_spend, measured=measured, retry=retry
    )
    sides = []
    for candidate in spec.candidates:
        config = candidate_config(candidate, manifest.parent)
        sides.append(
            PairedSide(
                open_adapter(candidate.adapter, manifest.parent),
                PipelineOptions(
                    k=spec.k,
                    params=candidate.params,
                    repeats=spec.repeats,
                    warmup=0,
                    label=candidate.name,
                    retry=retry,
                    max_rps=candidate.max_rps,
                ),
                config,
                spec.corpus_version or config.get("corpus_sha256"),
            )
        )
    runs = execute_paired(ds, sides[0], sides[1])
    incomplete = False
    for candidate, run in zip(spec.candidates, runs, strict=True):
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
        typer.echo(f"{candidate.name}: {run['status']} ({run['run_id']}, paired)")
        typer.echo(f"  {used}")
        incomplete |= run["status"] != "complete"
    if incomplete:
        raise typer.Exit(2)


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


@workbench_app.command("import-traces")
def import_traces_command(
    dataset: Annotated[Path, typer.Argument(help="Benchmark dataset (JSONL) with labels.")],
    traces: Annotated[
        Path,
        typer.Argument(
            help="Trace export: OpenInference (OTLP JSON or spans), LangSmith runs or "
            "Langfuse observations."
        ),
    ],
    out: Annotated[Path, typer.Option(help="Run file to write.")] = Path("traces.run.json"),
    report: Annotated[
        Path | None, typer.Option(help="Also write the capability report as JSON.")
    ] = None,
    label: Annotated[str | None, typer.Option(help="Run label.")] = None,
    k: Annotated[
        int | None,
        typer.Option(min=1, max=10000, help="Ranking depth; default the longest traced one."),
    ] = None,
    query_attribute: Annotated[
        str | None,
        typer.Option(
            help="Span attribute holding the question ID (metadata.query_id); "
            "without it, traces match questions by query text."
        ),
    ] = None,
    id_field: Annotated[
        str | None,
        typer.Option(help="Document metadata field to use as the document ID (doc_id)."),
    ] = None,
    final_span: Annotated[
        str | None,
        typer.Option(help="Span name whose documents are delivered, when traces have several."),
    ] = None,
    stage_kind: Annotated[
        list[str] | None,
        typer.Option(
            help="SPAN=KIND for retriever spans (bm25=keyword); default custom. In LangSmith, "
            "Langfuse and Opik exports a named span counts as a retriever (Opik needs it)."
        ),
    ] = None,
    latest: Annotated[
        bool,
        typer.Option("--latest", help="Use the most recent trace when a question has several."),
    ] = False,
) -> None:
    """Turn traced application requests into a run file and report what traces can show.

    The run is scored and checked like a live run (``vectraudit check``). It cannot be
    rerun: retesting a change needs a registered pipeline. Latency comes from the
    application's spans and is never compared with measured latency. Exits 2 on unusable
    input; a run with untraced or unusable questions is written as ``partial``.
    """
    from vectraudit.pipelines.traces import Settings, import_traces, render_report

    try:
        kinds = _stage_kinds(stage_kind)
        settings = Settings(
            query_attribute=query_attribute,
            id_field=id_field,
            final_span=final_span,
            stage_kinds=kinds,
            duplicates="latest" if latest else "error",
        )
        run, result = import_traces(load_dataset(dataset), traces, settings, k=k, label=label)
        _write(out, run)
        if report is not None:
            _write(report, result)
    except (OSError, ValueError) as exc:
        typer.echo(f"Import error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(render_report(result))
    typer.echo(f"\nWrote {out} ({run['status']}).")


@workbench_app.command("attach-answers")
def attach_answers_command(
    run: Annotated[Path, typer.Argument(help="Run file whose answers were judged.")],
    judgments: Annotated[Path, typer.Argument(help="Answer judgments (JSONL).")],
    out: Annotated[Path, typer.Option(help="Run file to write, with the judgments.")],
    threshold: Annotated[
        list[str] | None,
        typer.Option(
            help="EVALUATOR=VALUE: a score at or above VALUE is correct. Needed for judgments "
            "that give only a score; there is no default."
        ),
    ] = None,
) -> None:
    """Attach answer judgments from human review or tools such as Ragas, and diagnose them.

    Each judged question gets its verdict (human review decides when present; disagreement
    is reported as disputed) and the first observable loss between retrieval and answer.
    Release verdicts do not change. Writes a new file; the original run is left unchanged.
    """
    from vectraudit.evaluation.answers import attach_file, render_summary

    try:
        limits = {}
        for item in threshold or []:
            name, sep, value = item.rpartition("=")
            if not sep or not name:
                raise ValueError(f"--threshold expects EVALUATOR=VALUE, got {item!r}")
            limits[name] = float(value)
        if out.resolve() == run.resolve():
            raise ValueError("--out must differ from the run file; the original stays unchanged")
        result = attach_file(load_run(run), judgments, limits)
        _write(out, result)
    except (OSError, ValueError) as exc:
        typer.echo(f"Attach error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(render_summary(result))
    typer.echo(f"\nWrote {out}.")


def _candidate(spec: Any, name: str | None) -> Any:
    names = [c.name for c in spec.candidates]
    if name is None:
        if len(names) != 1:
            raise ValueError(f"--candidate is required: one of {', '.join(names)}")
        return spec.candidates[0]
    for candidate in spec.candidates:
        if candidate.name == name:
            return candidate
    raise ValueError(f"no candidate {name!r} in the manifest: {', '.join(names)}")


@workbench_app.command("investigate")
def investigate_command(
    manifest: Path,
    run: Annotated[Path, typer.Argument(help="Run file holding the failed question.")],
    query: Annotated[str, typer.Argument(help="Question ID to investigate.")],
    candidate: Annotated[
        str | None, typer.Option(help="Manifest candidate that produced the run.")
    ] = None,
    baseline: Annotated[
        Path | None, typer.Option(help="Earlier run, to show which losses are new.")
    ] = None,
    lineage: Annotated[
        Path | None, typer.Option(help="Lineage export, to name ingestion losses.")
    ] = None,
    lineage_ids: Annotated[
        str, typer.Option(help="What the run's result IDs are: chunk or source.")
    ] = "chunk",
    changes: Annotated[
        Path | None,
        typer.Option(help="Content change log, to flag stale content (needs --lineage)."),
    ] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
) -> None:
    """What one question lost and where, possible causes, and the declared fixes to try.

    Reads files only: no adapter is opened. Fixes come from the candidate's ``fixes`` in the
    manifest (and built-in ones for the hybrid baseline); nothing else is suggested.
    """
    from vectraudit.pipelines.investigate import available_fixes, investigate

    try:
        spec = load_manifest(manifest)
        record = load_run(run)
        chosen = _candidate(spec, candidate or record.get("label"))
        rows = None
        if lineage is not None:
            from vectraudit.evaluation.lineage import check_lineage, read_manifest

            report = check_lineage(
                load_dataset(manifest.parent / spec.dataset),
                read_manifest(lineage),
                record,
                ids=lineage_ids,
            )
            rows = [r for r in report["rows"] if r["query_id"] == query]
        freshness = None
        if changes is not None:
            from vectraudit.evaluation.freshness import question_freshness, read_changes
            from vectraudit.evaluation.lineage import read_manifest

            if lineage is None:
                raise ValueError("--changes needs the --lineage export it is checked against")
            freshness = question_freshness(
                read_changes(changes),
                read_manifest(lineage),
                record,
                query,
                dataset=load_dataset(manifest.parent / spec.dataset),
                ids=lineage_ids,
            )
        result = investigate(
            record,
            query,
            available_fixes(chosen, manifest.parent),
            baseline=load_run(baseline) if baseline else None,
            lineage=rows,
            freshness=freshness,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Investigation error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"{query} in {chosen.name}:")
    for loss in result["losses"] or [{"class": "none"}]:
        where = f" at {loss['stage'].replace('_', ' ')}" if loss.get("stage") else ""
        doc = f" {loss['doc_id']}" if loss.get("doc_id") else ""
        detail = f" ({loss['detail']})" if loss.get("detail") else ""
        typer.echo(f"  lost{doc}: {loss['class'].replace('_', ' ')}{where}{detail}")
    if result["regression"] is not None:
        typer.echo(f"  new since the baseline: {len(result['regression']['new_losses'])}")
    for title, key in (
        ("Possible causes (not established)", "hypotheses"),
        ("Cautions", "cautions"),
        ("Uncertain", "uncertainty"),
    ):
        if result[key]:
            typer.echo(f"{title}:")
            for line in result[key]:
                typer.echo(f"  - {line}")
    if result["fixes"]:
        typer.echo("Fixes to try (declared by the pipeline):")
        for fix in result["fixes"]:
            typer.echo(f"  - {fix['name']}: {json.dumps(fix['params'])}")
        typer.echo(
            f"Try one: vectraudit workbench try-fix {manifest} --candidate "
            f"'{chosen.name}' --fix '<name>' --query {query}"
        )
    elif result["unaddressed"]:
        typer.echo(
            "No declared fix addresses: " + ", ".join(result["unaddressed"]) + ". Declare "
            'fixes the pipeline supports under the candidate\'s "fixes" in the manifest.'
        )
    if json_out is not None:
        _write(json_out, result)


@workbench_app.command("try-fix")
def try_fix_command(
    manifest: Path,
    fix: Annotated[str, typer.Option(help="Name of a declared fix.")],
    query: Annotated[list[str], typer.Option(help="Failing question ID (repeatable).")],
    candidate: Annotated[str | None, typer.Option(help="Manifest candidate to fix.")] = None,
    sample: Annotated[
        int, typer.Option(min=0, max=1000, help="Other exploratory questions checked.")
    ] = 20,
    seed: Annotated[int, typer.Option(help="Seed for choosing the sample.")] = 0,
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Then judge the fix on held-out questions under --policy."),
    ] = False,
    policy: Annotated[Path | None, typer.Option(help="Release policy for --validate.")] = None,
    out: Annotated[Path, typer.Option(help="Directory for results and runs.")] = Path("fix-trials"),
    max_calls: MaxCalls = None,
) -> None:
    """Try a declared fix: current and fixed configurations, interleaved, on failing and
    sampled exploratory questions; optionally validate it under the release policy.

    The trial is exploratory and never uses held-out questions. ``--validate`` runs both
    configurations on held-out questions and approves only if the policy passes. Exit 3
    when the trial shows regressions or validation is not approved.
    """
    from vectraudit.pipelines.investigate import (
        available_fixes,
        try_fix,
        try_plan,
        validate_fix,
    )

    try:
        spec = load_manifest(manifest)
        chosen = _candidate(spec, candidate)
        fixes = {f.name: f for f in available_fixes(chosen, manifest.parent)}
        if fix not in fixes:
            raise ValueError(f"{chosen.name} declares no fix {fix!r}: {', '.join(fixes) or 'none'}")
        if validate and policy is None:
            raise ValueError("--validate needs --policy; approval is only under a policy")
        ds = load_dataset(manifest.parent / spec.dataset)
        planned = try_plan(ds, query, sample, seed)
        held = sum(q.split == "heldout" for q in ds.queries) if validate else 0
        calls = 2 * (len(planned) + held) * spec.repeats
        typer.echo(
            f"Plan: {len(planned)} exploratory questions"
            + (f" and {held} held-out questions" if validate else "")
            + f", two configurations, {spec.repeats} repeats: {calls} calls."
        )
        if max_calls is not None and calls > max_calls:
            raise ValueError(f"{calls} planned calls exceed --max-calls {max_calls}")
        rules = load_policy(policy) if policy else None
        config = candidate_config(chosen, manifest.parent)
        pipeline = open_adapter(chosen.adapter, manifest.parent)
        version = spec.corpus_version or config.get("corpus_sha256")
        trial = try_fix(
            ds,
            pipeline,
            chosen,
            fixes[fix],
            query,
            k=spec.k,
            repeats=spec.repeats,
            sample=sample,
            seed=seed,
            config=config,
            corpus_version=version,
        )
        out.mkdir(parents=True, exist_ok=True)
        for side, run in trial.pop("runs").items():
            write_run(out / f"trial-{side}.run.json", run)
        _write(out / "trial.json", trial)
        result = None
        if validate:
            result = validate_fix(
                ds,
                pipeline,
                chosen,
                fixes[fix],
                rules,
                k=spec.k,
                repeats=spec.repeats,
                config=config,
                corpus_version=version,
            )
            for side, run in result.pop("runs").items():
                write_run(out / f"validation-{side}.run.json", run)
            _write(out / "validation.json", result)
    except (OSError, ValueError, PolicyError, ImportError, AttributeError) as exc:
        typer.echo(f"Fix trial error: {exc}", err=True)
        raise typer.Exit(2) from exc
    focus = len(trial["focus"])
    typer.echo(
        f"Exploratory trial of {fix!r}: {trial['recovered_focus']} of {focus} failing "
        f"question(s) fully recovered; {len(trial['regressed'])} of {trial['questions']} "
        "questions regressed."
    )
    for row in trial["rows"]:
        if row["focus"] or row["regressed"]:
            marks = []
            if row["recovered"]:
                marks.append(f"recovered {len(row['recovered'])}")
            if row["moved"]:
                marks.append(f"moved {len(row['moved'])} to a later stage")
            if row["introduced"]:
                marks.append(f"{len(row['introduced'])} new loss(es)")
            typer.echo(
                f"  {row['query_id']}: nDCG {row['ndcg_before']} -> {row['ndcg_after']}"
                + (f"; {', '.join(marks)}" if marks else "")
            )
    if trial["untargeted_focus"]:
        typer.echo(
            "The fix targets none of the recorded losses of: "
            + ", ".join(trial["untargeted_focus"])
        )
    typer.echo(trial["status_label"])
    if result is not None:
        typer.echo(result["status_label"])
        for reason in (result["check"] or {}).get("reasons", []):
            typer.echo(f"  - {reason}")
        if result["approved"]:
            typer.echo(f"Promote: {json.dumps(result['promote']['params'])}")
    typer.echo(f"Wrote {out}.")
    if trial["regressed"] or (result is not None and not result["approved"]):
        raise typer.Exit(3)


@workbench_app.command("investigations")
def investigations_command(
    workspace: Annotated[Path, typer.Argument(help="Workbench directory (workbench serve).")],
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
) -> None:
    """Pilot measures from investigation records: diagnoses, verified fixes, times.

    Counts are given with their denominators and medians with their sample sizes. Names in
    the records are self-reported in the local workbench.
    """
    from vectraudit.pipelines.investigation_records import RECORD_FORMAT, summarize

    folder = workspace / ".workbench" / "investigations"
    try:
        records = []
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            if path.is_symlink():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("format") == RECORD_FORMAT:
                records.append(record)
        result = summarize(records)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"Investigation records error: {exc}", err=True)
        raise typer.Exit(2) from exc
    n = result["investigations"]
    diagnosis = result["diagnosis"]
    typer.echo(f"{n} investigations ({result['open']} open)")
    typer.echo(
        f"Diagnoses: {diagnosis['accepted']} of {n} accepted, {diagnosis['disputed']} "
        f"disputed, {diagnosis['pending']} pending"
    )
    typer.echo(
        f"Fixes verified under policy: {result['fixes_verified']} of {n}; reproduced by a "
        f"second engineer: {result['reproduced_by_second_engineer']}"
    )
    for key, label in (
        ("minutes_to_accepted_diagnosis", "Minutes to an accepted diagnosis"),
        ("minutes_to_verified_fix", "Minutes to a verified fix"),
    ):
        stat = result[key]
        if stat["n"]:
            typer.echo(f"{label}: median {stat['median']} (n={stat['n']})")
    if json_out is not None:
        _write(json_out, result)


@workbench_app.command("lineage")
def lineage_command(
    manifest: Annotated[Path, typer.Argument(help="Lineage manifest (JSONL) you exported.")],
    dataset: Annotated[Path, typer.Argument(help="Benchmark with relevant_spans.")],
    run: Annotated[Path | None, typer.Option(help="Also follow passages through this run.")] = None,
    ids: Annotated[
        str, typer.Option(help="What the run's result IDs are: chunk or source.")
    ] = "chunk",
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    fail_on_ingestion: Annotated[
        bool,
        typer.Option(
            "--fail-on-ingestion",
            help="Exit 3 when any required passage is lost before retrieval.",
        ),
    ] = False,
) -> None:
    """Follow each required passage from source to model context; name its first loss.

    Reads your export and benchmark (and a run, if given); nothing is re-ingested and no
    document text is read. Losses before retrieval are ingestion problems that a different
    embedding model cannot fix.
    """
    from vectraudit.evaluation.lineage import check_lineage, read_manifest, render

    try:
        report = check_lineage(
            load_dataset(dataset),
            read_manifest(manifest),
            load_run(run) if run else None,
            ids=ids,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Lineage error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not report["passages"]:
        typer.echo("The benchmark labels no relevant_spans, so there is nothing to follow.")
    else:
        typer.echo(render(report))
    if json_out is not None:
        _write(json_out, report)
    if fail_on_ingestion and report["ingestion_losses"]:
        raise typer.Exit(3)


@workbench_app.command("freshness")
def freshness_command(
    changes: Annotated[Path, typer.Argument(help="Content change log (JSONL).")],
    manifest: Annotated[
        Path, typer.Argument(help="Lineage export taken after the changes (JSONL).")
    ],
    run: Annotated[Path | None, typer.Option(help="A run recorded after the changes.")] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(help="The run's benchmark, so questions tagged historical are exempt."),
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option(help="ISO time the export reflects; default the run's time, else now."),
    ] = None,
    window_minutes: Annotated[
        int, typer.Option(min=0, help="Propagation window for changes that declare none.")
    ] = 60,
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    fail_on_stale: Annotated[
        bool,
        typer.Option(
            "--fail-on-stale",
            help="Exit 3 when content is overdue, lingering, stale in results or conflicting.",
        ),
    ] = False,
) -> None:
    """Knowledge freshness: did each content change reach the index, results and context?

    Changes still inside their propagation window are pending, not failures. Reads your
    exports only; nothing is re-indexed and no document text is read.
    """
    from datetime import datetime, timezone

    from vectraudit.evaluation.freshness import check_freshness, read_changes, render
    from vectraudit.evaluation.lineage import read_manifest

    try:
        recorded = load_run(run) if run else None
        if as_of:
            moment = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        elif recorded is not None:
            moment = datetime.fromisoformat(recorded["created_at"])
        else:
            moment = datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        report = check_freshness(
            read_changes(changes),
            read_manifest(manifest),
            as_of=moment,
            window_minutes=window_minutes,
            run=recorded,
            dataset=load_dataset(dataset) if dataset else None,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Freshness error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(render(report))
    if json_out is not None:
        _write(json_out, report)
    if fail_on_stale and report["failing"]:
        raise typer.Exit(3)


@workbench_app.command("access")
def access_command(
    changes: Annotated[Path, typer.Argument(help="Access change log (JSONL).")],
    dataset: Annotated[Path, typer.Argument(help="The benchmark whose contexts name principals.")],
    run: Annotated[
        Path | None, typer.Option(help="A run recorded after the changes, to check.")
    ] = None,
    write_dataset: Annotated[
        Path | None,
        typer.Option(help="Write the benchmark with the expected access applied (JSONL)."),
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option(help="ISO time to judge at; default the run's time, else now."),
    ] = None,
    window_minutes: Annotated[
        int, typer.Option(min=0, help="Propagation window for changes that declare none.")
    ] = 60,
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    fail_on_leak: Annotated[
        bool,
        typer.Option("--fail-on-leak", help="Exit 3 when a revoked document was still served."),
    ] = False,
) -> None:
    """Permission lifecycle: apply an access change log to a benchmark, or check a run.

    Expected access comes from your identity system's change log, never from the
    application's own answers. Changes inside their propagation window are pending, not
    failures. A passing check is evidence about these principals, not proof of authorization.
    """
    from datetime import datetime, timezone

    from vectraudit.evaluation.access import (
        apply_access_changes,
        check_access,
        read_access_changes,
        render,
    )

    if run is None and write_dataset is None:
        typer.echo("Give --run to check a run, --write-dataset to apply the changes, or both.")
        raise typer.Exit(2)
    try:
        log = read_access_changes(changes)
        data = load_dataset(dataset)
        recorded = load_run(run) if run else None
        if as_of:
            moment = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        elif recorded is not None:
            moment = datetime.fromisoformat(recorded["created_at"])
        else:
            moment = datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        report: dict[str, Any] = {"format": "vectraudit.access", "format_version": 1}
        if write_dataset is not None:
            text, applied = apply_access_changes(
                data, log, as_of=moment, window_minutes=window_minutes
            )
            write_dataset.write_text(text, encoding="utf-8")
            report["applied"] = applied
            typer.echo(
                f"Wrote {write_dataset}: {len(applied['affected'])} question(s) changed access, "
                f"{len(applied['pending'])} with changes still pending."
            )
            for item in applied["labels_to_review"]:
                typer.echo(
                    f"  review {item['query_id']}: removed revoked label(s) "
                    + ", ".join(item["removed_labels"])
                )
        if recorded is not None:
            checked = check_access(log, recorded, data, as_of=moment, window_minutes=window_minutes)
            report["check"] = checked
            typer.echo(render(checked))
    except (OSError, ValueError) as exc:
        typer.echo(f"Access error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if json_out is not None:
        _write(json_out, report)
    if fail_on_leak and report.get("check", {}).get("failing"):
        raise typer.Exit(3)


@workbench_app.command("cache-probe")
def cache_probe_command(
    manifest: Path,
    candidate: Annotated[
        str | None, typer.Option(help="Candidate to probe (required if several).")
    ] = None,
    max_pairs: Annotated[
        int, typer.Option(min=1, help="Refuse more target/primer pairs than this (2 calls each).")
    ] = 500,
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    fail_on_leak: Annotated[
        bool,
        typer.Option(
            "--fail-on-leak", help="Exit 3 on any leak, or when calls failed (incomplete)."
        ),
    ] = False,
) -> None:
    """Cache isolation: does one context's question warm another context's answer?

    For each question asked under several contexts with forbidden documents (permission
    scenarios), calls it under another context first, then under its own, in order. Probe a
    fresh deployment: answers cached before the probe can mask a leak.
    """
    from vectraudit.pipelines.cache_probe import cache_probe, probe_pairs, render

    try:
        spec = load_manifest(manifest)
        chosen = _candidate(spec, candidate)
        ds = load_dataset(manifest.parent / spec.dataset)
        pairs = len(probe_pairs(ds))
        typer.echo(f"Plan: {pairs} target/primer pair(s), {2 * pairs} calls, serially.")
        report = cache_probe(
            ds,
            open_adapter(chosen.adapter, manifest.parent),
            k=spec.k,
            params=chosen.params,
            max_pairs=max_pairs,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Cache probe error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(render(report))
    if json_out is not None:
        _write(json_out, report)
    if fail_on_leak and (report["failing"] or not report["complete"]):
        raise typer.Exit(3)


@workbench_app.command("migration")
def migration_command(
    current: Annotated[Path, typer.Argument(help="Run of the current system.")],
    proposed: Annotated[Path, typer.Argument(help="Run of the proposed system, same benchmark.")],
    dataset: Annotated[Path, typer.Argument(help="The labelled benchmark both runs answered.")],
    policy: Annotated[
        Path | None, typer.Option(help="Release policy; without it the result is exploratory.")
    ] = None,
    current_inventory: Annotated[
        Path | None, typer.Option(help="Current system's documents (JSONL: id, sha256, metadata).")
    ] = None,
    proposed_inventory: Annotated[
        Path | None, typer.Option(help="Proposed system's documents (JSONL: id, sha256, metadata).")
    ] = None,
    costs: Annotated[
        Path | None,
        typer.Option(help='Declared cost estimates: {"current": {"reindex_usd": 0}, ...}.'),
    ] = None,
    out: Annotated[Path, typer.Option(help="Markdown report.")] = Path("migration-report.md"),
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    fail_unless_ready: Annotated[
        bool,
        typer.Option(
            "--fail-unless-ready",
            help="Exit 3 unless the policy passes and no migration concern remains.",
        ),
    ] = False,
) -> None:
    """Migration assurance: compare two systems on one benchmark under the release policy.

    Adds the checks a migration needs: corpus equivalence and ID mapping from each system's
    inventory, filter semantics against document metadata, environment differences, and
    measured versus declared costs. Reads your exports only; nothing is copied or re-indexed.
    """
    from vectraudit.evaluation.migration import (
        migration_report,
        read_inventory,
        render_migration,
    )

    try:
        report = migration_report(
            load_run(current),
            load_run(proposed),
            load_dataset(dataset),
            load_policy(policy) if policy else DEFAULT_POLICY,
            current_inventory=read_inventory(current_inventory) if current_inventory else None,
            proposed_inventory=read_inventory(proposed_inventory) if proposed_inventory else None,
            declared_costs=json.loads(costs.read_text(encoding="utf-8")) if costs else None,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Migration error: {exc}", err=True)
        raise typer.Exit(2) from exc
    out.write_text(render_migration(report), encoding="utf-8")
    if json_out is not None:
        _write(json_out, report)
    typer.echo(
        f"{'Ready' if report['migration_ready'] else 'Not ready'} to migrate: policy verdict "
        f"{report['verdict']}, {len(report['concerns'])} concern(s). Report: {out}"
    )
    for concern in report["concerns"]:
        typer.echo(f"  - {concern}")
    if fail_unless_ready and not report["migration_ready"]:
        raise typer.Exit(3)


@workbench_app.command("langfuse-pull")
def langfuse_pull_command(
    since: Annotated[str, typer.Option(help="Start of the window (ISO 8601, inclusive).")],
    until: Annotated[str, typer.Option(help="End of the window (ISO 8601, exclusive).")],
    traces: Annotated[
        Path | None,
        typer.Option(help="Write observations here (JSONL) for import-traces or intake."),
    ] = None,
    judgments: Annotated[
        Path | None,
        typer.Option(help="Write scores here as answer judgments (JSONL) for attach-answers."),
    ] = None,
    score_name: Annotated[
        list[str] | None, typer.Option(help="Only scores with this name (repeatable).")
    ] = None,
    environment: Annotated[
        list[str] | None, typer.Option(help="Only observations in this environment (repeatable).")
    ] = None,
    human_source: Annotated[
        list[str] | None,
        typer.Option(help="Score sources counted as human review (default: ANNOTATION)."),
    ] = None,
) -> None:
    """Pull observations and scores from a Langfuse project for the given time window.

    Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST (or LANGFUSE_BASE_URL;
    default https://cloud.langfuse.com). Read-only: nothing is written to Langfuse, and
    score comments are never requested.
    """
    import os

    from vectraudit.pipelines.langfuse_api import (
        LangfuseSource,
        fetch_observations,
        fetch_scores,
        scores_to_judgments,
        write_jsonl,
    )

    if traces is None and judgments is None:
        typer.echo("Pull error: give --traces, --judgments or both.", err=True)
        raise typer.Exit(2)
    try:
        source = LangfuseSource(
            host=os.environ.get("LANGFUSE_HOST")
            or os.environ.get("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com",
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        )
        lines = []
        if traces is not None:
            observations = fetch_observations(
                source, start=since, end=until, environments=environment or ()
            )
            write_jsonl(traces, observations)
            n_traces = len({o["traceId"] for o in observations})
            lines.append(
                f"Wrote {len(observations)} observations from {n_traces} traces to {traces}."
            )
        if judgments is not None:
            scores = fetch_scores(source, start=since, end=until, names=score_name or ())
            rows, skipped = scores_to_judgments(
                scores, human_sources=human_source or ("ANNOTATION",)
            )
            write_jsonl(judgments, rows)
            lines.append(f"Wrote {len(rows)} of {len(scores)} scores as judgments to {judgments}.")
            if skipped:
                lines.append(
                    "Skipped: "
                    + ", ".join(
                        f"{n} {why.replace('_', ' ')}" for why, n in sorted(skipped.items())
                    )
                )
            if any("score" in r for r in rows):
                lines.append("Numeric scores need --threshold EVALUATOR=VALUE when attached.")
    except (OSError, ValueError) as exc:
        typer.echo(f"Pull error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo("\n".join(lines))
    typer.echo(f"({source.calls} API requests.)")


@workbench_app.command("langfuse-push")
def langfuse_push_command(
    run: Annotated[
        Path, typer.Argument(help="Run imported from Langfuse, with attach-answers applied.")
    ],
    send: Annotated[
        bool, typer.Option("--send", help="Write the scores. Without it, only preview.")
    ] = False,
    name: Annotated[str, typer.Option(help="Score name in Langfuse.")] = "vectraudit.diagnosis",
    environment: Annotated[
        str | None, typer.Option(help="Langfuse environment for the scores.")
    ] = None,
) -> None:
    """Write each judged trace's diagnosis back to Langfuse as a categorical score.

    Previews by default; --send writes. Only the diagnosis class, a stable score ID and a run
    digest are sent: no text, document IDs or facts. Re-sending replaces a trace's earlier
    score of the same name. Reads the same LANGFUSE_* variables as langfuse-pull.
    """
    import os

    from vectraudit.pipelines.langfuse_api import LangfuseSource
    from vectraudit.pipelines.langfuse_push import diagnosis_scores, preview, push_scores

    try:
        scores, skipped = diagnosis_scores(load_run(run), name=name, environment=environment)
        typer.echo(preview(scores, skipped))
        if not send:
            typer.echo("\nPreview only: nothing was sent. Add --send to write these scores.")
            return
        if not scores:
            typer.echo("\nNothing to send.")
            return
        source = LangfuseSource(
            host=os.environ.get("LANGFUSE_HOST")
            or os.environ.get("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com",
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        )
        totals = push_scores(source, scores)
    except (OSError, ValueError) as exc:
        typer.echo(f"Push error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        f"\nSent {totals['sent']} scores in {totals['requests']} requests: "
        f"{totals['accepted']} accepted, {totals['rejected']} rejected."
    )
    if totals["rejected"]:
        typer.echo(
            "Langfuse rejected some scores; check the score name against any score config "
            "with that name. Rejected batches are not retried.",
            err=True,
        )
        raise typer.Exit(3)


@workbench_app.command("opik-pull")
def opik_pull_command(
    since: Annotated[str, typer.Option(help="Start of the window (ISO 8601, inclusive).")],
    until: Annotated[str, typer.Option(help="End of the window (ISO 8601).")],
    project: Annotated[
        str | None, typer.Option(help="Opik project (default: OPIK_PROJECT_NAME).")
    ] = None,
    traces: Annotated[
        Path | None,
        typer.Option(help="Write spans here (JSONL) for import-traces or intake."),
    ] = None,
    judgments: Annotated[
        Path | None,
        typer.Option(help="Write feedback scores as answer judgments (JSONL) for attach-answers."),
    ] = None,
    grace_minutes: Annotated[
        int, typer.Option(help="Also read spans created this long after the window.")
    ] = 60,
    human_source: Annotated[
        list[str] | None,
        typer.Option(help="Score sources counted as human review (default: ui)."),
    ] = None,
) -> None:
    """Pull traces, spans and feedback scores from an Opik project for the given window.

    Reads OPIK_URL_OVERRIDE (default Opik Cloud), OPIK_API_KEY, OPIK_WORKSPACE and
    OPIK_PROJECT_NAME. Opik has no retriever span type: name retriever spans with
    --stage-kind when importing. Read-only; score reasons are never written out.
    """
    import os

    from vectraudit.pipelines.langfuse_api import write_jsonl
    from vectraudit.pipelines.opik_api import (
        OPIK_CLOUD,
        OpikSource,
        fetch,
        records,
        scores_to_judgments,
    )

    if traces is None and judgments is None:
        typer.echo("Pull error: give --traces, --judgments or both.", err=True)
        raise typer.Exit(2)
    name = project or os.environ.get("OPIK_PROJECT_NAME")
    if not name:
        typer.echo("Pull error: give --project or set OPIK_PROJECT_NAME.", err=True)
        raise typer.Exit(2)
    try:
        source = OpikSource(
            url=os.environ.get("OPIK_URL_OVERRIDE") or OPIK_CLOUD,
            api_key=os.environ.get("OPIK_API_KEY") or None,
            workspace=os.environ.get("OPIK_WORKSPACE") or None,
        )
        found, spans = fetch(source, name, start=since, end=until, grace_minutes=grace_minutes)
        lines = []
        if traces is not None:
            write_jsonl(traces, records(found, spans))
            lines.append(f"Wrote {len(found)} traces and {len(spans)} spans to {traces}.")
            if found and not spans:
                lines.append("No spans were found for these traces; retrieval will be missing.")
        if judgments is not None:
            rows, skipped = scores_to_judgments(found, spans, human_sources=human_source or ("ui",))
            write_jsonl(judgments, rows)
            lines.append(f"Wrote {len(rows)} feedback scores as judgments to {judgments}.")
            if skipped:
                lines.append(
                    "Skipped: "
                    + ", ".join(
                        f"{n} {why.replace('_', ' ')}" for why, n in sorted(skipped.items())
                    )
                )
            if any("score" in r for r in rows):
                lines.append("Numeric scores need --threshold EVALUATOR=VALUE when attached.")
    except (OSError, ValueError) as exc:
        typer.echo(f"Pull error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo("\n".join(lines))
    typer.echo(f"({source.calls} API requests.)")


@workbench_app.command("openobserve-pull")
def openobserve_pull_command(
    since: Annotated[str, typer.Option(help="Start of the window (ISO 8601, inclusive).")],
    until: Annotated[str, typer.Option(help="End of the window (ISO 8601).")],
    traces: Annotated[
        Path, typer.Option(help="Write the traces here (OTLP JSON) for import-traces or intake.")
    ],
    org: Annotated[
        str | None, typer.Option(help="Organisation (default: OPENOBSERVE_ORG).")
    ] = None,
    stream: Annotated[str, typer.Option(help="Trace stream name.")] = "default",
    grace_minutes: Annotated[
        int, typer.Option(help="Also read spans up to this long after the window.")
    ] = 60,
) -> None:
    """Pull the traces that started in a window from OpenObserve, as OTLP JSON.

    Reads OPENOBSERVE_URL, OPENOBSERVE_ORG, and OPENOBSERVE_TOKEN or OPENOBSERVE_USER and
    OPENOBSERVE_PASSWORD. OpenObserve flattens attribute keys on ingest; the keys Vectraudit
    reads (OpenInference and OpenLLMetry) are restored. Read-only.
    """
    import os

    from vectraudit.labelling.task import atomic_write
    from vectraudit.pipelines.openobserve_api import OpenObserveSource, fetch, to_otlp

    name = org or os.environ.get("OPENOBSERVE_ORG")
    url = os.environ.get("OPENOBSERVE_URL")
    if not name or not url:
        typer.echo("Pull error: set OPENOBSERVE_URL, and --org or OPENOBSERVE_ORG.", err=True)
        raise typer.Exit(2)
    try:
        source = OpenObserveSource(
            url=url,
            org=name,
            stream=stream,
            user=os.environ.get("OPENOBSERVE_USER") or None,
            password=os.environ.get("OPENOBSERVE_PASSWORD") or None,
            token=os.environ.get("OPENOBSERVE_TOKEN") or None,
        )
        trace_ids, rows = fetch(source, start=since, end=until, grace_minutes=grace_minutes)
        atomic_write(traces, json.dumps(to_otlp(rows)) + "\n")
    except (OSError, ValueError) as exc:
        typer.echo(f"Pull error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Wrote {len(trace_ids)} traces and {len(rows)} spans to {traces}.")
    if trace_ids and not rows:
        typer.echo("No spans were returned for these traces; check the stream name.")
    typer.echo(f"({source.calls} API requests.)")


@workbench_app.command("langsmith-pull")
def langsmith_pull_command(
    project: Annotated[str, typer.Option(help="LangSmith tracing project name.")],
    since: Annotated[str, typer.Option(help="Start of the window (ISO 8601, inclusive).")],
    until: Annotated[str, typer.Option(help="End of the window (ISO 8601, exclusive).")],
    traces: Annotated[
        Path | None,
        typer.Option(help="Write runs here (JSONL) for import-traces or intake."),
    ] = None,
    judgments: Annotated[
        Path | None,
        typer.Option(help="Write feedback here as answer judgments (JSONL) for attach-answers."),
    ] = None,
    human_source: Annotated[
        list[str] | None,
        typer.Option(help="Feedback source types counted as human review (default: app)."),
    ] = None,
) -> None:
    """Pull runs and feedback from a LangSmith project for the given time window.

    The window applies to each trace's root run, so traces are never split. Reads
    LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) and LANGSMITH_ENDPOINT (default
    https://api.smith.langchain.com). Read-only: nothing is written to LangSmith, and
    feedback comments are never written to the output files.
    """
    import os

    from vectraudit.pipelines.langfuse_api import write_jsonl
    from vectraudit.pipelines.langsmith_api import (
        LangSmithSource,
        feedback_to_judgments,
        fetch_feedback,
        fetch_runs,
    )

    if traces is None and judgments is None:
        typer.echo("Pull error: give --traces, --judgments or both.", err=True)
        raise typer.Exit(2)
    try:
        source = LangSmithSource(
            api_url=os.environ.get("LANGSMITH_ENDPOINT")
            or os.environ.get("LANGCHAIN_ENDPOINT")
            or "https://api.smith.langchain.com",
            api_key=os.environ.get("LANGSMITH_API_KEY")
            or os.environ.get("LANGCHAIN_API_KEY")
            or "",
        )
        runs = fetch_runs(source, project, start=since, end=until)
        n_traces = len({r["trace_id"] for r in runs})
        lines = []
        if traces is not None:
            write_jsonl(traces, runs)
            lines.append(f"Wrote {len(runs)} runs from {n_traces} traces to {traces}.")
        if judgments is not None:
            feedback = fetch_feedback(source, [str(r["id"]) for r in runs])
            rows, skipped = feedback_to_judgments(
                feedback, runs, human_sources=human_source or ("app",)
            )
            write_jsonl(judgments, rows)
            lines.append(
                f"Wrote {len(rows)} of {len(feedback)} feedback records as judgments to "
                f"{judgments}."
            )
            if skipped:
                lines.append(
                    "Skipped: "
                    + ", ".join(
                        f"{n} {why.replace('_', ' ')}" for why, n in sorted(skipped.items())
                    )
                )
            if any("score" in r for r in rows):
                lines.append("Numeric scores need --threshold EVALUATOR=VALUE when attached.")
    except (OSError, ValueError) as exc:
        typer.echo(f"Pull error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo("\n".join(lines))
    typer.echo(f"({source.calls} API requests.)")


@workbench_app.command("drift")
def drift_command(
    traces: Annotated[
        Path, typer.Argument(help="Production traces (any format import-traces reads).")
    ],
    dataset: Annotated[Path, typer.Argument(help="The benchmark to compare against.")],
    context_field: Annotated[
        list[str] | None,
        typer.Option(help="Context field to compare, e.g. tenant or language (repeatable)."),
    ] = None,
    stage_kind: Annotated[
        list[str] | None, typer.Option(help="SPAN=KIND, as in import-traces.")
    ] = None,
    top: Annotated[
        int, typer.Option(min=1, max=200, help="How many values and terms to list.")
    ] = 20,
    method: Annotated[
        str,
        typer.Option(
            help="auto: Evidently's default tests (K-S, chi-square, Z, Wasserstein, "
            "Jensen-Shannon); psi: population stability index (noisy below ~1,000 values)."
        ),
    ] = "auto",
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    fail_on_drift: Annotated[
        bool, typer.Option("--fail-on-drift", help="Exit 3 when any descriptor drifted.")
    ] = False,
) -> None:
    """Is the benchmark still representative of production questions?

    Compares production questions from traces with the benchmark's on word count, script,
    digits, question form, unseen terms and declared context fields, with the tests
    Evidently selects by default (--method psi for the population stability index).
    Local only.
    """
    from vectraudit.pipelines.drift import production_questions, query_drift, render
    from vectraudit.pipelines.traces import Settings

    fields = context_field or []
    try:
        bench = load_dataset(dataset)
        current, unreadable = production_questions(
            traces, Settings(stage_kinds=_stage_kinds(stage_kind)), fields
        )
        report = query_drift(
            [(q.text or "", dict(q.context)) for q in bench.queries],
            current,
            context_fields=fields,
            top=top,
            method=method,
        )
        report["unreadable_traces"] = unreadable
    except (OSError, ValueError) as exc:
        typer.echo(f"Drift error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(render(report))
    if unreadable:
        typer.echo(f"{unreadable} traces had no readable question and were not counted.")
    if json_out is not None:
        _write(json_out, report)
    if fail_on_drift and report["status"] == "drift":
        raise typer.Exit(3)


@workbench_app.command("score-drift")
def score_drift_command(
    earlier: Annotated[Path, typer.Argument(help="Traces from an earlier window (reference).")],
    later: Annotated[Path, typer.Argument(help="Traces from a later window (current).")],
    stage: Annotated[
        str | None, typer.Option(help="Span name of the retriever (default: the first one).")
    ] = None,
    stage_kind: Annotated[
        list[str] | None, typer.Option(help="SPAN=KIND, as in import-traces.")
    ] = None,
    id_field: Annotated[
        str | None, typer.Option(help="Document metadata field to use as the document ID.")
    ] = None,
    method: Annotated[
        str,
        typer.Option(
            help="auto: Evidently's default tests (K-S, chi-square, Z, Wasserstein, "
            "Jensen-Shannon); psi: population stability index (noisy below ~1,000 values)."
        ),
    ] = "auto",
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    fail_on_drift: Annotated[
        bool, typer.Option("--fail-on-drift", help="Exit 3 when the retriever likely changed.")
    ] = False,
) -> None:
    """Did the retriever's scores move between two windows of production traces?

    Compares the first document's score, the mean, the spread and how many documents came
    back, with the tests Evidently selects by default, and checks the questions too, so a
    retriever change (scores move, questions do not) is told apart from changed questions.
    """
    from vectraudit.pipelines.drift import render_scores, score_drift
    from vectraudit.pipelines.traces import Settings

    try:
        settings = Settings(id_field=id_field, stage_kinds=_stage_kinds(stage_kind))
        report = score_drift(earlier, later, settings, stage=stage, method=method)
    except (OSError, ValueError) as exc:
        typer.echo(f"Drift error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(render_scores(report))
    if json_out is not None:
        _write(json_out, report)
    if fail_on_drift and report["status"] == "retriever_shift_likely":
        raise typer.Exit(3)


@workbench_app.command("intake")
def intake_command(
    traces: Annotated[
        Path,
        typer.Argument(help="New trace export (OpenInference, LangSmith runs or Langfuse)."),
    ],
    dataset: Annotated[Path, typer.Argument(help="The benchmark, with labels.")],
    state: Annotated[
        Path, typer.Option(help="Folder holding the cursor and review queue (kept local).")
    ] = Path("intake"),
    feedback: Annotated[
        Path | None,
        typer.Option(help="Feedback per trace (JSONL: trace_id, signal); selects, never labels."),
    ] = None,
    routes: Annotated[
        Path | None,
        typer.Option(help='Reviewer rules: [{"when": {"tenant": "acme"}, "reviewer": "ana"}].'),
    ] = None,
    context_field: Annotated[
        list[str] | None,
        typer.Option(help="Trace metadata field kept as question context (repeatable)."),
    ] = None,
    sample_rate: Annotated[
        float,
        typer.Option(min=0, max=1, help="Share of new questions kept as an independent sample."),
    ] = 0.1,
    workspace: Annotated[
        Path | None, typer.Option(help="Workbench folder, to link recurrences to regression cases.")
    ] = None,
    export_review: Annotated[
        Path | None, typer.Option(help="Write pending questions as an unlabelled benchmark.")
    ] = None,
    reviewer: Annotated[
        str | None, typer.Option(help="With --export-review: only this reviewer's questions.")
    ] = None,
    id_field: Annotated[
        str | None, typer.Option(help="Document metadata field to use as the document ID.")
    ] = None,
    final_span: Annotated[
        str | None, typer.Option(help="Span name whose documents are delivered.")
    ] = None,
    stage_kind: Annotated[
        list[str] | None,
        typer.Option(
            help="SPAN=KIND for retriever spans, as in import-traces (needed for Opik, which "
            "has no retriever span type)."
        ),
    ] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write JSON.")] = None,
    fail_on_recurrence: Annotated[
        bool,
        typer.Option("--fail-on-recurrence", help="Exit 3 when a likely recurrence is found."),
    ] = False,
) -> None:
    """Production intake: take new traces since the last run into benchmark upkeep.

    Matched questions that missed their labelled documents are likely recurrences; new
    questions join a de-duplicated review queue with priority from feedback and frequency,
    an independent sample, and a routed reviewer. Run it on each new export.
    """
    from vectraudit.pipelines.intake import (
        intake,
        read_feedback,
        read_routes,
        render,
        review_dataset,
    )
    from vectraudit.pipelines.traces import Settings

    try:
        report = intake(
            traces,
            load_dataset(dataset),
            state,
            settings=Settings(
                id_field=id_field, final_span=final_span, stage_kinds=_stage_kinds(stage_kind)
            ),
            context_fields=context_field or (),
            feedback=read_feedback(feedback) if feedback else None,
            routes=read_routes(routes) if routes else (),
            sample_rate=sample_rate,
            workspace=workspace,
        )
        if export_review is not None:
            text = review_dataset(state, reviewer=reviewer)
            export_review.write_text(text, encoding="utf-8")
    except (OSError, ValueError) as exc:
        typer.echo(f"Intake error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(render(report))
    if export_review is not None:
        typer.echo(f"Review file: {export_review}")
    if json_out is not None:
        _write(json_out, report)
    if fail_on_recurrence and report["likely_recurrences"]:
        raise typer.Exit(3)


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
    lineage: Annotated[
        Path | None,
        typer.Option(help="Lineage export (JSONL) investigations use to name ingestion losses."),
    ] = None,
    lineage_ids: Annotated[
        str, typer.Option(help="What run result IDs are in the lineage export: chunk or source.")
    ] = "chunk",
    changes: Annotated[
        Path | None,
        typer.Option(help="Content change log (JSONL) investigations use to flag stale content."),
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
    try:
        app = create_workbench(directory, manifest, policy, lineage, lineage_ids, changes)
    except ValueError as exc:
        typer.echo(f"Workbench configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc
    uvicorn.run(app, host="127.0.0.1", port=port)


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
    """Share one saved case, benchmark snapshot, policy revision or investigation."""
    from typing import cast

    from fastapi import HTTPException

    from vectraudit.integrations.cloud import CloudClient, CloudClientError, CloudTarget
    from vectraudit.integrations.shared_records import preview_record
    from vectraudit.pipelines.guided import GuidedWorkspace
    from vectraudit.pipelines.share import CATEGORIES, GENERIC_ERROR, projection

    try:
        if kind not in CATEGORIES:
            raise ValueError("Kind must be case, benchmark, policy or investigation")
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
