"""Release-check commands: ``vectraudit dataset validate | record | check``.

``record`` runs a dataset against a connection and writes a run file; ``check`` compares a
candidate run with a baseline under a policy. Exit codes of ``check`` follow
``sweep --baseline``: 0 pass, 1 error, 3 fail (regression), 4 not comparable.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from vectraudit.connectors.base import ConnectorError, ParamValue
from vectraudit.connectors.config import load_connection, open_connector
from vectraudit.evaluation.compare import Verdict, compare_runs
from vectraudit.evaluation.dataset import DatasetError, load_dataset
from vectraudit.evaluation.policy import DEFAULT_POLICY, PolicyError, load_policy
from vectraudit.evaluation.report import render_html, render_markdown
from vectraudit.evaluation.runfile import RunFileError, load_run, save_run
from vectraudit.evaluation.runner import RunOptions, TextEmbedder, execute_run, failed_run

# Accepted means every failing rule is covered by an active audited exception: it unblocks
# the release, and every report still says "fail, accepted", never "pass".
EXIT_CODES = {
    Verdict.PASS: 0,
    Verdict.ACCEPTED: 0,
    Verdict.ERROR: 1,
    Verdict.FAIL: 3,
    Verdict.NOT_COMPARABLE: 4,
}
_VERDICT_STYLE = {
    Verdict.PASS: "bold green",
    Verdict.FAIL: "bold red",
    Verdict.ERROR: "bold yellow",
    Verdict.NOT_COMPARABLE: "bold magenta",
    Verdict.ACCEPTED: "bold yellow",
}

console = Console()
err_console = Console(stderr=True)

dataset_app = typer.Typer(help="Work with evaluation datasets (JSONL).", no_args_is_help=True)


cloud_app = typer.Typer(
    help="Work with the hosted Vectraudit service (VECTRAUDIT_CLOUD_URL / _TOKEN).",
    no_args_is_help=True,
)


def _cloud_client(project: str | None) -> Any:
    from vectraudit.integrations.cloud import CloudClient, CloudClientError, CloudTarget

    try:
        return CloudClient(CloudTarget.from_env(project))
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e


def _upload(client: Any, run: dict[str, Any]) -> bool:
    """Upload one run; report the outcome. Returns False on failure."""
    from vectraudit.integrations.cloud import CloudClientError

    try:
        stored = client.upload_run(run)
    except CloudClientError as e:
        err_console.print(f"[bold red]Upload failed:[/] {escape(e.message)}")
        if e.hint:
            err_console.print(f"[dim]Hint: {escape(e.hint)}[/]")
        return False
    console.print(f"Uploaded run {escape(stored['run_id'])} ({escape(stored['status'])})")
    return True


@cloud_app.command("push")
def cloud_push(
    runs: Annotated[list[Path], typer.Argument(help="Run files to upload.")],
    project: Annotated[
        str | None, typer.Option(help="workspace/project (or VECTRAUDIT_CLOUD_PROJECT).")
    ] = None,
    share_query_text: Annotated[
        bool, typer.Option(help="Explicitly share recorded query text/history. Preview first.")
    ] = False,
) -> None:
    """Upload run files. Retrying is safe: a run is stored once."""
    client = _cloud_client(project)
    ok = True
    for path in runs:
        try:
            run = load_run(path)
        except RunFileError as e:
            err_console.print(f"[bold red]Error:[/] {escape(str(e))}")
            ok = False
            continue
        from vectraudit.integrations.sharing import project_run

        try:
            projected = project_run(run, share_query_text=share_query_text)
        except ValueError as e:
            err_console.print(f"[bold red]Error:[/] {escape(str(e))}")
            ok = False
            continue
        ok = _upload(client, projected) and ok
    if not ok:
        raise typer.Exit(code=1)


@cloud_app.command("preview")
def cloud_preview(
    run: Path,
    share_query_text: Annotated[
        bool, typer.Option(help="Preview explicit query text/history sharing.")
    ] = False,
    out: Annotated[
        Path | None, typer.Option(help="Write the exact projection to a separate local file.")
    ] = None,
) -> None:
    """Preview a transfer without credentials or a network request. Raw run stays unchanged."""
    from vectraudit.integrations.sharing import project_run, transfer_preview

    try:
        record = load_run(run)
        projected = project_run(record, share_query_text=share_query_text)
        if out is not None:
            if out.resolve() == run.resolve() or out.exists():
                raise ValueError(
                    "Choose a new output file; existing artifacts are never overwritten."
                )
            save_run(projected, out)
        console.print_json(data=transfer_preview(projected))
    except (ValueError, OSError) as e:
        raise _fail(str(e)) from e


def _fail(message: str, hint: str | None = None) -> typer.Exit:
    err_console.print(f"[bold red]Error:[/] {escape(message)}")
    if hint:
        err_console.print(f"[dim]Hint: {escape(hint)}[/]")
    return typer.Exit(code=1)


@dataset_app.command("validate")
def dataset_validate(
    path: Annotated[Path, typer.Argument(help="Dataset JSONL file.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Validate a dataset and print its version (content hash)."""
    try:
        ds = load_dataset(path)
    except DatasetError as e:
        raise _fail(str(e)) from e
    summary = ds.summary()
    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return
    table = Table(show_header=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Version", summary["sha256"])
    table.add_row("Queries", f"{summary['n_queries']:,}")
    table.add_row("Labelled", f"{summary['n_labelled']:,}")
    table.add_row("Filtered", f"{summary['n_filtered']:,}")
    table.add_row("Dimension", str(summary["dim"] or "text only (needs --embedder)"))
    table.add_row("Tags", ", ".join(summary["tags"]) or "none")
    console.print(table)


def init(
    backend: Annotated[
        str,
        typer.Option(
            help="faiss, qdrant, pgvector, pinecone, weaviate, elasticsearch or opensearch."
        ),
    ] = "faiss",
    directory: Annotated[
        Path, typer.Option("--dir", help="Repository root to scaffold into.")
    ] = Path(),
    force: Annotated[bool, typer.Option(help="Overwrite files that already exist.")] = False,
    ci_only: Annotated[
        bool,
        typer.Option(
            "--ci-only",
            help=(
                "Write only the GitHub workflow and actions (with --force, upgrades them); "
                "retrieval/ is left alone."
            ),
        ),
    ] = False,
) -> None:
    """Scaffold retrieval/ (connection, policy, queries) and a GitHub workflow."""
    from vectraudit.scaffold import BACKENDS, init_project

    if backend not in BACKENDS:
        raise _fail(f"Unknown backend {backend!r}.", f"Use one of: {', '.join(BACKENDS)}.")
    written, skipped = init_project(directory, backend, force=force, ci_only=ci_only)
    for path in written:
        console.print(f"  [green]created[/] {escape(str(path))}")
    for path in skipped:
        console.print(f"  [yellow]kept[/]    {escape(str(path))} (exists; --force overwrites)")
    console.print("Next: follow retrieval/README.md.")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataset_app.command("import")
def dataset_import(
    source: Annotated[
        Path,
        typer.Argument(
            help=(
                "CSV (with a header row), JSON array or JSONL file. JSON columns may be dotted "
                "paths (inputs.question). With --format: a BEIR folder, a TREC query file "
                "(qid<TAB>text or JSONL), or a Ragas/DeepEval test set."
            )
        ),
    ],
    out: Annotated[Path, typer.Option(help="Dataset JSONL to write.")],
    source_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            help="beir, trec, ragas or deepeval. Omit for CSV/JSON with column mapping.",
        ),
    ] = None,
    qrels: Annotated[
        Path | None, typer.Option(help="TREC qrels file (qid iter docno grade), for --format trec.")
    ] = None,
    split: Annotated[str, typer.Option(help="BEIR qrels split, for --format beir.")] = "test",
    corpus: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Documents the backend returns (JSONL/JSON/CSV with id and text), to locate "
                "Ragas/DeepEval reference passages given as text."
            )
        ),
    ] = None,
    corpus_id_field: Annotated[
        str | None, typer.Option(help="Corpus id field (default: _id, doc_id or id).")
    ] = None,
    corpus_text_field: Annotated[
        str | None, typer.Option(help="Corpus text field (default: text, content, page_content).")
    ] = None,
    max_matches: Annotated[
        int,
        typer.Option(
            min=1, help="A passage found in more documents than this is ambiguous and skipped."
        ),
    ] = 5,
    id_column: Annotated[str, typer.Option(help="Column with the query id.")] = "query_id",
    text_column: Annotated[str, typer.Option(help="Column with the query text.")] = "text",
    vector_column: Annotated[
        str | None, typer.Option(help="Column with the vector as a JSON list.")
    ] = None,
    relevant_column: Annotated[
        str,
        typer.Option(help='Relevant documents: "doc-1:2; doc-7" (grade defaults to 1).'),
    ] = "relevant",
    tags_column: Annotated[str | None, typer.Option(help='Cohort tags, ";"-separated.')] = None,
    filter_column: Annotated[
        list[str] | None,
        typer.Option(
            "--filter",
            help="Filter column, optionally typed: lang, year:int, price:float, public:bool.",
        ),
    ] = None,
    name: Annotated[str | None, typer.Option(help="Dataset name for the header.")] = None,
) -> None:
    """Convert queries and relevance labels you already have into a dataset."""
    from vectraudit.evaluation.dataset_formats import import_format
    from vectraudit.evaluation.dataset_tools import (
        ColumnMap,
        import_rows,
        read_records,
        render_dataset,
    )

    meta: dict[str, Any] = {"name": name} if name else {}
    warnings: list[str] = []
    try:
        if source_format:
            imported = import_format(
                source_format.lower(),
                source,
                qrels=qrels,
                split=split,
                corpus=corpus,
                corpus_id_field=corpus_id_field,
                corpus_text_field=corpus_text_field,
                max_matches=max_matches,
            )
            rows, warnings = imported.rows, imported.warnings
            meta["provenance"] = imported.provenance
        else:
            filters = {}
            for spec in filter_column or []:
                col, _, kind = spec.partition(":")
                filters[col.strip()] = kind.strip() or "str"
            columns = ColumnMap(
                query_id=id_column,
                text=text_column,
                vector=vector_column,
                relevant=relevant_column,
                tags=tags_column,
                filters=filters,
            )
            rows = import_rows(read_records(source), columns)
        text = render_dataset(rows, meta or None)
    except DatasetError as e:
        raise _fail(str(e)) from e
    _write_text(out, text)
    ds = load_dataset(out)
    for warning in warnings:
        err_console.print(f"[yellow]Warning:[/] {escape(warning)}")
    console.print(
        f"Wrote {len(ds):,} queries ({ds.n_labelled:,} labelled) to {escape(str(out))}, "
        f"version {ds.sha256[:12]}"
    )


@dataset_app.command("embed")
def dataset_embed(
    path: Annotated[Path, typer.Argument(help="Dataset JSONL.")],
    model: Annotated[str, typer.Option(help="sentence-transformers model of the corpus.")],
    out: Annotated[Path | None, typer.Option(help="Output (default: overwrite input).")] = None,
    normalize: Annotated[
        bool, typer.Option(help="L2-normalise vectors (cosine / inner-product corpora).")
    ] = False,
    overwrite: Annotated[bool, typer.Option(help="Re-embed queries that have vectors.")] = False,
) -> None:
    """Add query vectors from text, and record the model in the dataset header."""
    from vectraudit.evaluation.dataset_tools import embed_dataset

    embed = _embedder(model, normalize)
    assert embed is not None
    try:
        text = embed_dataset(
            load_dataset(path), embed, model=model, normalized=normalize, overwrite=overwrite
        )
    except DatasetError as e:
        raise _fail(str(e)) from e
    target = out or path
    _write_text(target, text)
    ds = load_dataset(target)
    console.print(
        f"Embedded with {escape(model)}: {len(ds):,} queries, dimension {ds.dim}, "
        f"version {ds.sha256[:12]} → {escape(str(target))}"
    )


@dataset_app.command("diff")
def dataset_diff(
    old: Annotated[Path, typer.Argument(help="Earlier dataset.")],
    new: Annotated[Path, typer.Argument(help="Later dataset.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Show queries added, removed and changed between two dataset versions."""
    from vectraudit.evaluation.dataset_tools import diff_datasets

    try:
        diff = diff_datasets(load_dataset(old), load_dataset(new))
    except DatasetError as e:
        raise _fail(str(e)) from e
    if as_json:
        typer.echo(json.dumps(diff, indent=2))
        return
    if diff["same"]:
        console.print(f"Identical: version {diff['from'][:12]}")
        return
    console.print(f"{diff['from'][:12]} → {diff['to'][:12]}")
    console.print(
        f"Added {len(diff['added'])}, removed {len(diff['removed'])}, "
        f"changed {len(diff['changed'])}"
    )
    for qid, fields in list(diff["changed"].items())[:20]:
        console.print(f"  ~ {escape(qid)}: {', '.join(fields)}")
    for qid in diff["added"][:10]:
        console.print(f"  + {escape(qid)}")
    for qid in diff["removed"][:10]:
        console.print(f"  - {escape(qid)}")
    if diff["meta_changed"]:
        console.print(f"Header changed: {', '.join(diff['meta_changed'])}")


@dataset_app.command("stats")
def dataset_stats_cmd(
    path: Annotated[Path, typer.Argument(help="Dataset JSONL.")],
    policy: Annotated[
        Path | None, typer.Option(help="Policy file, to check cohort sizes against it.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Label coverage, grades, cohorts and readiness warnings."""
    from vectraudit.evaluation.dataset_tools import dataset_stats

    try:
        ds = load_dataset(path)
        minimum = load_policy(policy).cohorts.min_queries if policy else None
    except (DatasetError, PolicyError) as e:
        raise _fail(str(e)) from e
    stats = dataset_stats(ds, cohort_minimum=minimum)
    if as_json:
        typer.echo(json.dumps(stats, indent=2))
        return
    rel = stats["relevant_per_query"]
    console.print(
        f"Version {stats['sha256'][:12]} · {stats['n_queries']:,} queries · "
        f"{stats['n_labelled']:,} labelled · {stats['n_filtered']:,} filtered"
    )
    console.print(
        f"Relevant per labelled query: min {rel['min']}, mean {rel['mean']:.1f}, max {rel['max']}"
    )
    if stats["grades"]:
        console.print("Grades: " + ", ".join(f"{g}: {n}" for g, n in stats["grades"].items()))
    if stats["tags"]:
        console.print(
            "Cohorts: " + ", ".join(f"{escape(t)} ({n})" for t, n in stats["tags"].items())
        )
    for w in stats["warnings"]:
        console.print(f"[yellow]Warning:[/] {escape(w)}")


def _parse_pairs(values: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise _fail(f"{what} must look like key=value, got {item!r}.")
        out[key.strip()] = value.strip()
    return out


def _parse_params(values: Sequence[str]) -> dict[str, ParamValue]:
    out: dict[str, ParamValue] = {}
    for key, raw in _parse_pairs(values, "--param").items():
        try:
            out[key] = int(raw)
        except ValueError:
            try:
                out[key] = float(raw)
            except ValueError:
                raise _fail(f"--param {key} must be a number, got {raw!r}.") from None
    return out


def _embedder(model: str | None, normalize: bool) -> TextEmbedder | None:
    if model is None:
        return None
    from vectraudit.core.embed import EmbedderUnavailableError, sentence_transformer_embedder

    try:
        embed = sentence_transformer_embedder(model, normalize=normalize)
    except EmbedderUnavailableError as e:
        raise _fail(str(e), e.hint) from e

    def batch(texts: Sequence[str]) -> Any:
        return np.stack([np.asarray(embed(t), dtype=np.float32) for t in texts])

    return batch


def record(
    connection: Annotated[Path, typer.Option(help="Connection file (JSON).")],
    dataset: Annotated[Path, typer.Option(help="Dataset file (JSONL).")],
    out: Annotated[Path, typer.Option(help="Where to write the run file (JSON).")],
    k: Annotated[int, typer.Option(min=1, help="Results per query.")] = 10,
    param: Annotated[
        list[str] | None,
        typer.Option(help="Search parameter, e.g. nprobe=16. Overrides the connection file."),
    ] = None,
    warmup: Annotated[int, typer.Option(min=0, help="Untimed warm-up queries.")] = 5,
    repeats: Annotated[
        int, typer.Option(min=1, help="Timed searches per query (median kept).")
    ] = 1,
    exact: Annotated[
        bool | None,
        typer.Option(
            "--exact/--no-exact",
            help="Measure ANN recall against an exact search (default: when available).",
        ),
    ] = None,
    label: Annotated[str | None, typer.Option(help="Display name, e.g. main or pr-42.")] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option(help="Provenance label, e.g. embedding_model=e5-large. Repeatable."),
    ] = None,
    embedder: Annotated[
        str | None, typer.Option(help="sentence-transformers model for text-only queries.")
    ] = None,
    evidence: Annotated[
        bool,
        typer.Option(
            "--evidence/--no-evidence",
            help="Record per-query backend evidence (IVF miss reasons, query plans).",
        ),
    ] = True,
    upload: Annotated[
        bool, typer.Option(help="Also upload the run to the hosted service.")
    ] = False,
    project: Annotated[
        str | None,
        typer.Option(help="Cloud workspace/project for --upload (or VECTRAUDIT_CLOUD_PROJECT)."),
    ] = None,
    normalize_text: Annotated[
        bool, typer.Option(help="L2-normalise text embeddings (for cosine / IP corpora).")
    ] = False,
) -> None:
    """Run a dataset against a backend and write a run file.

    Exits 0 when every query succeeded, 1 otherwise. The run file is written either way,
    so `vectraudit check` can report the failure as an error.
    """
    try:
        ds = load_dataset(dataset)
        cfg = load_connection(connection)
    except DatasetError as e:
        raise _fail(str(e)) from e
    except ConnectorError as e:
        raise _fail(e.message, e.hint) from e
    params = {**cfg.params, **_parse_params(param or [])}
    options = RunOptions(
        k=k,
        params=params,
        warmup=warmup,
        repeats=repeats,
        exact=exact,
        label=label,
        labels={
            **(
                {"dataset_embedding_model": ds.meta["embedding_model"]}
                if ds.meta.get("embedding_model")
                else {}
            ),
            **_parse_pairs(tag or [], "--tag"),
        },
        evidence=evidence,
    )
    client = _cloud_client(project) if upload else None  # fail on bad config before running
    dataset_model = ds.meta.get("embedding_model")
    if embedder and dataset_model and embedder != dataset_model:
        err_console.print(
            f"[yellow]Warning:[/] the dataset's vectors come from {escape(dataset_model)}, "
            f"but text queries will be embedded with {escape(embedder)}."
        )
    embed = _embedder(embedder, normalize_text)
    try:
        connector = open_connector(cfg)
    except ConnectorError as e:
        # Still write a (failed) run file, so `check` reports the error on the release.
        run = failed_run(ds, e, options, connection=cfg.public())
        save_run(run, out)
        _print_run(run, out)
        if client is not None:
            _upload(client, run)
        raise typer.Exit(code=1) from e
    with connector, console.status("Running queries…") as status:

        def progress(done: int, total: int) -> None:
            status.update(f"Running queries… {done:,} / {total:,}")

        run = execute_run(
            ds,
            connector,
            options,
            connection=cfg.public(),
            embed=embed,
            embedder_name=embedder,
            progress=progress,
        )
    save_run(run, out)
    _print_run(run, out)
    uploaded = _upload(client, run) if client is not None else True
    if run["status"] == "cancelled":
        raise typer.Exit(code=130)
    if run["status"] != "complete" or not uploaded:
        raise typer.Exit(code=1)


def _print_run(run: dict[str, Any], out: Path) -> None:
    s = run["summary"]
    backend = (run.get("backend") or {}).get("backend") or run["connection"].get("backend")
    style = "green" if run["status"] == "complete" else "red"
    console.print(
        f"[{style}]{escape(run['status'])}[/]: {s['n_ok']:,} of {s['n_queries']:,} queries "
        f"on {escape(str(backend))} → {escape(str(out))}"
    )
    if run["error"]:
        err_console.print(
            f"[bold red]Error ({run['error']['code']}):[/] {escape(run['error']['message'])}"
        )
        if run["error"].get("hint"):
            err_console.print(f"[dim]Hint: {escape(run['error']['hint'])}[/]")
    elif s["n_errors"]:
        first = next(q["error"] for q in run["queries"] if q["error"])
        err_console.print(
            f"[red]{s['n_errors']:,} queries failed[/], e.g. {escape(first['message'])}"
        )
    if s["metrics"]:
        parts = ", ".join(f"{k} {v:.4f}" for k, v in s["metrics"].items())
        console.print(f"Labelled @{run['search']['k']}: {parts}")
    if s["ann_recall"] is not None:
        console.print(f"ANN recall @{run['search']['k']}: {s['ann_recall']:.4f}")
    if s["latency_ms"]:
        lat = s["latency_ms"]
        console.print(f"Latency: p50 {lat['p50']:.2f} ms, p95 {lat['p95']:.2f} ms")


def check(
    baseline: Annotated[Path, typer.Argument(help="Baseline run file.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate run file.")],
    policy: Annotated[
        Path | None, typer.Option(help="Policy file (JSON). Default: built-in policy.")
    ] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write the result JSON.")] = None,
    html: Annotated[Path | None, typer.Option(help="Write an HTML report.")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Write a Markdown summary.")] = None,
    github_summary: Annotated[
        bool,
        typer.Option(help="Append the Markdown summary to $GITHUB_STEP_SUMMARY when set."),
    ] = True,
    report_url: Annotated[
        str | None, typer.Option(help="Link to the full report, shown in the summary.")
    ] = None,
    github_check: Annotated[
        bool,
        typer.Option(help="Publish a check run on the commit (needs checks: write)."),
    ] = False,
    github_comment: Annotated[
        bool,
        typer.Option(help="Create or update one PR comment (needs pull-requests: write)."),
    ] = False,
    check_name: Annotated[
        str, typer.Option(help="Check run name, e.g. one per backend.")
    ] = "Retrieval check",
    project: Annotated[
        str | None,
        typer.Option(help="Cloud workspace/project for cloud:<baseline> or --upload."),
    ] = None,
    upload: Annotated[
        bool,
        typer.Option(help="Upload the candidate and record the check on the hosted service."),
    ] = False,
) -> None:
    """Compare a candidate run with a baseline and decide.

    Exit codes: 0 pass, 1 error (a run is incomplete or a file is invalid), 3 fail
    (a policy rule is broken), 4 not comparable (different data or missing metrics).
    """
    cloud_baseline = str(baseline).startswith("cloud:")
    client = _cloud_client(project) if cloud_baseline or upload else None
    if upload and not cloud_baseline:
        raise _fail("--upload compares with a pinned baseline: pass cloud:<name> as baseline.")
    try:
        cand = load_run(candidate)
        if cloud_baseline:
            base, pol = _cloud_baseline_and_policy(client, str(baseline)[6:], policy)
            # The hosted baseline is a sharing projection; project the candidate the same way
            # so this report and the hosted check compare identical fields.
            from vectraudit.integrations.sharing import project_run

            cand = project_run(cand)
        else:
            pol = DEFAULT_POLICY if policy is None else load_policy(policy)
            base = load_run(baseline)
    except (PolicyError, RunFileError, ValueError) as e:
        raise _fail(str(e)) from e
    result = compare_runs(base, cand, pol)
    if cloud_baseline:
        result = _apply_cloud_exceptions(client, result, pol)
    verdict = Verdict(result["verdict"])
    if upload:
        hosted = _record_cloud_check(client, cand, str(baseline)[6:], verdict)
        report_url = report_url or hosted
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
    raise typer.Exit(code=EXIT_CODES[verdict])


def emit_check(
    result: dict[str, Any],
    *,
    json_out: Path | None,
    html: Path | None,
    markdown: Path | None,
    github_summary: bool,
    report_url: str | None,
    github_check: bool,
    github_comment: bool,
    check_name: str,
) -> None:
    """Write the reports, print the verdict and publish to GitHub (best effort)."""
    verdict = Verdict(result["verdict"])
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(render_html(result), encoding="utf-8")
    md = render_markdown(result, report_url=report_url)
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(md, encoding="utf-8")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary and summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md)
    _print_check(result, verdict)
    if github_check or github_comment:
        _publish_github(result, md, github_check, github_comment, check_name, report_url)


def _publish_github(
    result: dict[str, Any],
    markdown: str,
    check_run: bool,
    comment: bool,
    name: str,
    report_url: str | None,
) -> None:
    """Publishing problems are warnings: they must never change the check's exit code."""
    from vectraudit.integrations.github import (
        GitHubContext,
        GitHubError,
        publish_check_run,
        upsert_pr_comment,
    )

    try:
        ctx = GitHubContext.from_env()
    except GitHubError as e:
        err_console.print(f"[yellow]Warning:[/] not published to GitHub: {escape(str(e))}")
        return
    actions: list[tuple[str, Any]] = []
    if check_run:
        actions.append(
            (
                "check run",
                lambda: publish_check_run(ctx, result, markdown, name=name, details_url=report_url),
            )
        )
    if comment:
        actions.append(("pull request comment", lambda: upsert_pr_comment(ctx, markdown)))
    for what, publish in actions:
        try:
            url = publish()
        except GitHubError as e:
            hint = (
                " (read-only token: pull requests from forks can't publish; the job summary "
                "still has the result)"
                if e.forbidden
                else ""
            )
            err_console.print(f"[yellow]Warning:[/] {what} not published{hint}: {escape(str(e))}")
        else:
            console.print(f"Published {what}" + (f": {escape(url)}" if url else "."))


def _cloud_baseline_and_policy(
    client: Any, name: str, policy: Path | None
) -> tuple[dict[str, Any], Any]:
    """The pinned baseline run, and the project's policy unless --policy overrides it."""
    from vectraudit.evaluation.policy import parse_policy
    from vectraudit.evaluation.runfile import check_run
    from vectraudit.integrations.cloud import CloudClientError

    try:
        base = check_run(client.get_baseline_file(name))
        if policy is not None:
            return base, load_policy(policy)
        remote = client.get_policy()["policy"]
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    return base, DEFAULT_POLICY if remote is None else parse_policy(remote)


def _apply_cloud_exceptions(client: Any, result: dict[str, Any], policy: Any) -> dict[str, Any]:
    """Apply the project's active audited exceptions, as the hosted check will.

    If they cannot be read, the verdict is left as computed: a failure stays a failure.
    """
    from vectraudit.evaluation.exceptions import apply_exceptions
    from vectraudit.integrations.cloud import CloudClientError

    if result["verdict"] != Verdict.FAIL.value:
        return result
    try:
        active = client.list_exceptions(active=True)
    except CloudClientError as e:
        err_console.print(
            f"[yellow]Warning:[/] could not read audited exceptions ({escape(e.message)}); "
            "the verdict ignores them."
        )
        return result
    return apply_exceptions(result, active, policy)


def _record_cloud_check(
    client: Any, cand: dict[str, Any], baseline: str, verdict: Verdict, trigger: str = "release"
) -> str | None:
    """Upload the candidate and store the hosted check; returns the hosted report URL."""
    from vectraudit.integrations.cloud import CloudClientError

    if not _upload(client, cand):
        raise typer.Exit(code=1)
    try:
        stored = client.create_check(cand["run_id"], baseline, trigger=trigger)
    except CloudClientError as e:
        err_console.print(f"[bold red]Hosted check failed:[/] {escape(e.message)}")
        raise typer.Exit(code=1) from e
    console.print(f"Hosted check: {escape(stored['url'])}")
    console.print(f"Plain-language summary: vectraudit cloud summary {escape(stored['id'])}")
    if stored["verdict"] != verdict.value:
        err_console.print(
            f"[bold red]Warning:[/] the hosted verdict is {stored['verdict']} but the local one "
            f"is {verdict.value} (different policy or baseline?)."
        )
    report: str | None = stored.get("report_url")
    return report


@cloud_app.command("pin")
def cloud_pin(
    run_id: Annotated[str, typer.Argument(help="Uploaded run id to pin.")],
    name: Annotated[str, typer.Option(help="Baseline name, e.g. main.")] = "main",
    note: Annotated[str, typer.Option(help="Why this run is the new baseline.")] = "",
    project: Annotated[str | None, typer.Option(help="workspace/project.")] = None,
) -> None:
    """Pin an uploaded, complete run as a baseline (engineer or owner; audited)."""
    from vectraudit.integrations.cloud import CloudClientError

    client = _cloud_client(project)
    try:
        b = client.pin_baseline(name, run_id, note)
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    console.print(f"Baseline {escape(b['name'])} → run {escape(b['run_id'])}")


@cloud_app.command("runs")
def cloud_runs(
    project: Annotated[str | None, typer.Option(help="workspace/project.")] = None,
    status: Annotated[str | None, typer.Option(help="complete, partial, failed...")] = None,
    commit: Annotated[str | None, typer.Option(help="Commit SHA or prefix.")] = None,
    label: Annotated[str | None, typer.Option(help="Run label.")] = None,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 20,
) -> None:
    """Recent uploaded runs, newest first."""
    from vectraudit.integrations.cloud import CloudClientError

    try:
        page = _cloud_client(project).list_runs(
            limit, status=status or "", commit=commit or "", label=label or ""
        )
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    table = Table("Run", "Status", "Label", "Commit", "Backend", "Uploaded")
    for r in page["runs"]:
        table.add_row(
            r["run_id"][:12],
            r["status"],
            r["label"] or "",
            (r["commit_sha"] or "")[:10],
            r["backend"] or "",
            str(r["uploaded_at"])[:19],
        )
    console.print(table)


@cloud_app.command("baselines")
def cloud_baselines(
    project: Annotated[str | None, typer.Option(help="workspace/project.")] = None,
) -> None:
    """List the project's pinned baselines."""
    from vectraudit.integrations.cloud import CloudClientError

    try:
        rows = _cloud_client(project).list_baselines()
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    if not rows:
        console.print("No baselines pinned yet: `vectraudit cloud pin RUN_ID --name main`.")
    for b in rows:
        console.print(
            f"{escape(b['name'])}: run {escape(b['run_id'])} "
            f"({escape(b['run_label'] or '')}), pinned by {escape(b['pinned_by'])} "
            f"at {b['pinned_at']}"
        )


def _print_check(result: dict[str, Any], verdict: Verdict) -> None:
    label = verdict.value.replace("_", " ").upper()
    if verdict == Verdict.ACCEPTED:
        label = "FAIL, ACCEPTED"
    console.print(f"[{_VERDICT_STYLE[verdict]}]{label}[/] · {result['n_queries']} paired queries")
    for r in result["reasons"]:
        accepted = r.get("accepted")
        note = (
            f" [yellow](accepted by {escape(accepted['accepted_by'])} until "
            f"{escape(accepted['expires_at'][:10])}: {escape(accepted['reason'])})[/]"
            if accepted
            else ""
        )
        console.print(f"  • {escape(r['message'])}{note}")
    if result["metrics"]:
        table = Table("Metric", "Baseline", "Candidate", "Change", "95% CI")
        for name, m in result["metrics"].items():
            table.add_row(
                name,
                f"{m['baseline']:.4f}",
                f"{m['candidate']:.4f}",
                f"{m['delta']:+.4f}",
                f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]",
            )
        console.print(table)
    for w in result["warnings"]:
        console.print(f"[yellow]Warning:[/] {escape(w)}")


@cloud_app.command("accept")
def cloud_accept(
    check_id: Annotated[str, typer.Argument(help="Hosted check in which the rule failed.")],
    rule: Annotated[str, typer.Argument(help="Failing rule key, e.g. metric:ndcg.max_drop.")],
    reason: Annotated[str, typer.Option(help="Why this regression is acceptable.")],
    owner: Annotated[str, typer.Option(help="Who follows up, e.g. an email.")],
    expires: Annotated[str, typer.Option(help="Expiry date YYYY-MM-DD (UTC end of day).")],
    max_checks: Annotated[
        int | None, typer.Option(min=1, max=1000, help="Stop after covering N checks.")
    ] = None,
    project: Annotated[str | None, typer.Option(help="workspace/project.")] = None,
) -> None:
    """Accept one failing rule through an audited, expiring exception (as the policy allows)."""
    from datetime import date, datetime, time, timezone

    from vectraudit.integrations.cloud import CloudClientError

    try:
        day = date.fromisoformat(expires)
    except ValueError as e:
        raise _fail("--expires must be a date such as 2026-11-01.") from e
    until = datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc).isoformat()
    try:
        granted = _cloud_client(project).grant_exception(
            check_id,
            rule=rule,
            reason=reason,
            owner=owner,
            expires_at=until,
            max_checks=max_checks,
        )
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    console.print(
        f"Accepted {escape(granted['rule'])} until {escape(granted['expires_at'][:10])} "
        f"(exception {escape(granted['id'])}). Checks it covers read FAIL, ACCEPTED."
    )


@cloud_app.command("summary")
def cloud_summary(
    check_id: Annotated[str, typer.Argument(help="Check id, as `check --upload` prints it.")],
    project: Annotated[str | None, typer.Option(help="workspace/project.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the structured summary.")] = False,
) -> None:
    """A release decision in plain language, as Markdown for an issue, chat or CI summary.

    Works after the check's runs expire under retention. In GitHub Actions:
    ``vectraudit cloud summary "$CHECK_ID" >> "$GITHUB_STEP_SUMMARY"``.
    """
    from vectraudit.integrations.cloud import CloudClientError

    try:
        summary = _cloud_client(project).decision_summary(check_id)
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    typer.echo(json.dumps(summary, indent=2) if as_json else summary["markdown"], nl=False)
    if as_json:
        typer.echo()


def _minutes(text: str, option: str) -> int:
    """``90``, ``90m``, ``6h`` or ``1d`` as whole minutes."""
    import re

    match = re.fullmatch(r"\s*(\d+)\s*([mhd]?)\s*", text.lower())
    if not match:
        raise _fail(f"{option} must look like 30m, 6h or 1d.", None)
    return int(match[1]) * {"": 1, "m": 1, "h": 60, "d": 1440}[match[2]]


@cloud_app.command("monitor")
def cloud_monitor(
    every: Annotated[
        str | None,
        typer.Option(help="Expect a scheduled monitor run this often (30m, 6h, 1d)."),
    ] = None,
    grace: Annotated[
        str, typer.Option(help="Extra time before a late run counts as missing.")
    ] = "1h",
    clear: Annotated[
        bool, typer.Option("--clear", help="Stop expecting runs; closes an open alert.")
    ] = False,
    project: Annotated[str | None, typer.Option(help="workspace/project.")] = None,
) -> None:
    """Scheduled-monitor status; with --every, declare how often a run is expected.

    When no run arrives in time, subscribed webhooks get one "monitor overdue" event and
    the next run resolves it. Needs the engineer or owner role to change.
    """
    from vectraudit.integrations.cloud import CloudClientError

    client = _cloud_client(project)
    try:
        if clear:
            client.clear_monitor_schedule()
            console.print("Monitor schedule removed.")
            return
        if every is not None:
            status = client.set_monitor_schedule(
                _minutes(every, "--every"), _minutes(grace, "--grace")
            )
        else:
            status = client.monitor_status()
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    last = status.get("last")
    console.print(
        f"Monitor: {status['state']}"
        + (
            f" · last run {last['at'][:16].replace('T', ' ')} UTC ({last['verdict']})"
            if last
            else ""
        )
    )
    schedule = status.get("schedule")
    if schedule is None:
        console.print("No schedule declared; a monitor that stops running is not noticed.")
        return
    due = schedule["due_at"][:16].replace("T", " ")
    console.print(
        f"Expected every {schedule['every_minutes']} min (+{schedule['grace_minutes']} min "
        f"grace) · next due by {due} UTC"
    )
    if schedule["overdue"]:
        console.print(f"[red]Overdue[/]: no monitor run since it was due at {due} UTC.")


@cloud_app.command("exceptions")
def cloud_exceptions(
    all_: Annotated[bool, typer.Option("--all", help="Include revoked and expired ones.")] = False,
    project: Annotated[str | None, typer.Option(help="workspace/project.")] = None,
) -> None:
    """Audited exceptions of the project, newest first (active ones by default)."""
    from vectraudit.integrations.cloud import CloudClientError

    try:
        rows = _cloud_client(project).list_exceptions(active=not all_)
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    if not rows:
        console.print("No exceptions.")
        return
    table = Table("Rule", "Status", "Accepted by", "Owner", "Expires", "Uses", "Reason", "ID")
    for row in rows:
        uses = f"{row['uses']}/{row['max_checks']}" if row["max_checks"] else str(row["uses"])
        table.add_row(
            escape(row["rule"]),
            row["status"],
            escape(row["accepted_by"]),
            escape(row["owner"]),
            row["expires_at"][:10],
            uses,
            escape(row["reason"]),
            row["id"],
        )
    console.print(table)


@cloud_app.command("revoke-exception")
def cloud_revoke_exception(
    exception_id: Annotated[str, typer.Argument(help="Exception id from `cloud exceptions`.")],
    project: Annotated[str | None, typer.Option(help="workspace/project.")] = None,
) -> None:
    """Stop an exception covering future checks; earlier decisions keep their record."""
    from vectraudit.integrations.cloud import CloudClientError

    try:
        revoked = _cloud_client(project).revoke_exception(exception_id)
    except CloudClientError as e:
        raise _fail(e.message, e.hint) from e
    console.print(f"Revoked exception {escape(revoked['id'])} ({escape(revoked['rule'])}).")


def calibrate(
    runs: Annotated[
        list[Path],
        typer.Argument(help="Run files of the same, unchanged setup (record it 5+ times)."),
    ],
    policy: Annotated[Path, typer.Option(help="The policy to calibrate.")],
    margin: Annotated[
        float, typer.Option(min=1.0, max=5.0, help="Safety factor over the worst noise.")
    ] = 1.5,
    draft: Annotated[
        Path | None, typer.Option(help="Write the proposed policy to this new file.")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the full report to this new file.")
    ] = None,
) -> None:
    """A/A calibration: how often the policy fails an unchanged system, and a reviewed draft.

    Every run is compared with every other under the policy; since nothing changed, each
    failure is a false alarm. The draft only raises noise-driven thresholds and is never
    applied for you. Exits 1 on invalid input.
    """
    from vectraudit.evaluation.calibration import calibrate as run_calibration

    for target in (draft, json_out):
        if target is not None and target.exists():
            raise _fail(f"{target} exists; calibration never overwrites files.")
    try:
        raw = json.loads(policy.read_text(encoding="utf-8"))
        records = [load_run(path) for path in runs]
        report = run_calibration(records, raw, margin=margin)
    except (OSError, json.JSONDecodeError, RunFileError, PolicyError, ValueError) as e:
        raise _fail(str(e)) from e
    current, proposed = report["current"], report["draft"]
    console.print(
        f"{len(records)} runs, {current['comparisons']} A/A comparisons. "
        f"False alarms under the current policy: [bold]{current['false_alarms']}[/]"
        + (f"; under the draft: [bold]{proposed['false_alarms']}[/]." if report["changes"] else ".")
    )
    if current["by_rule"]:
        table = Table("Rule", "Comparisons failing now", "With draft")
        for rule, count in current["by_rule"].items():
            table.add_row(escape(rule), str(count), str(proposed["by_rule"].get(rule, 0)))
        console.print(table)
    if report["changes"]:
        table = Table("Policy setting", "Now", "Proposed", "Worst observed")
        for change in report["changes"]:
            table.add_row(
                escape(change["path"]),
                str(change["from"]),
                str(change["to"]),
                f"{change['observed']:.4g}",
            )
        console.print(table)
    else:
        console.print("No threshold needs to change for the observed noise.")
    for rule in report["not_noise"]:
        console.print(f"[bold red]Not noise:[/] {escape(rule)} fails on an unchanged system.")
    for note in report["notes"]:
        console.print(f"[yellow]Note:[/] {escape(note)}")
    if draft is not None and report["draft_policy"] is not None:
        draft.write_text(json.dumps(report["draft_policy"], indent=2) + "\n", encoding="utf-8")
        console.print(f"Draft policy written to {escape(str(draft))}. Review it before use.")
    if json_out is not None:
        json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
