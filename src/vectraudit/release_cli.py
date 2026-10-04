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

EXIT_CODES = {Verdict.PASS: 0, Verdict.ERROR: 1, Verdict.FAIL: 3, Verdict.NOT_COMPARABLE: 4}
_VERDICT_STYLE = {
    Verdict.PASS: "bold green",
    Verdict.FAIL: "bold red",
    Verdict.ERROR: "bold yellow",
    Verdict.NOT_COMPARABLE: "bold magenta",
}

console = Console()
err_console = Console(stderr=True)

dataset_app = typer.Typer(help="Work with evaluation datasets (JSONL).", no_args_is_help=True)


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
    backend: Annotated[str, typer.Option(help="faiss, qdrant or pgvector.")] = "faiss",
    directory: Annotated[
        Path, typer.Option("--dir", help="Repository root to scaffold into.")
    ] = Path(),
    force: Annotated[bool, typer.Option(help="Overwrite files that already exist.")] = False,
) -> None:
    """Scaffold retrieval/ (connection, policy, queries) and a GitHub workflow."""
    from vectraudit.scaffold import BACKENDS, init_project

    if backend not in BACKENDS:
        raise _fail(f"Unknown backend {backend!r}.", f"Use one of: {', '.join(BACKENDS)}.")
    written, skipped = init_project(directory, backend, force=force)
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
    source: Annotated[Path, typer.Argument(help="CSV (with a header row) or JSONL file.")],
    out: Annotated[Path, typer.Option(help="Dataset JSONL to write.")],
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
    """Convert a CSV or JSONL file of queries and labels into a dataset."""
    from vectraudit.evaluation.dataset_tools import (
        ColumnMap,
        import_rows,
        read_records,
        render_dataset,
    )

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
    try:
        rows = import_rows(read_records(source), columns)
        text = render_dataset(rows, {"name": name} if name else None)
    except DatasetError as e:
        raise _fail(str(e)) from e
    _write_text(out, text)
    ds = load_dataset(out)
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
    if run["status"] == "cancelled":
        raise typer.Exit(code=130)
    if run["status"] != "complete":
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
) -> None:
    """Compare a candidate run with a baseline and decide.

    Exit codes: 0 pass, 1 error (a run is incomplete or a file is invalid), 3 fail
    (a policy rule is broken), 4 not comparable (different data or missing metrics).
    """
    try:
        pol = DEFAULT_POLICY if policy is None else load_policy(policy)
        base, cand = load_run(baseline), load_run(candidate)
    except (PolicyError, RunFileError) as e:
        raise _fail(str(e)) from e
    result = compare_runs(base, cand, pol)
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
    raise typer.Exit(code=EXIT_CODES[verdict])


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


def _print_check(result: dict[str, Any], verdict: Verdict) -> None:
    console.print(
        f"[{_VERDICT_STYLE[verdict]}]{verdict.value.replace('_', ' ').upper()}[/] · "
        f"{result['n_queries']} paired queries"
    )
    for r in result["reasons"]:
        console.print(f"  • {escape(r['message'])}")
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
