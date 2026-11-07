"""``sweep --filter`` and ``--pass-rates``: parsing, JSON and the table of the result."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from vectraudit.core.advice import Suggestion
from vectraudit.core.filtering import FilterError
from vectraudit.core.selectivity import (
    FilterSpec,
    SelectivityResult,
    metadata_spec,
    synthetic_spec,
)

FORMAT = "vectraudit.selectivity-sweep"
FORMAT_VERSION = 1


def parse_filter(text: str) -> FilterSpec:
    """``lang=de`` or ``lang=de;year={"gte":2020}``: conditions joined by ``;`` are ANDed.

    A value that is valid JSON is read as JSON (numbers, lists, ranges); anything else is a
    string.
    """
    conditions: dict[str, Any] = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        key, sep, raw = part.partition("=")
        if not sep or not key.strip():
            raise FilterError(
                f"Cannot read the filter {part!r}.",
                'Write conditions as key=value, e.g. "lang=de".',
            )
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        conditions[key.strip()] = value
    return metadata_spec(conditions)


def parse_pass_rates(text: str) -> list[FilterSpec]:
    """``0.001,0.01,0.1`` -> synthetic id filters. A pass rate is the fraction that passes."""
    try:
        rates = [float(v) for v in text.replace(" ", "").split(",") if v]
    except ValueError:
        raise FilterError(
            f"Cannot read the pass rates {text!r}.", "Use comma-separated numbers, e.g. 0.01,0.1."
        ) from None
    if not rates:
        raise FilterError("Give at least one pass rate.", "e.g. --pass-rates 0.01,0.1,0.5")
    return [synthetic_spec(r) for r in rates]


def result_json(
    result: SelectivityResult,
    target: float,
    suggestions: list[Suggestion],
    *,
    truth_reconstructed: bool,
) -> dict[str, Any]:
    """The sweep as a JSON document: one entry per filter, no document text."""
    runs = []
    for run, choice in result.choose(target):
        sw = run.sweep
        runs.append(
            {
                "filter": run.spec.as_dict(),
                "pass_rate": run.pass_rate,
                "allowed_count": run.allowed_count,
                "status": choice.status,
                "recommended": choice.by_recall.value if choice.by_recall else None,
                "points": [
                    {
                        "value": p.value,
                        "recall": p.recall,
                        "recall_ci_low": p.recall_ci_low,
                        "recall_ci_high": p.recall_ci_high,
                        "latency_mean_ms": p.latency_mean_ms,
                        "latency_p95_ms": p.latency_p95_ms,
                        "probe_coverage": p.probe_coverage,
                    }
                    for p in sw.points
                ],
                "n_queries": sw.n_queries,
                "query_sha256": sw.query_sha256,
            }
        )
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "param": result.param.value,
        "k": result.k,
        "target": target,
        "truth_source": "reconstructed" if truth_reconstructed else "raw",
        "pass_rate_means": "fraction of the corpus that satisfies the filter",
        "environment": result.baseline.sweep.environment,
        "runs": runs,
        "suggestions": [{**asdict(s), "kind": s.kind.value} for s in suggestions],
    }


def print_table(console: Console, result: SelectivityResult, target: float) -> None:
    """Recall by parameter value, one row per pass rate (most permissive first)."""
    name = result.param.value
    values = [p.value for p in result.baseline.sweep.points]
    console.print(
        f"[bold]{name} sweep by pass rate[/] · recall@{result.k} against the exact neighbours "
        f"among the documents each filter allows · columns are {name} values"
    )
    table = Table()
    table.add_column("filter")
    table.add_column("pass", justify="right")
    table.add_column("docs", justify="right")
    for v in values:
        table.add_column(str(v), justify="right")
    table.add_column(f"smallest {name} for {target:g}", justify="right")
    for run, choice in result.choose(target):
        by_value = {p.value: p for p in run.sweep.points}
        cells = []
        for v in values:
            p = by_value.get(v)
            cells.append("" if p is None else f"{p.recall:.3f}")
        rec = choice.by_recall
        note = str(rec.value) if rec else "[red]not reached[/]"
        table.add_row(
            escape(run.spec.label),
            f"{run.pass_rate:.3%}" if run.pass_rate < 0.01 else f"{run.pass_rate:.1%}",
            f"{run.allowed_count:,}",
            *cells,
            note,
        )
    console.print(table)
