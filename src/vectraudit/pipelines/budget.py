"""Calls and spend before running anything: candidates x questions x repeats.

A preview reads the manifest and dataset only; it opens no adapter and imports no code.
Cost per call comes from what an adapter reported in an earlier run of the same candidate
(``measured``), from ``cost_per_call_usd`` declared in the manifest (``declared``), or is
``unknown``; unknown costs are listed, never counted as zero. One production query is one
call, so the cost per call is also the per-production-query cost. Automatic retries can add
calls up to the run's retry budget; they are reported separately as an allowance.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from vectraudit.evaluation.dataset import Dataset
from vectraudit.pipelines.manifest import Manifest

Mode = Literal["run", "optimize", "monitor"]


def measured_cost(run: Mapping[str, Any]) -> dict[str, Any] | None:
    """Mean adapter-reported cost per call in a recorded run, with its coverage."""
    queries = [q for q in run.get("queries") or [] if q.get("error") is None]
    costs = [q["cost_usd"] for q in queries if q.get("cost_usd") is not None]
    if not costs:
        return None
    return {
        "usd": statistics.fmean(costs),
        "queries": len(costs),
        "of": len(queries),
        "run_id": run.get("run_id"),
    }


def _measured_by_candidate(runs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for run in runs:
        name = run.get("label") or (run.get("backend") or {}).get("config", {}).get("name")
        cost = measured_cost(run)
        if name and cost is not None:
            found[str(name)] = cost  # later files win: pass the newest last
    return found


def _money(calls: int, usd: float | None) -> float | None:
    return None if usd is None else round(calls * usd, 6)


def plan(
    spec: Manifest,
    dataset: Dataset,
    *,
    mode: Mode = "run",
    measured: Sequence[Mapping[str, Any]] = (),
    baseline: str | None = None,
    retry_budget: int = 100,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """How many pipeline calls ``mode`` makes, and what they cost where known.

    ``optimize`` runs every candidate on exploratory questions, then the chosen one (and,
    with a policy, the ``baseline``) on held-out questions; which one is chosen is not
    known in advance, so held-out calls and spend are given as a range.
    """
    if mode == "monitor" and len(spec.candidates) != 1:
        raise ValueError("monitor manifest must select exactly one candidate")
    observed = _measured_by_candidate(measured)
    repeats = spec.repeats
    explore = sum(1 for q in dataset.queries if q.split != "heldout")
    heldout = len(dataset.queries) - explore
    if mode == "optimize" and (explore == 0 or heldout == 0):
        raise ValueError("optimization requires both explore and heldout queries")
    rows = []
    for candidate in spec.candidates:
        cost = observed.get(candidate.name)
        if cost is not None and cost["queries"] == cost["of"]:
            usd, source = cost["usd"], "measured"
        elif candidate.cost_per_call_usd is not None:
            usd, source = candidate.cost_per_call_usd, "declared"
        elif candidate.adapter.kind == "hybrid":
            usd, source = 0.0, "built-in, runs locally"
        elif cost is not None:
            usd, source = cost["usd"], "measured (partial)"
        else:
            usd, source = None, "unknown"
        questions = explore if mode == "optimize" else len(dataset.queries)
        calls = questions * repeats
        rows.append(
            {
                "name": candidate.name,
                "questions": questions,
                "calls": calls,
                "cost_per_call_usd": usd,
                "cost_source": source,
                "measured": cost,
                "declared_usd": candidate.cost_per_call_usd,
                "spend_usd": _money(calls, usd),
            }
        )
    calls_min = calls_max = sum(r["calls"] for r in rows)
    known = [r["spend_usd"] for r in rows if r["spend_usd"] is not None]
    spend_min = spend_max = sum(known)
    heldout_info = None
    if mode == "optimize":
        per_run = heldout * repeats
        by_name = {r["name"]: r for r in rows}
        base = by_name.get(baseline) if baseline else None
        # Fewest: no candidate is feasible (nothing held out) ... most: a challenger and the
        # baseline both run on held-out questions.
        runs_max = 2 if base is not None and len(rows) > 1 else 1
        calls_max += per_run * runs_max
        prices = [r["cost_per_call_usd"] for r in rows]
        if all(p is not None for p in prices):
            dearest = max(p for p in prices if p is not None)
            extra = per_run * dearest
            if runs_max == 2 and base is not None:
                challengers = [r["cost_per_call_usd"] for r in rows if r is not base]
                extra = per_run * (max(challengers) + base["cost_per_call_usd"])
            spend_max += extra
        heldout_info = {"questions": heldout, "calls_per_run": per_run, "runs_up_to": runs_max}
    unknown = [r["name"] for r in rows if r["cost_per_call_usd"] is None]
    # The retry budget applies to each run (each candidate, and each held-out run).
    run_calls = [r["calls"] for r in rows]
    if heldout_info:
        run_calls += [heldout_info["calls_per_run"]] * heldout_info["runs_up_to"]
    retry_allowance = sum(min(retry_budget, n * (max_attempts - 1)) for n in run_calls)
    return {
        "format": "vectraudit.budget",
        "format_version": 1,
        "mode": mode,
        "dataset_sha256": dataset.sha256,
        "repeats": repeats,
        "candidates": rows,
        "heldout": heldout_info,
        "calls": {"min": calls_min, "max": calls_max, "retry_allowance": retry_allowance},
        "spend_usd": {
            "min": round(spend_min, 6),
            "max": round(spend_max, 6) if not (mode == "optimize" and unknown) else None,
            "complete": not unknown,
        },
        "unknown_cost": unknown,
        "notes": _notes(mode, unknown, rows, retry_allowance),
    }


def _notes(
    mode: Mode, unknown: list[str], rows: list[dict[str, Any]], retry_allowance: int
) -> list[str]:
    notes = []
    if unknown:
        notes.append(
            "Cost per call is unknown for "
            + ", ".join(unknown)
            + ": the spend shown excludes them. Pass --measured with an earlier run whose "
            "adapter reports cost_usd, or declare cost_per_call_usd in the manifest."
        )
    if any(r["cost_source"] == "measured (partial)" for r in rows):
        notes.append(
            "Some measured costs cover only part of the earlier run's questions; the average "
            "of the reported ones is used."
        )
    if retry_allowance:
        notes.append(
            f"Automatic retries can add up to {retry_allowance} calls (the retry budget of "
            "each run), not included above."
        )
    if mode == "optimize":
        notes.append(
            "Held-out calls run only for the chosen candidate (and the baseline under a "
            "policy), so the maximum assumes the most expensive choice."
        )
    notes.append(
        "Measured and declared costs cover pipeline calls only; model, database and hosting "
        "costs on the customer's side are not included unless the adapter reports them."
    )
    return notes


def check_limits(
    budget: Mapping[str, Any], *, max_calls: int | None = None, max_spend: float | None = None
) -> None:
    """Refuse a run whose planned calls or spend exceed the given limits."""
    calls = budget["calls"]["max"]
    if max_calls is not None and calls > max_calls:
        raise ValueError(
            f"Planned {calls} calls exceed --max-calls {max_calls}; "
            "reduce candidates, questions or repeats, or raise the limit."
        )
    if max_spend is not None:
        if budget["unknown_cost"]:
            raise ValueError(
                "--max-spend needs a known cost per call for every candidate; unknown for "
                + ", ".join(budget["unknown_cost"])
                + ". Declare cost_per_call_usd or pass --measured."
            )
        spend = budget["spend_usd"]["max"]
        if spend is not None and spend > max_spend + 1e-12:
            raise ValueError(
                f"Planned spend up to ${spend:.4f} exceeds --max-spend ${max_spend:.4f}."
            )


def format_usd(value: float | None) -> str:
    """``$12.50``; below a dollar, two significant figures (``$0.0042``)."""
    if value is None:
        return "unknown"
    if value == 0:
        return "$0"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    decimals = min(6, max(2, 1 - math.floor(math.log10(abs(value)))))
    return f"${value:.{decimals}f}"
