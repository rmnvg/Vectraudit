"""Calls and spend before running anything: candidates x questions x repeats.

A preview reads the manifest and dataset only; it opens no adapter and imports no code.
Cost per call comes from what an adapter reported in an earlier run of the same candidate
(``measured``), from ``cost_per_call_usd`` declared in the manifest (``declared``), or is
``unknown``; unknown costs are listed, never counted as zero. One production query is one
call, so the cost per call is also the per-production-query cost. Automatic retries can add
calls up to the run's retry budget; they are reported separately as an allowance.

After a run, :func:`usage` totals what it actually consumed (every call, including retries
and failures, and the adapter-reported cost of successful calls) and :func:`reconcile`
sets that beside the plan. Calls with no reported cost, including failed calls, stay
unknown rather than free.
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


def usage(queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What a run actually consumed, from each query's recorded ``usage``.

    A resumed query keeps every earlier attempt in ``attempts``; their calls count too. An
    interrupted attempt recorded no usage, and neither did runs made before usage was
    recorded: both are counted, so the totals say when they are incomplete.
    """
    calls = failed = unpriced = retries = interrupted = unrecorded = 0
    spend: list[float] = []
    for query in queries:
        attempts = query.get("attempts")
        results = [a.get("result") for a in attempts] if attempts else [query]
        for result in results:
            if result is None:
                interrupted += 1
                continue
            used = result.get("usage")
            if not isinstance(used, Mapping):
                unrecorded += 1
                continue
            calls += int(used.get("calls", 0))
            failed += int(used.get("failed_calls", 0))
            unpriced += int(used.get("unpriced_calls", 0))
            spend.append(float(used.get("spend_usd", 0.0)))
            retries += len(result.get("retries") or [])
    return {
        "format_version": 1,
        "calls": calls,
        "failed_calls": failed,
        "retries": retries,
        "spend_usd": round(math.fsum(spend), 6),
        "unpriced_calls": unpriced,
        "interrupted_attempts": interrupted,
        "unrecorded_queries": unrecorded,
        "complete": not (unpriced or interrupted or unrecorded),
    }


def combine(usages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals of several runs' :func:`usage` (an optimization's tuning and held-out runs)."""
    total: dict[str, Any] = {"format_version": 1}
    for key in (
        "calls",
        "failed_calls",
        "retries",
        "unpriced_calls",
        "interrupted_attempts",
        "unrecorded_queries",
    ):
        total[key] = sum(int(u.get(key, 0)) for u in usages)
    total["spend_usd"] = round(math.fsum(float(u.get("spend_usd", 0.0)) for u in usages), 6)
    total["complete"] = all(u.get("complete", False) for u in usages)
    return total


def reconcile(planned: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    """The plan beside what was used, for one run or a whole batch.

    ``planned`` has ``calls`` (a number, or ``{"min", "max"}``), ``retry_allowance`` and
    ``spend_usd`` (a number, ``{"min", "max"}`` or ``None`` when a cost was unknown).
    """

    def bounds(value: Any) -> tuple[float | None, float | None]:
        if isinstance(value, Mapping):
            return value.get("min"), value.get("max")
        return value, value

    calls_min, calls_max = bounds(planned.get("calls"))
    _, spend_max = bounds(planned.get("spend_usd"))
    calls = actual["calls"]
    over_calls = calls_max is not None and calls > calls_max
    within_allowance = over_calls and calls <= (calls_max or 0) + int(
        planned.get("retry_allowance") or 0
    )
    spend = actual["spend_usd"]
    notes = []
    if actual["retries"]:
        notes.append(f"{actual['retries']} automatic retries")
    if actual["failed_calls"]:
        notes.append(f"{actual['failed_calls']} failed calls (cost unknown)")
    priced_unknown = actual["unpriced_calls"] - actual["failed_calls"]
    if priced_unknown > 0:
        notes.append(f"{priced_unknown} successful calls reported no cost")
    if actual["interrupted_attempts"] or actual["unrecorded_queries"]:
        notes.append(
            "calls from interrupted attempts or earlier versions were not recorded; the "
            "totals are a lower bound"
        )
    if calls_min is not None and calls < calls_min:
        notes.append("fewer calls than planned (cancelled, or failed before every repeat)")
    return {
        "planned": {
            "calls": planned.get("calls"),
            "retry_allowance": planned.get("retry_allowance"),
            "spend_usd": planned.get("spend_usd"),
        },
        "actual": dict(actual),
        "calls_over_plan": max(0, calls - int(calls_max)) if calls_max is not None else None,
        "calls_within_retry_allowance": (not over_calls) or within_allowance,
        "spend_over_plan_usd": (
            round(spend - spend_max, 6)
            if spend_max is not None and spend > spend_max
            else (0.0 if spend_max is not None else None)
        ),
        "spend_complete": actual["complete"],
        "notes": notes,
    }


def planned_for(
    budget: Mapping[str, Any], name: str, *, retry_budget: int = 100, max_attempts: int = 3
) -> dict[str, Any]:
    """One candidate's planned calls, retry allowance and spend, from :func:`plan`."""
    row = next(r for r in budget["candidates"] if r["name"] == name)
    return {
        "calls": row["calls"],
        "retry_allowance": min(retry_budget, row["calls"] * (max_attempts - 1)),
        "spend_usd": row["spend_usd"],
        "cost_source": row["cost_source"],
    }


def planned_total(budget: Mapping[str, Any]) -> dict[str, Any]:
    """The whole plan's calls and spend; spend has no upper bound when a cost is unknown."""
    spend = budget["spend_usd"]
    return {
        "calls": {"min": budget["calls"]["min"], "max": budget["calls"]["max"]},
        "retry_allowance": budget["calls"]["retry_allowance"],
        "spend_usd": {"min": spend["min"], "max": spend["max"] if spend["complete"] else None},
    }


def reconcile_line(rec: Mapping[str, Any]) -> str:
    """``120 calls (planned 120) · $0.0240 reported (planned $0.0240)`` and any caveats."""
    planned = rec["planned"]
    actual = rec["actual"]

    def amount(value: Any) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, Mapping):
            low, high = value.get("min"), value.get("max")
            if high is None:
                return f"{format_usd(low)} or more" if low is not None else "unknown"
            return format_usd(low) if low == high else f"{format_usd(low)}-{format_usd(high)}"
        return format_usd(value)

    calls = planned["calls"]
    planned_calls = (
        (str(calls["min"]) if calls["min"] == calls["max"] else f"{calls['min']}-{calls['max']}")
        if isinstance(calls, Mapping)
        else str(calls)
    )
    spend = format_usd(actual["spend_usd"]) + (
        " reported" if actual["complete"] else " reported, incomplete"
    )
    line = (
        f"{actual['calls']} calls (planned {planned_calls}) · {spend} "
        f"(planned {amount(planned['spend_usd'])})"
    )
    if rec["notes"]:
        line += "; " + "; ".join(rec["notes"])
    return line


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
