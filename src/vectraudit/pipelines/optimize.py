"""Bounded measured candidate search with a single held-out validation."""

from __future__ import annotations

import statistics
from typing import Any

from vectraudit.evaluation.dataset import Dataset, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline
from vectraudit.pipelines.runtime import Pipeline


def optimize(
    dataset: Dataset,
    candidates: list[tuple[str, Pipeline, dict[str, Any]]],
    *,
    k: int = 10,
    max_candidates: int = 20,
    max_p95_ms: float = 150,
    max_cost_usd: float | None = None,
    max_memory_bytes: int | None = None,
    corpus_version: str | None = None,
    repeats: int = 3,
) -> dict[str, Any]:
    """Rank exploratory candidates; validate only the chosen one on unseen held-out rows.

    Memory and monetary cost are adapter-reported and remain unknown unless supplied.
    Unknown measurements cannot satisfy a requested constraint. Nothing is deployed.
    """
    if not 1 <= len(candidates) <= max_candidates <= 100:
        raise ValueError("candidate count must fit the explicit budget (maximum 100)")
    if len({name for name, _, _ in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    if not 0 < max_p95_ms < float("inf"):
        raise ValueError("max_p95_ms must be finite and positive")
    if max_cost_usd is not None and not 0 <= max_cost_usd < float("inf"):
        raise ValueError("max_cost_usd must be finite and non-negative")
    if max_memory_bytes is not None and max_memory_bytes < 0:
        raise ValueError("max_memory_bytes must be non-negative")
    rows = dataset_rows(dataset)
    explore = [r for r in rows if r.get("split", "explore") == "explore"]
    heldout = [r for r in rows if r.get("split") == "heldout"]
    if not explore or not heldout:
        raise ValueError("optimization requires both explore and heldout queries")

    # Query ids alone do not protect against reusing the same question under a new id.
    def identities(rows: list[dict[str, Any]]) -> set[str]:
        import json

        found = set()
        for row in rows:
            if row.get("text"):
                found.add("text:" + " ".join(row["text"].casefold().split()))
            elif row.get("vector"):
                found.add("vector:" + json.dumps(row["vector"]))
        return found

    if identities(explore) & identities(heldout):
        raise ValueError("duplicate questions cross explore/heldout boundary")
    tuning = parse_dataset(render_dataset(explore, dataset.meta))
    validation = parse_dataset(render_dataset(heldout, dataset.meta))
    reports = []
    for name, pipeline, params in candidates:
        run = execute_pipeline(
            tuning,
            pipeline,
            PipelineOptions(k=k, params=params, warmup=0, repeats=repeats),
            config={"candidate": name, "params": params},
            corpus_version=corpus_version,
        )
        reports.append(
            {"name": name, "run": run, **_measure(run, max_p95_ms, max_cost_usd, max_memory_bytes)}
        )
    feasible = sorted(
        [r for r in reports if r["feasible"]], key=lambda r: (-r["ndcg"], r["p95_ms"], r["name"])
    )
    chosen = feasible[0]["name"] if feasible else None
    validation_report = None
    if chosen is not None:
        _, pipeline, params = next(c for c in candidates if c[0] == chosen)
        run = execute_pipeline(
            validation,
            pipeline,
            PipelineOptions(k=k, params=params, warmup=0, repeats=repeats),
            config={"candidate": chosen, "params": params},
            corpus_version=corpus_version,
        )
        validation_report = {
            "run": run,
            **_measure(run, max_p95_ms, max_cost_usd, max_memory_bytes),
        }
    return {
        "format": "vectraudit.optimization",
        "format_version": 1,
        "dataset_sha256": dataset.sha256,
        "objective": "maximize exploratory nDCG subject to measured constraints",
        "constraints": {
            "max_p95_ms": max_p95_ms,
            "max_cost_usd": max_cost_usd,
            "max_memory_bytes": max_memory_bytes,
        },
        "candidates": reports,
        "selected": chosen,
        "heldout": validation_report,
        "requires_review": True,
        "sample_counts": {"explore": len(tuning), "heldout": len(validation)},
        "warning": "Small benchmarks provide indicative evidence, not production guarantees.",
        "eligible": bool(validation_report and validation_report["feasible"]),
        "measurement_note": "p95 across per-query median latencies; cost/memory adapter-reported",
    }


def _measure(
    run: dict[str, Any], latency: float, cost: float | None, memory: int | None
) -> dict[str, Any]:
    summary = run["summary"]
    metrics = summary["metrics"]
    p95 = (summary["latency_ms"] or {}).get("p95")
    costs = [q.get("cost_usd") for q in run["queries"]]
    memories = [q.get("memory_bytes") for q in run["queries"]]
    measured_cost = statistics.fmean(costs) if costs and all(c is not None for c in costs) else None
    measured_memory = max(memories) if memories and all(m is not None for m in memories) else None
    reasons = []
    if run["status"] != "complete":
        reasons.append("incomplete run")
    if metrics is None:
        reasons.append("nDCG not measured")
    if p95 is None or p95 > latency:
        reasons.append("latency constraint unmet")
    if cost is not None and (measured_cost is None or measured_cost > cost):
        reasons.append("cost constraint unmet or unmeasured")
    if memory is not None and (measured_memory is None or measured_memory > memory):
        reasons.append("memory constraint unmet or unmeasured")
    if summary["permissions"]["violations"] or any(
        q.get("no_answer_correct") is False for q in run["queries"]
    ):
        reasons.append("safety assertion failed")
    return {
        "ndcg": metrics["ndcg"] if metrics else None,
        "p95_ms": p95,
        "cost_usd": measured_cost,
        "memory_bytes": measured_memory,
        "feasible": not reasons,
        "reasons": reasons,
    }
