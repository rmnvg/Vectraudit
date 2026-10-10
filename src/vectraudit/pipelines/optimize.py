"""Bounded measured candidate search with a single held-out validation.

With a release policy and a baseline candidate, candidates that fail the policy against the
baseline on exploratory questions are excluded, and the chosen candidate is approved only if
the full policy passes against the baseline on held-out questions. Without a policy the
result is labelled exploratory: nothing here presents a candidate as release-approved
unless that held-out policy check passed.
"""

from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Any

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import Dataset, parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.exceptions import failing_rules
from vectraudit.evaluation.policy import Policy, policy_sha256
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
    policy: Policy | None = None,
    baseline: str | None = None,
) -> dict[str, Any]:
    """Rank exploratory candidates; validate only the chosen one on unseen held-out rows.

    Memory and monetary cost are adapter-reported and remain unknown unless supplied.
    Unknown measurements cannot satisfy a requested constraint. Nothing is deployed.
    """
    if (policy is None) != (baseline is None):
        raise ValueError("A release policy needs a baseline candidate, and a baseline a policy")
    if baseline is not None and baseline not in {name for name, _, _ in candidates}:
        raise ValueError(f"baseline {baseline!r} is not one of the candidates")
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
    if policy is not None:
        _screen(reports, baseline, _restrict(policy, {q.query_id for q in tuning.queries}))
    # With a baseline, a candidate must beat it on quality: on a tie the current configuration
    # stays, so latency jitter alone never recommends a change.
    feasible = sorted(
        [r for r in reports if r["feasible"]],
        key=lambda r: (-r["ndcg"], r["name"] != baseline, r["p95_ms"], r["name"]),
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
            "check": None,
        }
        if policy is not None and chosen != baseline:
            # The full release decision: the chosen candidate against the current baseline on
            # held-out questions neither was tuned on, under the release policy.
            _, base_pipeline, base_params = next(c for c in candidates if c[0] == baseline)
            base_run = execute_pipeline(
                validation,
                base_pipeline,
                PipelineOptions(k=k, params=base_params, warmup=0, repeats=repeats),
                config={"candidate": baseline, "params": base_params},
                corpus_version=corpus_version,
            )
            validation_report["baseline_run"] = base_run
            validation_report["check"] = _brief(
                compare_runs(
                    base_run, run, _restrict(policy, {q.query_id for q in validation.queries})
                )
            )
    eligible = bool(validation_report and validation_report["feasible"])
    status = _status(policy, baseline, chosen, eligible, validation_report)
    return {
        "format": "vectraudit.optimization",
        "format_version": 2,
        "status": status,
        "status_label": STATUS_LABELS[status],
        "approved": status == "approved_under_policy",
        "policy": None
        if policy is None
        else {
            "sha256": policy_sha256(policy),
            "baseline": baseline,
            "critical_queries": {
                "screened_on_exploratory": sorted(
                    r.query_id
                    for r in policy.critical_queries
                    if r.query_id in {q.query_id for q in tuning.queries}
                ),
                "checked_on_heldout": sorted(
                    r.query_id
                    for r in policy.critical_queries
                    if r.query_id in {q.query_id for q in validation.queries}
                ),
            },
        },
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
        "eligible": eligible,
        "measurement_note": "p95 across per-query median latencies; cost/memory adapter-reported",
    }


STATUS_LABELS = {
    "exploratory_only": "Best exploratory candidate; not checked under a release policy",
    "no_feasible_candidate": "No candidate met the constraints and the release policy",
    "baseline_remains_best": "The baseline remains the best candidate; nothing to release",
    "approved_under_policy": "Approved under the release policy on held-out questions",
    "not_approved": "Best exploratory candidate; not approved under the release policy",
}


def _status(
    policy: Policy | None,
    baseline: str | None,
    chosen: str | None,
    eligible: bool,
    validation: dict[str, Any] | None,
) -> str:
    if policy is None:
        return "exploratory_only"
    if chosen is None:
        return "no_feasible_candidate"
    if chosen == baseline:
        return "baseline_remains_best"
    check = (validation or {}).get("check") or {}
    if eligible and check.get("verdict") == "pass":
        return "approved_under_policy"
    return "not_approved"


def _restrict(policy: Policy, query_ids: set[str]) -> Policy:
    """Critical rules apply in the split that holds their question."""
    return replace(
        policy,
        critical_queries=tuple(r for r in policy.critical_queries if r.query_id in query_ids),
    )


def _brief(result: dict[str, Any]) -> dict[str, Any]:
    failing = [r for r in result["reasons"] if r["verdict"] == "fail"]
    return {
        "verdict": result["verdict"],
        "failing_rules": failing_rules(result),
        "reasons": [r["message"] for r in result["reasons"]][:10],
        # Under a non-inferiority policy, "not shown within tolerance" is a lack of evidence,
        # not a measured regression.
        "inconclusive": bool(failing) and all(r.get("inconclusive") for r in failing),
    }


def _screen(reports: list[dict[str, Any]], baseline: str | None, policy: Policy) -> None:
    """Exclude candidates that the release policy fails against the baseline on explore data.

    An inconclusive screen (for example too few exploratory questions for the policy's
    minimum) is recorded but does not exclude: the held-out check still decides approval.
    """
    base = next(r for r in reports if r["name"] == baseline)
    for report in reports:
        if report is base:
            report["policy_screen"] = None
            continue
        screen = _brief(compare_runs(base["run"], report["run"], policy))
        report["policy_screen"] = screen
        if screen["verdict"] == "fail" and not screen["inconclusive"]:
            report["feasible"] = False
            report["reasons"].append(
                "fails the release policy on exploratory questions: "
                + ", ".join(screen["failing_rules"])
            )


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
