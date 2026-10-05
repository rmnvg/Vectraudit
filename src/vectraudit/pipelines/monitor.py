"""Frozen-benchmark checks, durable local history, and machine-readable alerts."""

from __future__ import annotations

from typing import Any

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy


def monitor_check(
    baseline: dict[str, Any], candidate: dict[str, Any], policy: Policy = DEFAULT_POLICY
) -> dict[str, Any]:
    """Classify observed changes independently from their cause; never roll the baseline."""
    check = compare_runs(baseline, candidate, policy)
    old = baseline.get("backend") or {}
    new = candidate.get("backend") or {}
    corpus_old, corpus_new = old.get("corpus_version"), new.get("corpus_version")
    config_old, config_new = old.get("config"), new.get("config")
    # Corpus content is its own drift dimension, not a retrieval configuration change.
    if config_old is not None:
        config_old = {k: v for k, v in config_old.items() if k != "corpus_sha256"}
    if config_new is not None:
        config_new = {k: v for k, v in config_new.items() if k != "corpus_sha256"}
    before = {q["query_id"]: q for q in baseline["queries"]}
    changes = []
    for query in candidate["queries"]:
        previous = before.get(query["query_id"])
        if previous is not None and query.get("ids") != previous.get("ids"):
            changes.append(
                {
                    "query_id": query["query_id"],
                    "removed": sorted(set(previous.get("ids") or []) - set(query.get("ids") or [])),
                    "added": sorted(set(query.get("ids") or []) - set(previous.get("ids") or [])),
                }
            )
    alerts = [reason for reason in check["reasons"] if reason["verdict"] != "pass"]
    return {
        "format": "vectraudit.monitor",
        "format_version": 1,
        "created_at": candidate["created_at"],
        "run_id": candidate["run_id"],
        "baseline_id": baseline["run_id"],
        "dataset_sha256": candidate["dataset"]["sha256"],
        "verdict": check["verdict"],
        "alerts": alerts,
        "check": check,
        "drift": {
            "corpus": "unknown"
            if corpus_old is None or corpus_new is None
            else "changed"
            if corpus_old != corpus_new
            else "unchanged",
            "configuration": "unknown"
            if config_old is None or config_new is None
            else "changed"
            if config_old != config_new
            else "unchanged",
            "results": changes,
            "coverage_note": (
                "Result coverage only; corpus inventory requires an adapter-supplied version."
            ),
        },
        "summary": candidate["summary"],
    }
