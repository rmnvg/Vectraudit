"""A/A calibration: how often would a policy fail an unchanged system, and what would not?

Record the same, unchanged retrieval setup several times on the runner and endpoint that CI
uses, then compare every run with every other under the current policy. Each comparison is
between identical systems, so every failure is a false alarm, named by its rule key.

From the observed noise this module drafts a policy that only *raises* noise-driven
thresholds (metric and cohort ``max_drop``, ``latency.min_increase_ms`` and
``max_regressed_queries``) to the worst observed variation times a safety margin. It never
loosens a requirement: an unchanged system that misses a critical question, delivers a
forbidden document, fails a no-answer assertion or breaks an absolute floor (``min``,
``p95_max_ms``) has a real problem, reported as such. The draft is a proposal for review;
nothing is applied automatically.

Few runs underestimate rare noise, and comparisons between the same runs are not
independent. The report states how many runs and comparisons it rests on and makes no claim
beyond them.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.exceptions import failing_rules, rule_kind
from vectraudit.evaluation.policy import parse_policy, policy_sha256

FORMAT = "vectraudit.calibration"
MIN_RUNS = 2
RECOMMENDED_RUNS = 5
METRIC_STEP = 0.005
MS_STEP = 0.5


def _round_up(value: float, step: float) -> float:
    rounded = math.ceil(value / step - 1e-9) * step
    return round(rounded, 6)


def _check_inputs(runs: Sequence[Mapping[str, Any]]) -> None:
    if len(runs) < MIN_RUNS:
        raise ValueError(f"Calibration needs at least {MIN_RUNS} runs of the unchanged system.")
    first = runs[0]
    for run in runs:
        if run.get("status") != "complete":
            raise ValueError(f"Run {run.get('run_id')} is {run.get('status')}; use complete runs.")
        if run["dataset"]["sha256"] != first["dataset"]["sha256"]:
            raise ValueError("All runs must use the same dataset version.")
        if run["search"]["k"] != first["search"]["k"]:
            raise ValueError("All runs must use the same k.")
    if len({run.get("run_id") for run in runs}) != len(runs):
        raise ValueError("Each run must be a separate recording (run IDs repeat).")


def _comparisons(
    runs: Sequence[Mapping[str, Any]], raw_policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    policy = parse_policy(raw_policy)
    return [
        compare_runs(runs[i], runs[j], policy)
        for i in range(len(runs))
        for j in range(len(runs))
        if i != j
    ]


def _alarms(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_rule: Counter[str] = Counter()
    for result in results:
        by_rule.update(set(failing_rules(result)))
    verdicts = Counter(result["verdict"] for result in results)
    return {
        "comparisons": len(results),
        "false_alarms": sum(result["verdict"] != "pass" for result in results),
        "verdicts": dict(sorted(verdicts.items())),
        "by_rule": dict(sorted(by_rule.items())),
    }


def _noise(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[float]] = {}
    cohorts: dict[str, dict[str, list[float]]] = {}
    rises_ms: list[float] = []
    rises_pct: list[float] = []
    regressed: list[int] = []
    for result in results:
        for name, stats in (result.get("metrics") or {}).items():
            metrics.setdefault(name, []).append(-stats["delta"])
        for tag, cohort in (result.get("cohorts") or {}).items():
            for name, stats in (cohort.get("metrics") or {}).items():
                cohorts.setdefault(tag, {}).setdefault(name, []).append(-stats["delta"])
        latency = result.get("latency")
        if latency and latency.get("compared"):
            rises_ms.append(latency["candidate"]["p95"] - latency["baseline"]["p95"])
            if latency.get("p95_change_pct") is not None:
                rises_pct.append(latency["p95_change_pct"])
        queries = result.get("queries")
        if queries:
            regressed.append(queries["regressed_count"])
    return {
        "metrics": {
            name: {"worst_drop": max(drops), "mean_abs_change": sum(map(abs, drops)) / len(drops)}
            for name, drops in sorted(metrics.items())
        },
        "cohorts": {
            tag: {name: {"worst_drop": max(drops)} for name, drops in sorted(values.items())}
            for tag, values in sorted(cohorts.items())
        },
        "latency": {
            "worst_p95_rise_ms": max(rises_ms) if rises_ms else None,
            "worst_p95_rise_pct": max(rises_pct) if rises_pct else None,
        },
        "worst_regressed_queries": max(regressed) if regressed else None,
    }


def _draft(
    raw: Mapping[str, Any], noise: Mapping[str, Any], margin: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    draft = copy.deepcopy(dict(raw))
    changes: list[dict[str, Any]] = []

    def raise_to(
        path: str,
        current: float | None,
        observed: float,
        step: float,
        target: dict[str, Any],
        key: str,
    ) -> None:
        proposed = _round_up(observed * margin, step)
        if observed > 0 and (current is None or proposed > current):
            target[key] = proposed
            changes.append({"path": path, "from": current, "to": proposed, "observed": observed})

    for name, rule in (draft.get("metrics") or {}).items():
        observed = noise["metrics"].get(name, {}).get("worst_drop")
        if observed is not None and "max_drop" in rule:
            raise_to(
                f"metrics.{name}.max_drop",
                rule["max_drop"],
                observed,
                METRIC_STEP,
                rule,
                "max_drop",
            )
    for name, rule in ((draft.get("cohorts") or {}).get("metrics") or {}).items():
        worst = [
            values[name]["worst_drop"] for values in noise["cohorts"].values() if name in values
        ]
        if worst and "max_drop" in rule:
            raise_to(
                f"cohorts.metrics.{name}.max_drop",
                rule["max_drop"],
                max(worst),
                METRIC_STEP,
                rule,
                "max_drop",
            )
    latency = draft.get("latency")
    rise = noise["latency"]["worst_p95_rise_ms"]
    if (
        latency
        and rise is not None
        and ("p95_max_increase_pct" in latency or "p50_max_increase_pct" in latency)
    ):
        raise_to(
            "latency.min_increase_ms",
            latency.get("min_increase_ms", 2.0),
            rise,
            MS_STEP,
            latency,
            "min_increase_ms",
        )
    worst_regressed = noise["worst_regressed_queries"]
    if draft.get("max_regressed_queries") is not None and worst_regressed:
        proposed = math.ceil(worst_regressed * margin)
        if proposed > draft["max_regressed_queries"]:
            changes.append(
                {
                    "path": "max_regressed_queries",
                    "from": draft["max_regressed_queries"],
                    "to": proposed,
                    "observed": worst_regressed,
                }
            )
            draft["max_regressed_queries"] = proposed
    return draft, changes


def calibrate(
    runs: Sequence[Mapping[str, Any]], raw_policy: Mapping[str, Any], *, margin: float = 1.5
) -> dict[str, Any]:
    """Measure false alarms of ``raw_policy`` on runs of an unchanged system; draft a policy.

    ``runs`` are validated run records of the *same, unchanged* setup. Returns a report with
    the current false alarms by rule, observed noise, the proposed changes and draft policy
    (``None`` if nothing needs to change), the draft's own false alarms, and failures that
    no threshold should absorb.
    """
    if not 1.0 <= margin <= 5.0:
        raise ValueError("margin must be between 1 and 5")
    _check_inputs(runs)
    parse_policy(raw_policy)  # fail early on an invalid policy
    current = _comparisons(runs, raw_policy)
    noise = _noise(current)
    draft, changes = _draft(raw_policy, noise, margin)
    current_alarms = _alarms(current)
    real = sorted(
        rule
        for rule in current_alarms["by_rule"]
        if rule_kind(rule) in ("critical", "safety")
        or rule.endswith(".min")
        or rule == "latency.p95_max_ms"
    )
    notes = []
    if len(runs) < RECOMMENDED_RUNS:
        notes.append(
            f"Only {len(runs)} runs: rare noise is likely underestimated. Record at least "
            f"{RECOMMENDED_RUNS} on the same runner and endpoint that CI uses."
        )
    notes.append(
        "Comparisons reuse the same runs, so they are not independent; a low false-alarm count "
        "is evidence for this setup, not a guarantee. Re-run after infrastructure changes."
    )
    if real:
        notes.append(
            "Some failures happen on an unchanged system and are not noise to absorb: an "
            "intermittent critical question, permission or no-answer failure, or a floor the "
            "system already misses. Investigate these instead of loosening the policy."
        )
    return {
        "format": FORMAT,
        "format_version": 1,
        "runs": [
            {
                "run_id": run.get("run_id"),
                "label": run.get("label"),
                "created_at": run.get("created_at"),
            }
            for run in runs
        ],
        "dataset_sha256": runs[0]["dataset"]["sha256"],
        "margin": margin,
        "policy_sha256": policy_sha256(parse_policy(raw_policy)),
        "current": current_alarms,
        "noise": noise,
        "changes": changes,
        "draft_policy": draft if changes else None,
        "draft": _alarms(_comparisons(runs, draft)) if changes else current_alarms,
        "not_noise": real,
        "notes": notes,
    }
