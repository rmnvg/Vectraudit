"""Regression policies: when a candidate run is acceptable relative to its baseline.

A policy is JSON::

    {"version": 1,
     "metrics": {"ndcg": {"max_drop": 0.02}, "ann_recall": {"min": 0.9}},
     "latency": {"p95_max_increase_pct": 25, "p95_max_ms": 80, "min_increase_ms": 2},
     "min_queries": 30,
     "max_regressed_queries": 10,
     "primary_metric": "ndcg",
     "cohorts": {"min_queries": 10, "metrics": {"ndcg": {"max_drop": 0.05}}},
     "require_significance": false}

Every metric a policy names must be measurable in both runs, otherwise the check is not
comparable: a missing label set or exact reference never passes silently. Without a
policy file, :data:`DEFAULT_POLICY` applies its rules to whichever metrics both runs have.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.evaluation.metrics import QUALITY_METRICS

POLICY_VERSION = 1
_TOP_KEYS = {
    "version",
    "metrics",
    "latency",
    "min_queries",
    "max_regressed_queries",
    "primary_metric",
    "cohorts",
    "require_significance",
}
_LATENCY_KEYS = {
    "p50_max_increase_pct",
    "p95_max_increase_pct",
    "p95_max_ms",
    "min_increase_ms",
    "require_same_environment",
}


class PolicyError(ValueError):
    """The policy file is invalid."""


@dataclass(frozen=True)
class MetricRule:
    """Limits for one metric. ``max_drop`` is absolute (0.02 = two points of nDCG)."""

    max_drop: float | None = None
    min: float | None = None
    required: bool = True
    """When False (default policy only), a metric neither run measured is skipped."""


@dataclass(frozen=True)
class LatencyRule:
    """Latency limits, compared only between runs from matching environments by default."""

    p50_max_increase_pct: float | None = None
    p95_max_increase_pct: float | None = None
    p95_max_ms: float | None = None
    min_increase_ms: float = 2.0
    """Percentage rules ignore rises smaller than this in absolute terms: shared CI runners
    move p95 by ~0.1 ms on local indexes and ~1.5 ms against networked backends."""
    require_same_environment: bool = True


@dataclass(frozen=True)
class CohortRule:
    """Per-tag limits; tags with fewer than ``min_queries`` queries are reported, not judged."""

    min_queries: int = 10
    metrics: dict[str, MetricRule] = field(default_factory=dict)


@dataclass(frozen=True)
class Policy:
    metrics: dict[str, MetricRule]
    latency: LatencyRule = LatencyRule()
    min_queries: int = 1
    """Minimum suite size and measured paired queries for each judged quality metric."""
    max_regressed_queries: int | None = None
    primary_metric: str | None = None
    """Metric whose per-query drops define "regressed queries"; default: first available."""
    cohorts: CohortRule = CohortRule()
    require_significance: bool = False
    """Fail a max_drop rule only when the whole 95% interval of the drop exceeds the limit."""
    is_default: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY = Policy(
    metrics={
        "ndcg": MetricRule(max_drop=0.01, required=False),
        "recall": MetricRule(max_drop=0.01, required=False),
        "mrr": MetricRule(max_drop=0.02, required=False),
        "ann_recall": MetricRule(max_drop=0.01, required=False),
    },
    latency=LatencyRule(p95_max_increase_pct=50.0),
    is_default=True,
)


def load_policy(path: str | Path) -> Policy:
    """Read and validate a policy file."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise PolicyError(f"Policy file not found: {p}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise PolicyError(f"{p} is not valid JSON: {e}") from e
    return parse_policy(raw)


def parse_policy(raw: Any) -> Policy:
    """Validate a policy mapping (see the module docstring)."""
    _expect(isinstance(raw, dict), "a policy must be a JSON object")
    _no_unknown(raw, _TOP_KEYS, "policy")
    _expect(raw.get("version", POLICY_VERSION) == POLICY_VERSION, "unsupported policy version")
    metrics = _metric_rules(raw.get("metrics", {}), "metrics")
    latency_raw = raw.get("latency", {})
    _expect(isinstance(latency_raw, dict), "latency must be an object")
    _no_unknown(latency_raw, _LATENCY_KEYS, "latency")
    latency = LatencyRule(
        p50_max_increase_pct=_number(latency_raw, "p50_max_increase_pct"),
        p95_max_increase_pct=_number(latency_raw, "p95_max_increase_pct"),
        p95_max_ms=_number(latency_raw, "p95_max_ms"),
        min_increase_ms=_number(latency_raw, "min_increase_ms") or 0.0
        if "min_increase_ms" in latency_raw
        else LatencyRule.min_increase_ms,
        require_same_environment=_bool(latency_raw, "require_same_environment", True),
    )
    cohorts_raw = raw.get("cohorts", {})
    _expect(isinstance(cohorts_raw, dict), "cohorts must be an object")
    _no_unknown(cohorts_raw, {"min_queries", "metrics"}, "cohorts")
    cohorts = CohortRule(
        min_queries=_count(cohorts_raw, "min_queries", 10, minimum=1),
        metrics=_metric_rules(cohorts_raw.get("metrics", {}), "cohorts.metrics"),
    )
    primary = raw.get("primary_metric")
    _expect(
        primary is None or primary in QUALITY_METRICS,
        f"primary_metric must be one of {list(QUALITY_METRICS)}",
    )
    max_regressed = raw.get("max_regressed_queries")
    if max_regressed is not None:
        max_regressed = _count(raw, "max_regressed_queries", 0)
    policy = Policy(
        metrics=metrics,
        latency=latency,
        min_queries=_count(raw, "min_queries", 1, minimum=1),
        max_regressed_queries=max_regressed,
        primary_metric=primary,
        cohorts=cohorts,
        require_significance=_bool(raw, "require_significance", False),
    )
    has_quality_rule = bool(metrics) or max_regressed is not None or bool(cohorts.metrics)
    _expect(has_quality_rule, "a policy needs at least one metric or regressed-query rule")
    return policy


def _metric_rules(raw: Any, where: str) -> dict[str, MetricRule]:
    _expect(isinstance(raw, dict), f"{where} must be an object")
    rules = {}
    for name, rule in raw.items():
        _expect(name in QUALITY_METRICS, f"{where}: unknown metric {name!r}")
        _expect(isinstance(rule, dict), f"{where}.{name} must be an object")
        _no_unknown(rule, {"max_drop", "min"}, f"{where}.{name}")
        max_drop = _number(rule, "max_drop")
        minimum = _number(rule, "min")
        _expect(max_drop is not None or minimum is not None, f"{where}.{name} sets no limit")
        _expect(minimum is None or minimum <= 1, f"{where}.{name}.min must be <= 1")
        rules[name] = MetricRule(max_drop=max_drop, min=minimum)
    return rules


def _number(raw: dict[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    _expect(ok and math.isfinite(value) and value >= 0, f"{key} must be a number >= 0")
    return float(value)


def _count(raw: dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    value = raw.get(key, default)
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= minimum
    _expect(ok, f"{key} must be an integer >= {minimum}")
    return int(value)


def _bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    _expect(isinstance(value, bool), f"{key} must be true or false")
    return bool(value)


def _no_unknown(raw: dict[str, Any], known: set[str], where: str) -> None:
    unknown = set(raw) - known
    _expect(not unknown, f"{where}: unknown field(s) {sorted(unknown)}; allowed {sorted(known)}")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyError(message)
