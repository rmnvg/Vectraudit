"""Equal-weight query families for quality aggregation and paired resampling."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def family_id(query: Mapping[str, Any]) -> str:
    """Rows without an explicit family remain independent, including legacy run files."""
    value: str = query.get("family_id") or query["query_id"]
    return value


def family_count(queries: Iterable[Mapping[str, Any]]) -> int:
    return len({family_id(q) for q in queries})


def family_means(values: Iterable[tuple[str, float]]) -> list[float]:
    """Mean of measured rows within each family, in first-observed family order."""
    groups: dict[str, list[float]] = defaultdict(list)
    for key, value in values:
        groups[key].append(value)
    return [statistics.fmean(group) for group in groups.values()]


def family_mean(values: Iterable[tuple[str, float]]) -> float | None:
    means = family_means(values)
    return statistics.fmean(means) if means else None
