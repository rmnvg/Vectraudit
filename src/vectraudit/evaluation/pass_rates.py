"""The share of the corpus a query's filter lets through, measured where the backend can.

A filtered ANN query is hardest when its filter removes almost everything. Whether a query was
in that regime is a property of the filter and the corpus, not of the question, so the pass rate
is **counted** (the backend's exact count of documents matching the filter, over its exact
total) and never estimated. A backend that cannot count leaves the pass rate ``None``:
unknown, not "high" and not "low". An unknown pass rate puts the query in no selectivity
cohort, and the run says how many filtered queries were counted.

A dataset header may declare ``"selectivity_buckets": [0.001, 0.01, 0.1]``. Each query is then
tagged with the single bucket its pass rate falls in, named by the bucket's upper bound
(``pass_rate<1%``), exactly like a ``cohort_fields`` cohort, so the existing cohort rules
apply. Buckets are exclusive, so one query is never counted in two selectivity cohorts; a pass
rate at or above the largest bound is in none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.connectors.base import Connector, ConnectorError, ErrorCode, FilterValue

# Failures of one count that leave the run going, as for a search (see the runner).
_PER_QUERY = frozenset({ErrorCode.QUERY, ErrorCode.UNSUPPORTED, ErrorCode.TIMEOUT})


def bucket_label(upper: float) -> str:
    """``pass_rate<1%`` for 0.01."""
    return f"pass_rate<{upper * 100:g}%"


def bucket_for(rate: float | None, buckets: Sequence[float]) -> str | None:
    """The one bucket ``rate`` falls in, or ``None`` (unknown, or not selective enough)."""
    if rate is None:
        return None
    for upper in sorted(buckets):
        if rate < upper:
            return bucket_label(upper)
    return None


class PassRates:
    """Counts filters once each and remembers the answers for a run."""

    def __init__(self, connector: Connector, buckets: Sequence[float] = ()) -> None:
        self.connector = connector
        self.buckets = tuple(sorted(buckets))
        self.supported = connector.capabilities().filter_count
        self._total: int | None = None
        self._rates: dict[str, float | None] = {}
        self.filtered_queries = 0
        self.counted = 0

    def _count(self, filter: Mapping[str, FilterValue]) -> int | None:
        try:
            return self.connector.count(filter)
        except ConnectorError as e:
            if e.code not in _PER_QUERY:
                raise
            return None

    def of(self, filter: Mapping[str, FilterValue]) -> float | None:
        """The pass rate of ``filter`` over the whole corpus, or ``None`` if unknown."""
        self.filtered_queries += 1
        if not self.supported:
            return None
        key = repr(sorted(filter.items()))
        if key not in self._rates:
            if self._total is None:
                self._total = self._count({})
            matching = self._count(filter)
            self._rates[key] = (
                matching / self._total if matching is not None and self._total else None
            )
        rate = self._rates[key]
        self.counted += rate is not None
        return rate

    def summary(self) -> dict[str, Any] | None:
        """What was counted, for the run record; ``None`` when no query was filtered."""
        if not self.filtered_queries:
            return None
        return {
            "supported": self.supported,
            "filtered_queries": self.filtered_queries,
            "counted": self.counted,
            "buckets": list(self.buckets),
            "meaning": "fraction of the corpus that satisfies the filter, counted by the backend",
        }
