"""Recall against filter selectivity: the same sweep at several pass rates.

A filter that removes most of the corpus is the hard case for an approximate index: the
search visits cells or nodes chosen by distance to the query, and few of the vectors in them
pass. This runs the ordinary sweep once per filter, each with its own exact truth over the
allowed ids, so a recall-versus-pass-rate table is a measurement rather than an expectation.

Two kinds of filter are supported:

- **metadata** filters, evaluated against real metadata with the portable matcher, so the
  pass rate is whatever the data says it is;
- the **synthetic id filter** of VectorDBBench, ``id >= cut``: the highest ids pass, chosen to
  hit a target pass rate exactly. It is deliberately uncorrelated with vector content in
  typical corpora, which isolates selectivity from filter/vector correlation.

``pass_rate`` is the fraction of the corpus that passes (VectorDBBench's "filter rate" is the
fraction removed).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from vectraudit.core import ivf
from vectraudit.core.filtering import (
    AllowedSet,
    FilterError,
    allowed_from_metadata,
    filter_fingerprint,
)
from vectraudit.core.metadata import Metadata
from vectraudit.core.search import GroundTruth
from vectraudit.core.sweep import (
    Choice,
    QuerySet,
    SweepParam,
    SweepResult,
    ground_truth_ids,
    sweep,
)
from vectraudit.core.types import LoadedIndex

IntArray = npt.NDArray[np.int64]
ProgressFn = Callable[[float, str], None]

FilterKind = Literal["all", "metadata", "synthetic"]
MIN_PASS_RATE = 1e-6


@dataclass(frozen=True)
class FilterSpec:
    """One filter to sweep: how to build its allowed set, and what to call it."""

    kind: FilterKind
    label: str
    filters: Mapping[str, Any] | None = None
    """``metadata``: the condition, e.g. ``{"lang": "de"}``."""
    pass_rate: float | None = None
    """``synthetic``: the target fraction of the corpus that passes."""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "label": self.label}
        if self.filters is not None:
            out["filters"] = dict(self.filters)
        if self.pass_rate is not None:
            out["pass_rate"] = self.pass_rate
        return out


ALL = FilterSpec("all", "all")


def metadata_spec(filters: Mapping[str, Any]) -> FilterSpec:
    """A metadata filter; its label is the filter written compactly."""
    if not filters:
        raise FilterError("A metadata filter needs at least one condition.", "Use --filter k=v.")
    label = ",".join(f"{k}={filters[k]}" for k in sorted(filters))
    return FilterSpec("metadata", label, dict(filters))


def synthetic_spec(pass_rate: float) -> FilterSpec:
    """The VectorDBBench-style integer filter at a target pass rate."""
    if not MIN_PASS_RATE <= pass_rate <= 1.0:
        raise FilterError(
            f"A pass rate must be between {MIN_PASS_RATE:g} and 1, got {pass_rate}.",
            "A pass rate is the fraction of documents that satisfy the filter, not the "
            "fraction removed.",
        )
    return FilterSpec("synthetic", f"synthetic@{pass_rate:g}", pass_rate=pass_rate)


def synthetic_allowed(ids: npt.ArrayLike, pass_rate: float) -> AllowedSet:
    """The highest ``round(n * pass_rate)`` ids: VectorDBBench's ``id >= cut`` filter.

    The cut is by rank among the stored ids, so sparse or offset ids still hit the rate.
    At least one id always passes.
    """
    corpus = np.unique(np.asarray(ids, dtype=np.int64))
    n = len(corpus)
    keep = min(n, max(1, round(n * pass_rate))) if n else 0
    return AllowedSet(corpus[n - keep :], n)


@dataclass(frozen=True)
class FilterRun:
    """A sweep at one filter."""

    spec: FilterSpec
    pass_rate: float
    """The measured fraction of the corpus that passes (1.0 for the unfiltered run)."""
    allowed_count: int
    sweep: SweepResult


@dataclass(frozen=True)
class SelectivityResult:
    """The same sweep at several filters, the unfiltered run first."""

    param: SweepParam
    k: int
    runs: list[FilterRun]

    @property
    def baseline(self) -> FilterRun:
        return self.runs[0]

    def choose(
        self, target_recall: float, *, confident: bool = False, max_p95_ms: float | None = None
    ) -> list[tuple[FilterRun, Choice]]:
        """The recommendation at each pass rate, tightest filter last."""
        return [
            (run, run.sweep.choose(target_recall, confident=confident, max_p95_ms=max_p95_ms))
            for run in sorted(self.runs, key=lambda r: -r.pass_rate)
        ]


def build_allowed(
    spec: FilterSpec, ids: npt.ArrayLike, metadata: Metadata | None, k: int
) -> AllowedSet | None:
    """The allowed set for ``spec`` (``None`` = unfiltered), refusing degenerate filters.

    A filter that allows fewer documents than ``k`` has no k true neighbours to find, so its
    recall would be computed over a shorter list and compare with nothing: it is refused.
    """
    if spec.kind == "all":
        return None
    if spec.kind == "synthetic":
        assert spec.pass_rate is not None
        allowed = synthetic_allowed(ids, spec.pass_rate)
    else:
        if metadata is None:
            raise FilterError(
                "A metadata filter needs metadata.", "Pass --meta with a file keyed by id."
            )
        assert spec.filters is not None
        allowed = allowed_from_metadata(metadata, spec.filters, ids)
    if len(allowed) < k:
        raise FilterError(
            f"The filter {spec.label!r} allows {len(allowed):,} documents, fewer than k={k}.",
            "Loosen the filter, lower k, or use a larger corpus.",
        )
    return allowed


def selectivity_sweep(
    li: LoadedIndex,
    queries: QuerySet,
    gt: GroundTruth,
    specs: Sequence[FilterSpec],
    *,
    metadata: Metadata | None = None,
    param: SweepParam | str | None = None,
    values: Sequence[int] | None = None,
    k: int = 10,
    repeats: int = 3,
    seed: int = 0,
    assignments: ivf.Assignments | None = None,
    progress: ProgressFn | None = None,
) -> SelectivityResult:
    """Sweep ``param`` at the unfiltered baseline and at each filter in ``specs``.

    Each filter gets its own exact ground truth (cached on ``gt`` per allowed set), so recall
    is always against the true neighbours *among the allowed ids*. Filters are built before
    anything runs, so a bad one fails fast.
    """
    report = progress or (lambda _f, _m: None)
    seen: set[str] = {ALL.label}
    todo: list[tuple[FilterSpec, AllowedSet | None]] = [(ALL, None)]
    for spec in specs:
        if spec.label in seen:
            continue
        seen.add(spec.label)
        todo.append((spec, build_allowed(spec, gt.source.ids, metadata, k)))
    runs: list[FilterRun] = []
    for i, (spec, allowed) in enumerate(todo):

        def scaled(frac: float, msg: str, i: int = i, label: str = spec.label) -> None:
            report((i + frac) / len(todo), f"[{label}] {msg}")

        truth = ground_truth_ids(gt, queries, k, scaled, allowed=allowed)
        result = sweep(
            li,
            queries,
            truth,
            param=param,
            values=values,
            k=k,
            truth_reconstructed=gt.reconstructed,
            progress=scaled,
            repeats=repeats,
            seed=seed,
            assignments=assignments if li.kind.is_ivf else None,
            allowed=allowed,
        )
        runs.append(
            FilterRun(
                spec,
                1.0 if allowed is None else allowed.pass_rate,
                li.ntotal if allowed is None else len(allowed),
                result,
            )
        )
    report(1.0, "Done")
    return SelectivityResult(runs[0].sweep.param, k, runs)


def filter_identity(spec: FilterSpec) -> str:
    """Stable id of a filter description, for run records and caching."""
    return filter_fingerprint(spec.as_dict())
