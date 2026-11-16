"""Tune a search-time setting on any backend: recall against latency, and a recommendation.

Most vector backends have a knob that trades recall for speed at query time (Qdrant
``hnsw_ef``, pgvector ``hnsw.ef_search`` and ``ivfflat.probes``, Elasticsearch
``num_candidates``, OpenSearch ``ef_search``, FAISS ``nprobe`` and ``efSearch``). This runs the
same queries at each value of one such parameter, measures ANN recall against one exact
reference, and recommends the smallest value that meets a recall target within an optional
latency budget, with the same rules the FAISS sweep uses (:mod:`vectraudit.core.sweep`).

What it is careful about:

- **Read-only and bounded.** Only searches are sent. The number of searches is computed and
  checked against a ceiling before the first one, they are paced under a requests-per-second
  ceiling, and the target must be confirmed by name (the rules of
  :mod:`vectraudit.evaluation.capacity`).
- **Recall needs a stated exact reference**: the backend's own exact search, run once per
  query, or an external reference. Without one the sweep refuses; settings are never scored
  by agreeing with each other.
- **Latency is client-side wall clock, round trip included.** It is comparable between values
  of this sweep, and not with another backend's or with FAISS in-process timings.
- **A missed neighbour has no cause here.** A database does not expose why its index skipped
  a vector. The report lists the neighbours the worst queries missed and the smallest swept
  value that returns each, and nothing more.

A parameter must be one the connector declares (``Capabilities.tuning_params``). Backends
whose setting is part of the collection's configuration (Weaviate ``ef``, Chroma
``ef_search``) or that expose none (Pinecone) declare none: changing a collection is a write,
which connectors never do.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from vectraudit.connectors.base import (
    Capabilities,
    Connector,
    ConnectorError,
    FilterValue,
    SearchRequest,
)
from vectraudit.core.sweep import (
    SweepPoint,
    choose_point,
    pareto_points,
    recommend_point,
    summarize_recalls,
)
from vectraudit.evaluation.capacity import safety_problems, target_name
from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.external_reference import ExternalReference, ExternalReferenceError
from vectraudit.evaluation.metrics import ann_recall
from vectraudit.evaluation.pass_rates import PassRates
from vectraudit.evaluation.ratelimit import RateLimiter
from vectraudit.evaluation.runner import _query_vectors

FORMAT = "vectraudit.tune"
FORMAT_VERSION = 1
DEFAULT_MAX_RPS = 50.0
DEFAULT_MAX_SEARCHES = 100_000
MAX_VALUES = 32
WORST_QUERIES = 10
WARMUP = 5
TIMING_SCOPE = (
    "client-side wall clock of one search call, network round trip included; one query at a "
    "time. Comparable between values of this sweep only."
)


BATCH_SCOPE = (
    "queries per second with every query in one batch call (best of the repeats). A "
    "throughput, never a latency: not compared with the latency columns."
)


class TuneRefused(ValueError):
    """The sweep may not start as asked; the message says why."""


@dataclass(frozen=True)
class TunePlan:
    """Everything a sweep will do, fixed before the first search."""

    param: str
    values: tuple[int, ...]
    k: int = 10
    repeats: int = 3
    max_rps: float = DEFAULT_MAX_RPS
    max_searches: int = DEFAULT_MAX_SEARCHES
    target: float = 0.95
    max_p95_ms: float | None = None
    filters: tuple[Mapping[str, Any], ...] = ()
    """Extra curves: the sweep repeated with each filter applied to every query."""
    batch: bool = False
    """Also measure throughput with every query in one batch call, where the backend has one."""

    def __post_init__(self) -> None:
        if not self.values or len(self.values) > MAX_VALUES:
            raise TuneRefused(f"Give 1 to {MAX_VALUES} values to sweep.")
        if any(type(v) is not int or v < 1 for v in self.values):
            raise TuneRefused("Swept values must be positive integers.")
        if len(set(self.values)) != len(self.values):
            raise TuneRefused("Swept values must be distinct.")
        if self.k < 1 or self.repeats < 1 or self.max_rps <= 0 or self.max_searches < 1:
            raise TuneRefused("k, repeats, max-rps and max-searches must be positive.")
        if not 0 < self.target <= 1:
            raise TuneRefused("The recall target must be above 0 and at most 1.")

    def searches(self, n_queries: int, *, exact: bool) -> int:
        """Searches the sweep sends: warm-up, every value and repeat, and one exact per query."""
        single = len(self.values) * (n_queries * self.repeats + min(WARMUP, n_queries))
        unfiltered = single + (n_queries if exact else 0)
        # Batch, on the unfiltered curve only: one untimed call, then one per repeat.
        batches = len(self.values) * n_queries * (self.repeats + 1) if self.batch else 0
        # A filtered curve always computes its own exact neighbours.
        return unfiltered + batches + len(self.filters) * (single + n_queries)


def describe_plan(
    plan: TunePlan, backend: str, target: str | None, n_queries: int, *, exact: bool
) -> list[str]:
    """What the sweep will send, in plain lines, for printing before it starts."""
    total = plan.searches(n_queries, exact=exact)
    return [
        f"{total:,} read-only searches to {backend} target {target or '(unnamed)'}: "
        f"{n_queries:,} queries x {len(plan.values)} values of {plan.param} x {plan.repeats} "
        f"repeats, plus warm-up" + (" and one exact search per query" if exact else ""),
        f"at most {plan.max_rps:g} searches per second (about {total / plan.max_rps:,.0f} s)",
        *(
            [
                f"the same sweep again under each of {len(plan.filters)} filters, each with its "
                "own exact search"
            ]
            if plan.filters
            else []
        ),
        *(
            ["batch calls holding every query at once (a burst; not paced within the batch)"]
            if plan.batch
            else []
        ),
        "nothing is written to the backend",
    ]


def check_plan(
    plan: TunePlan,
    caps: Capabilities,
    backend: str,
    n_queries: int,
    reference: ExternalReference | None,
    *,
    batch_search: bool = False,
) -> None:
    """Refuse a sweep that cannot be measured or exceeds its bound. Sends nothing."""
    if plan.param not in caps.tuning_params:
        allowed = ", ".join(caps.tuning_params)
        raise TuneRefused(
            f"{backend} has no search-time parameter {plan.param!r}. "
            + (
                f"It declares: {allowed}."
                if allowed
                else "It declares none: its setting belongs to the collection's configuration, "
                "which Vectraudit never changes. Measure recall at the current setting with "
                "`vectraudit record`."
            )
        )
    if reference is None and not caps.exact_reference:
        raise TuneRefused(
            f"{backend} has no exact search here, so recall cannot be measured. Pass an "
            "external exact reference with --reference."
        )
    if plan.batch and not batch_search:
        raise TuneRefused(
            f"{backend} has no batch search call in Vectraudit, so --batch cannot be measured."
        )
    if plan.filters and not caps.filters:
        raise TuneRefused(f"{backend} does not support filters here, so --filter cannot be swept.")
    if plan.filters and reference is not None:
        raise TuneRefused(
            "An external reference holds neighbours for its own filter only; --filter needs the "
            "backend's exact search."
        )
    if plan.filters and not caps.exact_reference:
        raise TuneRefused(f"{backend} has no exact search, so filtered recall cannot be measured.")
    total = plan.searches(n_queries, exact=reference is None)
    if total > plan.max_searches:
        raise TuneRefused(
            f"The sweep would send {total:,} searches, above the ceiling of "
            f"{plan.max_searches:,}. Use fewer values, repeats or queries, or raise "
            "--max-searches."
        )


class _Paced:
    """Searches under the rate ceiling; the wait is not part of the measured latency."""

    def __init__(self, connector: Connector, max_rps: float) -> None:
        self.connector = connector
        self.limiter = RateLimiter(max_rps)
        self.sent = 0

    def search(self, request: SearchRequest) -> tuple[list[str], float]:
        wait = self.limiter.reserve()
        if wait > 0:
            time.sleep(wait)
        started = time.perf_counter()
        ids = self.connector.search(request)
        self.sent += 1
        return ids, (time.perf_counter() - started) * 1000


def tune(
    dataset: Dataset,
    connector: Connector,
    plan: TunePlan,
    *,
    reference: ExternalReference | None = None,
    base_params: Mapping[str, Any] | None = None,
    confirm_target: str | None = None,
    allow_production: bool = False,
    connection_name: str | None = None,
    embed: Any = None,
    progress: Callable[[int, int], None] | None = None,
    watch: str | None = None,
) -> dict[str, Any]:
    """Run the sweep and return the ``vectraudit.tune`` report.

    Raises :class:`TuneRefused` before any search when the plan cannot be run. A connection
    failure, or an interrupt, ends the sweep with status ``failed`` or ``cancelled`` and no
    recommendation. ``watch`` names one question whose ranking at every value is kept.
    """
    if watch is not None and watch not in {q.query_id for q in dataset.queries}:
        raise TuneRefused(f"The dataset has no question {watch!r}.")
    info, caps = connector.describe(), connector.capabilities()
    check_plan(
        plan, caps, info.backend, len(dataset), reference, batch_search=connector.batch_search
    )
    problems = safety_problems(
        info, connection_name, confirm_target=confirm_target, allow_production=allow_production
    )
    if problems:
        raise TuneRefused(" ".join(problems))
    if any(q.context or q.history for q in dataset.queries):
        raise TuneRefused("Permission contexts and conversation history need a pipeline adapter.")
    if reference is not None:
        if not reference.ranks_as(info.distance.value):
            raise TuneRefused(
                f"The reference was computed under {reference.distance} distance; the backend "
                f"ranks by {info.distance.value}."
            )
        if plan.k > reference.k:
            raise TuneRefused(f"k={plan.k} is above the {reference.k} neighbours in the reference.")
        try:
            reference.check_dataset(
                [q.query_id for q in dataset.queries], [q.filter for q in dataset.queries]
            )
        except ExternalReferenceError as e:
            raise TuneRefused(str(e)) from e
    params = {k: v for k, v in (base_params or {}).items() if k != plan.param}
    record: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "failed",
        "error": None,
        "backend": info.as_dict(),
        "target": target_name(info),
        "dataset": {"sha256": dataset.sha256, "n_queries": len(dataset)},
        "param": plan.param,
        "values": sorted(plan.values),
        "k": plan.k,
        "repeats": plan.repeats,
        "other_params": dict(params),
        "reference": reference.identity()
        if reference is not None
        else {"source": "backend exact search", "note": caps.exact_reference_note},
        "timing": {"scope": TIMING_SCOPE, "max_rps": plan.max_rps},
        "target_recall": plan.target,
        "max_p95_ms": plan.max_p95_ms,
        "searches_planned": plan.searches(len(dataset), exact=reference is None),
        "points": [],
        "recommendation": None,
        "missed": [],
        "filtered": [],
        "batch": {"measured": plan.batch, "scope": BATCH_SCOPE if plan.batch else None},
    }
    paced = _Paced(connector, plan.max_rps)
    try:
        vectors, _ = _query_vectors(dataset, embed, info.dim)
        steps = len(plan.values) * (1 + len(plan.filters))
        done = 0

        def tick() -> None:
            nonlocal done
            if progress is not None:
                progress(done, steps)
            done += 1

        base = _curve(dataset, vectors, paced, plan, params, reference, None, tick, watch)
        record.update(base)
        counter = PassRates(connector)
        for flt in plan.filters:
            curve = _curve(dataset, vectors, paced, plan, params, None, flt, tick, None)
            rate = counter.of(cast(Mapping[str, FilterValue], flt))
            record["filtered"].append(
                {
                    "filter": dict(flt),
                    "pass_rate": rate,
                    "pass_rate_note": None
                    if rate is not None
                    else f"{info.backend} cannot count the documents a filter allows.",
                    **curve,
                }
            )
        record["status"] = "complete"
    except ConnectorError as e:
        record["error"] = e.as_dict()
    except KeyboardInterrupt:
        record["status"] = "cancelled"
        record["error"] = {
            "code": "cancelled",
            "message": "The sweep was interrupted; a partial sweep recommends nothing.",
            "hint": None,
        }
    record["searches_sent"] = paced.sent
    if progress is not None:
        total = len(plan.values) * (1 + len(plan.filters))
        progress(total, total)
    return record


def _curve(
    dataset: Dataset,
    vectors: Sequence[Any],
    paced: _Paced,
    plan: TunePlan,
    params: Mapping[str, Any],
    reference: ExternalReference | None,
    flt: Mapping[str, Any] | None,
    tick: Callable[[], None],
    watch: str | None,
) -> dict[str, Any]:
    """One sweep: every value against one exact truth, under ``flt`` when given (else each
    query's own filter). Batch throughput is measured on the unfiltered curve only."""

    def request(i: int, value: int | None, *, exact: bool = False) -> SearchRequest:
        q = dataset.queries[i]
        tuned = params if value is None else {**params, plan.param: value}
        return SearchRequest(
            vectors[i],
            plan.k,
            filter=cast(Mapping[str, FilterValue], q.filter if flt is None else flt),
            params={} if exact else tuned,
            exact=exact,
        )

    n = len(dataset)
    truth = [
        reference.neighbours(q.query_id, plan.k)
        if reference is not None
        else paced.search(request(i, None, exact=True))[0]
        for i, q in enumerate(dataset.queries)
    ]

    def recall_of(found: Sequence[Sequence[str]]) -> Any:
        # A query whose filter allows nothing has no neighbours to miss: recall 1, as in sweep.
        values = [ann_recall(ids, exact) for ids, exact in zip(found, truth, strict=True)]
        return np.asarray([1.0 if v is None else v for v in values], dtype=np.float64)

    rankings: dict[int, list[list[str]]] = {}
    points: list[SweepPoint] = []
    rows = []
    for value in sorted(plan.values):
        tick()
        for i in range(min(WARMUP, n)):
            paced.search(request(i, value))
        found: list[list[str]] = []
        latencies: list[float] = []
        for i in range(n):
            for repeat in range(plan.repeats):
                ids, ms = paced.search(request(i, value))
                latencies.append(ms)
                if repeat == 0:
                    found.append(ids)
        rankings[value] = found
        recalls = recall_of(found)
        low, high, distribution, worst = summarize_recalls(recalls, None)
        points.append(
            SweepPoint(
                value=value,
                recall=float(recalls.mean()),
                latency_mean_ms=float(np.mean(latencies)),
                latency_p95_ms=float(np.percentile(latencies, 95)),
                recall_ci_low=low,
                recall_ci_high=high,
                recall_distribution=distribution,
                worst_queries=worst,
            )
        )
        row = _point(points[-1], dataset)
        if plan.batch and flt is None:
            requests = [request(i, value) for i in range(n)]
            paced.connector.search_batch(requests)  # untimed: connections and caches warm
            best, answers = float("inf"), None
            for _ in range(plan.repeats):
                started = time.perf_counter()
                answers = paced.connector.search_batch(requests)
                best = min(best, time.perf_counter() - started)
            paced.sent += n * (plan.repeats + 1)
            assert answers is not None
            row["batch_qps"] = n / max(best, 1e-9)
            row["batch_recall"] = float(recall_of(answers).mean())
        rows.append(row)
    chosen = recommend_point(points, plan.target, False, plan.max_p95_ms)
    at = chosen.value if chosen is not None else max(plan.values)
    out: dict[str, Any] = {
        "points": rows,
        "recommendation": _recommendation(points, plan),
        "missed": missed_neighbours(dataset, truth, rankings, at),
        "missed_at": at,
    }
    if watch is not None:
        i = next(k for k, q in enumerate(dataset.queries) if q.query_id == watch)
        out["watched"] = {
            "query_id": watch,
            "exact_ids": list(truth[i]),
            "by_value": {str(v): list(rankings[v][i]) for v in sorted(rankings)},
        }
    return out


def _point(p: SweepPoint, dataset: Dataset) -> dict[str, Any]:
    return {
        "value": p.value,
        "recall": p.recall,
        "recall_ci_low": p.recall_ci_low,
        "recall_ci_high": p.recall_ci_high,
        "latency_mean_ms": p.latency_mean_ms,
        "latency_p95_ms": p.latency_p95_ms,
        "recall_distribution": [list(pair) for pair in p.recall_distribution],
        "worst_queries": [
            {"query_id": dataset.queries[w.query_no].query_id, "recall": w.recall}
            for w in p.worst_queries[:WORST_QUERIES]
        ],
    }


def _recommendation(points: Sequence[SweepPoint], plan: TunePlan) -> dict[str, Any]:
    choice = choose_point(points, plan.target, False, plan.max_p95_ms)
    return {
        "status": choice.status,
        "value": choice.point.value if choice.point else None,
        "recall_needs": choice.by_recall.value if choice.by_recall else None,
        "best_in_budget": choice.best_in_budget.value if choice.best_in_budget else None,
        "pareto": [p.value for p in pareto_points(points)],
        "rule": "the smallest swept value whose mean recall meets the target"
        + (" with p95 latency within the budget" if plan.max_p95_ms is not None else ""),
    }


def missed_neighbours(
    dataset: Dataset,
    truth: Sequence[Sequence[str]],
    rankings: Mapping[int, Sequence[Sequence[str]]],
    at: int,
    *,
    limit: int = WORST_QUERIES,
) -> list[dict[str, Any]]:
    """For the lowest-recall queries at value ``at``: each true neighbour they missed and the
    smallest swept value that returns it (``None``: at no swept value). No cause is given."""
    values = sorted(rankings)
    rows = []
    for i, exact in enumerate(truth):
        lost = [d for d in exact if d not in set(rankings[at][i])]
        if lost:
            rows.append((len(lost) / max(len(exact), 1), i, lost))
    rows.sort(key=lambda r: (-r[0], r[1]))
    out = []
    for _, i, lost in rows[:limit]:
        out.append(
            {
                "query_id": dataset.queries[i].query_id,
                "recall": 1 - len(lost) / max(len(truth[i]), 1),
                "missed": [
                    {
                        "doc_id": doc,
                        "found_from": next((v for v in values if doc in rankings[v][i]), None),
                    }
                    for doc in lost
                ],
            }
        )
    return out


def render(report: Mapping[str, Any]) -> str:
    """The sweep as plain text for the terminal."""
    backend = report["backend"]["backend"]
    lines = [
        f"{report['param']} sweep on {backend} target {report['target']}: ANN recall@"
        f"{report['k']} over {report['dataset']['n_queries']} queries against "
        f"{report['reference']['source']}; latency is client-side, round trip included.",
        f"  {report['param']:>12}  recall  95% interval    mean ms   p95 ms",
    ]
    rec = report["recommendation"] or {}
    for p in report["points"]:
        mark = "  <- recommended" if p["value"] == rec.get("value") else ""
        if "batch_qps" in p:
            mark = f"  batch {p['batch_qps']:,.0f} q/s{mark}"
        lines.append(
            f"  {p['value']:>12}  {p['recall']:.3f}   {p['recall_ci_low']:.3f}-"
            f"{p['recall_ci_high']:.3f}  {p['latency_mean_ms']:8.2f} {p['latency_p95_ms']:8.2f}"
            f"{mark}"
        )
    if report["status"] != "complete":
        message = (report["error"] or {}).get("message", "")
        lines.append(f"{report['status'].capitalize()}: {message} No recommendation.")
        return "\n".join(lines)
    target = report["target_recall"]
    if rec["status"] == "ok":
        lines.append(f"Recommended {report['param']}={rec['value']}: {rec['rule']} ({target:g}).")
    elif rec["status"] == "latency":
        lines.append(
            f"No value reached recall {target:g} within p95 {report['max_p95_ms']:g} ms: recall "
            f"needs {report['param']}={rec['recall_needs']}."
        )
    else:
        best = max(report["points"], key=lambda p: p["recall"])
        lines.append(
            f"No swept value reached recall {target:g} (best {best['recall']:.3f} at "
            f"{report['param']}={best['value']})."
        )
    for row in report["missed"][:5]:
        found = ", ".join(
            f"{m['doc_id']} (from {m['found_from']})"
            if m["found_from"] is not None
            else f"{m['doc_id']} (at no swept value)"
            for m in row["missed"][:5]
        )
        lines.append(
            f"  {row['query_id']} at {report['param']}={report['missed_at']}: recall "
            f"{row['recall']:.2f}, missed {found}"
        )
    if report["missed"]:
        lines.append("  A backend does not expose why a neighbour was missed; none is given.")
    if (report.get("batch") or {}).get("measured"):
        lines.append("Batch q/s is a throughput with every query in one call, not a latency.")
    for curve in report.get("filtered") or []:
        rate = curve["pass_rate"]
        passing = "pass rate unknown" if rate is None else f"pass rate {rate:.2%}"
        chosen = curve["recommendation"]["value"]
        best = max(curve["points"], key=lambda p: p["recall"])
        lines.append(
            f"Filter {json.dumps(curve['filter'], sort_keys=True)} ({passing}): "
            + (
                f"recommended {report['param']}={chosen}"
                if chosen is not None
                else f"no value reached the target (best recall {best['recall']:.3f} at "
                f"{report['param']}={best['value']})"
            )
            + "; recall "
            + ", ".join(f"{p['value']}: {p['recall']:.3f}" for p in curve["points"])
        )
    watched = report.get("watched")
    if watched:
        exact = set(watched["exact_ids"])
        lines.append(f"Question {watched['query_id']} ({len(exact)} exact neighbours):")
        for value, ids in watched["by_value"].items():
            got = sum(1 for d in ids if d in exact)
            lost = ", ".join(d for d in watched["exact_ids"] if d not in ids) or "none"
            lines.append(f"  {report['param']}={value}: {got} of {len(exact)} found; missed {lost}")
    return "\n".join(lines)


# --- comparing two sweeps ----------------------------------------------------------------------


def _reference_key(report: Mapping[str, Any]) -> tuple[str, str | None]:
    ref = report["reference"]
    return (str(ref.get("source")), ref.get("sha256"))


def compare_tunes(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Two sweeps side by side: recall at matched values, each recommendation, and the queries
    whose missed neighbours differ.

    Recall is compared only when both sweeps used the same questions, ``k`` and kind of exact
    reference (the same external reference, or each backend's own exact search). Latency is
    shown per side, and its difference only when both sides are the same backend: client-side
    timings of different backends are not comparable.
    """
    for report in (a, b):
        if report.get("format") != FORMAT:
            raise TuneRefused("Both files must be vectraudit.tune reports.")
    reasons = []
    for name, report in (("first", a), ("second", b)):
        if report["status"] != "complete":
            reasons.append(f"The {name} sweep is {report['status']}, not complete.")
    if a["dataset"]["sha256"] != b["dataset"]["sha256"]:
        reasons.append("The sweeps used different question sets.")
    if a["k"] != b["k"]:
        reasons.append(f"The sweeps used k={a['k']} and k={b['k']}.")
    if _reference_key(a) != _reference_key(b):
        reasons.append(
            "The sweeps measured recall against different exact references "
            f"({_reference_key(a)[0]} and {_reference_key(b)[0]})."
        )
    same_backend = a["backend"]["backend"] == b["backend"]["backend"]
    same_param = a["param"] == b["param"]

    def side(report: Mapping[str, Any]) -> dict[str, Any]:
        rec = report.get("recommendation") or {}
        chosen = next((p for p in report["points"] if p["value"] == rec.get("value")), None)
        return {
            "backend": report["backend"]["backend"],
            "target": report["target"],
            "param": report["param"],
            "recommended": rec.get("value"),
            "recall": chosen["recall"] if chosen else None,
            "latency_p95_ms": chosen["latency_p95_ms"] if chosen else None,
        }

    out: dict[str, Any] = {
        "format": "vectraudit.tune_comparison",
        "format_version": 1,
        "comparable": not reasons,
        "reasons": reasons,
        "a": side(a),
        "b": side(b),
        "matched": [],
        "missed_changed": None,
        "latency_note": None
        if same_backend
        else "Different backends: latency is shown per side and not subtracted.",
    }
    if reasons:
        return out
    if same_param:
        points_b = {p["value"]: p for p in b["points"]}
        for p in a["points"]:
            q = points_b.get(p["value"])
            if q is None:
                continue
            out["matched"].append(
                {
                    "value": p["value"],
                    "recall_a": p["recall"],
                    "recall_b": q["recall"],
                    "recall_change": q["recall"] - p["recall"],
                    "p95_ms_a": p["latency_p95_ms"],
                    "p95_ms_b": q["latency_p95_ms"],
                    "p95_ms_change": q["latency_p95_ms"] - p["latency_p95_ms"]
                    if same_backend
                    else None,
                }
            )
    missed_a = {row["query_id"] for row in a.get("missed") or []}
    missed_b = {row["query_id"] for row in b.get("missed") or []}
    out["missed_changed"] = {
        "at": {"a": a.get("missed_at"), "b": b.get("missed_at")},
        "only_a": sorted(missed_a - missed_b),
        "only_b": sorted(missed_b - missed_a),
        "both": sorted(missed_a & missed_b),
        "note": "Among each sweep's lowest-recall questions at its own recommended value.",
    }
    return out


def render_comparison(result: Mapping[str, Any]) -> str:
    """The comparison as plain text for the terminal."""
    lines = []
    for key in ("a", "b"):
        s = result[key]
        chosen = (
            f"recommended {s['param']}={s['recommended']} (recall {s['recall']:.3f}, p95 "
            f"{s['latency_p95_ms']:.2f} ms)"
            if s["recommended"] is not None
            else "no value reached its target"
        )
        lines.append(f"{key.upper()}: {s['backend']} target {s['target']}: {chosen}")
    if not result["comparable"]:
        lines += [f"Not comparable: {reason}" for reason in result["reasons"]]
        return "\n".join(lines)
    for m in result["matched"]:
        latency = "" if m["p95_ms_change"] is None else f", p95 {m['p95_ms_change']:+.2f} ms"
        lines.append(
            f"  {result['a']['param']}={m['value']}: recall {m['recall_a']:.3f} -> "
            f"{m['recall_b']:.3f} ({m['recall_change']:+.3f}){latency}"
        )
    if not result["matched"]:
        lines.append("  No swept value is shared, so recall is shown per recommendation only.")
    if result["latency_note"]:
        lines.append(result["latency_note"])
    changed = result["missed_changed"]
    lines.append(
        f"Questions missing neighbours: {len(changed['only_a'])} only in A, "
        f"{len(changed['only_b'])} only in B, {len(changed['both'])} in both "
        "(each sweep's worst questions at its recommended value)."
    )
    return "\n".join(lines)
