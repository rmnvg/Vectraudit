"""A capacity curve: how a backend behaves as concurrent load grows.

The release checks measure one query at a time. This sends a bounded amount of load at one
named target and reports, per concurrency level, the achieved throughput, p50/p95/p99
latency, the error rate and the recall of a sample of the answers. It is a probe of a
**staging** target, not a load-testing tool, so every limit errs on the side of sending less:

- it will not start unless the caller names the target it believes it is hitting
  (``confirm_target``) and that name matches the connection's, so a copied command cannot hit
  another collection;
- a target or connection whose name contains ``prod`` is refused without an explicit override;
- every level runs for a fixed duration under a global requests-per-second ceiling
  (default 30 s and 50 requests/s), so the most it can send is known before it starts and
  is printed first.

Latency is measured by this process, around the connector call: it includes the network and
the client library, and it depends on the machine, so two reports compare only when their
runner environments match (:func:`compare_capacity`). Reaching the ceiling is reported as
such: a run limited by ``max_rps`` shows the backend kept up, not its limit.

Recall is not measured under load (an exact search would distort it). The answers to a
fixed sample of queries are kept during each level and compared, afterwards, with the exact
reference for those queries. Without an exact reference the recall is ``null``.
"""

from __future__ import annotations

import contextlib
import math
import random
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np

from vectraudit.connectors.base import (
    BackendInfo,
    Capabilities,
    Connector,
    ConnectorError,
    ErrorCode,
    FilterValue,
    SearchRequest,
)
from vectraudit.evaluation.compare import environment_differences
from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.metrics import ann_recall
from vectraudit.evaluation.ratelimit import RateLimiter
from vectraudit.evaluation.runner import _query_vectors, runner_environment

FORMAT = "vectraudit.capacity"
FORMAT_VERSION = 1

DEFAULT_DURATION_S = 30.0
DEFAULT_MAX_RPS = 50.0
MAX_DURATION_S = 600.0
MAX_RPS_LIMIT = 1000.0
MAX_CONCURRENCY = 32
MAX_LEVELS = 8
SERIAL_QUERIES = 100
RECALL_SAMPLE = 20
WARMUP = 5

# The primary name of a backend's target: the thing a person would call "the collection".
TARGET_KEYS = {
    "faiss": "index",
    "qdrant": "collection",
    "pgvector": "table",
    "pinecone": "index",
    "elasticsearch": "index",
    "opensearch": "index",
    "weaviate": "collection",
    "chroma": "collection",
}
# A request failing like this will fail every time: stop rather than count it thousands of times.
_FATAL = frozenset({ErrorCode.CONFIG, ErrorCode.AUTH, ErrorCode.DEPENDENCY})
# How close to the ceiling counts as "limited by the ceiling rather than by the backend".
_AT_CEILING = 0.9


class CapacityRefused(ValueError):
    """The run must not start; the message says what to change."""


@dataclass(frozen=True)
class CapacityPlan:
    """Everything that decides how much load is sent."""

    concurrency: tuple[int, ...] = (1, 4, 16)
    duration_s: float = DEFAULT_DURATION_S
    max_rps: float = DEFAULT_MAX_RPS
    k: int = 10
    params: Mapping[str, Any] = field(default_factory=dict)
    serial_queries: int = SERIAL_QUERIES
    recall_sample: int = RECALL_SAMPLE
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= len(self.concurrency) <= MAX_LEVELS or len(set(self.concurrency)) != len(
            self.concurrency
        ):
            raise CapacityRefused(f"Give 1 to {MAX_LEVELS} distinct concurrency levels.")
        if any(not 1 <= c <= MAX_CONCURRENCY for c in self.concurrency):
            raise CapacityRefused(f"Each concurrency level must be 1 to {MAX_CONCURRENCY}.")
        if not 1 <= self.duration_s <= MAX_DURATION_S:
            raise CapacityRefused(f"duration must be 1 to {MAX_DURATION_S:g} seconds.")
        if not 0 < self.max_rps <= MAX_RPS_LIMIT:
            raise CapacityRefused(f"max_rps must be above 0 and at most {MAX_RPS_LIMIT:g}.")
        if self.k < 1 or self.serial_queries < 1 or self.recall_sample < 0:
            raise CapacityRefused("k and serial_queries must be >= 1 and recall_sample >= 0.")

    @property
    def level_request_ceiling(self) -> int:
        """The most one level can send."""
        return math.ceil(self.max_rps * self.duration_s) + 1

    def request_ceiling(self, exact_reference: bool) -> int:
        """The most the whole run can send, counting exact-reference lookups."""
        total = self.serial_queries + WARMUP + len(self.concurrency) * self.level_request_ceiling
        return total + (self.recall_sample if exact_reference else 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concurrency": list(self.concurrency),
            "duration_s": self.duration_s,
            "max_rps": self.max_rps,
            "k": self.k,
            "params": dict(self.params),
            "serial_queries": self.serial_queries,
            "recall_sample": self.recall_sample,
            "seed": self.seed,
        }


# --- safety ------------------------------------------------------------------------------


def target_name(info: BackendInfo) -> str | None:
    """The name of the thing under test (collection, index or table), or ``None``."""
    key = TARGET_KEYS.get(info.backend)
    value = info.target.get(key) if key else None
    return value if isinstance(value, str) and value else None


def safety_problems(
    info: BackendInfo,
    connection_name: str | None,
    *,
    confirm_target: str | None,
    allow_production: bool,
) -> list[str]:
    """Why the run may not start (empty when it may). Deciding this sends no search."""
    target = target_name(info)
    problems = []
    if target is None:
        problems.append(
            f"The {info.backend} connection does not name a target "
            "(collection, index or table), so there is nothing to confirm."
        )
    elif confirm_target is None:
        problems.append(
            f"Pass --confirm-target {target} to confirm that {info.backend} target is the one "
            "to load."
        )
    elif confirm_target != target:
        problems.append(
            f"--confirm-target {confirm_target!r} does not match the connection's target "
            f"{target!r}."
        )
    names = [n for n in (target, connection_name) if n]
    flagged = [n for n in names if "prod" in n.lower()]
    if flagged and not allow_production:
        problems.append(
            f"{flagged[0]!r} looks like production. A capacity run is for a staging target; pass "
            "--allow-production only if you are sure."
        )
    return problems


def describe_plan(
    plan: CapacityPlan, info: BackendInfo, caps: Capabilities, n_queries: int
) -> list[str]:
    """What will be sent, as lines to print before anything is."""
    lines = [
        f"Backend {info.backend}, target {target_name(info)!r}: a read-only search of "
        f"{n_queries} dataset queries, k={plan.k}, repeated.",
        f"1. Serial: {plan.serial_queries} queries one after another.",
        f"2. {len(plan.concurrency)} concurrency level(s) {list(plan.concurrency)}, "
        f"{plan.duration_s:g} s each, "
        f"at most {plan.max_rps:g} requests per second in total.",
        f"The most this can send: {plan.request_ceiling(caps.exact_reference):,} requests"
        + (
            f" (including up to {plan.recall_sample} exact-reference searches)."
            if caps.exact_reference
            else "; no exact reference, so recall is not measured."
        ),
    ]
    return lines


# --- measurement -------------------------------------------------------------------------


def _percentiles(latencies_ms: Sequence[float]) -> dict[str, float | None]:
    if not latencies_ms:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    x = np.asarray(latencies_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(x, [50, 95, 99])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "mean_ms": float(x.mean()),
    }


@dataclass
class _Tally:
    """What the workers of one level recorded."""

    latencies_ms: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)
    samples: dict[int, list[str]] = field(default_factory=dict)
    fatal: ConnectorError | None = None

    def merge(self, other: _Tally) -> None:
        self.latencies_ms += other.latencies_ms
        for code, n in other.errors.items():
            self.errors[code] = self.errors.get(code, 0) + n
        for index, ids in other.samples.items():
            self.samples.setdefault(index, ids)
        self.fatal = self.fatal or other.fatal


def _worker(
    connector: Connector,
    requests: Sequence[SearchRequest],
    order: Sequence[int],
    start: int,
    stride: int,
    limiter: RateLimiter,
    deadline: float,
    stop: threading.Event,
    sampled: frozenset[int],
    max_calls: int | None,
) -> _Tally:
    tally = _Tally()
    i = start
    calls = 0
    while not stop.is_set() and (max_calls is None or calls < max_calls):
        delay = limiter.reserve()
        if time.monotonic() + delay >= deadline:
            break
        if delay and stop.wait(delay):
            break
        index = order[i % len(order)]
        i += stride
        calls += 1
        t0 = time.perf_counter()
        try:
            ids = connector.search(requests[index])
        except ConnectorError as e:
            tally.errors[e.code.value] = tally.errors.get(e.code.value, 0) + 1
            if e.code in _FATAL:
                tally.fatal = e
                stop.set()
            continue
        tally.latencies_ms.append((time.perf_counter() - t0) * 1000)
        if index in sampled and index not in tally.samples:
            tally.samples[index] = ids
    return tally


def _run_load(
    connectors: Sequence[Connector],
    requests: Sequence[SearchRequest],
    order: Sequence[int],
    limiter: RateLimiter,
    duration_s: float,
    sampled: frozenset[int],
    max_calls: int | None = None,
) -> tuple[_Tally, float]:
    """Run one worker per connector until the deadline, the call cap or a fatal error."""
    stop = threading.Event()
    started = time.monotonic()
    deadline = started + duration_s
    tallies: list[_Tally] = []
    n = len(connectors)

    def target(c: Connector, w: int) -> None:
        tallies.append(
            _worker(c, requests, order, w, n, limiter, deadline, stop, sampled, max_calls)
        )

    if n == 1:
        target(connectors[0], 0)
    else:
        threads = [threading.Thread(target=target, args=(c, w)) for w, c in enumerate(connectors)]
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            stop.set()
            for t in threads:
                t.join()
            raise
    elapsed = time.monotonic() - started
    total = _Tally()
    for part in tallies:
        total.merge(part)
    return total, elapsed


@dataclass(frozen=True)
class _Workload:
    requests: list[SearchRequest]
    order: list[int]
    sampled: frozenset[int]


def _workload(
    dataset: Dataset, vectors: Sequence[Any], plan: CapacityPlan, exact_reference: bool
) -> _Workload:
    requests = [
        SearchRequest(
            v,
            plan.k,
            filter=cast(Mapping[str, FilterValue], q.filter),
            params=plan.params,
            exact=False,
        )
        for q, v in zip(dataset.queries, vectors, strict=True)
    ]
    order = list(range(len(requests)))
    random.Random(plan.seed).shuffle(order)
    wanted = plan.recall_sample if exact_reference else 0
    step = max(1, len(order) // wanted) if wanted else 1
    sampled = frozenset(order[::step][:wanted]) if wanted else frozenset()
    return _Workload(requests, order, sampled)


def _recall(
    connector: Connector,
    work: _Workload,
    samples: Mapping[int, list[str]],
    exact_cache: dict[int, list[str]],
    limiter: RateLimiter,
) -> dict[str, Any]:
    """Recall of the answers kept during a level against the exact reference, measured now."""
    if not work.sampled:
        return {"basis": "none: no exact reference", "sample": 0, "mean": None}
    recalls = []
    for index, ids in samples.items():
        if index not in exact_cache:
            req = work.requests[index]
            exact = SearchRequest(req.vector, req.k, req.filter, req.params, exact=True)
            if delay := limiter.reserve():  # the ceiling covers these searches too
                time.sleep(delay)
            try:
                exact_cache[index] = connector.search(exact)
            except ConnectorError as e:
                if e.code in _FATAL:
                    raise
                continue
        r = ann_recall(ids, exact_cache[index])
        if r is not None:
            recalls.append(r)
    return {
        "basis": "exact reference, measured after the level, not under load",
        "sample": len(recalls),
        "mean": statistics.fmean(recalls) if recalls else None,
    }


def run_capacity(
    dataset: Dataset,
    open_worker: Callable[[], Connector],
    plan: CapacityPlan,
    *,
    connection: Mapping[str, Any] | None = None,
    embed: Callable[[Sequence[str]], Any] | None = None,
    embedder_name: str | None = None,
    share_connector: bool | None = None,
    safety: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Measure the curve and return the ``vectraudit.capacity`` record. Sends only what the
    plan allows; never raises for a backend failure (the record says what happened).

    ``open_worker`` opens a connector; it is called once per worker thread unless the
    connector declares itself ``thread_safe`` (or ``share_connector`` says so), in which case
    the first connector serves every worker.
    """
    say = progress or (lambda _m: None)
    record: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "failed",
        "error": None,
        "dataset": dataset.summary(),
        "connection": dict(connection or {}),
        "plan": plan.as_dict(),
        "embedder": embedder_name,
        "safety": dict(safety) if safety is not None else None,
        "runner": runner_environment(),
        "latency_scope": "measured by this process around the connector call (client side)",
        "backend": None,
        "capabilities": None,
        "serial": None,
        "levels": [],
        "notes": [],
        "comparison": None,
    }
    opened: list[Connector] = []

    def connectors(n: int) -> list[Connector]:
        if not opened:
            opened.append(open_worker())
        shared = opened[0].thread_safe if share_connector is None else share_connector
        while len(opened) < (1 if shared else n):
            opened.append(open_worker())
        return [opened[0]] * n if shared else opened[:n]

    try:
        main = connectors(1)[0]
        info, caps = main.describe(), main.capabilities()
        record["backend"], record["capabilities"] = info.as_dict(), caps.as_dict()
        if any(q.context or q.history for q in dataset.queries):
            raise ConnectorError(
                ErrorCode.UNSUPPORTED,
                "Permission contexts and conversation history need a full-pipeline adapter.",
            )
        if any(isinstance(v, (list, dict)) for q in dataset.queries for v in q.filter.values()):
            raise ConnectorError(
                ErrorCode.UNSUPPORTED, "List/range filters require a full-pipeline adapter."
            )
        vectors, _ = _query_vectors(dataset, embed, info.dim)
        work = _workload(dataset, vectors, plan, caps.exact_reference)
        limiter = RateLimiter(plan.max_rps)
        no_samples: frozenset[int] = frozenset()

        say("Warming up")
        _run_load([main], work.requests, work.order, limiter, 30.0, no_samples, max_calls=WARMUP)
        say(f"Serial: {plan.serial_queries} queries")
        serial, elapsed = _run_load(
            [main], work.requests, work.order, limiter, MAX_DURATION_S, no_samples,
            max_calls=plan.serial_queries,
        )  # fmt: skip
        if serial.fatal:
            raise serial.fatal
        record["serial"] = {
            "requests": len(serial.latencies_ms) + sum(serial.errors.values()),
            "errors": dict(serial.errors),
            **_percentiles(serial.latencies_ms),
        }
        exact_cache: dict[int, list[str]] = {}
        for level in plan.concurrency:
            say(f"Concurrency {level}: {plan.duration_s:g} s")
            workers = connectors(level)
            tally, elapsed = _run_load(
                workers, work.requests, work.order, limiter, plan.duration_s, work.sampled
            )
            if tally.fatal:
                raise tally.fatal
            attempts = len(tally.latencies_ms) + sum(tally.errors.values())
            row = {
                "concurrency": level,
                "duration_s": elapsed,
                "requests": attempts,
                "errors": dict(tally.errors),
                "error_rate": (sum(tally.errors.values()) / attempts) if attempts else None,
                "achieved_qps": (len(tally.latencies_ms) / elapsed) if elapsed else None,
                "attempts_per_s": (attempts / elapsed) if elapsed else None,
                **_percentiles(tally.latencies_ms),
                "at_ceiling": bool(attempts and attempts / elapsed >= _AT_CEILING * plan.max_rps),
                "recall": _recall(main, work, tally.samples, exact_cache, limiter),
            }
            record["levels"].append(row)
        record["status"] = "complete"
    except ConnectorError as e:
        record["error"] = e.as_dict()
    except KeyboardInterrupt:
        record["status"] = "cancelled"
    except Exception as e:  # a connector bug must still leave a record
        record["error"] = {"code": "internal", "message": f"{type(e).__name__}: {e}", "hint": None}
    finally:
        for c in dict.fromkeys(opened):
            with contextlib.suppress(Exception):  # closing must not hide the result
                c.close()
    record["notes"] = _notes(record, plan)
    return record


def _notes(record: Mapping[str, Any], plan: CapacityPlan) -> list[str]:
    notes = [
        "Client-side latency on this machine: it is comparable with another capacity report only "
        "when the runner environments match."
    ]
    if any(level["at_ceiling"] for level in record["levels"]):
        notes.append(
            f"Levels marked at_ceiling reached the {plan.max_rps:g} requests/s ceiling: the "
            "backend kept up, so its limit is at least that high, not exactly that."
        )
    if record["status"] == "cancelled":
        notes.append("Cancelled: only the levels completed so far are reported.")
    if any(level["recall"]["mean"] is None for level in record["levels"]):
        notes.append("Recall is null where no exact reference or no sampled answer was available.")
    return notes


# --- comparison --------------------------------------------------------------------------


def compare_capacity(base: Mapping[str, Any], cand: Mapping[str, Any]) -> dict[str, Any]:
    """Level-by-level changes between two capacity reports, or why they cannot be compared.

    Comparable only when both completed, ran on matching runner environments (the rule the
    release comparison applies to latency), and measured the same dataset, backend, k,
    search parameters and request ceiling. Otherwise ``comparable`` is false and no deltas are
    given: unknown is not a pass.
    """
    reasons: list[str] = []
    for name, report in (("baseline", base), ("current", cand)):
        if report.get("format") != FORMAT or report.get("format_version") != FORMAT_VERSION:
            reasons.append(f"The {name} is not a {FORMAT} v{FORMAT_VERSION} report.")
        elif report.get("status") != "complete":
            reasons.append(f"The {name} run did not complete ({report.get('status')}).")
    if reasons:
        return {"comparable": False, "reasons": reasons, "levels": []}
    for key in environment_differences((base.get("runner") or {}), (cand.get("runner") or {})):
        reasons.append(f"Runner environments differ in {key}: capacity is machine dependent.")
    if (base.get("dataset") or {}).get("sha256") != (cand.get("dataset") or {}).get("sha256"):
        reasons.append("The datasets differ.")
    if (base.get("backend") or {}).get("backend") != (cand.get("backend") or {}).get("backend"):
        reasons.append("The backends differ.")
    bp, cp = base["plan"], cand["plan"]
    for key in ("k", "params", "max_rps"):
        if bp.get(key) != cp.get(key):
            reasons.append(f"The plans differ in {key}.")
    if reasons:
        return {"comparable": False, "reasons": reasons, "levels": []}
    cur = {level["concurrency"]: level for level in cand["levels"]}
    levels = []
    for old in base["levels"]:
        new = cur.get(old["concurrency"])
        if new is None:
            continue
        levels.append(
            {
                "concurrency": old["concurrency"],
                **{
                    f"{key}": [old[key], new[key]]
                    for key in ("achieved_qps", "p50_ms", "p95_ms", "p99_ms", "error_rate")
                },
                "recall": [old["recall"]["mean"], new["recall"]["mean"]],
            }
        )
    if not levels:
        return {
            "comparable": False,
            "reasons": ["The two reports share no concurrency level."],
            "levels": [],
        }
    return {"comparable": True, "reasons": [], "levels": levels}
