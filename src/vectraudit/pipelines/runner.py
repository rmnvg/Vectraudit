"""Full-pipeline runs compatible with release checks, plus recorded stage evidence."""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.core.jobs import JobCancelledError
from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.evaluation.metrics import repeat_quality
from vectraudit.evaluation.runner import RunOptions, _skeleton, summarize
from vectraudit.labelling.storage import task_lock
from vectraudit.pipelines.budget import usage
from vectraudit.pipelines.checkpoint import Checkpoint, identity, validate_path
from vectraudit.pipelines.errors import RETRYABLE, PipelineCallError, run_failure
from vectraudit.pipelines.models import PipelineRequest, PipelineResult
from vectraudit.pipelines.runtime import Pipeline


@dataclass(frozen=True)
class RetryPolicy:
    """Automatic retries of transient call failures (:data:`RETRYABLE` codes only).

    Each pipeline call (warm-up or repeat) gets up to ``max_attempts`` tries, waiting
    ``base_delay_s * 2**n`` between them (capped at ``max_delay_s``), or longer when the
    endpoint sends ``Retry-After``. A ``Retry-After`` above ``max_delay_s`` is not waited for:
    the call fails with its cause. ``max_retries`` bounds retries across one
    run invocation, so an endpoint that is down fails quickly instead of stalling CI. A query
    whose call still fails stays failed: retries never turn missing evidence into a pass.
    """

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    max_retries: int = 100

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if not 0 <= self.base_delay_s <= self.max_delay_s <= 300:
            raise ValueError("Require 0 <= base_delay_s <= max_delay_s <= 300 seconds")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

    def delay(self, retry: int, requested: float | None) -> float | None:
        """Seconds before retry number ``retry`` (0-based); ``None`` if beyond the cap."""
        if requested is not None and requested > self.max_delay_s:
            return None
        return min(max(self.base_delay_s * 2.0**retry, requested or 0.0), self.max_delay_s)


NO_RETRY = RetryPolicy(max_attempts=1)


@dataclass(frozen=True)
class PipelineOptions(RunOptions):
    """Pipeline parameters may include named rerankers and other nonnumeric choices."""

    params: Mapping[str, Any] = field(default_factory=dict)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    concurrency: int = 1
    """Queries in flight at once. Above 1 the adapter must be safe to call from threads."""
    max_rps: float | None = None
    """Upper bound on pipeline calls per second (retries and warm-ups included)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 1 <= self.concurrency <= 32:
            raise ValueError("concurrency must be between 1 and 32")
        if self.max_rps is not None and not 0 < self.max_rps <= 1000:
            raise ValueError("max_rps must be in (0, 1000]")


class _Shared:
    """State shared by every query of one run invocation, safe across worker threads."""

    def __init__(self, policy: RetryPolicy, max_rps: float | None) -> None:
        self.policy = policy
        self._lock = threading.Lock()
        self._retries_left = policy.max_retries
        self._interval = 1 / max_rps if max_rps else 0.0
        self._next_call = 0.0

    def take_retry(self) -> bool:
        """Spend one retry from the run budget; False once it is exhausted."""
        with self._lock:
            if self._retries_left <= 0:
                return False
            self._retries_left -= 1
            return True

    def throttle(self, alive: Callable[[], None]) -> None:
        """Reserve the next call slot under ``max_rps`` and wait for it."""
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_call)
            self._next_call = slot + self._interval
        wait(slot - now, alive)


def wait(seconds: float, alive: Callable[[], None]) -> None:
    """Sleep in short slices; ``alive`` raises when the run is cancelled."""
    deadline = time.monotonic() + seconds
    while (left := deadline - time.monotonic()) > 0:
        alive()
        time.sleep(min(left, 0.2))


@dataclass
class _Calls:
    """Invoke one query's pipeline calls, retrying transient failures within the budget."""

    pipeline: Pipeline
    request: PipelineRequest
    shared: _Shared
    alive: Callable[[], None]
    retries: list[dict[str, Any]] = field(default_factory=list)
    made: int = 0
    """Every invocation, including warm-ups, retries and calls that failed."""
    failed: int = 0
    priced: list[float] = field(default_factory=list)
    """Adapter-reported ``cost_usd`` of successful calls that reported one."""

    def usage(self) -> dict[str, Any]:
        """What this query actually consumed. Failed and unpriced calls have no known cost."""
        return {
            "calls": self.made,
            "failed_calls": self.failed,
            "spend_usd": round(math.fsum(self.priced), 9),
            "unpriced_calls": self.made - len(self.priced),
        }

    def __call__(self, label: str) -> tuple[Any, float]:
        """Return the call's result and its start time; re-raise once retries are spent."""
        policy = self.shared.policy
        tries = 0
        while True:
            self.shared.throttle(self.alive)
            t0 = time.perf_counter()
            self.made += 1
            try:
                raw = self.pipeline(self.request.model_copy(deep=True))
            except BaseException as exc:
                self.failed += 1
                if not isinstance(exc, PipelineCallError):
                    raise
                tries += 1
                delay = (
                    policy.delay(tries - 1, exc.retry_after)
                    if exc.code in RETRYABLE and tries < policy.max_attempts
                    else None
                )
                if delay is None or not self.shared.take_retry():
                    raise
                self.retries.append({"call": label, "code": exc.code, "delay_s": delay})
                wait(delay, self.alive)
                continue
            cost = getattr(raw, "cost_usd", None)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
                self.priced.append(float(cost))
            return raw, t0


def failure_analysis(result: PipelineResult, relevant: dict[str, float]) -> list[dict[str, Any]]:
    """Locate observed losses on explicit edges, without guessing hidden causes."""
    delivered = {h.id for h in result.hits}
    stages = {s.name: s for s in result.stages}
    findings: list[dict[str, Any]] = []
    for doc, grade in relevant.items():
        if grade <= 0 or doc in delivered:
            continue
        appeared = False
        for stage in result.stages:
            if doc in {h.id for h in stage.hits}:
                appeared = True
                continue
            parents = [p for p in stage.parents if doc in {h.id for h in stages[p].hits}]
            if parents and (result.coverage is None or result.coverage.stage_trace == "observed"):
                findings.append(
                    {
                        "doc_id": doc,
                        "stage": stage.name,
                        "kind": stage.kind,
                        "parents": parents,
                        "evidence": stage.source,
                        "reason": "present in parent output; absent from this stage",
                    }
                )
        if not appeared:
            findings.append(
                {
                    "doc_id": doc,
                    "stage": None,
                    "kind": "not_observed",
                    "evidence": "unavailable" if not result.stages else "recorded",
                    "reason": "not present in recorded stages; internal cause unknown",
                }
            )
    return findings


def context_analysis(result: PipelineResult, relevant: dict[str, float]) -> dict[str, Any] | None:
    """What happened to relevant delivered documents between retrieval and the model.

    ``losses`` lists documents left out (only when coverage is ``observed``) or cut short;
    ``unknown`` lists documents whose fate the recorded context cannot show. ``None`` when
    the pipeline recorded no model context at all.
    """
    context = result.context
    if context is None:
        return None
    items = {item.id: item for item in context.items}
    losses: list[dict[str, Any]] = []
    unknown: list[str] = []
    for hit in result.hits:
        if relevant.get(hit.id, 0) <= 0:
            continue
        item = items.get(hit.id)
        if item is None and context.coverage == "observed":
            losses.append(
                {
                    "doc_id": hit.id,
                    "kind": "omitted",
                    "reason": "delivered by retrieval; absent from the model context",
                }
            )
        elif item is None:
            unknown.append(hit.id)
        elif item.truncated:
            losses.append(
                {
                    "doc_id": hit.id,
                    "kind": "truncated",
                    "reason": "only part of the document reached the model",
                }
            )
    return {
        "coverage": context.coverage,
        "method": context.method,
        "reached": [
            item.id for item in context.items if relevant.get(item.id, 0) > 0 and not item.truncated
        ],
        "losses": losses,
        "unknown": unknown,
    }


def evidence_sufficiency(
    groups: Sequence[Sequence[str]], result: PipelineResult
) -> dict[str, Any] | None:
    """Whether a complete required group was delivered, and whether it reached the model.

    ``groups`` are the question's alternatives (any one complete group suffices). Delivered
    is ``sufficient`` or ``insufficient``. Context is ``sufficient`` when every document of
    some group reached the model uncut, ``insufficient`` when every group lost a document
    (left out under observed coverage, or cut short), ``unknown`` otherwise, and
    ``unrecorded`` without a context record. ``None`` when the question declares no groups.
    """
    if not groups:
        return None
    delivered = {h.id for h in result.hits}
    closest = min(groups, key=lambda g: (len(set(g) - delivered), list(g)))
    out: dict[str, Any] = {
        "groups": len(groups),
        "delivered": "sufficient" if any(set(g) <= delivered for g in groups) else "insufficient",
        "missing_from_delivered": sorted(set(closest) - delivered),
    }
    context = result.context
    if context is None:
        out["context"] = "unrecorded"
        return out
    items = {item.id: item for item in context.items}

    def state(group: Sequence[str]) -> str:
        if any(
            (doc not in items and context.coverage == "observed")
            or (doc in items and items[doc].truncated is True)
            for doc in group
        ):
            return "broken"
        if all(doc in items and items[doc].truncated is False for doc in group):
            return "complete"
        return "unknown"

    states = [state(g) for g in groups]
    out["context"] = (
        "sufficient"
        if "complete" in states
        else "insufficient"
        if all(st == "broken" for st in states)
        else "unknown"
    )
    return out


def execute_pipeline(
    dataset: Dataset,
    pipeline: Pipeline,
    options: RunOptions | None = None,
    *,
    config: dict[str, Any] | None = None,
    corpus_version: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    checkpoint: Path | None = None,
    resume: bool = False,
    retry_errors: bool = False,
) -> dict[str, Any]:
    """Run text/vector requests locally, recording failures without inventing scores."""
    options = options or RunOptions(warmup=0)
    if retry_errors and not resume:
        raise ValueError("Retrying errors requires resume")
    if resume and checkpoint is None:
        raise ValueError("Resume requires a checkpoint")
    if checkpoint is None:
        return _execute(dataset, pipeline, options, config, corpus_version, progress, None, False)
    if (
        resume
        and (config or {}).get("adapter") != "hybrid"
        and (not corpus_version or not (config or {}).get("pipeline_version"))
    ):
        raise ValueError("Resuming custom adapters requires corpus_version and pipeline_version")
    validate_path(checkpoint)
    with task_lock(checkpoint, timeout=0):
        journal = Checkpoint(
            checkpoint, identity(dataset, pipeline, options, config, corpus_version), resume=resume
        )
        try:
            return _execute(
                dataset, pipeline, options, config, corpus_version, progress, journal, retry_errors
            )
        finally:
            journal.close()


def _execute(
    dataset: Dataset,
    pipeline: Pipeline,
    options: RunOptions,
    config: dict[str, Any] | None,
    corpus_version: str | None,
    progress: Callable[[int, int], None] | None,
    journal: Checkpoint | None,
    retry_errors: bool,
) -> dict[str, Any]:
    record, shared, workers = _open_record(
        dataset, pipeline, options, config, corpus_version, journal
    )
    start = time.perf_counter()
    results: dict[int, dict[str, Any]] = {}
    try:
        if workers == 1:
            _serial(dataset, pipeline, options, shared, progress, journal, retry_errors, results)
        else:
            _concurrent(
                dataset,
                pipeline,
                options,
                shared,
                workers,
                progress,
                journal,
                retry_errors,
                results,
            )
        if progress is not None:
            progress(len(dataset), len(dataset))
        status = "partial" if any(e["error"] for e in results.values()) else "complete"
    except (KeyboardInterrupt, JobCancelledError):
        status = "cancelled"
    return _close_record(record, results, status, start, dataset, journal)


def _open_record(
    dataset: Dataset,
    pipeline: Pipeline,
    options: RunOptions,
    config: dict[str, Any] | None,
    corpus_version: str | None,
    journal: Checkpoint | None,
) -> tuple[dict[str, Any], _Shared, int]:
    """A run record's header, before any query runs; returns it with its shared state."""
    record = (journal.record if journal else None) or _skeleton(dataset, options, None, None)
    record["queries"] = []
    record["status"] = "partial"
    record["backend"] = {
        "name": "pipeline",
        "backend": "pipeline",
        "snapshot": corpus_version,
        "config": config or {},
        "corpus_version": corpus_version,
    }
    record["capabilities"] = {
        "exact_reference": False,
        "stage_traces_supported": True,
        "stage_traces": False,
    }
    policy = options.retry if isinstance(options, PipelineOptions) else NO_RETRY
    workers = options.concurrency if isinstance(options, PipelineOptions) else 1
    max_rps = options.max_rps if isinstance(options, PipelineOptions) else None
    if workers > 1 and getattr(pipeline, "thread_safe", True) is False:
        raise ValueError("This adapter declares thread_safe = False; use concurrency 1")
    record["timing"]["scope"] = "complete pipeline call; client wall clock; " + (
        "serial queries" if workers == 1 else f"{workers} concurrent queries"
    )
    record["timing"]["concurrency"] = workers
    record["timing"]["max_rps"] = max_rps
    record["timing"]["retry"] = {
        **asdict(policy),
        "retryable": sorted(RETRYABLE),
        "latency": "measured on the successful call only",
    }
    if journal:
        journal.put("record", record)
    return record, _Shared(policy, max_rps), workers


def _close_record(
    record: dict[str, Any],
    results: dict[int, dict[str, Any]],
    status: str,
    start: float,
    dataset: Dataset,
    journal: Checkpoint | None,
) -> dict[str, Any]:
    """Attach the query results, summary and usage once a run stops."""
    record["queries"] = [results[i] for i in sorted(results)]
    record["status"] = status
    record["timing"]["wall_s"] = time.perf_counter() - start
    if journal:
        record["timing"]["scope"] += (
            "; wall_s covers this invocation; query samples retain original timing"
        )
    record["capabilities"]["stage_traces"] = any(
        (q.get("evidence") or {}).get("stages") for q in record["queries"]
    )
    record["summary"] = summarize(record["queries"], len(dataset))
    if journal:
        for entry in record["queries"]:
            entry["attempts"] = journal.history(entry["query_id"])
    record["usage"] = usage(record["queries"])
    if journal:
        journal.put("record", record)
    return record


@dataclass(frozen=True)
class PairedSide:
    """One side of a paired execution: the pipeline and how it was configured."""

    pipeline: Pipeline
    options: RunOptions
    config: dict[str, Any] | None = None
    corpus_version: str | None = None


def execute_paired(
    dataset: Dataset,
    first: PairedSide,
    second: PairedSide,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run two pipelines over the same questions, interleaved, and return both run records.

    Each question runs on both sides back to back, and the side that goes first alternates
    (A then B, then B then A), so drift in the endpoint, network or caches during the run
    falls on both sides alike instead of on whichever ran second. Both runs are serial: the
    pairing is per question. Each record names the other in ``timing.paired_with`` and
    ``timing.order`` is ``interleaved``. Checkpoints and resume are not supported here.
    """
    for side in (first, second):
        workers = side.options.concurrency if isinstance(side.options, PipelineOptions) else 1
        if workers != 1:
            raise ValueError("Paired execution runs one question at a time; use concurrency 1")
        if side.options.repeats != first.options.repeats or side.options.k != first.options.k:
            raise ValueError("Paired runs need the same k and repeats on both sides")
    sides: list[tuple[PairedSide, dict[str, Any], _Shared, dict[int, dict[str, Any]]]] = []
    for side in (first, second):
        record, shared, _ = _open_record(
            dataset, side.pipeline, side.options, side.config, side.corpus_version, None
        )
        sides.append((side, record, shared, {}))
    a, b = sides
    for record, other in ((a[1], b[1]), (b[1], a[1])):
        record["timing"]["order"] = "interleaved"
        record["timing"]["paired_with"] = other["run_id"]
        record["timing"]["scope"] += "; interleaved with a paired run, alternating which goes first"
    start = time.perf_counter()
    status = "complete"
    try:
        for index, q in enumerate(dataset.queries):
            if progress is not None:
                progress(index, len(dataset))
            order = (a, b) if index % 2 == 0 else (b, a)
            for side, _record, shared, results in order:

                def alive(index: int = index) -> None:
                    if progress is not None:
                        progress(index, len(dataset))

                entry = _measure(q, index, side.pipeline, side.options, shared, alive)
                _finish(entry, 1, None)
                results[index] = entry
        if progress is not None:
            progress(len(dataset), len(dataset))
    except (KeyboardInterrupt, JobCancelledError):
        status = "cancelled"
    out = []
    for _side, record, _shared, results in sides:
        side_status = status
        if status == "complete" and any(e["error"] for e in results.values()):
            side_status = "partial"
        out.append(_close_record(record, results, side_status, start, dataset, None))
    return out[0], out[1]


def _reuse(journal: Checkpoint | None, query_id: str, retry_errors: bool) -> dict[str, Any] | None:
    """A saved result to keep instead of calling the pipeline again, if there is one."""
    previous = journal.latest(query_id) if journal else None
    if previous is not None and (not previous["error"] or not retry_errors):
        return previous
    return None


def _serial(
    dataset: Dataset,
    pipeline: Pipeline,
    options: RunOptions,
    shared: _Shared,
    progress: Callable[[int, int], None] | None,
    journal: Checkpoint | None,
    retry_errors: bool,
    results: dict[int, dict[str, Any]],
) -> None:
    for index, q in enumerate(dataset.queries):
        if progress is not None:
            progress(index, len(dataset))
        if (previous := _reuse(journal, q.query_id, retry_errors)) is not None:
            results[index] = previous
            continue
        attempt = journal.begin(q.query_id) if journal else 1

        def alive(index: int = index) -> None:
            if progress is not None:
                progress(index, len(dataset))

        entry = _measure(q, index, pipeline, options, shared, alive)
        _finish(entry, attempt, journal)
        results[index] = entry


def _concurrent(
    dataset: Dataset,
    pipeline: Pipeline,
    options: RunOptions,
    shared: _Shared,
    workers: int,
    progress: Callable[[int, int], None] | None,
    journal: Checkpoint | None,
    retry_errors: bool,
    results: dict[int, dict[str, Any]],
) -> None:
    """Keep at most ``workers`` queries in flight; journal writes stay on this thread."""
    stop = threading.Event()

    def alive() -> None:
        if stop.is_set():
            raise JobCancelledError("Run cancelled.")

    queue: deque[tuple[int, DatasetQuery]] = deque()
    for index, q in enumerate(dataset.queries):
        if (previous := _reuse(journal, q.query_id, retry_errors)) is not None:
            results[index] = previous
        else:
            queue.append((index, q))
    pending: dict[Future[dict[str, Any]], tuple[int, int]] = {}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vectraudit-query")
    try:
        while queue or pending:
            if progress is not None:
                progress(len(results), len(dataset))
            while queue and len(pending) < workers:
                index, q = queue.popleft()
                attempt = journal.begin(q.query_id) if journal else 1
                future = pool.submit(_measure, q, index, pipeline, options, shared, alive)
                pending[future] = (index, attempt)
            finished, _ = futures_wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in finished:
                index, attempt = pending.pop(future)
                entry = future.result()
                _finish(entry, attempt, journal)
                results[index] = entry
    except BaseException:
        # Stop retry waits and new calls; calls already running cannot be interrupted, so
        # wait for them and keep whatever finished. Unfinished attempts stay "interrupted".
        stop.set()
        for future, (index, attempt) in pending.items():
            try:
                entry = future.result()
            except JobCancelledError:
                continue
            _finish(entry, attempt, journal)
            results[index] = entry
        raise
    finally:
        pool.shutdown(wait=True)


def _finish(entry: dict[str, Any], attempt: int, journal: Checkpoint | None) -> None:
    entry["attempt_count"] = attempt
    if journal:
        journal.finish(entry, attempt)


def _measure(
    q: DatasetQuery,
    index: int,
    pipeline: Pipeline,
    options: RunOptions,
    shared: _Shared,
    alive: Callable[[], None],
) -> dict[str, Any]:
    """Execute one query's calls and build its record; never touches the journal."""
    query_start = time.perf_counter()
    request = PipelineRequest(
        query_id=q.query_id,
        text=q.text,
        vector=None if q.vector is None else q.vector.tolist(),
        k=options.k,
        filters=q.filter,
        context=q.context,
        history=list(q.history),
        params=dict(options.params),
    )
    entry: dict[str, Any] = {
        "query_id": q.query_id,
        **({"family_id": q.family_id} if q.family_id is not None else {}),
        "tags": list(q.cohort_tags),
        "filtered": bool(q.filter),
        "ids": None,
        "exact_ids": None,
        "latency_ms": None,
        "metrics": None,
        "ann_recall": None,
        "nondeterministic": False,
        "evidence": None,
        "error": None,
    }
    rankings: list[list[str]] = []
    calls = _Calls(pipeline, request, shared, alive)
    try:
        if index < options.warmup:
            calls("warmup")
        samples, results = [], []
        for repeat in range(options.repeats):
            raw, t0 = calls(f"repeat {repeat + 1}")
            result = PipelineResult.model_validate(raw.model_dump())
            samples.append((time.perf_counter() - t0) * 1000)
            if len(result.hits) > options.k:
                raise ValueError("pipeline returned more than k results")
            results.append(result)
            rankings.append([h.id for h in result.hits])
            if any(r.coverage is not None or any(h.source for h in r.hits) for r in results):
                entry["observations"] = [
                    {
                        "format_version": 1,
                        "coverage": r.coverage.model_dump() if r.coverage else None,
                        "hits": [h.model_dump(exclude_none=True) for h in r.hits],
                    }
                    for r in results
                ]
            if result.coverage is not None and result.coverage.ranked_output != "observed":
                raise PipelineCallError(
                    "incomplete_evidence", "The final ranking was not fully observed"
                )
        result = results[0]
        ids = [h.id for h in result.hits]
        entry.update(
            ids=ids,
            repeat_ids=rankings,
            latency_ms=statistics.median(samples),
            latency_samples_ms=samples,
            **repeat_quality(rankings, q.relevant, options.k),
            nondeterministic=any(r.hits != result.hits for r in results[1:]),
            no_answer_correct=all(not r.hits for r in results) if q.no_answer else None,
            forbidden_ids=sorted({h.id for r in results for h in r.hits} & set(q.forbidden)),
            permission_checked=bool(q.forbidden),
            abstained=result.abstained,
            cost_usd=(
                statistics.fmean(r.cost_usd for r in results if r.cost_usd is not None)
                if all(r.cost_usd is not None for r in results)
                else None
            ),
            memory_bytes=(
                max(r.memory_bytes for r in results if r.memory_bytes is not None)
                if all(r.memory_bytes is not None for r in results)
                else None
            ),
            repeat_safety=[
                {
                    "forbidden_ids": sorted({h.id for h in r.hits} & set(q.forbidden)),
                    "no_answer_correct": not r.hits if q.no_answer else None,
                }
                for r in results
            ],
            evidence={
                "kind": "pipeline",
                "stages": [s.model_dump() for s in result.stages],
                "failures": failure_analysis(result, q.relevant),
                **(
                    {"context": context_analysis(result, q.relevant)}
                    if result.context is not None
                    else {}
                ),
            }
            if options.evidence
            else None,
        )
        if result.context is not None:
            # IDs and flags only; never text. Repeat 1, like the primary stage evidence.
            entry["model_context"] = result.context.model_dump(exclude_none=True)
        sufficiency = evidence_sufficiency(q.required_evidence, result)
        if sufficiency is not None:
            entry["sufficiency"] = sufficiency
        if options.evidence and len(results) > 1:
            # Store only changed stage outputs; unchanged repeats reference the first trace.
            # Ignore timing-only changes: end-to-end latency samples remain per execution.
            def trace_signature(r: PipelineResult) -> list[dict[str, Any]]:
                return [s.model_dump(exclude={"latency_ms"}) for s in r.stages]

            first = trace_signature(results[0])
            changed = {
                str(i): {
                    "kind": "pipeline",
                    "stages": [s.model_dump() for s in r.stages],
                    "failures": failure_analysis(r, q.relevant),
                    **(
                        {"context": context_analysis(r, q.relevant)}
                        if r.context is not None
                        else {}
                    ),
                }
                for i, r in enumerate(results[1:], 2)
                if trace_signature(r) != first
                or r.coverage != results[0].coverage
                or r.hits != results[0].hits
                or r.context != results[0].context
            }
            entry["repeat_evidence"] = changed
    except JobCancelledError:
        # Cancellation is not a pipeline failure: leave this attempt unfinished.
        raise
    except Exception as exc:
        # Do not persist exception text: user callables can expose secrets or queries.
        entry["error"] = run_failure(exc)
        # Rankings observed before the failure stay in the attempt history, so a
        # retry cannot hide a critical document that went missing.
        entry["repeat_ids"] = rankings or None
    if calls.retries:
        entry["retries"] = calls.retries
    entry["usage"] = calls.usage()
    entry["elapsed_ms"] = (time.perf_counter() - query_start) * 1000
    return entry
