"""Full-pipeline runs compatible with release checks, plus recorded stage evidence."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.core.jobs import JobCancelledError
from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.metrics import semantic_metrics
from vectraudit.evaluation.runner import RunOptions, _skeleton, summarize
from vectraudit.labelling.storage import task_lock
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


@dataclass
class _Calls:
    """Invoke one query's pipeline calls, retrying transient failures within the budget."""

    pipeline: Pipeline
    request: PipelineRequest
    policy: RetryPolicy
    budget: list[int]
    """Retries left in this run invocation, shared by every query."""
    retries: list[dict[str, Any]]
    progress: Callable[[int, int], None] | None
    index: int
    total: int

    def __call__(self, label: str) -> tuple[Any, float]:
        """Return the call's result and its start time; re-raise once retries are spent."""
        tries = 0
        while True:
            t0 = time.perf_counter()
            try:
                return self.pipeline(self.request.model_copy(deep=True)), t0
            except PipelineCallError as exc:
                tries += 1
                wait = (
                    self.policy.delay(tries - 1, exc.retry_after)
                    if exc.code in RETRYABLE and tries < self.policy.max_attempts
                    else None
                )
                if wait is None or self.budget[0] <= 0:
                    raise
                self.budget[0] -= 1
                self.retries.append({"call": label, "code": exc.code, "delay_s": wait})
                self._wait(wait)

    def _wait(self, seconds: float) -> None:
        # Sleep in short slices; progress callbacks raise when the job is cancelled.
        deadline = time.monotonic() + seconds
        while (left := deadline - time.monotonic()) > 0:
            if self.progress is not None:
                self.progress(self.index, self.total)
            time.sleep(min(left, 0.2))


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
            if parents:
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
    record["timing"]["scope"] = "complete pipeline call; client wall clock; serial queries"
    policy = options.retry if isinstance(options, PipelineOptions) else NO_RETRY
    record["timing"]["retry"] = {
        **asdict(policy),
        "retryable": sorted(RETRYABLE),
        "latency": "measured on the successful call only",
    }
    if journal:
        journal.put("record", record)
    start = time.perf_counter()
    budget = [policy.max_retries]
    try:
        for index, q in enumerate(dataset.queries):
            if progress is not None:
                progress(index, len(dataset))
            previous = journal.latest(q.query_id) if journal else None
            if previous is not None and (not previous["error"] or not retry_errors):
                record["queries"].append(previous)
                continue
            attempt = journal.begin(q.query_id) if journal else 1
            query_start = time.perf_counter()
            request = PipelineRequest(
                query_id=q.query_id,
                text=q.text,
                vector=None if q.vector is None else q.vector.tolist(),
                k=options.k,
                filters=q.filter,
                context=q.context,
                params=dict(options.params),
            )
            entry: dict[str, Any] = {
                "query_id": q.query_id,
                **({"family_id": q.family_id} if q.family_id is not None else {}),
                "tags": list(q.tags),
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
            retries: list[dict[str, Any]] = []
            calls = _Calls(
                pipeline, request, policy, budget, retries, progress, index, len(dataset)
            )
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
                result = results[0]
                ids = [h.id for h in result.hits]
                entry.update(
                    ids=ids,
                    repeat_ids=rankings,
                    latency_ms=statistics.median(samples),
                    latency_samples_ms=samples,
                    metrics=semantic_metrics(ids, q.relevant, options.k) if q.relevant else None,
                    nondeterministic=any(r.hits != result.hits for r in results[1:]),
                    no_answer_correct=all(not r.hits for r in results) if q.no_answer else None,
                    forbidden_ids=sorted(
                        {h.id for r in results for h in r.hits} & set(q.forbidden)
                    ),
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
                    }
                    if options.evidence
                    else None,
                )
            except Exception as exc:
                # Do not persist exception text: user callables can expose secrets or queries.
                entry["error"] = run_failure(exc)
                # Rankings observed before the failure stay in the attempt history, so a
                # retry cannot hide a critical document that went missing.
                entry["repeat_ids"] = rankings or None
            if retries:
                entry["retries"] = retries
            entry["attempt_count"] = attempt
            entry["elapsed_ms"] = (time.perf_counter() - query_start) * 1000
            if journal:
                journal.finish(entry, attempt)
            record["queries"].append(entry)
        if progress is not None:
            progress(len(dataset), len(dataset))
        record["status"] = "partial" if any(q["error"] for q in record["queries"]) else "complete"
    except (KeyboardInterrupt, JobCancelledError):
        record["status"] = "cancelled"
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
        journal.put("record", record)
    return record
