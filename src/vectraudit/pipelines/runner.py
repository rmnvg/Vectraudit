"""Full-pipeline runs compatible with release checks, plus recorded stage evidence."""

from __future__ import annotations

import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.metrics import semantic_metrics
from vectraudit.evaluation.runner import RunOptions, _skeleton, summarize
from vectraudit.pipelines.models import PipelineRequest, PipelineResult
from vectraudit.pipelines.runtime import Pipeline


@dataclass(frozen=True)
class PipelineOptions(RunOptions):
    """Pipeline parameters may include named rerankers and other nonnumeric choices."""

    params: Mapping[str, Any] = field(default_factory=dict)


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
) -> dict[str, Any]:
    """Run text/vector requests locally, recording failures without inventing scores."""
    options = options or RunOptions(warmup=0)
    record = _skeleton(dataset, options, None, None)
    record["backend"] = {
        "name": "pipeline",
        "backend": "pipeline",
        "snapshot": corpus_version,
        "config": config or {},
        "corpus_version": corpus_version,
    }
    record["capabilities"] = {"exact_reference": False, "stage_traces": True}
    record["timing"]["scope"] = "complete pipeline call; client wall clock; serial queries"
    start = time.perf_counter()
    try:
        for index, q in enumerate(dataset.queries):
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
            try:
                if index < options.warmup:
                    pipeline(request.model_copy(deep=True))
                samples, results = [], []
                for _ in range(options.repeats):
                    t0 = time.perf_counter()
                    result = PipelineResult.model_validate(
                        pipeline(request.model_copy(deep=True)).model_dump()
                    )
                    samples.append((time.perf_counter() - t0) * 1000)
                    if len(result.hits) > options.k:
                        raise ValueError("pipeline returned more than k results")
                    results.append(result)
                result = results[0]
                ids = [h.id for h in result.hits]
                entry.update(
                    ids=ids,
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
            except Exception:
                # Do not persist exception text: user callables can expose secrets or queries.
                entry["error"] = {
                    "code": "pipeline_error",
                    "message": "Pipeline call or response validation failed.",
                    "hint": "Validate the adapter locally against PipelineRequest/PipelineResult.",
                }
            record["queries"].append(entry)
        record["status"] = "partial" if any(q["error"] for q in record["queries"]) else "complete"
    except KeyboardInterrupt:
        record["status"] = "cancelled"
    record["timing"]["wall_s"] = time.perf_counter() - start
    record["summary"] = summarize(record["queries"], len(dataset))
    return record
