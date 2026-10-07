"""Local support search: actual BM25 ranking over a versioned fictional corpus."""

from __future__ import annotations

import json
from pathlib import Path

from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.models import PipelineRequest, PipelineResult

# The case-study runner creates independent source revisions by changing only this value.
SEARCH_MODE = "baseline"
DOCUMENTS = [
    json.loads(line)
    for line in Path(__file__).with_name("corpus.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
INDEX = hybrid_pipeline(DOCUMENTS)


def retrieve(request: PipelineRequest) -> PipelineResult:
    """Apply tenant/current-document restrictions before or after candidate selection."""
    if SEARCH_MODE not in {"baseline", "candidate", "fixed"}:
        raise ValueError("Unknown reference-search revision")
    required = dict(request.filters)
    if "tenant" in request.context:
        required["tenant"] = request.context["tenant"]
    pipeline = INDEX
    if SEARCH_MODE == "fixed":
        permitted = [
            doc
            for doc in DOCUMENTS
            if all(doc["metadata"].get(key) == value for key, value in required.items())
        ]
        if not permitted:
            return PipelineResult(abstained=True)
        # Tiny reference implementation: rebuilding is included in end-to-end latency.
        # A production backend would normally use its native pre-filtered index search.
        pipeline = hybrid_pipeline(permitted)
    depth = len(DOCUMENTS) if SEARCH_MODE == "baseline" else request.k
    result = pipeline(
        request.model_copy(
            update={
                "params": {"depth": max(request.k, depth), "vector_weight": 0, "reranker": "none"},
            }
        )
    )
    return result.model_copy(update={"abstained": not result.hits})
