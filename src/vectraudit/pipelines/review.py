"""Read-only passage pools from measured exploratory runs of a saved benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import Field

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.runfile import load_run
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.models import Contract

if TYPE_CHECKING:
    from vectraudit.pipelines.guided import GuidedWorkspace


class ReviewPoolIn(Contract):
    draft_id: str
    revision: str
    baseline: str
    candidate: str
    query_id: str = Field(min_length=1, max_length=512)


def review_router(workspace: GuidedWorkspace) -> APIRouter:
    """Inherit the guided router's token/origin checks; never execute a retriever."""
    router = APIRouter()

    @router.post("/review-pool")
    def pool(body: ReviewPoolIn) -> dict[str, Any]:
        draft = workspace.load_draft(body.draft_id)
        if draft["revision"] != body.revision:
            raise HTTPException(409, "Benchmark changed. Reload before building a review queue.")
        source = parse_dataset(draft["dataset"])
        rows = [r for r in dataset_rows(source) if r.get("split", "explore") == "explore"]
        if body.query_id not in {r["query_id"] for r in rows}:
            raise ValueError("Passage review queues require an exploratory question")
        dataset = parse_dataset(render_dataset(rows, source.meta))
        if body.baseline == body.candidate:
            raise ValueError("Choose two different runs to pool passages")
        runs = [load_run(workspace.run_path(key)) for key in (body.baseline, body.candidate)]
        expected = {q.query_id for q in dataset.queries}
        for run in runs:
            if run["dataset"]["sha256"] != dataset.sha256:
                raise ValueError(
                    "Choose exploratory runs from this exact saved benchmark version. "
                    "After changing labels, remeasure both pipelines."
                )
            if (
                run["status"] not in ("complete", "partial")
                or {q["query_id"] for q in run["queries"]} != expected
            ):
                raise ValueError("Review requires finished runs covering the exploratory questions")
        audit = audit_benchmark(dataset, runs, irrelevant=draft.get("irrelevant", {}))
        entries = [
            next(q for q in run["queries"] if q["query_id"] == body.query_id) for run in runs
        ]
        measured = [not q.get("error") and q.get("ids") is not None for q in entries]
        ranks = [
            {doc: rank for rank, doc in enumerate(q.get("ids") or [], 1)} if ok else {}
            for q, ok in zip(entries, measured, strict=True)
        ]
        items = [
            {
                **item,
                "baseline_rank": ranks[0].get(item["doc_id"]),
                "candidate_rank": ranks[1].get(item["doc_id"]),
            }
            for item in audit["judgment_queue"]
            if item["query_id"] == body.query_id
        ]
        return {
            "draft_id": draft["id"],
            "revision": draft["revision"],
            "dataset_sha256": dataset.sha256,
            "query_id": body.query_id,
            "baseline": body.baseline,
            "candidate": body.candidate,
            "baseline_measured": measured[0],
            "candidate_measured": measured[1],
            "items": items,
        }

    return router
