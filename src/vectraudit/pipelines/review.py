"""Read-only passage pools from measured exploratory runs of a saved benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import Field

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.decision_queue import DecisionQueueError, decision_queue
from vectraudit.evaluation.runfile import load_run
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.document_lookup import LookupRequest, LookupRunner, content_hash
from vectraudit.pipelines.models import Contract

if TYPE_CHECKING:
    from vectraudit.pipelines.guided import GuidedWorkspace


class ReviewPoolIn(Contract):
    draft_id: str
    revision: str
    baseline: str
    candidate: str
    query_id: str = Field(min_length=1, max_length=512)
    policy: str | None = None
    """Saved or startup policy id for the decision order; ``None`` uses the defaults."""


class ReviewDocumentsIn(ReviewPoolIn):
    doc_ids: list[str] = Field(min_length=1, max_length=10)
    source_run: str
    confirm_lookup: bool = False


def review_router(workspace: GuidedWorkspace) -> APIRouter:
    """Inherit the guided router's token/origin checks; never execute a retriever."""
    router = APIRouter()
    lookup = LookupRunner()

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
        decision = _decision(workspace, body, runs, dataset, draft.get("irrelevant", {}))
        swing = {
            i["doc_id"]: i
            for i in (decision or {}).pop("items", [])
            if i["query_id"] == body.query_id
        }
        items = [
            {
                **item,
                "baseline_rank": ranks[0].get(item["doc_id"]),
                "candidate_rank": ranks[1].get(item["doc_id"]),
                **({"swing": swing[item["doc_id"]]["swing"]} if item["doc_id"] in swing else {}),
            }
            for item in audit["judgment_queue"]
            if item["query_id"] == body.query_id
        ]
        if decision is not None:
            # The documents that could move the check most come first (stable otherwise).
            items.sort(key=lambda item: -float(item.get("swing", 0.0)))
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
            "decision": decision,
            "document_lookup": bool(workspace.manifest and workspace.manifest.document_lookup),
        }

    @router.post("/review-documents")
    def documents(body: ReviewDocumentsIn) -> dict[str, Any]:
        config = workspace.manifest.document_lookup if workspace.manifest else None
        if config is None:
            raise HTTPException(422, "No document resolver is registered at startup")
        if not body.confirm_lookup:
            raise HTTPException(
                422, "Confirm sending document IDs and saved context to the registered source"
            )
        reviewed = pool(body)
        allowed = {item["doc_id"] for item in reviewed["items"]}
        if len(body.doc_ids) != len(set(body.doc_ids)) or not set(body.doc_ids) <= allowed:
            raise HTTPException(422, "Choose distinct documents from this question's review pool")
        if body.source_run not in (body.baseline, body.candidate):
            raise HTTPException(
                422, "Choose the baseline or candidate as the document version source"
            )
        draft = workspace.load_draft(body.draft_id)
        if draft["revision"] != body.revision:
            raise HTTPException(409, "Benchmark changed. Reload before looking up passages.")
        query = next(
            q for q in parse_dataset(draft["dataset"]).queries if q.query_id == body.query_id
        )
        run = load_run(workspace.run_path(body.source_run))
        version = run["backend"].get("corpus_version") or run["backend"].get("snapshot")
        request = LookupRequest(
            query_id=query.query_id,
            doc_ids=body.doc_ids,
            context=query.context,
            filters=query.filter,
            corpus_version=version,
            run_id=body.source_run,
        )
        result = lookup.run(config, workspace.manifest_root, request)
        if workspace.load_draft(body.draft_id)["revision"] != body.revision:
            raise HTTPException(
                409, "Benchmark changed during lookup. Reload before reviewing passages."
            )
        return {
            "source_run": body.source_run,
            "query_id": query.query_id,
            "corpus_version": result.corpus_version,
            "version_verified": version is not None,
            "documents": [
                {
                    **doc.model_dump(),
                    "sha256": content_hash(doc.text) if doc.text is not None else None,
                }
                for doc in result.documents
            ],
        }

    return router


def _decision(
    workspace: GuidedWorkspace,
    body: ReviewPoolIn,
    runs: list[dict[str, Any]],
    dataset: Any,
    irrelevant: dict[str, list[str]],
) -> dict[str, Any] | None:
    """The pair's decision queue, without per-document items for other questions.

    Ranks and sources stay hidden in the grader, so only the pool-wide stop rule and which
    questions to review next leave here; ``None`` when the runs cannot be compared.
    """
    rules, _ = workspace.policies.resolve(body.policy)
    try:
        report = decision_queue(runs[0], runs[1], dataset, rules, irrelevant=irrelevant)
    except DecisionQueueError:
        return None
    per_question: dict[str, int] = {}
    for item in report["items"]:
        if item["could_change"] and item["query_id"] != body.query_id:
            per_question[item["query_id"]] = per_question.get(item["query_id"], 0) + 1
    order = list(dict.fromkeys(i["query_id"] for i in report["items"] if i["could_change"]))
    return {
        "state": report["state"],
        "verdict": report["verdict"],
        "message": report["message"],
        "n_could_change": report["n_could_change"],
        "saving": report["saving"],
        "next_questions": [
            {"query_id": qid, "documents": per_question[qid]}
            for qid in order
            if qid in per_question
        ][:5],
        "items": report["items"],
    }
