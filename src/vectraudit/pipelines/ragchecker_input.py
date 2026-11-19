"""Build RAGChecker's input from a run, so its checker can judge what the model was given.

RAGChecker (Ru et al., NeurIPS 2024 Datasets and Benchmarks; Apache-2.0) needs, per question,
the question, a reference answer, the system's response and the retrieved chunks with their
text. A run holds none of that text, so this joins it from sources the user controls:

- the **context** is the run's observed model context when it recorded one, else the
  documents retrieval delivered, in the recorded order;
- the **document text** comes from a corpus file (``doc_id`` and ``text`` per line) or from
  the registered document resolver (:mod:`vectraudit.pipelines.document_lookup`), which is
  scoped to the question's tenant, filters and the run's corpus version;
- the **reference answers** and the **responses** come from files the user supplies.

The file this writes **contains private text** (questions, answers and documents) and is for
local use: run RAGChecker on your own machine, then import its output with
``workbench ragchecker-judgments``. It is never uploaded, and Vectraudit never calls a model.

A question is exported only when everything RAGChecker judges is present: a text question, a
reference, a response, a context, and the text of every context document. A partial context
would make RAGChecker judge something other than what the model saw, so the question is
skipped and counted instead. Held-out questions are not exported, so claim-level evidence is
never gathered for them.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from fastapi import HTTPException

from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.pipelines.document_lookup import (
    LookupConfig,
    LookupRequest,
    LookupRunner,
)

MAX_TEXT = 200_000
LOOKUP_BATCH = 10


class RagCheckerInputError(ValueError):
    """The inputs cannot be combined; the message says what to fix."""


class DocumentSource(Protocol):
    """Where document text comes from."""

    def texts(
        self, query: DatasetQuery, doc_ids: Sequence[str], run: Mapping[str, Any]
    ) -> dict[str, str] | None:
        """Text by ID for every requested document, or ``None`` when any is unavailable."""
        ...


class CorpusSource:
    """A JSONL corpus: one object per line with ``doc_id`` and ``text``."""

    def __init__(self, path: str | Path) -> None:
        self.documents: dict[str, str] = {}
        try:
            lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            raise RagCheckerInputError(f"Cannot read {path}: {exc.strerror or exc}") from exc
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise RagCheckerInputError(f"{path} line {number}: not JSON") from exc
            doc, text = (
                (row.get("doc_id"), row.get("text")) if isinstance(row, dict) else (None, None)
            )
            if not isinstance(doc, str) or not doc or not isinstance(text, str) or not text:
                raise RagCheckerInputError(f"{path} line {number}: needs a doc_id and a text")
            if doc in self.documents:
                raise RagCheckerInputError(f"{path} line {number}: doc_id {doc!r} repeats")
            self.documents[doc] = text

    def texts(
        self, query: DatasetQuery, doc_ids: Sequence[str], run: Mapping[str, Any]
    ) -> dict[str, str] | None:
        found = {doc: self.documents[doc] for doc in doc_ids if doc in self.documents}
        return found if len(found) == len(set(doc_ids)) else None


class LookupSource:
    """The registered document resolver, scoped to each question's context and corpus version."""

    def __init__(self, config: LookupConfig, root: Path) -> None:
        self.config, self.root, self.runner = config, root, LookupRunner()

    def texts(
        self, query: DatasetQuery, doc_ids: Sequence[str], run: Mapping[str, Any]
    ) -> dict[str, str] | None:
        backend = run.get("backend") or {}
        version = backend.get("corpus_version") or backend.get("snapshot")
        unique = list(dict.fromkeys(doc_ids))
        found: dict[str, str] = {}
        for start in range(0, len(unique), LOOKUP_BATCH):
            request = LookupRequest(
                query_id=query.query_id,
                doc_ids=unique[start : start + LOOKUP_BATCH],
                context=query.context,
                filters=query.filter,
                corpus_version=version,
                run_id=str(run["run_id"]),
            )
            try:
                result = self.runner.run(self.config, self.root, request)
            except HTTPException as exc:
                raise RagCheckerInputError(f"Document lookup: {exc.detail}") from exc
            if any(d.status != "found" or d.text is None for d in result.documents):
                return None
            found.update({d.doc_id: d.text for d in result.documents if d.text is not None})
        return found


def read_texts(path: str | Path, field: str) -> dict[str, str]:
    """``{query_id: text}`` from JSONL lines ``{"query_id": ..., field: ...}``."""
    texts: dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise RagCheckerInputError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise RagCheckerInputError(f"{path} line {number}: not JSON") from exc
        qid = row.get("query_id") if isinstance(row, dict) else None
        text = row.get(field) if isinstance(row, dict) else None
        if not isinstance(qid, str) or not qid:
            raise RagCheckerInputError(f"{path} line {number}: needs a query_id")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            raise RagCheckerInputError(
                f"{path} line {number}: {field} must be non-empty text of at most {MAX_TEXT} "
                "characters"
            )
        if qid in texts:
            raise RagCheckerInputError(f"{path} line {number}: query_id {qid!r} repeats")
        texts[qid] = text
    if not texts:
        raise RagCheckerInputError(f"{path} contains no rows")
    return texts


def context_documents(entry: Mapping[str, Any]) -> tuple[list[str], str]:
    """The documents the model was given, in order, and where that list came from."""
    context = entry.get("model_context")
    if isinstance(context, Mapping) and context.get("coverage") == "observed":
        ids = [i["id"] for i in context.get("items") or [] if isinstance(i, Mapping) and "id" in i]
        return list(dict.fromkeys(ids)), "model_context"
    return list(dict.fromkeys(entry.get("ids") or [])), "delivered"


def build_input(
    run: Mapping[str, Any],
    dataset: Dataset,
    references: Mapping[str, str],
    responses: Mapping[str, str],
    source: DocumentSource,
) -> tuple[dict[str, Any], Counter[str]]:
    """RAGChecker's ``{"results": [...]}`` and counts of what was and was not exported."""
    if (run.get("dataset") or {}).get("sha256") != dataset.sha256:
        raise RagCheckerInputError(
            "The run was recorded on a different dataset version than --dataset."
        )
    by_id = {q.query_id: q for q in dataset.queries}
    counts: Counter[str] = Counter()
    results = []
    for entry in run["queries"]:
        qid = entry["query_id"]
        query = by_id.get(qid)
        if query is None:
            raise RagCheckerInputError(f"The run has question {qid!r}, not in --dataset.")
        if query.split == "heldout":
            counts["skipped_heldout"] += 1
            continue
        if entry.get("error") or entry.get("ids") is None:
            counts["skipped_errored"] += 1
            continue
        if not query.text:
            counts["skipped_no_text"] += 1
            continue
        if qid not in references:
            counts["skipped_no_reference"] += 1
            continue
        if qid not in responses:
            counts["skipped_no_response"] += 1
            continue
        docs, origin = context_documents(entry)
        if not docs:
            counts["skipped_no_context"] += 1
            continue
        texts = source.texts(query, docs, run)
        if texts is None:
            counts["skipped_incomplete_context"] += 1
            continue
        results.append(
            {
                "query_id": qid,
                "query": query.text,
                "gt_answer": references[qid],
                "response": responses[qid],
                "retrieved_context": [{"doc_id": doc, "text": texts[doc]} for doc in docs],
            }
        )
        counts["exported"] += 1
        counts[f"context_from_{origin}"] += 1
        if origin == "model_context":
            items = (entry.get("model_context") or {}).get("items") or []
            counts["truncated_context_items"] += sum(
                1 for i in items if isinstance(i, Mapping) and i.get("truncated") is True
            )
    if not results:
        raise RagCheckerInputError(
            "No question could be exported: "
            + (
                ", ".join(
                    f"{n} {k.removeprefix('skipped_').replace('_', ' ')}"
                    for k, n in sorted(counts.items())
                    if k.startswith("skipped_")
                )
                or "the run has no questions"
            )
        )
    return {"results": results}, counts


def render(counts: Mapping[str, int], out: str) -> str:
    """What was exported, what was skipped and why, and the warning about private text."""
    lines = [f"Wrote {counts.get('exported', 0)} questions to {out}."]
    skipped = {
        "skipped_heldout": "held-out (never exported)",
        "skipped_errored": "failed in the run",
        "skipped_no_text": "without question text",
        "skipped_no_reference": "without a reference answer",
        "skipped_no_response": "without a response",
        "skipped_no_context": "with no recorded context",
        "skipped_incomplete_context": "with a context document whose text was unavailable",
    }
    lines += [
        f"  {counts[k]} questions skipped: {why}" for k, why in skipped.items() if counts.get(k)
    ]
    if counts.get("context_from_delivered"):
        lines.append(
            f"  {counts['context_from_delivered']} questions use the delivered documents as the "
            "context (the run recorded no observed model context)."
        )
    if counts.get("truncated_context_items"):
        lines.append(
            f"  {counts['truncated_context_items']} context documents were cut short in the "
            "prompt; the export has their full text."
        )
    lines.append(
        "This file contains question, answer and document text. Keep it on this machine: run "
        "RAGChecker locally, then import its output with `workbench ragchecker-judgments`."
    )
    return "\n".join(lines)
