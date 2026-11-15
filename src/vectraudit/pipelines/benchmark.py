"""Local benchmark maintenance: coverage, duplicate questions and judgment queues."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit.core.coverage import coverage, public
from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.label_audit import queue_items
from vectraudit.pipelines.claim_review import claim_review

SUITE_TAGS = (
    "exact_identifier",
    "typo",
    "paraphrase",
    "multilingual",
    "ambiguous",
    "outdated",
    "no_answer",
    "permission",
)


@dataclass(frozen=True)
class CorpusVectors:
    """The corpus as the coverage analysis needs it."""

    ids: Sequence[str]
    vectors: npt.ArrayLike
    titles: Sequence[str | None] | None = None


def _embedding_coverage(dataset: Dataset, corpus: CorpusVectors | None) -> dict[str, Any] | None:
    """Coverage regions of the corpus against the exploratory questions (held-out never count)."""
    if corpus is None:
        return None
    asked = [q.vector for q in dataset.queries if q.split == "explore" and q.vector is not None]
    if not asked:
        return {
            "findings": [],
            "note": "No exploratory question has a vector, so coverage was not measured.",
        }
    relevant = frozenset(d for q in dataset.queries if q.split == "explore" for d in q.relevant)
    result = coverage(corpus.ids, corpus.vectors, np.asarray(asked), relevant, corpus.titles)
    return public(result)


def audit_benchmark(
    dataset: Dataset,
    runs: list[dict[str, Any]] | None = None,
    *,
    irrelevant: dict[str, list[str]] | None = None,
    corpus: CorpusVectors | None = None,
    label_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest human review; never infer relevance or modify a frozen benchmark.

    With ``corpus`` (document IDs, vectors, and titles) the report gains ``embedding_coverage``:
    groups of documents no exploratory question is close to, as hypotheses to look at. With
    ``label_audit`` (a ``vectraudit.label_audit`` report of this dataset version), its suspect
    labels join the judgment queue with their reasons.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    tags: Counter[str] = Counter()
    for q in dataset.queries:
        if q.text:
            groups[" ".join(q.text.casefold().split())].append(q.query_id)
        tags.update(q.tags)
        if q.no_answer:
            tags["no_answer"] += 1 if "no_answer" not in q.tags else 0
        if q.context or q.forbidden:
            tags["permission"] += 1 if "permission" not in q.tags else 0
    judgments: list[dict[str, Any]] = []
    by_query: dict[str, list[set[str]]] = defaultdict(list)
    for run in runs or []:
        if run["dataset"]["sha256"] != dataset.sha256:
            raise ValueError("judgment queues require runs from this exact dataset version")
        for entry in run["queries"]:
            if not entry.get("error") and entry.get("ids") is not None:
                by_query[entry["query_id"]].append(set(entry["ids"]))
    for q in dataset.queries:
        rankings = by_query[q.query_id]
        pool = set.union(*rankings) if rankings else set()
        common = set.intersection(*rankings) if rankings else set()
        judged = set(q.relevant) | {span.doc_id for span in q.relevant_spans}
        judged.update((irrelevant or {}).get(q.query_id, []))
        unjudged = pool - judged - set(q.forbidden)
        if q.split == "explore":
            judgments.extend(
                {
                    "query_id": q.query_id,
                    "doc_id": doc,
                    "reason": "candidate disagreement"
                    if doc not in common
                    else "unjudged pooled result",
                    "priority": 2 if doc not in common else 1,
                }
                for doc in sorted(unjudged)
            )
    # Claim-level evidence (RAGChecker via attach-answers) adds documents to judge and labels
    # to recheck; they merge with what the pool already queued, never become labels.
    claim_items, proposals = claim_review(dataset, runs or [], judged_extra=irrelevant)
    queued = {(j["query_id"], j["doc_id"]): j for j in judgments}
    audit_items = queue_items(label_audit, dataset) if label_audit is not None else []
    for item in [*claim_items, *audit_items]:
        key = (item["query_id"], item["doc_id"])
        if key in queued:
            queued[key].update({**item, "priority": max(queued[key]["priority"], item["priority"])})
        else:
            judgments.append(item)
            queued[key] = item
    return {
        "format": "vectraudit.benchmark",
        "dataset_sha256": dataset.sha256,
        "n_queries": len(dataset),
        "n_labelled": dataset.n_labelled,
        "splits": dict(Counter(q.split for q in dataset.queries)),
        "coverage": dict(tags),
        "embedding_coverage": _embedding_coverage(dataset, corpus),
        "missing_categories": [tag for tag in SUITE_TAGS if not tags[tag]],
        "duplicates": [ids for ids in groups.values() if len(ids) > 1],
        "unannotated": [q.query_id for q in dataset.queries if not q.labelled],
        "judgment_queue": sorted(
            judgments,
            key=lambda row: (
                -int(row["priority"]),
                -int(row.get("claims_entailed", 0)),
                str(row["query_id"]),
            ),
        ),
        "required_evidence_proposals": proposals,
        "suggestions": [
            {"category": tag, "instruction": instruction}
            for tag, instruction in zip(
                SUITE_TAGS,
                (
                    "Choose an actual SKU, error code or exact identifier from your corpus.",
                    "Make a spelling-error variant of a judged exploratory question.",
                    "Rewrite an exploratory question while preserving its intent.",
                    "Have a fluent reviewer translate a question into a supported language.",
                    "Write a question with two plausible interpretations and judge both.",
                    "Ask for current facts where old and new documents coexist.",
                    "Write an out-of-scope question and explicitly mark no_answer after review.",
                    "Repeat a question under allowed and forbidden tenant contexts.",
                ),
                strict=True,
            )
            if not tags[tag]
        ],
        "note": (
            "Suggestions require human judgment. "
            "Held-out queries are excluded from the judgment queue and from proposals."
        ),
    }
