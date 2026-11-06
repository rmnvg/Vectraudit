"""Review items from claim-level evidence: documents to judge, labels to recheck, evidence groups.

When a run has claim evidence (``answer.claim_evidence``, from ``attach-answers`` on a
RAGChecker import; see :mod:`vectraudit.evaluation.ragchecker_import`), the documents that
entailed the reference answer's claims say something about the benchmark's labels. They are
suggestions for a human, never labels (model output does not become a relevance label):

- **Possibly relevant, unjudged.** A delivered document that entails at least one reference
  claim but has no judgment for the question. Queued in the benchmark's judgment queue,
  ordered by how many claims it entails.
- **Label may be stale or wrong.** A document labelled relevant that entails none of the
  reference claims while another delivered document entails some. Queued the same way.
- **Proposed required evidence.** The smallest set of delivered documents that together
  entail the most reference claims (a greedy set cover over claim positions), with the
  fraction of claims covered. Status ``proposed``: a reviewer accepts it
  (``vectraudit workbench accept-evidence``) once every document in it is labelled relevant.

Only exploratory questions get items: held-out questions stay out of every queue, as they
stay out of the rest of the judgment queue. Only documents the run delivered or put in the
model context are named. Items carry IDs, counts and flags, never text, and the evidence
level is ``supported`` (model-judged).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset

SOURCE = "claim_evidence"
EVIDENCE = "supported"
PRIORITY_UNJUDGED = 3
PRIORITY_STALE = 4


def claim_review(
    dataset: Dataset,
    runs: Sequence[Mapping[str, Any]],
    *,
    judged_extra: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(queue items, proposed evidence groups)`` from the runs' claim evidence.

    ``judged_extra`` maps a question to documents already reviewed (for instance marked
    irrelevant in a draft), which are not queued again. Runs must be of this dataset.
    """
    by_query = {q.query_id: q for q in dataset.queries if q.split == "explore"}
    items: dict[tuple[str, str, str], dict[str, Any]] = {}
    proposals: dict[str, dict[str, Any]] = {}
    for run in runs:
        for entry in run["queries"]:
            query = by_query.get(entry["query_id"])
            evidence = (entry.get("answer") or {}).get("claim_evidence")
            if query is None or entry.get("error") or not evidence:
                continue
            support = _delivered(entry, evidence["claim_support"])
            reference = evidence["claims"]["reference"]
            judged = (
                set(query.relevant)
                | {s.doc_id for s in query.relevant_spans}
                | set(query.forbidden)
                | set((judged_extra or {}).get(query.query_id, ()))
            )
            for item in _unjudged(query, support, judged, reference):
                _keep(items, item, "claims_entailed")
            for item in _stale(query, support, reference):
                _keep(items, item, "claims_entailed")
            proposal = _proposal(query, support, reference, judged_extra)
            if proposal is not None:
                best = proposals.get(query.query_id)
                if best is None or (proposal["coverage"], -len(proposal["docs"])) > (
                    best["coverage"],
                    -len(best["docs"]),
                ):
                    proposals[query.query_id] = proposal
    queue = sorted(
        items.values(),
        key=lambda i: (-i["priority"], -i["claims_entailed"], i["query_id"], i["doc_id"]),
    )
    return queue, [proposals[q] for q in sorted(proposals)]


def _delivered(entry: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    known = set(entry.get("ids") or [])
    context = entry.get("model_context")
    if isinstance(context, Mapping):
        known |= {i.get("id") for i in context.get("items") or [] if isinstance(i, Mapping)}
    return {doc: s for doc, s in support.items() if doc in known}


def _unjudged(
    query: DatasetQuery, support: Mapping[str, Any], judged: set[str], reference: int
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": query.query_id,
            "doc_id": doc,
            "kind": "possibly_relevant_unjudged",
            "reason": (
                f"possibly relevant, unjudged: entails {s['reference']} of {reference} "
                "reference claims"
            ),
            "priority": PRIORITY_UNJUDGED,
            "claims_entailed": s["reference"],
            "reference_claims": reference,
            "evidence": EVIDENCE,
            "source": SOURCE,
        }
        for doc, s in sorted(support.items())
        if s["reference"] > 0 and doc not in judged
    ]


def _stale(query: DatasetQuery, support: Mapping[str, Any], reference: int) -> list[dict[str, Any]]:
    if not any(s["reference"] > 0 for s in support.values()):
        return []
    return [
        {
            "query_id": query.query_id,
            "doc_id": doc,
            "kind": "label_may_be_stale",
            "reason": (
                "label may be stale or wrong: labelled relevant, but entails none of the "
                "reference claims while another delivered document does"
            ),
            "priority": PRIORITY_STALE,
            "claims_entailed": 0,
            "reference_claims": reference,
            "grade": query.relevant[doc],
            "evidence": EVIDENCE,
            "source": SOURCE,
        }
        for doc, s in sorted(support.items())
        if s["reference"] == 0 and query.relevant.get(doc, 0) > 0
    ]


def _keep(
    items: dict[tuple[str, str, str], dict[str, Any]], item: Mapping[str, Any], more: str
) -> None:
    """One item per question, document and kind; across runs the larger count is kept."""
    key = (item["query_id"], item["doc_id"], item["kind"])
    if key not in items or item[more] > items[key][more]:
        items[key] = dict(item)


def _proposal(
    query: DatasetQuery,
    support: Mapping[str, Any],
    reference: int,
    marked: Mapping[str, Sequence[str]] | None,
) -> dict[str, Any] | None:
    """Greedy cover of the reference claims; ``None`` without positions or a new group.

    A forbidden document, or one a reviewer marked irrelevant, is never part of a group.
    """
    blocked = set(query.forbidden) | set((marked or {}).get(query.query_id, ()))
    positions = {
        doc: set(s["reference_claims"])
        for doc, s in support.items()
        if s.get("reference_claims") and doc not in blocked
    }
    if reference <= 0 or not positions:
        return None
    covered: set[int] = set()
    group: list[str] = []
    while True:
        gain = {doc: len(claims - covered) for doc, claims in positions.items() if doc not in group}
        best = max(gain.values(), default=0)
        if best == 0:
            break
        chosen = min(doc for doc, n in gain.items() if n == best)
        group.append(chosen)
        covered |= positions[chosen]
    docs = sorted(group)
    if tuple(docs) in query.required_evidence:
        return None
    return {
        "query_id": query.query_id,
        "docs": docs,
        "claims_covered": len(covered),
        "reference_claims": reference,
        "coverage": len(covered) / reference,
        "status": "proposed",
        "already_relevant": [d for d in docs if query.relevant.get(d, 0) > 0],
        "needs_judgment": [d for d in docs if query.relevant.get(d, 0) <= 0],
        "required_evidence": [docs],
        "evidence": EVIDENCE,
        "source": SOURCE,
    }


def accept_evidence(
    dataset: Dataset, audit: Mapping[str, Any], questions: Sequence[str]
) -> tuple[str, list[dict[str, Any]]]:
    """A new dataset version with the named questions' proposed required evidence added.

    Nothing is accepted implicitly: the reviewer names each question. The proposal must come
    from an audit of this exact dataset version, and every document in its group must already
    be labelled relevant (judge ``needs_judgment`` documents first), because a required
    document is a relevant one. The group is added beside any groups the question already
    has, since any one complete group suffices. The header records what was accepted. Returns
    the dataset text and the accepted proposals.
    """
    if audit.get("format") != "vectraudit.benchmark":
        raise ValueError("The proposals file is not a benchmark audit (workbench benchmark).")
    if audit.get("dataset_sha256") != dataset.sha256:
        raise ValueError(
            "The proposals were made for another version of this dataset; audit this version "
            "again with the runs that carry claim evidence."
        )
    if not questions:
        raise ValueError("Name the questions to accept; nothing is accepted automatically.")
    proposals = {p["query_id"]: p for p in audit.get("required_evidence_proposals") or []}
    exploratory = {q.query_id for q in dataset.queries if q.split == "explore"}
    accepted = []
    for qid in dict.fromkeys(questions):
        if qid not in proposals:
            raise ValueError(f"No proposed evidence for question {qid!r}.")
        if qid not in exploratory:
            raise ValueError(f"Question {qid!r} is held out; evidence is never proposed for it.")
        accepted.append(proposals[qid])
    rows = {r["query_id"]: r for r in dataset_rows(dataset)}
    for proposal in accepted:
        row = rows[proposal["query_id"]]
        unjudged = [d for d in proposal["docs"] if row.get("relevant", {}).get(d, 0) <= 0]
        if unjudged:
            raise ValueError(
                f"Question {proposal['query_id']!r}: judge {', '.join(unjudged)} as relevant "
                "first; required evidence must be made of relevant documents."
            )
        groups = [list(g) for g in row.get("required_evidence", [])]
        if proposal["docs"] not in groups:
            groups.append(list(proposal["docs"]))
        row["required_evidence"] = groups
    meta = dict(dataset.meta)
    provenance = dict(meta.get("provenance") or {})
    provenance["required_evidence_accepted"] = [
        *provenance.get("required_evidence_accepted", []),
        *(
            {
                "query_id": p["query_id"],
                "docs": p["docs"],
                "coverage": p["coverage"],
                "source": SOURCE,
                "evidence": EVIDENCE,
            }
            for p in accepted
        ),
    ]
    meta["provenance"] = provenance
    return render_dataset(list(rows.values()), meta), accepted
