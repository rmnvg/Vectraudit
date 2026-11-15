"""Label audit: which existing relevance labels are likely wrong or missing, for a person to check.

A benchmark's labels are themselves measurements, and some are wrong: a relevant document
never labelled, or a label on a document that does not answer the question. Every metric
inherits those errors. This module finds suspects from data Vectraudit already has, using
three signals, and puts them in front of a reviewer. It never changes a label: a reviewer's
decision makes a new dataset version (:func:`apply_label_decisions`), and the original file is
untouched.

1. **Rank (confident learning).** Each candidate run ranks the question's documents; the mean
   over runs of ``1 - (rank - 1) / k`` (0 when a run did not return the document) is a noisy
   score of relevance. As in confident learning, each class gets a per-question threshold:
   the mean score of the documents labelled with it. A document labelled relevant that scores
   no higher than the question's not-relevant candidates, or a not-relevant one (judged 0, or
   unjudged and so counted as not relevant by every metric) that scores as high as the
   relevant ones, is a suspect. Needs at least two runs on the question and at least two
   relevant labels (one relevant label gives no stable threshold). It cannot tell a wrong
   label from a document every run is wrong about: both look the same, and both deserve a
   look.
2. **Near-duplicate disagreement.** With corpus vectors: two documents at or above a cosine
   threshold labelled differently for one question. When both are judged, one of the two
   labels is wrong and both are listed; when one is relevant and its duplicate unjudged, the
   duplicate is listed as possibly relevant.
3. **Annotator split.** With a labelling task holding more than one judgment per item: items
   where annotators disagree on relevance and no reviewer has resolved them.

On planted errors in synthetic corpora (``scripts/label_audit_spike.py``; docs/DECISIONS.md,
2026-11-09) the rank signal flagged 72 to 86% of the errors with 19 to 25% of its flags false;
the other two are weaker alone (the duplicate signal sees only errors on duplicated
documents; a split vote usually has a right majority), so they order the queue beside it
rather than stand as verdicts. Real label errors correlate with retrieval errors more than
planted ones, so read these as upper bounds.

Every suspect has evidence level ``hypothesis`` (invariant 16). Held-out questions, and
exploratory questions that repeat a held-out one under another id, are left out unless
``include_heldout`` is set. The report holds IDs, labels, counts and scores, never text.

Credit: confident learning is Northcutt, Jiang and Chuang, "Confident learning: estimating
uncertainty in dataset labels" (JAIR, 2021), implemented in cleanlab; the multi-annotator
consensus idea follows cleanlab's multi-annotator module. Our own numpy implementation of a
rank-based variant; see docs/credits.md.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from vectraudit.evaluation.dataset import Dataset, DatasetQuery
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset

if TYPE_CHECKING:
    from vectraudit.labelling.task import Task

FORMAT = "vectraudit.label_audit"
FORMAT_VERSION = 1
EVIDENCE = "hypothesis"
SOURCE = "label_audit"
RANK = "rank"
DUPLICATE = "near_duplicate"
ANNOTATORS = "annotators"
MIN_RUNS = 2
DUPLICATE_COSINE = 0.995
# Queue priorities beside the benchmark's own (1 pooled, 2 disagreement, 3-4 claim evidence):
# the rank signal met the adoption bar alone, so it leads; corroboration ranks first.
PRIORITY_CORROBORATED = 5
PRIORITY_RANK = 4
PRIORITY_OTHER = 3
_BLOCK = 1 << 24  # similarity entries per block when matching duplicates


class LabelAuditError(ValueError):
    """The audit cannot run as asked; the message says why."""


@dataclass(frozen=True)
class CorpusVectors:
    """Document IDs and vectors, for the near-duplicate signal."""

    ids: Sequence[str]
    vectors: npt.ArrayLike


# --- labels ----------------------------------------------------------------------------------


def _labels(q: DatasetQuery, irrelevant: Iterable[str] = ()) -> dict[str, float]:
    """Each judged document's grade: span-only documents count as 1, extra judgments as 0."""
    grades = dict(q.relevant)
    for span in q.relevant_spans:
        grades.setdefault(span.doc_id, 1.0)
    for doc in irrelevant:
        grades.setdefault(doc, 0.0)
    return grades


def _state(grade: float | None) -> str:
    if grade is None:
        return "unjudged"
    return "relevant" if grade > 0 else "judged_not_relevant"


def _reviewed(dataset: Dataset) -> set[tuple[str, str]]:
    """Question and document pairs a reviewer already decided on (``provenance.label_review``)."""
    review = (dataset.meta.get("provenance") or {}).get("label_review") or []
    return {(r["query_id"], r["doc_id"]) for r in review if isinstance(r, Mapping)}


def _audited(dataset: Dataset, include_heldout: bool) -> tuple[list[DatasetQuery], int, int]:
    """The questions the audit may look at, held-out ones and their repeats left out."""
    from vectraudit.pipelines.optimize import question_identities

    held = [q for q in dataset.queries if q.split == "heldout"]
    if include_heldout:
        return list(dataset.queries), 0, 0
    rows = {r["query_id"]: r for r in dataset_rows(dataset)}
    held_ids = question_identities([rows[q.query_id] for q in held])
    kept, repeats = [], 0
    for q in dataset.queries:
        if q.split == "heldout":
            continue
        if held_ids and question_identities([rows[q.query_id]]) & held_ids:
            repeats += 1
            continue
        kept.append(q)
    return kept, len(held), repeats


# --- signal 1: rank-based confident learning -------------------------------------------------


def _rankings(
    dataset: Dataset, runs: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[list[str], int]]], int]:
    """The runs used, each question's ``(ranking, depth)`` per run, and runs ignored.

    Runs of another dataset version are ignored (their labels were different); the same run
    given twice counts once. Errored questions contribute nothing.
    """
    used: list[dict[str, Any]] = []
    seen: set[str] = set()
    ignored = 0
    by_query: dict[str, list[tuple[list[str], int]]] = defaultdict(list)
    for run in runs:
        if (run.get("dataset") or {}).get("sha256") != dataset.sha256:
            ignored += 1
            continue
        run_id = str(run.get("run_id") or id(run))
        if run_id in seen:
            continue
        seen.add(run_id)
        k = (run.get("search") or {}).get("k")
        used.append(
            {
                "run_id": run.get("run_id"),
                "label": run.get("label"),
                "k": k,
                "source": (run.get("timing") or {}).get("source"),
            }
        )
        for entry in run.get("queries") or []:
            ids = entry.get("ids")
            if entry.get("error") or ids is None:
                continue
            ranking = [str(d) for d in ids]
            by_query[entry["query_id"]].append((ranking, max(int(k or 0), len(ranking), 1)))
    return used, by_query, ignored


def _rank_signal(
    queries: Sequence[DatasetQuery],
    rankings: Mapping[str, list[tuple[list[str], int]]],
    irrelevant: Mapping[str, Sequence[str]],
    min_runs: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    flags: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    audited = 0
    for q in queries:
        if q.no_answer:
            skipped["no-answer question"] += 1
            continue
        labels = _labels(q, irrelevant.get(q.query_id, ()))
        labels = {d: g for d, g in labels.items() if d not in q.forbidden}
        positive = {d for d, g in labels.items() if g > 0}
        measured = rankings.get(q.query_id, [])
        if len(positive) < 2:
            skipped["fewer than two relevant labels"] += 1
            continue
        if len(measured) < min_runs:
            skipped[f"measured by fewer than {min_runs} runs"] += 1
            continue
        n = len(measured)
        score: dict[str, float] = defaultdict(float)
        best: dict[str, int] = {}
        hits: Counter[str] = Counter()
        for ranking, depth in measured:
            for rank, doc in enumerate(ranking, 1):
                if doc in q.forbidden:
                    continue
                score[doc] += (1 - (rank - 1) / depth) / n
                best[doc] = min(best.get(doc, rank), rank)
                hits[doc] += 1
        for doc in labels:
            score.setdefault(doc, 0.0)
        relevant_scores = [score[d] for d in positive]
        if not any(relevant_scores):
            # Every run misses every relevant document: a retrieval failure and a labelling
            # error look identical here, so the signal says nothing about this question.
            skipped["no run returns a relevant document"] += 1
            continue
        audited += 1
        other = [s for d, s in score.items() if d not in positive]
        t_rel = float(np.mean(relevant_scores))
        t_other = float(np.mean(other)) if other else 0.0
        for doc, s in score.items():
            grade = labels.get(doc)
            if doc in positive and s <= t_other:
                strength = (t_other - s) / t_other if t_other > 0 else 1.0
                threshold = t_other
            elif doc not in positive and s >= t_rel:
                strength = (s - t_rel) / (1 - t_rel) if t_rel < 1 else 1.0
                threshold = t_rel
            else:
                continue
            flags[(q.query_id, doc)] = {
                "signal": RANK,
                "strength": round(min(1.0, max(0.0, strength)), 4),
                "score": round(s, 4),
                "threshold": round(threshold, 4),
                "runs": n,
                "returned_by": hits[doc],
                "best_rank": best.get(doc),
                "label_state": _state(grade),
            }
    return flags, {
        "questions_audited": audited,
        "questions_skipped": dict(sorted(skipped.items())),
    }


# --- signal 2: near-duplicate disagreement ---------------------------------------------------


def _unit(vectors: npt.ArrayLike) -> npt.NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise LabelAuditError("Corpus vectors must be a non-empty 2-D matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    out: npt.NDArray[np.float32] = np.divide(
        matrix, norms, out=np.zeros_like(matrix), where=norms > 0
    )
    return out


def _duplicate_signal(
    queries: Sequence[DatasetQuery],
    corpus: CorpusVectors,
    irrelevant: Mapping[str, Sequence[str]],
    cosine: float,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    ids = [str(d) for d in corpus.ids]
    if len(set(ids)) != len(ids):
        raise LabelAuditError("The corpus lists a document ID twice.")
    unit = _unit(corpus.vectors)
    if len(unit) != len(ids):
        raise LabelAuditError("The corpus has a different number of IDs and vectors.")
    row = {d: i for i, d in enumerate(ids)}
    labels = {
        q.query_id: {
            d: g
            for d, g in _labels(q, irrelevant.get(q.query_id, ())).items()
            if d not in q.forbidden
        }
        for q in queries
    }
    labelled = sorted({d for grades in labels.values() for d in grades})
    present = [d for d in labelled if d in row]
    neighbours: dict[str, list[tuple[str, float]]] = {}
    step = max(1, _BLOCK // len(ids))
    for start in range(0, len(present), step):
        block = present[start : start + step]
        sims = unit[[row[d] for d in block]] @ unit.T
        for i, doc in enumerate(block):
            close = np.flatnonzero(sims[i] >= cosine)
            neighbours[doc] = [(ids[j], float(sims[i, j])) for j in close if ids[j] != doc]
    flags: dict[tuple[str, str], dict[str, Any]] = {}
    pairs: set[tuple[str, str, str]] = set()

    def flag(qid: str, doc: str, partner: str, sim: float, grade: float | None) -> None:
        current = flags.get((qid, doc))
        if current is None or sim > current["strength"]:
            flags[(qid, doc)] = {
                "signal": DUPLICATE,
                "strength": round(sim, 4),
                "cosine": round(sim, 4),
                "duplicate_of": partner,
                "duplicate_label_state": _state(labels[qid].get(partner)),
                "label_state": _state(grade),
            }

    for q in queries:
        grades = labels[q.query_id]
        for doc, grade in grades.items():
            for other, sim in neighbours.get(doc, []):
                if other in q.forbidden:
                    continue
                theirs = grades.get(other)
                mine_positive, theirs_positive = grade > 0, (theirs or 0) > 0
                if mine_positive == theirs_positive:
                    continue
                if theirs is None:
                    if not mine_positive:
                        continue  # judged 0 beside unjudged: both count as not relevant
                    flag(q.query_id, other, doc, sim, None)
                else:
                    flag(q.query_id, doc, other, sim, grade)
                    flag(q.query_id, other, doc, sim, theirs)
                first, second = sorted((doc, other))
                pairs.add((q.query_id, first, second))
    return flags, {
        "cosine": cosine,
        "labelled_documents": len(labelled),
        "not_in_corpus": len(labelled) - len(present),
        "pairs": len(pairs),
    }


# --- signal 3: annotator split ---------------------------------------------------------------


def _annotator_signal(
    queries: Sequence[DatasetQuery], task: Task
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    from vectraudit.labelling.agreement import decisions

    audited = {q.query_id: q for q in queries}
    shared = [tq for tq in task.queries if tq.query_id in audited]
    if task.queries and not shared:
        raise LabelAuditError(
            "The labelling task shares no question with the audited part of this dataset."
        )
    resolved = {(j.query_id, j.doc_id) for j in task.resolutions}
    flags: dict[tuple[str, str], dict[str, Any]] = {}
    several = 0
    for tq in shared:
        q = audited[tq.query_id]
        grades = _labels(q)
        for doc in tq.documents:
            rows = decisions(task, tq.query_id, doc)
            if len(rows) < 2:
                continue
            several += 1
            votes = sum(1 for j in rows if (j.grade or 0) > 0)
            if votes in (0, len(rows)) or (tq.query_id, doc) in resolved:
                continue
            majority = max(votes, len(rows) - votes) / len(rows)
            flags[(tq.query_id, doc)] = {
                "signal": ANNOTATORS,
                "strength": round(1 - majority, 4),
                "judgments": len(rows),
                "relevant_votes": votes,
                "label_state": _state(grades.get(doc)),
            }
    return flags, {"items_with_several_judgments": several}


# --- the report ------------------------------------------------------------------------------


def _reason(state: str, signals: Mapping[str, Mapping[str, Any]]) -> str:
    parts = []
    if RANK in signals:
        parts.append(
            "the candidate runs rank it no higher than the question's not-relevant results "
            "(or every run misses it)"
            if state == "relevant"
            else "the candidate runs rank it as high as the question's relevant documents"
        )
    if DUPLICATE in signals:
        d = signals[DUPLICATE]
        parts.append(
            f"a near-duplicate ({d['duplicate_of']}, cosine {d['cosine']:.3f}) is "
            f"{d['duplicate_label_state'].replace('_', ' ')}"
        )
    if ANNOTATORS in signals:
        a = signals[ANNOTATORS]
        parts.append(
            f"{a['relevant_votes']} of {a['judgments']} annotators judged it relevant, unresolved"
        )
    label = {
        "relevant": "labelled relevant",
        "judged_not_relevant": "judged not relevant",
        "unjudged": "unjudged (counted as not relevant)",
    }[state]
    return f"{label}, but " + "; ".join(parts)


def audit_labels(
    dataset: Dataset,
    runs: Sequence[Mapping[str, Any]] = (),
    *,
    corpus: CorpusVectors | None = None,
    task: Task | None = None,
    irrelevant: Mapping[str, Sequence[str]] | None = None,
    include_heldout: bool = False,
    min_runs: int = MIN_RUNS,
    duplicate_cosine: float = DUPLICATE_COSINE,
    recheck_reviewed: bool = False,
) -> dict[str, Any]:
    """The ``vectraudit.label_audit`` report: suspect labels, each with the signals that fired.

    ``runs`` are candidate runs of this exact dataset version (others are ignored and
    counted); ``corpus`` enables the near-duplicate signal and ``task`` (a labelling task) the
    annotator signal. ``irrelevant`` maps a question to documents a reviewer marked not
    relevant outside the dataset (a workbench draft's judgments). Pairs a reviewer already
    decided on in an earlier review (``provenance.label_review``) are left out unless
    ``recheck_reviewed``. Nothing is modified.
    """
    if min_runs < 2:
        raise LabelAuditError("The rank signal needs at least two runs per question.")
    if not 0 < duplicate_cosine <= 1:
        raise LabelAuditError("The duplicate cosine threshold must be above 0 and at most 1.")
    if not runs and corpus is None and task is None:
        raise LabelAuditError(
            "Nothing to audit with: give candidate runs, corpus vectors or a labelling task."
        )
    extra = irrelevant or {}
    queries, held, repeats = _audited(dataset, include_heldout)
    used, rankings, ignored = _rankings(dataset, runs)
    signals: dict[str, dict[str, Any]] = {}
    found: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    if len(used) >= min_runs:
        found[RANK], stats = _rank_signal(queries, rankings, extra, min_runs)
        signals[RANK] = {"status": "measured", **stats}
    else:
        signals[RANK] = {
            "status": "not_measured",
            "reason": f"{len(used)} usable runs of this dataset version; at least {min_runs} "
            "are needed.",
        }
    if corpus is not None:
        found[DUPLICATE], stats = _duplicate_signal(queries, corpus, extra, duplicate_cosine)
        signals[DUPLICATE] = {"status": "measured", **stats}
    else:
        signals[DUPLICATE] = {"status": "not_measured", "reason": "No corpus vectors given."}
    if task is not None:
        found[ANNOTATORS], stats = _annotator_signal(queries, task)
        signals[ANNOTATORS] = {"status": "measured", **stats}
    else:
        signals[ANNOTATORS] = {"status": "not_measured", "reason": "No labelling task given."}
    if not found:
        raise LabelAuditError(
            "No signal could be measured: "
            + " ".join(str(s.get("reason")) for s in signals.values())
        )
    reviewed = set() if recheck_reviewed else _reviewed(dataset)
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for name, flags in found.items():
        signals[name]["flagged"] = len(flags)
        for key, detail in flags.items():
            by_pair[key][name] = detail
    grades = {q.query_id: _labels(q, extra.get(q.query_id, ())) for q in queries}
    suspects = []
    skipped_reviewed = 0
    for (qid, doc), fired in by_pair.items():
        if (qid, doc) in reviewed:
            skipped_reviewed += 1
            continue
        grade = grades[qid].get(doc)
        state = _state(grade)
        lead = fired.get(RANK) or max(fired.values(), key=lambda s: s["strength"])
        priority = (
            PRIORITY_CORROBORATED
            if RANK in fired and len(fired) > 1
            else PRIORITY_RANK
            if RANK in fired
            else PRIORITY_OTHER
        )
        suspects.append(
            {
                "query_id": qid,
                "doc_id": doc,
                "label": {"state": state, "grade": grade},
                "kind": "label_may_be_missing" if state == "unjudged" else "label_may_be_wrong",
                "signals": [fired[name] for name in (RANK, DUPLICATE, ANNOTATORS) if name in fired],
                "score": lead["strength"],
                "priority": priority,
                "reason": _reason(state, fired),
                "evidence": EVIDENCE,
                "source": SOURCE,
            }
        )
    suspects.sort(key=lambda s: (-s["priority"], -s["score"], s["query_id"], s["doc_id"]))
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "dataset_sha256": dataset.sha256,
        "evidence": EVIDENCE,
        "include_heldout": include_heldout,
        "settings": {"min_runs": min_runs, "duplicate_cosine": duplicate_cosine},
        "runs": used,
        "runs_ignored": ignored,
        "excluded": {
            "heldout_questions": held,
            "exploratory_repeats_of_heldout": repeats,
            "already_reviewed": skipped_reviewed,
        },
        "signals": signals,
        "summary": {
            "suspects": len(suspects),
            "questions": len({s["query_id"] for s in suspects}),
            "by_kind": dict(sorted(Counter(s["kind"] for s in suspects).items())),
            "by_signal": {
                name: sum(1 for s in suspects if any(x["signal"] == name for x in s["signals"]))
                for name in (RANK, DUPLICATE, ANNOTATORS)
            },
        },
        "suspects": suspects,
        "note": (
            "Suspects are hypotheses for a reviewer, ordered by priority and score (a "
            "ranking, not a probability). Nothing was changed: a reviewer's decisions make a "
            "new dataset version with `vectraudit workbench review-labels`."
        ),
    }


def queue_items(
    report: Mapping[str, Any], dataset: Dataset, *, include_heldout: bool = False
) -> list[dict[str, Any]]:
    """The audit's suspects as judgment-queue items for the benchmark audit.

    The report must be of this dataset version. Held-out questions never enter the queue,
    whatever the report says, unless ``include_heldout``.
    """
    if report.get("format") != FORMAT:
        raise LabelAuditError("The file is not a label audit (workbench label-audit).")
    if report.get("dataset_sha256") != dataset.sha256:
        raise LabelAuditError(
            "The label audit was made for another version of this dataset; audit this version."
        )
    exploratory = {q.query_id for q in dataset.queries if q.split == "explore"}
    return [
        {
            "query_id": s["query_id"],
            "doc_id": s["doc_id"],
            "kind": s["kind"],
            "reason": s["reason"],
            "priority": s["priority"],
            "label_audit_score": s["score"],
            "signals": [x["signal"] for x in s["signals"]],
            "evidence": EVIDENCE,
            "source": SOURCE,
        }
        for s in report.get("suspects") or []
        if include_heldout or s["query_id"] in exploratory
    ]


# --- a reviewer's decisions ------------------------------------------------------------------


def apply_label_decisions(
    dataset: Dataset,
    report: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    *,
    reviewer: str,
) -> tuple[str, list[dict[str, Any]]]:
    """A new dataset version with a reviewer's decisions on suspects from this audit.

    Each decision names a suspect (``query_id``, ``doc_id``) and either a grade 0-3 or
    ``"keep"``. Nothing is decided implicitly; every decision, kept ones included, is recorded
    under ``provenance.label_review`` so a later audit does not raise the pair again. A
    document set to 0 loses its relevant spans; one that is part of required evidence, or the
    question's last relevant document, is refused (edit the question itself). Returns the
    dataset text and the recorded decisions.
    """
    if not reviewer.strip():
        raise LabelAuditError("Name the reviewer; decisions are recorded with who made them.")
    if report.get("format") != FORMAT:
        raise LabelAuditError("The file is not a label audit (workbench label-audit).")
    if report.get("dataset_sha256") != dataset.sha256:
        raise LabelAuditError(
            "The label audit was made for another version of this dataset; audit this version "
            "again before deciding."
        )
    if not decisions:
        raise LabelAuditError("No decisions given; nothing is changed automatically.")
    suspects = {(s["query_id"], s["doc_id"]): s for s in report.get("suspects") or []}
    rows = {r["query_id"]: r for r in dataset_rows(dataset)}
    recorded: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for decision in decisions:
        key = (str(decision.get("query_id")), str(decision.get("doc_id")))
        if key not in suspects:
            raise LabelAuditError(f"{key[0]} / {key[1]} is not a suspect in this audit.")
        if key in seen:
            raise LabelAuditError(f"{key[0]} / {key[1]} is decided twice.")
        seen.add(key)
        value = decision.get("decision")
        if value != "keep" and (type(value) is not int or not 0 <= value <= 3):
            raise LabelAuditError(
                f'{key[0]} / {key[1]}: a decision is a grade from 0 to 3 or "keep".'
            )
        row = rows[key[0]]
        relevant = dict(row.get("relevant") or {})
        before = relevant.get(key[1])
        if before is None and any(s["doc_id"] == key[1] for s in row.get("relevant_spans", [])):
            before = 1.0
        entry: dict[str, Any] = {
            "query_id": key[0],
            "doc_id": key[1],
            "from": before,
            "to": before if value == "keep" else float(value),
            "decision": "kept" if value == "keep" else "changed",
            "signals": [x["signal"] for x in suspects[key]["signals"]],
            "reviewer": reviewer.strip(),
            "audited_version": dataset.sha256,
        }
        if value != "keep":
            if value == 0 and any(key[1] in group for group in row.get("required_evidence", [])):
                raise LabelAuditError(
                    f"{key[0]} / {key[1]} is part of the question's required evidence; edit "
                    "the evidence groups before judging it not relevant."
                )
            relevant[key[1]] = value
            if not any(g > 0 for g in relevant.values()) and not any(
                s["doc_id"] != key[1] for s in row.get("relevant_spans", [])
            ):
                raise LabelAuditError(
                    f"{key[0]}: {key[1]} is its last relevant document; edit the question "
                    "(another relevant document, or no_answer) rather than leave it unlabelled."
                )
            row["relevant"] = relevant
            if value == 0 and row.get("relevant_spans"):
                kept = [s for s in row["relevant_spans"] if s["doc_id"] != key[1]]
                entry["spans_removed"] = len(row["relevant_spans"]) - len(kept)
                if kept:
                    row["relevant_spans"] = kept
                else:
                    del row["relevant_spans"]
        recorded.append(entry)
    meta = dict(dataset.meta)
    provenance = dict(meta.get("provenance") or {})
    provenance["label_review"] = [*provenance.get("label_review", []), *recorded]
    meta["provenance"] = provenance
    return render_dataset(list(rows.values()), meta), recorded
