"""Pairwise agreement and conservative human resolution."""

from __future__ import annotations

import itertools
import random
from collections import Counter
from typing import Any

from vectraudit.labelling.task import Judgment, Task


def kappa(pairs: list[tuple[int, int]], *, binary: bool = False) -> float | None:
    """Unweighted Cohen kappa; undefined without pairs or with constant marginals."""
    if not pairs:
        return None
    if binary:
        pairs = [(int(a > 0), int(b > 0)) for a, b in pairs]
    n = len(pairs)
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(left[c] * right[c] for c in set(left) | set(right)) / n**2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def decisions(task: Task, qid: str, doc: str) -> list[Judgment]:
    """Non-skipped, latest judgments on one document/question pair."""
    return [
        j for j in task.judgments if j.query_id == qid and j.doc_id == doc and j.grade is not None
    ]


def resolve(task: Task, qid: str, doc: str) -> Judgment | None:
    """Reviewer wins; otherwise require a strict majority on both grade and span."""
    reviewed = next((j for j in task.resolutions if j.query_id == qid and j.doc_id == doc), None)
    if reviewed is not None:
        return reviewed
    rows = decisions(task, qid, doc)
    if not rows:
        return None
    counts = Counter((j.grade, j.start, j.end) for j in rows)
    winner, count = counts.most_common(1)[0]
    if count <= len(rows) / 2:
        return None
    return next(j for j in rows if (j.grade, j.start, j.end) == winner)


def agreement(task: Task) -> dict[str, Any]:
    """All annotator-pair statistics, plus disagreements (including span differences)."""
    people = sorted({j.annotator for j in task.judgments})
    stats = []
    for a, b in itertools.combinations(people, 2):
        left = {
            (j.query_id, j.doc_id): j.grade
            for j in task.judgments
            if j.annotator == a and j.grade is not None
        }
        right = {
            (j.query_id, j.doc_id): j.grade
            for j in task.judgments
            if j.annotator == b and j.grade is not None
        }
        pairs = [(int(left[key]), int(right[key])) for key in left.keys() & right.keys()]
        stats.append(
            {
                "annotators": [a, b],
                "n": len(pairs),
                "binary_kappa": kappa(pairs, binary=True),
                "graded_kappa": kappa(pairs),
            }
        )
    disagreements = []
    for q in task.queries:
        for doc in q.documents:
            rows = decisions(task, q.query_id, doc)
            if len({(j.grade, j.start, j.end) for j in rows}) > 1:
                disagreements.append(
                    {
                        "query_id": q.query_id,
                        "doc_id": doc,
                        "judgments": [j.model_dump() for j in rows],
                        "resolved": resolve(task, q.query_id, doc) is not None,
                    }
                )
    return {"pairs": stats, "disagreements": disagreements}


def assign_sample(task: Task, annotator: str, fraction: float, *, seed: int = 0) -> list[str]:
    """Assign a reproducible sample of questions for a blind second pass."""
    if not annotator.strip() or not 0 < fraction <= 1:
        raise ValueError("Name and sample fraction in (0,1] required")
    ids = [q.query_id for q in task.queries]
    random.Random(seed).shuffle(ids)
    selected = ids[: max(1, int(len(ids) * fraction))]
    task.assignments[annotator] = selected
    return selected
