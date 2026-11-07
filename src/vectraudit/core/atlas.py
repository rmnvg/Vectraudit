"""Atlas data: a 2D map of the corpus and the benchmark questions, ready for a browser.

Documents and questions are projected into the same plane (PCA, so a question is placed
exactly, by :func:`vectraudit.core.projection.place_points`) and each point gets a category
that says what happened to it in a run:

- documents: unlabelled, relevant to a question, **lost** (relevant but missing from the
  delivered results, split by where it was lost), a retrieval **hub**, or **flagged** by the
  corpus scan or a poisoning suite;
- questions: passed, failed, regressed (worse than in the baseline run), no-answer, or not
  judged.

Everything shown is measured from the run records: nothing is inferred from the picture. A 2D
layout distorts distance, so the map is for finding where to look; coverage findings come from
the full-dimensional analysis (:mod:`vectraudit.core.coverage`).

**Held-out questions are hidden** unless the caller opts in, and even then they are one
uncoloured category: they never colour themselves or any document by result, so the map cannot
be used to tune against the held-out set.

Sampling is seeded and keeps every question and every document in a non-default category, so a
failure is never sampled out; only plain, unlabelled documents are dropped to meet
``max_points`` (which may therefore be exceeded when there are more notable points than that).

The payload is binary (little-endian): ``x`` float32, ``y`` float32, ``category`` uint8 padded
to a 4-byte boundary, ``row`` uint32 (the point's row in the corpus or the dataset), each of
length ``n``, plus a JSON sidecar with identifiers, titles, the legend and the counts. This
module imports nothing web-related.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from vectraudit.core.coverage import coverage
from vectraudit.core.projection import PcaModel, Projection, ProjectionMethod, fit_pca, place_points

FORMAT = "vectraudit.atlas"
FORMAT_VERSION = 1
DEFAULT_MAX_POINTS = 50_000
HARD_MAX_POINTS = 1_000_000
TITLE_CHARS = 120
TITLES_MAX_POINTS = 200_000
"""Above this many points the sidecar omits titles, keeping it small enough to serve."""

FloatArray = npt.NDArray[np.float32]


class Category(IntEnum):
    """Point categories; the value is the byte written to the payload (at most 256)."""

    DOCUMENT = 0
    RELEVANT = 1
    LOST_NEVER_RETRIEVED = 2
    LOST_DROPPED = 3
    HUB = 4
    FLAGGED = 5
    QUESTION_PASSED = 6
    QUESTION_FAILED = 7
    QUESTION_REGRESSED = 8
    QUESTION_NO_ANSWER = 9
    QUESTION_UNJUDGED = 10


_LEGEND_ROWS = (
    (Category.DOCUMENT, "document", "Document"),
    (Category.RELEVANT, "relevant", "Relevant to a question"),
    (Category.LOST_NEVER_RETRIEVED, "lost_never_retrieved", "Lost: never retrieved"),
    (Category.LOST_DROPPED, "lost_dropped", "Lost: retrieved, then dropped"),
    (Category.HUB, "hub", "Hub (retrieved for many questions)"),
    (Category.FLAGGED, "flagged", "Flagged by the corpus scan or a poison suite"),
    (Category.QUESTION_PASSED, "question_passed", "Question: passed"),
    (Category.QUESTION_FAILED, "question_failed", "Question: failed"),
    (Category.QUESTION_REGRESSED, "question_regressed", "Question: regressed"),
    (Category.QUESTION_NO_ANSWER, "question_no_answer", "Question: no answer expected"),
    (Category.QUESTION_UNJUDGED, "question_unjudged", "Question: not judged"),
)
LEGEND: list[dict[str, Any]] = [
    {
        "id": int(c),
        "key": key,
        "kind": "question" if c >= Category.QUESTION_PASSED else "document",
        "label": label,
    }
    for c, key, label in _LEGEND_ROWS
]

QUESTION_CATEGORIES = frozenset(c for c in Category if c >= Category.QUESTION_PASSED)
# Default-colour documents are the only points sampling may drop.
_ALWAYS_KEPT = frozenset(c for c in Category if c is not Category.DOCUMENT)


class AtlasError(ValueError):
    """The atlas cannot be built; the message says what to change."""


@dataclass(frozen=True)
class AtlasQuestion:
    """One benchmark question as the atlas needs it."""

    query_id: str
    vector: npt.ArrayLike | None
    relevant: frozenset[str] = frozenset()
    """Documents with a positive relevance label."""
    no_answer: bool = False
    heldout: bool = False


@dataclass
class Atlas:
    """A built atlas: the arrays and the sidecar describing them."""

    x: FloatArray
    y: FloatArray
    category: npt.NDArray[np.uint8]
    row: npt.NDArray[np.uint32]
    sidecar: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.x)

    def payload(self) -> bytes:
        """The binary payload (see the module docstring for the layout)."""
        return encode_payload(self.x, self.y, self.category, self.row)


def encode_payload(
    x: FloatArray, y: FloatArray, category: npt.NDArray[np.uint8], row: npt.NDArray[np.uint32]
) -> bytes:
    """``x``, ``y``, ``category`` (padded to 4 bytes) and ``row``, little-endian, in that order."""
    pad = (-len(category)) % 4
    return b"".join(
        (
            np.ascontiguousarray(x, dtype="<f4").tobytes(),
            np.ascontiguousarray(y, dtype="<f4").tobytes(),
            np.ascontiguousarray(category, dtype=np.uint8).tobytes() + b"\x00" * pad,
            np.ascontiguousarray(row, dtype="<u4").tobytes(),
        )
    )


def decode_payload(data: bytes, n: int) -> tuple[FloatArray, FloatArray, Any, Any]:
    """The inverse of :func:`encode_payload` for ``n`` points."""
    padded = n + (-n) % 4
    if len(data) != 8 * n + padded + 4 * n:
        raise AtlasError(f"The payload is {len(data)} bytes, expected {12 * n + padded - n}.")
    x = np.frombuffer(data, dtype="<f4", count=n, offset=0)
    y = np.frombuffer(data, dtype="<f4", count=n, offset=4 * n)
    category = np.frombuffer(data, dtype=np.uint8, count=n, offset=8 * n)
    row = np.frombuffer(data, dtype="<u4", count=n, offset=8 * n + padded)
    return x, y, category, row


# --- what happened to each document and question -------------------------------------------


def _delivered(entry: Mapping[str, Any]) -> list[str]:
    return [str(i) for i in (entry.get("ids") or [])]


def _staged(entry: Mapping[str, Any]) -> set[str]:
    stages = (entry.get("evidence") or {}).get("stages") or []
    return {str(h["id"]) for s in stages for h in (s.get("hits") or []) if "id" in h}


def _recall(entry: Mapping[str, Any] | None, relevant: frozenset[str]) -> float | None:
    """Share of the relevant documents delivered; ``None`` when it cannot be judged."""
    if entry is None or entry.get("error") or entry.get("ids") is None or not relevant:
        return None
    return len(relevant & set(_delivered(entry))) / len(relevant)


def question_category(
    q: AtlasQuestion,
    selected: Mapping[str, Mapping[str, Any]] | None,
    baseline: Mapping[str, Mapping[str, Any]] | None,
) -> Category:
    """The category of a question; held-out questions are never judged here."""
    if q.heldout:
        return Category.QUESTION_UNJUDGED
    if q.no_answer:
        return Category.QUESTION_NO_ANSWER
    if selected is None or not q.relevant:
        return Category.QUESTION_UNJUDGED
    entry = selected.get(q.query_id)
    if entry is None:
        return Category.QUESTION_UNJUDGED
    now = _recall(entry, q.relevant)
    if entry.get("error") or now is None:
        return Category.QUESTION_FAILED  # an unanswered question is never counted as passed
    before = _recall(baseline.get(q.query_id), q.relevant) if baseline is not None else None
    if before is not None and now < before - 1e-9:
        return Category.QUESTION_REGRESSED
    return Category.QUESTION_PASSED if now >= 1 - 1e-9 else Category.QUESTION_FAILED


def document_categories(
    doc_ids: Sequence[str],
    questions: Sequence[AtlasQuestion],
    selected: Mapping[str, Mapping[str, Any]] | None,
    hubs: set[str],
    flagged: Mapping[str, str],
) -> list[Category]:
    """The category of each document. Held-out questions contribute nothing."""
    relevant: set[str] = set()
    lost_never: set[str] = set()
    lost_dropped: set[str] = set()
    for q in questions:
        if q.heldout or q.no_answer:
            continue
        relevant |= q.relevant
        entry = (selected or {}).get(q.query_id)
        if entry is None or entry.get("error") or entry.get("ids") is None:
            continue
        found = set(_delivered(entry))
        staged = _staged(entry)
        for doc in q.relevant - found:
            (lost_dropped if doc in staged else lost_never).add(doc)
    out = []
    for doc in doc_ids:
        if doc in flagged:
            out.append(Category.FLAGGED)
        elif doc in lost_never:
            out.append(Category.LOST_NEVER_RETRIEVED)
        elif doc in lost_dropped:
            out.append(Category.LOST_DROPPED)
        elif doc in hubs:
            out.append(Category.HUB)
        elif doc in relevant:
            out.append(Category.RELEVANT)
        else:
            out.append(Category.DOCUMENT)
    return out


# --- building ------------------------------------------------------------------------------


def _title(text: str | None, limit: int = TITLE_CHARS) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def build_atlas(
    doc_ids: Sequence[str],
    doc_vectors: npt.ArrayLike,
    questions: Sequence[AtlasQuestion],
    *,
    doc_titles: Sequence[str | None] | None = None,
    selected_run: Mapping[str, Any] | None = None,
    baseline_run: Mapping[str, Any] | None = None,
    hubs: set[str] | None = None,
    flagged: Mapping[str, str] | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 0,
    show_heldout: bool = False,
    name: str = "atlas",
    inputs: Mapping[str, Any] | None = None,
) -> Atlas:
    """Project the corpus and the questions and categorise every point.

    ``selected_run`` is the run whose outcomes colour the map; ``baseline_run`` (the earlier
    run) makes a question that got worse a *regression*. ``flagged`` maps a document ID to why
    it is flagged. ``max_points`` bounds the plain documents kept (see the module docstring).
    """
    if not 1 <= max_points <= HARD_MAX_POINTS:
        raise AtlasError(
            f"max_points must be between 1 and {HARD_MAX_POINTS:,}, got {max_points}.",
        )
    x = np.ascontiguousarray(doc_vectors, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != len(doc_ids) or x.shape[0] == 0:
        raise AtlasError("The corpus needs one vector per document, and at least one document.")
    if x.shape[1] < 2:
        raise AtlasError("Vectors need at least 2 dimensions to be drawn in 2D.")
    if not np.isfinite(x).all():
        raise AtlasError("The corpus vectors contain NaN or infinite values.")
    if len(set(doc_ids)) != len(doc_ids):
        raise AtlasError("Document IDs must be unique.")
    titles = list(doc_titles) if doc_titles is not None else [None] * len(doc_ids)

    hidden = [q for q in questions if q.heldout and not show_heldout]
    visible = [q for q in questions if not (q.heldout and not show_heldout)]
    with_vectors = [q for q in visible if q.vector is not None]
    qvec = (
        np.ascontiguousarray([q.vector for q in with_vectors], dtype=np.float32)
        if with_vectors
        else np.empty((0, x.shape[1]), dtype=np.float32)
    )
    if len(qvec) and (qvec.ndim != 2 or qvec.shape[1] != x.shape[1]):
        raise AtlasError("Question vectors must have the same dimension as the documents.")
    if not np.isfinite(qvec).all():
        raise AtlasError("The question vectors contain NaN or infinite values.")

    pca: PcaModel = fit_pca(x, 2)
    proj = Projection(
        ProjectionMethod.PCA, 2, np.arange(len(x), dtype=np.int64), pca.transform(x), None, None,
        len(x), pca,
    )  # fmt: skip
    doc_xy = proj.coords
    q_xy = place_points(proj, qvec, None) if len(qvec) else np.empty((0, 2), dtype=np.float32)

    selected = _by_query(selected_run)
    baseline = _by_query(baseline_run)
    doc_cat = document_categories(list(doc_ids), questions, selected, hubs or set(), flagged or {})
    q_cat = [question_category(q, selected, baseline) for q in with_vectors]

    # --- sampling: only plain documents may be dropped -----------------------------------
    plain = [i for i, c in enumerate(doc_cat) if c is Category.DOCUMENT]
    kept_plain = len(plain)
    fixed = len(doc_cat) - len(plain) + len(with_vectors)
    room = max(0, max_points - fixed)
    if len(plain) > room:
        rng = np.random.default_rng(seed)
        chosen: list[int] = (
            np.sort(rng.choice(len(plain), size=room, replace=False)).tolist() if room else []
        )
        keep_plain = {plain[int(i)] for i in chosen}
        kept_plain = len(keep_plain)
    else:
        keep_plain = set(plain)
    doc_rows = [i for i, c in enumerate(doc_cat) if c is not Category.DOCUMENT or i in keep_plain]

    px = np.concatenate([doc_xy[doc_rows, 0], q_xy[:, 0]]).astype(np.float32)
    py = np.concatenate([doc_xy[doc_rows, 1], q_xy[:, 1]]).astype(np.float32)
    cat = np.array([int(doc_cat[i]) for i in doc_rows] + [int(c) for c in q_cat], dtype=np.uint8)
    qrow = {q.query_id: r for r, q in enumerate(questions)}
    row = np.array([*doc_rows, *(qrow[q.query_id] for q in with_vectors)], dtype=np.uint32)
    ids = [str(doc_ids[i]) for i in doc_rows] + [q.query_id for q in with_vectors]
    point_titles = [_title(titles[i]) if titles[i] else None for i in doc_rows] + [None] * len(
        with_vectors
    )

    # Regions come from the full-dimensional grouping that the coverage findings use, so the
    # picture and the findings agree. Held-out questions never count as covering a region.
    judged = [q for q in with_vectors if not q.heldout]
    cov = coverage(
        doc_ids,
        x,
        np.ascontiguousarray([q.vector for q in judged], dtype=np.float32)
        if judged
        else np.empty((0, x.shape[1]), dtype=np.float32),
        frozenset(d for q in questions if not q.heldout for d in q.relevant),
        titles,
        seed=seed,
    )
    uncovered = {c for f in cov["findings"] for c in f["clusters"]}
    regions = [
        {
            **{k: v for k, v in r.items() if k != "member_rows"},
            "centre": [float(v) for v in doc_xy[r["member_rows"]].mean(axis=0)],
            "uncovered": r["cluster"] in uncovered,
        }
        for r in cov["regions"]
    ]
    atlas = Atlas(px, py, cat, row)
    payload = atlas.payload()
    counts = {
        "documents_total": len(doc_ids),
        "documents_shown": len(doc_rows),
        "plain_documents_total": len(plain),
        "plain_documents_shown": kept_plain,
        "questions_total": len(questions),
        "questions_shown": len(with_vectors),
        "questions_without_vectors": len(visible) - len(with_vectors),
        "heldout_hidden": len(hidden),
        "heldout_shown": sum(q.heldout for q in with_vectors),
        "by_category": {str(item["key"]): int((cat == item["id"]).sum()) for item in LEGEND},
    }
    atlas.sidecar = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "local_only": True,
        "contains_text": ["titles"],
        "projection": {
            "method": "pca",
            "dims": 2,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio],
            "note": (
                "A 2D layout distorts distance: points that look close may not be, and the "
                "reverse. Use the map to decide where to look, not as evidence."
            ),
        },
        "counts": counts,
        "legend": LEGEND,
        "sampling": {
            "seed": seed,
            "max_points": max_points,
            "over_cap": len(doc_rows) + len(with_vectors) > max_points,
            "rule": "every question and every document in a non-default category is kept",
        },
        "heldout": {
            "shown": bool(show_heldout),
            "note": "Held-out questions never colour a point or a document by result.",
        },
        "runs": {
            "selected": _brief(selected_run),
            "baseline": _brief(baseline_run),
        },
        "inputs": dict(inputs or {}),
        "extent": {
            "x": [float(px.min()), float(px.max())],
            "y": [float(py.min()), float(py.max())],
        },
        "points": {
            "n": len(px),
            "ids": ids,
            "titles": point_titles if len(px) <= TITLES_MAX_POINTS else None,
            "titles_omitted": len(px) > TITLES_MAX_POINTS,
        },
        "bin": {
            "n": len(px),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "layout": "float32 x[n], float32 y[n], uint8 category[n] (padded to 4), uint32 row[n]; "
            "little-endian",
        },
        "regions": regions,
        "findings": cov["findings"],
        "coverage": {k: cov[k] for k in ("method", "k", "seed", "min_share", "far_factor", "note")},
    }
    return atlas


def _by_query(run: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]] | None:
    if run is None:
        return None
    return {str(e["query_id"]): e for e in run.get("queries") or []}


def _brief(run: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {"run_id": run.get("run_id"), "label": run.get("label")}
