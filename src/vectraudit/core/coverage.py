"""Where the corpus has no benchmark question: coverage regions in the embedding space.

Documents are grouped by seeded k-means on the **full-dimensional** vectors (not on a 2D map,
which distorts distance). For each group this counts its documents, the questions whose nearest
group it is, and the documents a question is labelled relevant to. A group holding at least
``min_share`` of the corpus, with no question nearest to it and whose documents are far from
every question (see :func:`coverage`), is a finding at evidence level ``hypothesis``: the
benchmark may not ask anything about that part of the corpus. It is a pointer for a person to
read, never proof the benchmark is wrong, and a question can be far from its answers in an
embedding space.

Groups are labelled with their most characteristic terms (TF-IDF, treating each group as one
document) from the local titles and first lines, so a finding says what the documents are about
without carrying their text anywhere. Held-out questions never count as covering a group.

Pure numpy; imports nothing web-related.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

DEFAULT_MIN_SHARE = 0.05
FAR_FACTOR = 2.0
"""A group is far from the questions when its documents' median distance to their nearest
question is more than this many times that of the groups that have a question."""
MAX_CLUSTERS = 48
FIT_ROWS = 50_000
ITERATIONS = 25
_CHUNK = 20_000
_TERM = re.compile(r"[a-z][a-z0-9]{2,}")
_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "not",
        "but",
        "you",
        "your",
        "can",
        "will",
        "all",
        "any",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "into",
        "out",
        "one",
        "two",
        "use",
        "used",
        "using",
        "per",
        "via",
        "its",
        "our",
        "their",
    ]
)


def default_k(n: int) -> int:
    """About the square root of the corpus size, within ``[2, MAX_CLUSTERS]`` and at most ``n``."""
    return int(min(n, max(2, min(MAX_CLUSTERS, round(math.sqrt(n))))))


def _nearest(x: FloatArray, centroids: FloatArray) -> IntArray:
    """Index of the nearest centroid (squared L2) for each row, in bounded chunks."""
    c2 = (centroids.astype(np.float64) ** 2).sum(axis=1)
    out = np.empty(len(x), dtype=np.int64)
    for start in range(0, len(x), _CHUNK):
        block = x[start : start + _CHUNK].astype(np.float64)
        d = c2[None, :] - 2.0 * block @ centroids.astype(np.float64).T
        out[start : start + _CHUNK] = d.argmin(axis=1)
    return out


def kmeans(x: FloatArray, k: int, seed: int = 0) -> tuple[FloatArray, IntArray]:
    """Seeded k-means (k-means++ start, Lloyd iterations); ``(centroids, labels)`` for all rows.

    The centroids are fitted on at most :data:`FIT_ROWS` sampled rows, then every row is
    assigned to its nearest centroid, so large corpora stay fast and the result is deterministic.
    """
    n = len(x)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and the number of vectors ({n}), got {k}.")
    rng = np.random.default_rng(seed)
    fit = x if n <= FIT_ROWS else x[np.sort(rng.choice(n, size=FIT_ROWS, replace=False))]
    fit64 = fit.astype(np.float64)
    first = int(rng.integers(len(fit)))
    centres = [fit64[first]]
    closest = ((fit64 - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = closest.sum()
        pick = (
            int(rng.choice(len(fit), p=closest / total))
            if total > 0
            else int(rng.integers(len(fit)))
        )
        centres.append(fit64[pick])
        closest = np.minimum(closest, ((fit64 - fit64[pick]) ** 2).sum(axis=1))
    centroids = np.asarray(centres, dtype=np.float32)
    for _ in range(ITERATIONS):
        labels = _nearest(fit, centroids)
        moved = centroids.copy()
        for j in range(k):
            members = fit64[labels == j]
            if len(members):
                moved[j] = members.mean(axis=0).astype(np.float32)
        if np.allclose(moved, centroids, atol=1e-6):
            break
        centroids = moved
    return centroids, _nearest(x, centroids)


def nearest_question_distance(x: FloatArray, q: FloatArray) -> npt.NDArray[np.float64]:
    """Each document's L2 distance to its nearest question, in bounded chunks."""
    q64 = q.astype(np.float64)
    q2 = (q64**2).sum(axis=1)
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), 2_000):
        block = x[start : start + 2_000].astype(np.float64)
        d2 = (block**2).sum(axis=1)[:, None] + q2[None, :] - 2.0 * block @ q64.T
        out[start : start + 2_000] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


LINK_RADII = 3.0
"""Two far regions belong to one group when their centroids are within this many radii."""


def _radii(x: FloatArray, centroids: FloatArray, labels: IntArray) -> npt.NDArray[np.float64]:
    """Root-mean-square distance of each cluster's members to its centroid."""
    total = np.zeros(len(centroids))
    for start in range(0, len(x), _CHUNK):
        diff = (
            x[start : start + _CHUNK].astype(np.float64) - centroids[labels[start : start + _CHUNK]]
        )
        total += np.bincount(
            labels[start : start + _CHUNK], weights=(diff**2).sum(axis=1), minlength=len(centroids)
        )
    counts = np.maximum(np.bincount(labels, minlength=len(centroids)), 1)
    radii: npt.NDArray[np.float64] = np.sqrt(total / counts)
    return radii


def _neighbouring(
    regions: list[dict[str, Any]], centroids: FloatArray, radii: npt.NDArray[np.float64]
) -> list[list[dict[str, Any]]]:
    """Connected groups of regions whose centroids lie within :data:`LINK_RADII` radii."""
    parent = list(range(len(regions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(regions):
        for j in range(i + 1, len(regions)):
            b = regions[j]
            gap = float(np.linalg.norm(centroids[a["cluster"]] - centroids[b["cluster"]]))
            if gap <= LINK_RADII * max(radii[a["cluster"]], radii[b["cluster"]]):
                parent[find(i)] = find(j)
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, r in enumerate(regions):
        groups.setdefault(find(i), []).append(r)
    return list(groups.values())


def top_terms(texts_by_cluster: Mapping[int, Sequence[str]], n: int = 4) -> dict[int, list[str]]:
    """The most characteristic terms of each cluster: term frequency times inverse cluster
    frequency, treating a cluster's texts as one document."""
    counts: dict[int, Counter[str]] = {}
    for cluster, texts in texts_by_cluster.items():
        tally: Counter[str] = Counter()
        for text in texts:
            tally.update(t for t in _TERM.findall((text or "").lower()) if t not in _STOP)
        counts[cluster] = tally
    n_clusters = len(counts)
    spread: Counter[str] = Counter()
    for tally in counts.values():
        spread.update(tally.keys())
    out: dict[int, list[str]] = {}
    for cluster, tally in counts.items():
        scored = sorted(
            tally.items(),
            key=lambda kv: (
                -(kv[1] * (math.log((1 + n_clusters) / (1 + spread[kv[0]])) + 1)),
                kv[0],
            ),
        )
        out[cluster] = [term for term, _ in scored[:n]]
    return out


def coverage(
    doc_ids: Sequence[str],
    doc_vectors: npt.ArrayLike,
    question_vectors: npt.ArrayLike,
    relevant: frozenset[str] | set[str],
    texts: Sequence[str | None] | None = None,
    *,
    k: int | None = None,
    seed: int = 0,
    min_share: float = DEFAULT_MIN_SHARE,
    far_factor: float = FAR_FACTOR,
) -> dict[str, Any]:
    """Coverage regions of the corpus and the findings among them.

    A region is a finding when it holds at least ``min_share`` of the corpus, no question's
    nearest region is it, **and** its documents are far from every question: their median
    distance to the nearest question exceeds ``far_factor`` times that of the regions that do
    have a question. The last condition keeps a topic that has questions from being flagged
    just because k-means cut it into several pieces.

    ``question_vectors`` must exclude held-out questions. ``relevant`` is the set of documents
    some question is labelled relevant to. ``texts`` (titles or first lines) only label groups.
    """
    if not 0 < min_share <= 1:
        raise ValueError("min_share must be in (0, 1].")
    x = np.ascontiguousarray(doc_vectors, dtype=np.float32)
    if x.ndim != 2 or len(x) != len(doc_ids) or len(x) == 0:
        raise ValueError("Give one vector per document, and at least one document.")
    q = np.ascontiguousarray(question_vectors, dtype=np.float32).reshape(-1, x.shape[1])
    kk = default_k(len(x)) if k is None else k
    centroids, labels = kmeans(x, kk, seed)
    q_labels = _nearest(q, centroids) if len(q) else np.empty(0, dtype=np.int64)
    near = nearest_question_distance(x, q) if len(q) else None
    rel_mask = np.fromiter((d in relevant for d in doc_ids), dtype=bool, count=len(doc_ids))
    present = sorted(int(c) for c in np.unique(labels))
    by_text: dict[int, list[str]] = {c: [] for c in present}
    if texts is not None:
        for i, text in enumerate(texts):
            if text:
                by_text[int(labels[i])].append(text)
    terms = top_terms(by_text)
    regions: list[dict[str, Any]] = []
    for c in present:
        members = labels == c
        n_docs = int(members.sum())
        regions.append(
            {
                "cluster": c,
                "documents": n_docs,
                "share": n_docs / len(x),
                "questions": int((q_labels == c).sum()),
                "relevant_documents": int((members & rel_mask).sum()),
                "median_question_distance": (
                    float(np.median(near[members])) if near is not None else None
                ),
                "terms": terms[c],
                "member_rows": np.flatnonzero(members),
            }
        )
    findings: list[dict[str, Any]] = []
    if len(q):
        covered = [r["median_question_distance"] for r in regions if r["questions"] > 0]
        reference = max(float(np.median(covered)), 1e-9) if covered else 1e-9
        for r in regions:
            r["distance_ratio"] = round(r["median_question_distance"] / reference, 3)
        far = [r for r in regions if r["questions"] == 0 and r["distance_ratio"] > far_factor]
        for group in _neighbouring(far, centroids, _radii(x, centroids, labels)):
            documents = sum(r["documents"] for r in group)
            share = documents / len(x)
            if share < min_share:
                continue
            clusters = sorted(r["cluster"] for r in group)
            pooled = [t for r in group for t in by_text[r["cluster"]]]
            others = {c: by_text[c] for c in present if c not in clusters}
            terms_here = top_terms({-1: pooled, **others})[-1]
            label = f" (about: {', '.join(terms_here)})" if terms_here else ""
            findings.append(
                {
                    "kind": "uncovered_region",
                    "evidence": "hypothesis",
                    "clusters": clusters,
                    "documents": documents,
                    "share": round(share, 4),
                    "distance_ratio": max(r["distance_ratio"] for r in group),
                    "terms": terms_here,
                    "message": f"About {round(share * 100)}% of documents form a group "
                    f"no benchmark question is close to{label}.",
                }
            )
    return {
        "method": "k-means on the full-dimensional vectors",
        "k": len(present),
        "seed": seed,
        "min_share": min_share,
        "far_factor": far_factor,
        "documents": len(x),
        "questions": len(q),
        "regions": regions,
        "findings": findings,
        "note": (
            "A finding is a hypothesis: the benchmark may not ask about that part of the corpus. "
            "2D maps distort distance, so findings come from this full-dimensional grouping, and "
            "a question can sit far from its answers in an embedding space."
        ),
    }


def public(result: Mapping[str, Any]) -> dict[str, Any]:
    """The result without the per-document member rows, ready for JSON."""
    return {
        **result,
        "regions": [{k: v for k, v in r.items() if k != "member_rows"} for r in result["regions"]],
    }
