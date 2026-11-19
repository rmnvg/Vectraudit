"""Embedding drift: have production questions, or the corpus, moved away from the reference?

The descriptor tests in :mod:`vectraudit.pipelines.drift` catch drift that shows in length,
script, vocabulary or context. A new topic phrased in ordinary words shows in none of them,
but it moves the questions' embeddings. This compares two sets of vectors: the benchmark's
exploratory questions against production questions, two windows of production questions, or
two exports of the corpus.

Four methods, as Evidently defines them (read in Evidently 0.7.22, commit 34771fe41c08, file
``legacy/metrics/data_drift/embedding_drift_methods.py``; its current ``EmbeddingsDrift``
metric wraps the same code):

- **model** (the default, as in Evidently): a classifier is trained to tell the two sets
  apart on a seeded half of the pooled vectors and scored by ROC AUC on the other half; 0.5
  means it cannot. With fewer than 1,000 reference vectors the AUC is compared with the 95th
  percentile of 100 AUCs of random scores on the same held-out labels (a test at about the
  5% level); otherwise drift is an AUC above 0.55. Evidently uses scikit-learn's SGD
  logistic regression; this is a ridge-penalised logistic regression fitted by accelerated
  gradient descent in numpy. The test stays valid whatever the classifier, because it is
  scored on vectors it never saw.
- **ratio**: the share of components (dimensions, or principal components) whose normed
  Wasserstein distance is at least 0.1; drift when more than 20% of them moved.
- **distance**: the Euclidean distance between the two means; drift above 0.2, or, with
  ``bootstrap``, above the 95th percentile of the distance between two resamples of the
  reference. The fixed threshold depends on the vectors' scale.
- **mmd**: the unbiased squared maximum mean discrepancy with a Gaussian kernel whose width
  is the median distance between the sets. Evidently's bootstrap compares the statistic with
  the 5th percentile of its resamples, which flags drift almost every time, so this uses a
  seeded permutation p-value instead: drift at p < 0.05.

``pca_components`` projects both sets on the reference's principal components first, as
Evidently's option does. Large sets are sampled (seeded) to keep the classifier and the
kernel bounded; the report says when.

A drift result is evidence level ``hypothesis``: it says the sets differ, not that retrieval
got worse. The report holds counts and statistics; group terms and identifiers appear only
when the drift is located (:func:`where_drift`), and the report stays local.

Credit: Evidently (Apache-2.0), whose embedding drift methods and defaults these follow;
Gretton, Borgwardt, Rasch, Schoelkopf and Smola, "A kernel two-sample test" (JMLR 2012) for
the MMD; Lopez-Paz and Oquab, "Revisiting classifier two-sample tests" (ICLR 2017) for the
classifier test. Our own numpy implementation; see docs/credits.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from vectraudit.pipelines.drift import wasserstein_normed

if TYPE_CHECKING:
    from vectraudit.evaluation.dataset import Dataset

FORMAT = "vectraudit.embedding_drift"
FORMAT_VERSION = 1
METHODS = ("model", "ratio", "distance", "mmd")
DEFAULT_METHOD = "model"  # Evidently's default
EVIDENTLY_READ = "Evidently 0.7.22 (commit 34771fe41c08)"
MODEL_THRESHOLD = 0.55
BOOTSTRAP_BELOW = 1000  # Evidently tests the classifier against random scores below this
NULL_DRAWS = 100
NULL_QUANTILE = 0.95
COMPONENT_THRESHOLD = 0.1
RATIO_THRESHOLD = 0.2
DISTANCE_THRESHOLD = 0.2
MMD_PERMUTATIONS = 200
MMD_ALPHA = 0.05
MMD_MAX_ROWS = 1000
MAX_ROWS = 10_000
MIN_ROWS = 10
SMALL_REFERENCE = 200  # below this the classifier test misses a topic a fifth of the traffic
RIDGE = 1.0
ITERATIONS = 200

FloatMatrix = npt.NDArray[np.float64]


class EmbeddingDriftError(ValueError):
    """The comparison cannot be made as asked; the message says why."""


def _matrix(vectors: npt.ArrayLike, name: str) -> FloatMatrix:
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or len(x) < MIN_ROWS:
        raise EmbeddingDriftError(f"The {name} set needs at least {MIN_ROWS} vectors.")
    if not np.isfinite(x).all():
        raise EmbeddingDriftError(f"The {name} vectors contain NaN or infinite values.")
    return x


def _sample(x: FloatMatrix, limit: int, rng: np.random.Generator) -> FloatMatrix:
    if len(x) <= limit:
        return x
    return x[np.sort(rng.choice(len(x), limit, replace=False))]


def _pca(reference: FloatMatrix, current: FloatMatrix, k: int) -> tuple[FloatMatrix, FloatMatrix]:
    """Both sets on the reference's first ``k`` principal components."""
    if not 1 <= k <= min(reference.shape):
        raise EmbeddingDriftError(
            f"pca_components must be between 1 and {min(reference.shape)} for these vectors."
        )
    mean = reference.mean(axis=0)
    _, _, vt = np.linalg.svd(reference - mean, full_matrices=False)
    basis = vt[:k].T
    return (reference - mean) @ basis, (current - mean) @ basis


def roc_auc(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """ROC AUC of ``scores`` for the positive ``labels`` (Mann-Whitney, ties counted half)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        raise EmbeddingDriftError("ROC AUC needs both sets among the held-out vectors.")
    order = np.argsort(s, kind="mergesort")
    _, first, counts = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _top_singular_value(x: FloatMatrix, rng: np.random.Generator) -> float:
    v = np.asarray(rng.normal(size=x.shape[1]), dtype=np.float64)
    for _ in range(30):
        v = x.T @ (x @ v)
        norm = float(np.linalg.norm(v))
        if norm == 0:
            return 0.0
        v /= norm
    return math.sqrt(float(np.linalg.norm(x.T @ (x @ v))))


def _logistic(x: FloatMatrix, y: npt.NDArray[np.float64], rng: np.random.Generator) -> FloatMatrix:
    """Ridge-penalised logistic regression (weights then intercept) by Nesterov descent."""
    n = len(x)
    aug = np.hstack([x, np.ones((n, 1))])
    step = 1 / (_top_singular_value(aug, rng) ** 2 / (4 * n) + RIDGE)
    penalty = np.ones(aug.shape[1])
    penalty[-1] = 0.0  # the intercept is not penalised
    w: FloatMatrix = np.zeros(aug.shape[1])
    v: FloatMatrix = w.copy()
    t = 1.0
    for _ in range(ITERATIONS):
        p = 0.5 * (1 + np.tanh(0.5 * (aug @ v)))
        grad = aug.T @ (p - y) / n + RIDGE * penalty * v
        w_next = v - step * grad
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        v = w_next + ((t - 1) / t_next) * (w_next - w)
        w, t = w_next, t_next
    return w


def _grouped_halves(
    x: FloatMatrix, rng: np.random.Generator
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """A seeded half/half split that keeps identical vectors on the same side.

    A question asked verbatim in both sets (or a document unchanged between two exports)
    would otherwise be learned from one copy and scored on the other, pulling the held-out AUC
    below 0.5 and hiding real drift. Kept together, the copies tie and count as 0.5.
    """
    _, group = np.unique(x, axis=0, return_inverse=True)
    group = group.reshape(-1)
    order = rng.permutation(int(group.max()) + 1)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    rows = np.argsort(rank[group], kind="stable")
    sizes = np.bincount(group)[order]
    cut = int(np.searchsorted(np.cumsum(sizes), len(x) / 2))
    first = int(np.sum(sizes[: cut + 1]))
    return rows[:first].astype(np.int64), rows[first:].astype(np.int64)


def domain_classifier(
    reference: FloatMatrix, current: FloatMatrix, *, bootstrap: bool, seed: int
) -> dict[str, Any]:
    """Evidently's ``model`` method: held-out ROC AUC of a classifier telling the sets apart."""
    rng = np.random.default_rng(seed)
    x = np.vstack([reference, current])
    y = np.concatenate([np.ones(len(reference)), np.zeros(len(current))])
    train, test = _grouped_halves(x, rng)
    mean, std = x[train].mean(axis=0), x[train].std(axis=0)
    std[std < 1e-12] = 1.0
    z = (x - mean) / std
    w = _logistic(z[train], y[train], rng)
    scores = np.hstack([z[test], np.ones((len(test), 1))]) @ w
    auc = roc_auc(scores, y[test])
    if bootstrap:
        null = [roc_auc(rng.random(len(test)), y[test]) for _ in range(NULL_DRAWS)]
        threshold = float(np.percentile(null, 100 * NULL_QUANTILE))
        rule = (
            f"drift when the ROC AUC is above the {NULL_QUANTILE:.0%} quantile of "
            f"{NULL_DRAWS} AUCs of random scores ({threshold:.3f})"
        )
    else:
        threshold = MODEL_THRESHOLD
        rule = f"drift when the ROC AUC is above {MODEL_THRESHOLD}"
    return {
        "statistic": "domain classifier ROC AUC",
        "value": round(auc, 4),
        "threshold": round(threshold, 4),
        "p_value": None,
        "drift": auc > threshold,
        "rule": rule,
        "held_out": len(test),
    }


def drifted_share(reference: FloatMatrix, current: FloatMatrix) -> dict[str, Any]:
    """Evidently's ``ratio`` method: the share of components whose distribution moved."""
    moved = sum(
        wasserstein_normed(reference[:, i].tolist(), current[:, i].tolist()) >= COMPONENT_THRESHOLD
        for i in range(reference.shape[1])
    )
    share = moved / reference.shape[1]
    return {
        "statistic": "share of drifted components",
        "value": round(share, 4),
        "threshold": RATIO_THRESHOLD,
        "p_value": None,
        "drift": share > RATIO_THRESHOLD,
        "rule": (
            f"drift when more than {RATIO_THRESHOLD:.0%} of components have a normed "
            f"Wasserstein distance of at least {COMPONENT_THRESHOLD}"
        ),
        "components_drifted": int(moved),
        "components": reference.shape[1],
    }


def mean_distance(
    reference: FloatMatrix, current: FloatMatrix, *, bootstrap: bool, seed: int
) -> dict[str, Any]:
    """Evidently's ``distance`` method: Euclidean distance between the means."""
    value = float(np.linalg.norm(reference.mean(axis=0) - current.mean(axis=0)))
    if bootstrap:
        rng = np.random.default_rng(seed)
        m, n = len(reference), len(current)
        a, b = max(1, m * m // (m + n)), max(1, n * m // (m + n))
        null = [
            float(
                np.linalg.norm(
                    reference[rng.integers(0, m, a)].mean(axis=0)
                    - reference[rng.integers(0, m, b)].mean(axis=0)
                )
            )
            for _ in range(NULL_DRAWS)
        ]
        threshold = float(np.percentile(null, 100 * NULL_QUANTILE))
        rule = (
            f"drift when the distance is above the {NULL_QUANTILE:.0%} quantile of "
            f"{NULL_DRAWS} distances between resamples of the reference ({threshold:.4f})"
        )
    else:
        threshold = DISTANCE_THRESHOLD
        rule = f"drift when the distance is above {DISTANCE_THRESHOLD} (depends on scale)"
    return {
        "statistic": "Euclidean distance between means",
        "value": round(value, 6),
        "threshold": round(threshold, 6),
        "p_value": None,
        "drift": value > threshold,
        "rule": rule,
    }


def mmd(reference: FloatMatrix, current: FloatMatrix, *, seed: int) -> dict[str, Any]:
    """Unbiased squared MMD (Gaussian kernel, median width) with a permutation p-value."""
    rng = np.random.default_rng(seed)
    x = _sample(reference, MMD_MAX_ROWS, rng)
    y = _sample(current, MMD_MAX_ROWS, rng)
    m, n = len(x), len(y)
    pooled = np.vstack([x, y])
    sq = (pooled**2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * pooled @ pooled.T, 0.0)
    width = float(np.median(np.sqrt(d2[:m, m:]))) ** 2
    kernel = np.exp(-d2 / max(width, 1e-12))

    def statistic(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """MMD^2_u for each column of ``u`` (1 marks the reference rows)."""
        v = 1 - u
        ku = kernel @ u
        s_xx = (u * ku).sum(axis=0)
        s_xy = (v * ku).sum(axis=0)
        s_yy = (v * (kernel @ v)).sum(axis=0)
        out: npt.NDArray[np.float64] = (
            (s_xx - m) / (m * (m - 1)) + (s_yy - n) / (n * (n - 1)) - 2 * s_xy / (m * n)
        )
        return out

    observed = float(statistic(np.r_[np.ones(m), np.zeros(n)][:, None])[0])
    marks = np.zeros((m + n, MMD_PERMUTATIONS))
    for j in range(MMD_PERMUTATIONS):
        marks[rng.permutation(m + n)[:m], j] = 1.0
    null = statistic(marks)
    p_value = (1 + int((null >= observed).sum())) / (1 + MMD_PERMUTATIONS)
    return {
        "statistic": "squared maximum mean discrepancy",
        "value": round(max(observed, 0.0), 6),
        "threshold": MMD_ALPHA,
        "p_value": round(p_value, 4),
        "drift": p_value < MMD_ALPHA,
        "rule": f"drift when the permutation p-value ({MMD_PERMUTATIONS} permutations) is "
        f"below {MMD_ALPHA}",
        "kernel_width": round(math.sqrt(width), 6),
    }


def embedding_drift(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    *,
    method: str = DEFAULT_METHOD,
    seed: int = 0,
    pca_components: int | None = None,
    bootstrap: bool | None = None,
) -> dict[str, Any]:
    """Compare two sets of vectors with one of Evidently's embedding drift methods.

    ``bootstrap=None`` follows Evidently: the ``model`` method tests against random scores
    when the reference has fewer than 1,000 vectors, and ``distance`` uses its fixed
    threshold. ``mmd`` always uses a permutation p-value and ``ratio`` a fixed share.
    """
    if method not in METHODS:
        raise EmbeddingDriftError(f"method must be one of {', '.join(METHODS)}.")
    ref = _matrix(reference, "reference")
    cur = _matrix(current, "current")
    if ref.shape[1] != cur.shape[1]:
        raise EmbeddingDriftError(
            f"The sets have different dimensions ({ref.shape[1]} and {cur.shape[1]}): they were "
            "not embedded with the same model."
        )
    n_ref, n_cur = len(ref), len(cur)
    rng = np.random.default_rng(seed)
    ref, cur = _sample(ref, MAX_ROWS, rng), _sample(cur, MAX_ROWS, rng)
    if pca_components:
        ref, cur = _pca(ref, cur, pca_components)
    if method == "model":
        boot = n_ref < BOOTSTRAP_BELOW if bootstrap is None else bootstrap
        result = domain_classifier(ref, cur, bootstrap=boot, seed=seed)
    elif method == "ratio":
        result = drifted_share(ref, cur)
    elif method == "distance":
        result = mean_distance(ref, cur, bootstrap=bool(bootstrap), seed=seed)
    else:
        result = mmd(ref, cur, seed=seed)
    sampled = len(ref) < n_ref or len(cur) < n_cur
    warnings = []
    if method == "model" and n_ref < SMALL_REFERENCE:
        warnings.append(
            f"With fewer than {SMALL_REFERENCE} reference vectors the classifier test has little "
            "power: when the rest of the traffic is unchanged, a new topic must be a large share "
            "of it to show (a perfect classifier reaches only 0.5 + share / 2). On synthetic "
            "questions mmd found a topic making up a fifth of the traffic far more often; "
            "consider --method mmd."
        )
    if method == "ratio" and n_ref < BOOTSTRAP_BELOW:
        warnings.append(
            "With fewer than 1,000 reference vectors the per-component distances are noisy: "
            "on synthetic windows of 300 unchanged vectors this method flagged every one."
        )
    if method == "distance" and not bootstrap:
        warnings.append(
            "The fixed threshold depends on the vectors' scale (on unit vectors it rarely "
            "fires); pass bootstrap to compare with resamples of the reference instead."
        )
    return {
        "method": method,
        "evidently_default": method == DEFAULT_METHOD,
        **result,
        "n_reference": n_ref,
        "n_current": n_cur,
        "dimensions": int(np.asarray(reference).shape[1]),
        "pca_components": pca_components,
        "sampled": {"reference": len(ref), "current": len(cur)} if sampled else None,
        "seed": seed,
        "warnings": warnings,
    }


def drift_report(
    result: Mapping[str, Any],
    *,
    kind: str,
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    where: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``vectraudit.embedding_drift`` report around one :func:`embedding_drift` result."""
    if kind not in ("questions", "corpus"):
        raise EmbeddingDriftError("kind must be questions or corpus.")
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "status": "drift" if result["drift"] else "no_drift",
        "evidence": "hypothesis",
        "reference": dict(reference),
        "current": dict(current),
        "result": dict(result),
        "where": dict(where) if where is not None else None,
        "method_source": f"{EVIDENTLY_READ}; see docs/production-intake.md",
        "note": (
            "Drift says the two sets of vectors differ, not that retrieval got worse. Compare "
            "the release check on the new traffic before acting on it. Local only."
        ),
    }


def render(report: Mapping[str, Any]) -> str:
    """The embedding drift report as plain text for the terminal."""
    r = report["result"]
    what = "questions" if report["kind"] == "questions" else "documents"
    head = (
        f"Embedding drift: {'DRIFT' if r['drift'] else 'no drift'} ({r['n_current']} "
        f"{report['current']['label']} vs {r['n_reference']} {report['reference']['label']} "
        f"{what}, method {r['method']})"
    )
    value = f"{r['statistic']}: {r['value']}"
    if r["p_value"] is not None:
        value += f", p = {r['p_value']}"
    lines = [head, f"  {value}; {r['rule']}."]
    if r["sampled"]:
        s = r["sampled"]
        lines.append(f"  Sampled {s['reference']} and {s['current']} vectors (seed {r['seed']}).")
    lines += [f"  Warning: {w}" for w in r["warnings"]]
    where = report.get("where") or {}
    for group in where.get("groups") or []:
        lines.append(f"  Hypothesis: {group['message']}")
    if where.get("note"):
        lines.append(f"  {where['note']}")
    return "\n".join(lines)


def where_drift(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    current_ids: Sequence[str],
    current_texts: Sequence[str | None] | None,
    *,
    reference_texts: Sequence[str | None] | None = None,
    nouns: tuple[str, str] = ("production questions", "benchmark question"),
    seed: int = 0,
    listed: int = 1000,
) -> dict[str, Any]:
    """Where the current set moved: groups of it far from every reference vector.

    Clusters the reference and current vectors together by k-means and keeps the groups no
    reference vector is near, by the coverage rule of :func:`vectraudit.core.coverage.coverage`
    (the reference vectors play the questions). Pooling keeps the rule meaningful when every
    current item is new, as with the documents a corpus export added. Groups are labelled with
    their most characteristic terms (texts only label groups and are not kept); each lists up to
    ``listed`` of its current identifiers, so its items can be queued for review.
    """
    from vectraudit.core.coverage import coverage

    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32)
    n_ref = len(ref)
    texts: list[str | None] = [
        *(reference_texts or [None] * n_ref),
        *(current_texts or [None] * len(cur)),
    ]
    found = coverage(
        [f"r{i}" for i in range(n_ref)] + [f"c{i}" for i in range(len(cur))],
        np.vstack([ref, cur]),
        ref,
        frozenset(),
        texts,
        seed=seed,
    )
    rows = {r["cluster"]: r["member_rows"] for r in found["regions"]}
    groups = []
    for finding in found["findings"]:
        members = np.sort(np.concatenate([rows[c] for c in finding["clusters"]]))
        members = members[members >= n_ref] - n_ref
        if not len(members):
            continue
        share = len(members) / len(cur)
        about = f" (about: {', '.join(finding['terms'])})" if finding["terms"] else ""
        groups.append(
            {
                "share": round(share, 4),
                "items": len(members),
                "terms": finding["terms"],
                "distance_ratio": finding["distance_ratio"],
                "ids": [current_ids[int(i)] for i in members[:listed]],
                "evidence": "hypothesis",
                "message": f"About {round(share * 100)}% of the {nouns[0]} form a group no "
                f"{nouns[1]} is close to{about}.",
            }
        )
    return {
        "method": "k-means on the reference and current vectors together",
        "k": found["k"],
        "seed": seed,
        "groups": groups,
        "note": None
        if groups
        else f"No group of the {nouns[0]} stands apart from every {nouns[1]}; the drift is "
        "spread out rather than one new topic.",
    }


# --- inputs ---------------------------------------------------------------------------------

Embed = Callable[[str], npt.ArrayLike]


def question_model(dataset: Dataset, embedder: str | None) -> str:
    """The model to embed questions with: ``embedder``, else the benchmark's own.

    Refuses a model other than the one the benchmark's vectors were made with: comparing
    vectors from two models measures the models, not the questions.
    """
    recorded = dataset.meta.get("embedding_model")
    has_vectors = any(q.vector is not None for q in dataset.queries if q.split == "explore")
    if embedder and recorded and embedder != recorded and has_vectors:
        raise EmbeddingDriftError(
            f"The benchmark's vectors were made with {recorded}; embedding production questions "
            f"with {embedder} would compare two models. Leave --embedder out, or re-embed the "
            "benchmark with `vectraudit dataset embed`."
        )
    model = embedder or recorded
    if not model:
        raise EmbeddingDriftError(
            "Name the embedding model with --embedder: the benchmark records none "
            "(`vectraudit dataset embed` records it)."
        )
    return str(model)


def wants_normalized(dataset: Dataset) -> bool:
    """Whether new embeddings should be unit length, to match the benchmark's vectors."""
    from vectraudit.core.embed import looks_normalized

    if "normalized" in dataset.meta:
        return bool(dataset.meta["normalized"])
    vectors = [q.vector for q in dataset.queries if q.vector is not None]
    return bool(vectors) and looks_normalized(np.asarray(vectors))


def _embed(texts: Sequence[str], embed: Embed) -> FloatMatrix:
    return np.asarray([np.asarray(embed(t), dtype=np.float64) for t in texts])


def benchmark_drift(
    dataset: Dataset,
    production: Sequence[tuple[str, str]],
    embed: Embed,
    *,
    model: str,
    method: str = DEFAULT_METHOD,
    seed: int = 0,
    source: str = "traces",
) -> dict[str, Any]:
    """Benchmark questions (reference) against production questions (``(trace ID, text)``).

    Only exploratory questions are the reference; held-out ones stay out of exploratory work.
    A question without a vector is embedded locally with ``embed``.
    """
    explore = [
        q for q in dataset.queries if q.split == "explore" and (q.vector is not None or q.text)
    ]
    if not explore:
        raise EmbeddingDriftError("The benchmark has no exploratory question to compare with.")
    missing = [q.text or "" for q in explore if q.vector is None]
    vectors = iter(_embed(missing, embed)) if missing else iter(())
    reference = np.asarray(
        [q.vector.astype(np.float64) if q.vector is not None else next(vectors) for q in explore]
    )
    current = _embed([text for _, text in production], embed)
    if not len(current):
        raise EmbeddingDriftError("No production question text to compare.")
    result = embedding_drift(reference, current, method=method, seed=seed)
    where = None
    if result["drift"]:
        where = where_drift(
            reference,
            current,
            [trace_id for trace_id, _ in production],
            [text for _, text in production],
            reference_texts=[q.text for q in explore],
            seed=seed,
        )
    return drift_report(
        result,
        kind="questions",
        reference={
            "label": "benchmark",
            "source": "exploratory benchmark questions",
            "dataset_sha256": dataset.sha256,
            "count": len(explore),
            "embedded_here": len(missing),
            "model": model,
        },
        current={
            "label": "production",
            "source": source,
            "count": len(production),
            "embedded_here": len(production),
            "model": model,
            "ids": "trace",
        },
        where=where,
    )


def window_drift(
    earlier: Sequence[tuple[str, str]],
    later: Sequence[tuple[str, str]],
    embed: Embed,
    *,
    model: str,
    method: str = DEFAULT_METHOD,
    seed: int = 0,
    sources: tuple[str, str] = ("earlier traces", "later traces"),
) -> dict[str, Any]:
    """Two windows of production questions (``(trace ID, text)``), both embedded locally."""
    if not earlier or not later:
        raise EmbeddingDriftError("Both windows need production questions with text.")
    old, new = _embed([t for _, t in earlier], embed), _embed([t for _, t in later], embed)
    result = embedding_drift(old, new, method=method, seed=seed)
    where = None
    if result["drift"]:
        where = where_drift(
            old,
            new,
            [trace_id for trace_id, _ in later],
            [t for _, t in later],
            reference_texts=[t for _, t in earlier],
            nouns=("later questions", "earlier question"),
            seed=seed,
        )
    return drift_report(
        result,
        kind="questions",
        reference={"label": "earlier", "source": sources[0], "count": len(earlier), "model": model},
        current={
            "label": "later",
            "source": sources[1],
            "count": len(later),
            "model": model,
            "ids": "trace",
        },
        where=where,
    )


def corpus_drift(
    earlier: tuple[Sequence[str], npt.ArrayLike],
    later: tuple[Sequence[str], npt.ArrayLike],
    *,
    method: str = DEFAULT_METHOD,
    seed: int = 0,
    sources: tuple[str, str] = ("earlier corpus", "later corpus"),
    later_texts: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    """Two exports of the corpus, as ``(document IDs, vectors)``.

    The earlier export is compared with the documents the later one **added or changed**
    (a new ID, or a new vector under an old ID): unchanged documents are the same in both and
    would only dilute the comparison. ``later_texts`` (titles or first lines) only label the
    groups of new content.
    """
    (old_ids, old_vectors), (new_ids, new_vectors) = earlier, later
    old_matrix = np.asarray(old_vectors, dtype=np.float64)
    new_matrix = np.asarray(new_vectors, dtype=np.float64)
    before = {d: old_matrix[i] for i, d in enumerate(old_ids)}
    fresh = [
        i
        for i, d in enumerate(new_ids)
        if d not in before or not np.array_equal(before[d], new_matrix[i])
    ]
    if len(fresh) < MIN_ROWS:
        raise EmbeddingDriftError(
            f"Only {len(fresh)} documents were added or changed between the exports; at least "
            f"{MIN_ROWS} are needed to compare."
        )
    added = new_matrix[fresh]
    result = embedding_drift(old_matrix, added, method=method, seed=seed)
    where = None
    if result["drift"]:
        where = where_drift(
            old_matrix,
            added,
            [str(new_ids[i]) for i in fresh],
            [later_texts[i] for i in fresh] if later_texts is not None else None,
            nouns=("added or changed documents", "earlier document"),
            seed=seed,
        )
    old, new = set(old_ids), set(new_ids)
    return drift_report(
        result,
        kind="corpus",
        reference={"label": "earlier", "source": sources[0], "count": len(old_ids)},
        current={
            "label": "added or changed",
            "source": sources[1],
            "count": len(new_ids),
            "compared": "documents added or changed",
            "documents_added": len(new - old),
            "documents_changed": len(fresh) - len(new - old),
            "documents_removed": len(old - new),
            "ids": "document",
        },
        where=where,
    )


def new_topics(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Trace ID -> the new-topic group it is in, from an embedding drift report of questions."""
    if report.get("format") != FORMAT:
        raise EmbeddingDriftError("The file is not an embedding drift report.")
    if report.get("kind") != "questions" or (report.get("current") or {}).get("ids") != "trace":
        raise EmbeddingDriftError(
            "Only an embedding drift report of production questions names traces."
        )
    out: dict[str, dict[str, Any]] = {}
    for group in (report.get("where") or {}).get("groups") or []:
        for trace_id in group["ids"]:
            out[str(trace_id)] = {"terms": list(group["terms"]), "share": group["share"]}
    return out
