"""Spike (not shipped): can three signals find planted label errors? Phase 35.0, ends at D8.

Plants label errors in synthetic corpora and measures how well each signal finds them, using
only data Vectraudit already has:

1. rank-based confident learning over several candidate runs' rankings (Northcutt et al.),
2. near-duplicate documents labelled differently for the same question,
3. annotator agreement when a labelling task holds more than one judgment per item.

    uv run python scripts/label_audit_spike.py [--seeds 1 2 3]

The synthetic corpora are friendlier than real ones (the planted documents are cleanly
relevant or irrelevant, and hard negatives are the only trap), so the numbers are an upper
bound on what to expect, not a promise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

K = 20  # depth of each candidate run's ranking
RUNS = 5  # candidate runs (differently noisy retrievers)
DIM = 48
QUESTIONS = 120
RELEVANT_PER_QUESTION = 4
HARD_NEGATIVES = 6  # near the question, not relevant: the honest source of false alarms
CLUTTER = 500
FLIP_TO_IRRELEVANT = 0.12  # share of questions that lose one relevant label
FLIP_TO_RELEVANT = 0.12  # share of questions that gain one wrong relevant label
DUPLICATED = 0.35  # share of relevant documents that also exist as a near-duplicate
DUP_COSINE = 0.995  # the cosine at or above which two documents count as near-duplicates
ANNOTATOR_ERRORS = (0.05, 0.10, 0.20)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@dataclass
class Corpus:
    docs: np.ndarray
    queries: np.ndarray
    truth: dict[int, set[int]]  # question -> truly relevant documents
    labels: dict[int, set[int]]  # question -> documents labelled relevant (with planted errors)
    planted: dict[tuple[int, int], str]  # (question, document) -> kind of planted error
    dup_of: dict[int, int]  # duplicate document -> its original


def make_corpus(seed: int) -> Corpus:
    rng = np.random.default_rng(seed)
    docs: list[np.ndarray] = []
    queries = unit(rng.normal(size=(QUESTIONS, DIM)))
    truth: dict[int, set[int]] = {}
    for q in range(QUESTIONS):
        truth[q] = set()
        for _ in range(RELEVANT_PER_QUESTION):
            truth[q].add(len(docs))
            docs.append(unit(queries[q] + 0.35 * rng.normal(size=DIM) / np.sqrt(DIM) * 3))
        for _ in range(HARD_NEGATIVES):
            docs.append(unit(queries[q] + 0.9 * rng.normal(size=DIM) / np.sqrt(DIM) * 3))
    for _ in range(CLUTTER):
        docs.append(unit(rng.normal(size=DIM)))
    dup_of: dict[int, int] = {}
    for q in range(QUESTIONS):
        for d in sorted(truth[q]):
            if rng.random() < DUPLICATED:
                dup_of[len(docs)] = d
                docs.append(unit(docs[d] + 0.01 * rng.normal(size=DIM) / np.sqrt(DIM)))
                truth[q].add(len(docs) - 1)
    arr = np.array(docs)
    labels = {q: set(ds) for q, ds in truth.items()}
    planted: dict[tuple[int, int], str] = {}
    for q in range(QUESTIONS):
        kept = sorted(labels[q])
        if rng.random() < FLIP_TO_IRRELEVANT and len(kept) > 2:
            d = int(rng.choice(kept))
            labels[q].discard(d)
            planted[(q, d)] = "relevant-labelled-irrelevant"
        if rng.random() < FLIP_TO_RELEVANT:
            d = int(rng.integers(0, len(arr)))
            if d not in truth[q] and (q, d) not in planted:
                labels[q].add(d)
                planted[(q, d)] = "irrelevant-labelled-relevant"
    return Corpus(arr, queries, truth, labels, planted, dup_of)


def run_rankings(corpus: Corpus, seed: int) -> list[dict[int, list[int]]]:
    """RUNS differently noisy retrievers: each sees a noisy view of queries and documents."""
    rng = np.random.default_rng(seed + 1000)
    runs = []
    for r in range(RUNS):
        noise = 0.05 + 0.04 * r
        d = unit(corpus.docs + noise * rng.normal(size=corpus.docs.shape) / np.sqrt(DIM) * 3)
        q = unit(corpus.queries + noise * rng.normal(size=corpus.queries.shape) / np.sqrt(DIM) * 3)
        order = np.argsort(-(q @ d.T), axis=1)[:, :K]
        runs.append({i: [int(x) for x in order[i]] for i in range(len(q))})
    return runs


# --- signal 1: rank-based confident learning ---------------------------------------------------


def rank_scores(runs: list[dict[int, list[int]]], q: int) -> dict[int, float]:
    """A noisy relevance score per document: the mean over runs of 1 - (rank-1)/K, 0 if absent."""
    scores: dict[int, float] = {}
    for run in runs:
        for rank, d in enumerate(run[q]):
            scores[d] = scores.get(d, 0.0) + (1 - rank / K) / len(runs)
    return scores


def signal_rank(corpus: Corpus, runs: list[dict[int, list[int]]]) -> dict[tuple[int, int], float]:
    """Per-question confident-learning thresholds; the score is how far on the wrong side.

    As in confident learning, the threshold for a class is the mean predicted score of the
    examples labelled with it: a document labelled irrelevant whose score is at least the mean
    score of this question's relevant documents is suspect, and a document labelled relevant
    whose score is no more than the mean score of its irrelevant candidates is suspect.
    Questions with fewer than two relevant labels have no stable threshold and are skipped.
    """
    suspects: dict[tuple[int, int], float] = {}
    for q in range(QUESTIONS):
        scores = rank_scores(runs, q)
        relevant = corpus.labels[q]
        if len(relevant) < 2:
            continue
        for d in relevant:
            scores.setdefault(d, 0.0)
        rel = [scores[d] for d in relevant]
        irr = [s for d, s in scores.items() if d not in relevant]
        t_rel, t_irr = float(np.mean(rel)), float(np.mean(irr)) if irr else 0.0
        for d, s in scores.items():
            if d in relevant and s <= t_irr:
                suspects[(q, d)] = (t_irr - s) / max(t_irr, 1e-9)
            elif d not in relevant and s >= t_rel:
                suspects[(q, d)] = (s - t_rel) / max(1 - t_rel, 1e-9) + 1e-6
    return suspects


# --- signal 2: near-duplicate disagreement -----------------------------------------------------


def signal_duplicates(corpus: Corpus) -> dict[tuple[int, int], float]:
    """Documents with cosine >= DUP_COSINE labelled differently for the same question."""
    sims = corpus.docs @ corpus.docs.T
    np.fill_diagonal(sims, 0)
    pairs = np.argwhere(np.triu(sims >= DUP_COSINE))
    suspects: dict[tuple[int, int], float] = {}
    for a, b in pairs:
        for q in range(QUESTIONS):
            if (int(a) in corpus.labels[q]) != (int(b) in corpus.labels[q]):
                for d in (int(a), int(b)):  # which of the two is wrong is not known
                    suspects[(q, d)] = float(sims[a, b])
    return suspects


# --- signal 3: annotator agreement -------------------------------------------------------------


def signal_annotators(corpus: Corpus, seed: int) -> dict[str, float]:
    """Three simulated annotators label a pool of (question, document) pairs.

    The planted truth here is each annotator's own error, so the numbers answer two different
    questions: does a split vote find the consensus errors, and how many individual errors
    does review of split votes cover.
    """
    rng = np.random.default_rng(seed + 2000)
    pool = [
        (q, d)
        for q in range(QUESTIONS)
        for d in set(corpus.truth[q]) | {int(x) for x in rng.integers(0, len(corpus.docs), 8)}
    ]
    truth = np.array([d in corpus.truth[q] for q, d in pool])
    votes = np.array([truth ^ (rng.random(len(pool)) < e) for e in ANNOTATOR_ERRORS])
    consensus = votes.sum(axis=0) >= 2
    split = ~((votes.all(axis=0)) | (~votes.any(axis=0)))
    consensus_wrong = consensus != truth
    any_wrong = (votes != truth).any(axis=0)
    flagged = int(split.sum())
    return {
        "items": float(len(pool)),
        "flagged": float(flagged),
        "consensus_wrong": float(consensus_wrong.sum()),
        "precision_consensus_wrong": float((split & consensus_wrong).sum() / max(flagged, 1)),
        "recall_consensus_wrong": float(
            (split & consensus_wrong).sum() / max(consensus_wrong.sum(), 1)
        ),
        "recall_any_annotator_error": float((split & any_wrong).sum() / max(any_wrong.sum(), 1)),
    }


# --- scoring -----------------------------------------------------------------------------------


def score(
    suspects: dict[tuple[int, int], float], planted: dict[tuple[int, int], str]
) -> dict[str, float]:
    hits = [k for k in suspects if k in planted]
    ranked = sorted(suspects, key=suspects.get, reverse=True)  # type: ignore[arg-type]
    top = ranked[: len(planted)]
    return {
        "planted": float(len(planted)),
        "flagged": float(len(suspects)),
        "precision": len(hits) / max(len(suspects), 1),
        "recall": len(hits) / max(len(planted), 1),
        "precision_at_planted": sum(k in planted for k in top) / max(len(top), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    for seed in args.seeds:
        corpus = make_corpus(seed)
        runs = run_rankings(corpus, seed)
        print(f"\nseed {seed}: {len(corpus.docs)} documents, {len(corpus.planted)} planted errors")
        rank = signal_rank(corpus, runs)
        dup = signal_duplicates(corpus)
        both = {k: rank.get(k, 0) + dup.get(k, 0) for k in set(rank) | set(dup)}
        for name, suspects in (
            ("rank (confident learning)", rank),
            ("near-duplicate disagreement", dup),
            ("rank + duplicates", both),
        ):
            s = score(suspects, corpus.planted)
            print(
                f"  {name:30s} flagged {s['flagged']:5.0f}  precision {s['precision']:.2f}  "
                f"recall {s['recall']:.2f}  p@planted {s['precision_at_planted']:.2f}"
            )
        for kind in ("relevant-labelled-irrelevant", "irrelevant-labelled-relevant"):
            planted = {k: v for k, v in corpus.planted.items() if v == kind}
            hit = sum(k in rank for k in planted)
            print(f"    rank recall on {kind}: {hit}/{len(planted)}")
        a = signal_annotators(corpus, seed)
        print(
            f"  annotators (3, error rates {ANNOTATOR_ERRORS}): "
            f"flagged {a['flagged']:.0f}/{a['items']:.0f} items; "
            f"consensus wrong {a['consensus_wrong']:.0f}, "
            f"precision {a['precision_consensus_wrong']:.2f}, "
            f"recall {a['recall_consensus_wrong']:.2f}; "
            f"individual-error recall {a['recall_any_annotator_error']:.2f}"
        )


if __name__ == "__main__":
    main()
