"""Spike (not shipped): project recall from a corpus sample to the full corpus. Phase 48.0.

Recall measured on a sample of the corpus is inflated: fewer distractors compete with the
relevant document. Every sample here contains every labelled document (the relevant documents
of the questions asked) plus a seeded random share of the other documents, nested, as
``experiment project`` would draw them. Three projections of full-corpus recall@k from samples
of at most 10% of the corpus are compared:

- ``naive``: recall measured on the largest sample, as if it held at full size;
- ``curve``: recall against log(corpus size) over the nested samples, fitted by a line and
  extrapolated; interval by bootstrap over questions;
- ``thinning``: for each question, count the unlabelled documents in the sample that score above
  its relevant document (``u``) and the labelled ones (``a``, always present). An unlabelled
  document is in the sample with known probability ``g``, so ``u`` is a thinned draw of the
  question's full-corpus count. The counts across questions are modelled as a Poisson mixture
  whose mixing distribution is estimated without a parametric form (the nonparametric maximum
  likelihood estimate on a grid, by EM; Kiefer and Wolfowitz 1956, Koenker and Mizera 2014).
  Each question's chance that fewer than ``k`` documents beat its relevant one at full size
  follows from its posterior; the projection is their mean. Interval: refit the mixture on
  questions resampled with replacement, then draw each question's outcome, 2,000 times.

    uv run python scripts/corpus_projection_spike.py --embed DIR --parquet PARQUET...
    uv run python scripts/corpus_projection_spike.py --data DIR [--seeds 20] [--first-seed 0]

``--embed`` writes ``relevant.npy`` and ``<model>.queries.npy`` / ``<model>.docs.npy`` into
``DIR``: normalised embeddings, by all-MiniLM-L6-v2 and bge-small-en-v1.5, of
neural-bridge/rag-dataset-12000 (Apache-2.0; both parquet splits), where each question's
context is its one relevant document and identical contexts are one document.

A fourth projection, ``gmodel``, replaces the nonparametric prior with Efron's smooth
exponential-family prior (g-modeling, 2016). It was tried after ``npmle`` under-covered on one
synthetic corpus; its intervals are half as wide and too narrow on fresh seeds, so it is kept
here only as the alternative that failed.

Bar (the plan's, made stricter): for the synthetic corpus and the real one, with two embedders
each, the projection from at most 10% of the corpus is within its stated 95% interval of the
measured full-corpus recall, in at least 90% of sample draws (seeds).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

K = 10
FRACTIONS = (0.04, 0.06, 0.08, 0.10)
DRAWS = 2000
GRID = 160
EM_STEPS = 400


@dataclass
class Corpus:
    name: str
    n_docs: int
    labelled: np.ndarray  # doc indices always in the sample
    beaters: list[np.ndarray]  # per question: docs scoring above its relevant document


def build(name: str, queries: np.ndarray, docs: np.ndarray, relevant: np.ndarray) -> Corpus:
    beaters = []
    for start in range(0, len(queries), 256):
        scores = queries[start : start + 256] @ docs.T
        rel = relevant[start : start + 256]
        mine = scores[np.arange(len(rel)), rel][:, None]
        beaters += [np.flatnonzero(row) for row in scores > mine]
    return Corpus(name, len(docs), np.unique(relevant), beaters)


def synthetic(dim: int, noise: float, seed: int, n_docs: int, n_q: int) -> Corpus:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(500, dim))
    docs = centers[rng.integers(0, 500, n_docs)] + 0.8 * rng.normal(size=(n_docs, dim))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    relevant = rng.choice(n_docs, n_q, replace=False)
    spread = noise * np.exp(0.35 * rng.normal(size=(n_q, 1)))
    queries = docs[relevant] + spread * rng.normal(size=(n_q, dim)) / math.sqrt(dim) * 4
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return build(
        f"synthetic d={dim}", queries.astype(np.float32), docs.astype(np.float32), relevant
    )


def real(data: Path, model: str, n_q: int, seed: int) -> Corpus:
    relevant_all = np.load(data / "relevant.npy")
    queries_all = np.load(data / f"{model}.queries.npy")
    docs = np.load(data / f"{model}.docs.npy")
    pick = np.random.default_rng(seed).choice(len(relevant_all), n_q, replace=False)
    return build(f"rag-12000 {model}", queries_all[pick], docs, relevant_all[pick])


@dataclass
class Sample:
    a: np.ndarray  # labelled beaters per question (always present)
    u: np.ndarray  # unlabelled beaters in the sample
    g: float  # chance an unlabelled document is in the sample
    size: int


def samples(c: Corpus, seed: int) -> list[Sample]:
    labelled = np.zeros(c.n_docs, dtype=bool)
    labelled[c.labelled] = True
    unl = np.flatnonzero(~labelled)
    position = np.full(c.n_docs, -1)
    position[unl] = np.random.default_rng(seed).permutation(len(unl))
    out = []
    for f in FRACTIONS:
        take = max(0, round(f * c.n_docs) - len(c.labelled))
        a = np.array([int(labelled[b].sum()) for b in c.beaters])
        u = np.array([int(((position[b] >= 0) & (position[b] < take)).sum()) for b in c.beaters])
        out.append(Sample(a, u, take / len(unl), len(c.labelled) + take))
    return out


def truth(c: Corpus) -> float:
    return float(np.mean([len(b) < K for b in c.beaters]))


def curve(c: Corpus, ss: list[Sample], rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.log([s.size for s in ss])
    hits = np.stack([(s.a + s.u) < K for s in ss])  # (sizes, questions)

    def fit(cols: np.ndarray) -> float:
        slope, icept = np.polyfit(x, hits[:, cols].mean(1), 1)
        return float(min(1.0, max(0.0, icept + slope * math.log(c.n_docs))))

    n = hits.shape[1]
    point = fit(np.arange(n))
    boot = [fit(rng.integers(0, n, n)) for _ in range(DRAWS // 4)]
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _poisson_logpmf(u: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lam = np.maximum(lam, 1e-300)
    lgam = np.array([math.lgamma(v + 1) for v in u])[:, None]
    return u[:, None] * np.log(lam)[None, :] - lam[None, :] - lgam


def npmle(u: np.ndarray, g: float, grid: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    like = np.exp(_poisson_logpmf(u, g * grid))  # (questions, grid)
    w = np.full(len(grid), 1 / len(grid)) if w is None else w.copy()
    for _ in range(EM_STEPS):
        post = like * w
        post /= post.sum(1, keepdims=True) + 1e-300
        w = post.mean(0)
    return w


def _poisson_cdf_below(m: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """P(V < m) for V ~ Poisson(lam): (questions, grid); m per question, 0 when m <= 0."""
    top = int(max(1, m.max()))
    out = np.zeros((len(m), len(lam)))
    term = np.exp(-lam)[None, :].repeat(len(m), 0)
    for v in range(top):
        out += np.where((v < m)[:, None], term, 0.0)
        term = term * lam[None, :] / (v + 1)
    return out


def chances(s: Sample, w: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Each question's posterior chance of fewer than K beaters at full size."""
    like = np.exp(_poisson_logpmf(s.u, s.g * grid))
    post = like * w
    post /= post.sum(1, keepdims=True) + 1e-300
    room = K - s.a - s.u
    return (post * _poisson_cdf_below(room, (1 - s.g) * grid)).sum(1)


def _basis(grid: np.ndarray, degree: int = 5) -> np.ndarray:
    z = np.log1p(grid)
    z = (z - z.mean()) / z.std()
    q, _ = np.linalg.qr(np.vander(z, degree + 1, increasing=True)[:, 1:])
    return q * math.sqrt(len(grid))


def gmodel(
    u: np.ndarray, g: float, grid: np.ndarray, basis: np.ndarray, penalty: float = 1.0
) -> np.ndarray:
    """Efron's g-modeling (2016): a smooth exponential-family prior on the grid, fitted by
    penalised maximum likelihood (BHHH steps with step halving)."""
    like = np.exp(_poisson_logpmf(u, g * grid))
    alpha = np.zeros(basis.shape[1])

    def weights(a: np.ndarray) -> np.ndarray:
        x = basis @ a
        e = np.exp(x - x.max())
        return e / e.sum()

    def objective(a: np.ndarray) -> float:
        return float(np.log(like @ weights(a) + 1e-300).sum() - penalty * a @ a)

    current = objective(alpha)
    for _ in range(60):
        w = weights(alpha)
        jac = (np.diag(w) - np.outer(w, w)) @ basis  # (grid, p)
        r = like / (like @ w + 1e-300)[:, None]  # (questions, grid)
        per = r @ jac  # (questions, p)
        grad = per.sum(0) - 2 * penalty * alpha
        hess = per.T @ per + 2 * penalty * np.eye(len(alpha))
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-4:
            trial = objective(alpha + t * step)
            if trial >= current:
                break
            t /= 2
        if t <= 1e-4 or trial - current < 1e-8:
            break
        alpha, current = alpha + t * step, trial
    return weights(alpha)


def thinning(
    c: Corpus, s: Sample, rng: np.random.Generator, prior: str = "npmle"
) -> tuple[float, float, float]:
    top = max(10.0, (c.n_docs - len(c.labelled)) * 1.0)
    grid = np.concatenate([[0.0], np.geomspace(0.01, top, GRID - 1)])
    basis = _basis(grid)

    def fit(u: np.ndarray) -> np.ndarray:
        return npmle(u, s.g, grid) if prior == "npmle" else gmodel(u, s.g, grid, basis)

    w = fit(s.u)
    p = chances(s, w, grid)
    n = len(p)
    draws = []
    for _ in range(DRAWS // 10):
        idx = rng.integers(0, n, n)
        pb = chances(s, fit(s.u[idx]), grid)
        draws += list((rng.random((10, n)) < pb).mean(1))
    return float(p.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def embed(out: Path, parquet: list[Path]) -> None:
    import pyarrow.parquet as pq  # spike only
    from sentence_transformers import SentenceTransformer  # spike only

    contexts: list[str] = []
    index: dict[str, int] = {}
    questions: list[str] = []
    relevant: list[int] = []
    for path in sorted(parquet):
        for row in pq.read_table(path).to_pylist():
            context, question = row["context"] or "", row["question"] or ""
            if not context.strip() or not question.strip():
                continue
            if context not in index:
                index[context] = len(contexts)
                contexts.append(context)
            questions.append(question)
            relevant.append(index[context])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "relevant.npy", np.array(relevant))
    for name in ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"):
        model = SentenceTransformer(name)
        tag = name.split("/")[1]
        for what, texts in (("queries", questions), ("docs", contexts)):
            vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)
            np.save(out / f"{tag}.{what}.npy", np.asarray(vectors, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--embed", type=Path, help="Write embeddings into this folder.")
    parser.add_argument("--parquet", type=Path, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--questions", type=int, default=300)
    parser.add_argument("--synthetic-docs", type=int, default=60_000)
    parser.add_argument("--first-seed", type=int, default=0)
    args = parser.parse_args()
    if args.embed:
        embed(args.embed, args.parquet)
        return
    corpora = [
        synthetic(64, 0.45, 1, args.synthetic_docs, args.questions),
        synthetic(32, 0.35, 2, args.synthetic_docs, args.questions),
    ]
    if args.data:
        corpora += [
            real(args.data, m, args.questions, 0)
            for m in ("all-MiniLM-L6-v2", "bge-small-en-v1.5")
            if (args.data / f"{m}.docs.npy").exists()
        ]
    print(f"recall@{K}; samples of {', '.join(f'{f:.0%}' for f in FRACTIONS)} of the corpus")
    for c in corpora:
        full = truth(c)
        covered = {"naive": 0, "curve": 0, "npmle": 0, "gmodel": 0}
        errors: dict[str, list[float]] = {m: [] for m in covered}
        widths: dict[str, list[float]] = {m: [] for m in covered}
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            rng = np.random.default_rng(seed)
            ss = samples(c, seed)
            last = ss[-1]
            naive = float(((last.a + last.u) < K).mean())
            results = {
                "naive": (naive, naive, naive),
                "curve": curve(c, ss, rng),
                "npmle": thinning(c, last, rng, "npmle"),
                "gmodel": thinning(c, last, rng, "gmodel"),
            }
            for m, (point, lo, hi) in results.items():
                covered[m] += lo <= full <= hi
                errors[m].append(point - full)
                widths[m].append(hi - lo)
            if seed == args.first_seed:
                print(
                    f"\n{c.name}: {c.n_docs} documents, {len(c.beaters)} questions, full "
                    f"recall {full:.3f}; largest sample {last.size} documents, sample recall "
                    f"{naive:.3f}"
                )
                for m, (point, lo, hi) in results.items():
                    print(f"  seed {seed} {m:<9} {point:.3f} ({lo:.3f} to {hi:.3f})")
        for m in covered:
            ok = covered[m] >= 0.9 * args.seeds
            print(
                f"  {m:<9} covers {covered[m]}/{args.seeds}, mean error "
                f"{np.mean(errors[m]):+.3f}, mean width {np.mean(widths[m]):.3f}: "
                f"{'holds' if ok else 'missed'}"
            )


if __name__ == "__main__":
    main()
