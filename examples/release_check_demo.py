"""A before/after release-check investigation on a real RAG corpus.

Three pull requests against the same labelled questions:

1. ``lower-nprobe``: cut IVF nprobe from 16 to 2 to save latency. Expected: fail, with
   the diagnosis that missed passages sat in unprobed lists and the nprobe that fixes it.
2. ``fix-nprobe``: apply the nprobe the diagnosis suggested. Expected: pass.
3. ``pq-compression``: switch to 8-byte product quantization to save memory. Expected:
   fail, diagnosed as quantization loss that raising nprobe cannot recover.

Corpus: "RAG Dataset 12000" (Neural Bridge AI, Apache-2.0; passages from Falcon
RefinedWeb, ODC-By 1.0), chunked and embedded with all-MiniLM-L6-v2. Each question's
relevant chunks are the chunks of the passage it was written for, so the labels are real.
Needs ``vectraudit[all]`` and network access the first time (data and model are cached).

    uv run python examples/release_check_demo.py --out release-check-demo
    uv run python examples/release_check_demo.py --synthetic   # offline, no downloads

Outputs ``dataset.jsonl``, one run file per setup, ``report-<scenario>.html`` and a
``SUMMARY.md`` with each verdict and its findings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# FAISS before torch (sentence-transformers): on macOS both bundle OpenMP, and loading
# torch's first makes FAISS's multi-threaded training segfault.
import faiss
import numpy as np

from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import render_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.report import render_html, render_markdown
from vectraudit.evaluation.runfile import save_run
from vectraudit.evaluation.runner import RunOptions, execute_run

POLICY = {
    "metrics": {
        "recall": {"max_drop": 0.02},
        "ndcg": {"max_drop": 0.02},
        "ann_recall": {"max_drop": 0.02},
    },
    "min_queries": 50,
    "latency": {"p95_max_increase_pct": 100},
}
K = 10


def real_corpus(out: Path, n_queries: int, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Chunk vectors and labelled question rows from the RAG dataset."""
    import pyarrow.parquet as pq

    from vectraudit import demo

    paths = demo.download(demo.default_dir())
    contexts: dict[str, int] = {}
    pairs: list[tuple[str, str]] = []
    for path in paths:
        table = pq.read_table(path, columns=["context", "question"]).to_pydict()
        for ctx, question in zip(table["context"], table["question"], strict=True):
            if ctx and ctx.strip() and question and question.strip():
                key = " ".join(ctx.split())
                contexts.setdefault(key, len(contexts))
                pairs.append((key, question.strip()))
    chunks = demo.chunk(list(contexts), max_chunks=10_000)
    by_doc: dict[int, list[int]] = {}
    for c in chunks:
        by_doc.setdefault(int(c["doc_id"]), []).append(int(c["id"]))  # type: ignore[call-overload]
    cached = demo.default_dir() / "vectors.npy"
    vectors = np.load(cached) if cached.exists() else None
    if vectors is None or len(vectors) != len(chunks):
        demo.log("embedding chunks")
        vectors = demo.embed([str(c["text"]) for c in chunks])
    usable = [(q, by_doc[contexts[ctx]]) for ctx, q in pairs if contexts[ctx] in by_doc]
    rng = np.random.default_rng(seed)
    picked = [usable[i] for i in sorted(rng.choice(len(usable), n_queries, replace=False))]
    demo.log(f"embedding {len(picked)} questions")
    q_vectors = demo.embed([q for q, _ in picked])
    rows = [
        {
            "query_id": f"q{i:03d}",
            "text": question,
            "vector": [float(x) for x in v],
            # The chunk that opens the passage usually carries the answer: grade it higher.
            "relevant": {str(c): (2 if j == 0 else 1) for j, c in enumerate(chunk_ids)},
            "tags": ["long-passage" if len(chunk_ids) > 1 else "short-passage"],
        }
        for i, ((question, chunk_ids), v) in enumerate(zip(picked, q_vectors, strict=True))
    ]
    return vectors, rows


def synthetic_corpus(n_queries: int, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Offline stand-in: unclustered vectors, relevant = the 3 exact neighbours."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(10_000, 64)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = rng.normal(size=(n_queries, 64)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    _, nn = faiss.knn(q, x, 3, metric=faiss.METRIC_INNER_PRODUCT)
    rows = [
        {
            "query_id": f"q{i:03d}",
            "vector": v.tolist(),
            "relevant": {str(d): 3 - r for r, d in enumerate(ids)},
            "tags": ["even" if i % 2 == 0 else "odd"],
        }
        for i, (v, ids) in enumerate(zip(q, nn, strict=True))
    ]
    return x, rows


def build_indexes(out: Path, x: np.ndarray) -> dict[str, Path]:

    d = x.shape[1]
    nlist = max(16, len(x) // 40)
    ip = faiss.METRIC_INNER_PRODUCT
    flat = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, ip)
    pq = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 8, 8, ip)
    paths = {}
    for name, index in (("ivf_flat", flat), ("ivf_pq8", pq)):
        index.train(x)
        index.add(x)
        paths[name] = out / f"{name}.index"
        faiss.write_index(index, str(paths[name]))
    np.save(out / "vectors.npy", x)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("release-check-demo"))
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--synthetic", action="store_true", help="Offline data, no downloads.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    x, rows = (synthetic_corpus if args.synthetic else lambda n, s: real_corpus(out, n, s))(
        args.queries, args.seed
    )
    meta = {
        "name": "rag-12000 questions" if not args.synthetic else "synthetic",
        "embedding_model": "synthetic" if args.synthetic else "all-MiniLM-L6-v2",
        "normalized": True,
    }
    text = render_dataset(rows, meta)
    (out / "dataset.jsonl").write_text(text)
    ds = parse_dataset(text)
    indexes = build_indexes(out, x)
    vectors = out / "vectors.npy"
    # Synthetic searches take microseconds: too noisy to judge latency, so only quality counts.
    policy = parse_policy(
        {k: v for k, v in POLICY.items() if k != "latency"} if args.synthetic else POLICY
    )

    def run(index: str, nprobe: int, label: str) -> dict[str, Any]:
        conn = FaissConnector(indexes[index], vectors=vectors)
        record = execute_run(
            ds, conn, RunOptions(k=K, params={"nprobe": nprobe}, label=label, repeats=3)
        )
        save_run(record, out / f"run-{label}.json")
        return record

    base = run("ivf_flat", 16, "main")
    scenarios: list[tuple[str, dict[str, Any]]] = [
        ("lower-nprobe", run("ivf_flat", 2, "lower-nprobe"))
    ]
    first = compare_runs(base, scenarios[0][1], policy)
    unprobed = next((f for f in first["findings"] if f["id"] == "ivf-unprobed"), None)
    suggested = int((unprobed or {}).get("data", {}).get("recommended_nprobe", 16))
    scenarios.append(("fix-nprobe", run("ivf_flat", suggested, f"fix-nprobe-{suggested}")))
    scenarios.append(("pq-compression", run("ivf_pq8", 16, "pq-compression")))

    lines = ["# Release-check demo", "", f"{len(ds)} labelled queries, k = {K}.", ""]
    for name, cand in scenarios:
        result = compare_runs(base, cand, policy)
        (out / f"report-{name}.html").write_text(render_html(result))
        (out / f"result-{name}.json").write_text(json.dumps(result, indent=2))
        lines += [f"## {name}: {result['verdict']}", ""]
        lines += [f"- [{f['level']}] {f['title']}" for f in result["findings"]]
        lines += ["", render_markdown(result), ""]
        print(
            f"{name:16s} {result['verdict']:15s} " + "; ".join(f["id"] for f in result["findings"])
        )
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print(f"Reports in {out}/")


if __name__ == "__main__":
    main()
