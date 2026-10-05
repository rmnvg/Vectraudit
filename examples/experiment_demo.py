"""Embedding and chunking bake-off on a real RAG corpus.

Candidates: a lexical baseline, all-MiniLM-L6-v2 with 150- and 400-word chunks, BGE-small
(with its query instruction), and MiniLM on an approximate HNSW index. Each question is
labelled with the passage it was written for, so labels are real and survive re-chunking.

Corpus: "RAG Dataset 12000" (Neural Bridge AI, Apache-2.0; passages from Falcon
RefinedWeb, ODC-By 1.0). Needs ``vectraudit[all]`` and network access the first time.

    uv run python examples/experiment_demo.py --out experiment-demo
    uv run python examples/experiment_demo.py --synthetic --out /tmp/x   # offline smoke

Writes the corpus, queries, manifest, and the experiment outputs (report.html, result.json,
summary.md, runs/) into ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# FAISS before torch: on macOS both bundle OpenMP and the other order can segfault.
import faiss  # noqa: F401
import numpy as np

from vectraudit.evaluation.runfile import save_run
from vectraudit.experiments.manifest import load_manifest
from vectraudit.experiments.report import rank, render_html, render_markdown, to_json
from vectraudit.experiments.runner import run_experiment

BGE_QUERY = "Represent this sentence for searching relevant passages: "
CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "lexical-baseline",
        "embedder": {"type": "hashing", "dim": 2048},
        "chunking": {"method": "words", "size": 150, "overlap": 30},
    },
    {
        "name": "minilm-150w",
        "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "chunking": {"method": "words", "size": 150, "overlap": 30},
    },
    {
        "name": "minilm-400w",
        "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "chunking": {"method": "words", "size": 400, "overlap": 50},
    },
    {
        "name": "bge-small-150w",
        "embedder": {"model": "BAAI/bge-small-en-v1.5", "query_prefix": BGE_QUERY},
        "chunking": {"method": "words", "size": 150, "overlap": 30},
    },
    {
        "name": "minilm-150w-hnsw",
        "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "chunking": {"method": "words", "size": 150, "overlap": 30},
        "index": {"factory": "HNSW16", "params": {"ef_search": 16}},
    },
]


def real_inputs(
    n_docs: int, n_queries: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    from vectraudit import demo

    contexts: dict[str, int] = {}
    pairs: list[tuple[int, str]] = []
    for path in demo.download(demo.default_dir()):
        table = pq.read_table(path, columns=["context", "question"]).to_pydict()
        for ctx, question in zip(table["context"], table["question"], strict=True):
            if ctx and ctx.strip() and question and question.strip():
                key = " ".join(ctx.split())
                doc = contexts.setdefault(key, len(contexts))
                pairs.append((doc, question.strip()))
    docs = [{"doc_id": f"p{i}", "text": text} for text, i in contexts.items() if i < n_docs]
    usable = [(d, q) for d, q in pairs if d < n_docs]
    rng = np.random.default_rng(seed)
    picked = [usable[i] for i in sorted(rng.choice(len(usable), n_queries, replace=False))]
    queries = [
        {
            "query_id": f"q{i:03d}",
            "text": q,
            "relevant": {f"p{d}": 1},
            "tags": ["long-passage" if len(docs[d]["text"].split()) > 300 else "short-passage"],
        }
        for i, (d, q) in enumerate(picked)
    ]
    return docs, queries


def synthetic_inputs(
    n_queries: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    vocab = [f"term{i}" for i in range(400)]
    docs = []
    for i in range(200):
        words = rng.choice(vocab, size=120)
        docs.append({"doc_id": f"p{i}", "text": " ".join(words)})
    queries = []
    for n in range(n_queries):
        d = int(rng.integers(0, len(docs)))
        words = docs[d]["text"].split()
        queries.append(
            {
                "query_id": f"q{n:03d}",
                "text": " ".join(rng.choice(words, size=8)),
                "relevant": {f"p{d}": 1},
                "tags": ["even" if n % 2 == 0 else "odd"],
            }
        )
    return docs, queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("experiment-demo"))
    parser.add_argument("--docs", type=int, default=3000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true", help="Offline, lexical only.")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        docs, queries = synthetic_inputs(args.queries, args.seed)
        candidates = [
            {**CANDIDATES[0], "name": "lexical-2048"},
            {**CANDIDATES[0], "name": "lexical-64", "embedder": {"type": "hashing", "dim": 64}},
            {
                **CANDIDATES[0],
                "name": "lexical-2048-hnsw",
                "index": {"factory": "HNSW16", "params": {"ef_search": 16}},
            },
        ]
    else:
        docs, queries = real_inputs(args.docs, args.queries, args.seed)
        candidates = CANDIDATES
    (out / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    (out / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    manifest = {
        "name": "rag-bakeoff" if not args.synthetic else "synthetic-bakeoff",
        "corpus": "corpus.jsonl",
        "queries": "queries.jsonl",
        "k": 10,
        "candidates": candidates,
    }
    (out / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")

    from vectraudit.core.projection import default_cache_root

    summary, results = run_experiment(
        load_manifest(out / "experiment.json"),
        cache_root=default_cache_root() / "embeddings",
        progress=lambda msg: print(msg, flush=True),
    )
    ranked = rank(summary, results)
    for r in results:
        save_run(r.run, out / "runs" / f"{r.spec.name}.json")
    (out / "result.json").write_text(to_json(ranked) + "\n")
    (out / "report.html").write_text(render_html(ranked))
    (out / "summary.md").write_text(render_markdown(ranked))
    print(render_markdown(ranked))


if __name__ == "__main__":
    main()
