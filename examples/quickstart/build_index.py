"""Build the quickstart's FAISS index from a BEIR folder.

Embeds every document (title and text) with a sentence-transformers model, normalised so
inner product is cosine similarity, and writes into ``OUT``:

- ``index.faiss``: an IVF-Flat index (inner product), keyed by the BEIR corpus ids
- ``vectors.npy`` and ``ids.npy``: the raw vectors and their ids, so release checks can
  measure ANN recall against exact search as well as relevance against the labels

The ids must be integers, as they are in SciFact. Needs ``faiss-cpu`` and
``sentence-transformers`` (``pip install "vectraudit[faiss-cpu,text]"``)::

    python build_index.py scifact --out retrieval/index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# FAISS before torch (sentence-transformers): on macOS both bundle OpenMP, and loading
# torch's first makes FAISS's multi-threaded training crash.
import faiss
import numpy as np


def read_corpus(folder: Path) -> tuple[np.ndarray, list[str]]:
    """The corpus ids (as int64) and each document's title and text."""
    ids, texts = [], []
    with (folder / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if not str(doc["_id"]).isdigit():
                raise SystemExit(f"Corpus id {doc['_id']!r} is not an integer; FAISS needs one.")
            ids.append(int(doc["_id"]))
            texts.append(f"{doc.get('title', '')} {doc.get('text', '')}".strip())
    return np.asarray(ids, dtype=np.int64), texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("beir", type=Path, help="BEIR folder with corpus.jsonl")
    parser.add_argument("--out", type=Path, default=Path("retrieval/index"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nlist", type=int, default=64, help="IVF lists")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    ids, texts = read_corpus(args.beir)
    print(f"Embedding {len(texts):,} documents with {args.model} ...")
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], args.nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add_with_ids(vectors, ids)

    args.out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out / "index.faiss"))
    np.save(args.out / "vectors.npy", vectors)
    np.save(args.out / "ids.npy", ids)
    print(f"Wrote {args.out}/index.faiss ({index.ntotal:,} vectors, {args.nlist} lists)")


if __name__ == "__main__":
    main()
