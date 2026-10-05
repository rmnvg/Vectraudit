"""Pool source documents without disclosing retrieval ranks to annotators."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from vectraudit.evaluation.dataset import Dataset, parse_dataset
from vectraudit.evaluation.runfile import load_run
from vectraudit.experiments.corpus import Corpus
from vectraudit.experiments.embedders import HashingEmbedder
from vectraudit.experiments.manifest import load_manifest
from vectraudit.experiments.runner import run_candidate
from vectraudit.labelling.task import Question, Task


def load_questions(path: Path) -> Dataset:
    """Read JSONL or CSV with text and optional query_id (stable row ids otherwise)."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [
                {"query_id": r.get("query_id") or f"q{i}", "text": r.get("text")}
                for i, r in enumerate(csv.DictReader(stream), 1)
            ]
        dataset = parse_dataset("\n".join(json.dumps(r) for r in rows))
    else:
        dataset = parse_dataset(path.read_text(encoding="utf-8"))
    if any(not q.text or q.filter for q in dataset.queries):
        raise ValueError("Pooling requires text queries without filters")
    return dataset


def pool(
    dataset: Dataset,
    corpus: Corpus,
    *,
    k: int = 10,
    runs: list[Path] | None = None,
    manifest: Path | None = None,
    lexical: bool = True,
    name: str = "labels",
    seed: int | None = None,
) -> Task:
    """Union top-k documents per source, validating every id against the corpus."""
    if any(
        q.no_answer or q.context or q.forbidden or q.split == "heldout" for q in dataset.queries
    ):
        raise ValueError(
            "This annotation workflow accepts exploratory positive-label queries only. "
            "Keep no-answer, permission and held-out assertions in the source benchmark JSONL."
        )
    if k < 1:
        raise ValueError("k must be positive")
    pools: dict[str, set[str]] = {q.query_id: set() for q in dataset.queries}
    sources: list[str] = []

    def add(source: str, rankings: dict[str, list[str]]) -> None:
        if set(rankings) != set(pools):
            raise ValueError("Pooling source must contain exactly the question ids")
        for qid, ids in rankings.items():
            selected = list(dict.fromkeys(ids))[:k]
            if any(corpus.get(d) is None for d in selected):
                raise ValueError("Run ids must be source documents in the corpus, never chunk ids")
            pools[qid].update(selected)
        sources.append(source)

    if lexical:
        embedder = HashingEmbedder()
        vectors = embedder.embed_documents([d.text for d in corpus.documents])
        rankings = {}
        for q in dataset.queries:
            scores = vectors @ embedder.embed_queries([q.text or ""])[0]
            rankings[q.query_id] = [
                corpus.documents[int(i)].doc_id for i in np.argsort(-scores, kind="stable")[:k]
            ]
        add("hashing-1024", rankings)
    for path in runs or []:
        run = load_run(path)
        if run["status"] != "complete" or any(q.get("error") for q in run["queries"]):
            raise ValueError("Only complete, successful runs can be pooled")
        add(
            f"run:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            {q["query_id"]: q["ids"] for q in run["queries"]},
        )
    if manifest is not None:
        spec = load_manifest(manifest)
        if any(c.embedder.type != "hashing" for c in spec.candidates):
            raise ValueError(
                "Local pooling only loads hashing candidates; pool precomputed runs otherwise"
            )
        for candidate in spec.candidates:
            result = run_candidate(candidate, corpus, dataset, k, experiment=spec.name)
            add(
                f"candidate:{json.dumps(candidate.describe(), sort_keys=True)}",
                {q["query_id"]: q["ids"] for q in result.run["queries"]},
            )
    if not sources:
        raise ValueError("Select at least one pooling source")
    rng = random.Random(seed)
    questions = []
    for q in dataset.queries:
        ids = sorted(pools[q.query_id])
        rng.shuffle(ids)
        questions.append(Question(query_id=q.query_id, text=q.text or "", documents=ids))
    used = {d for q in questions for d in q.documents}
    return Task(
        name=name,
        corpus_sha256=corpus.sha256,
        sources=sources,
        documents={d.doc_id: d.text for d in corpus.documents if d.doc_id in used},
        queries=questions,
    )
