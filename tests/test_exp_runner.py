"""Experiment manifests and the runner, on a fixture whose right answer is known.

Documents are "<topic> ... <plan>" (e.g. refund policy for plan alpha). Queries ask about a
topic with a synonym ("money back") for one plan. A lexical model matches the plan name in
every topic's document; a model that knows the synonyms finds the right document. So the
semantic candidate must win, at document level, whatever the chunking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.runfile import check_run
from vectraudit.experiments.embedders import HashingEmbedder
from vectraudit.experiments.manifest import ExperimentError, load_manifest, parse_manifest
from vectraudit.experiments.runner import run_experiment

TOPICS = {
    "refund": "money back",
    "password": "login credentials",
    "invoice": "billing statement",
    "shipping": "delivery time",
    "warranty": "repair guarantee",
}
PLANS = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
FILLER = "the team reviews every request carefully and answers within one business day"


class SynonymEmbedder:
    """Knows that "money back" means refund: topic one-hot plus a lexical part."""

    def __init__(self) -> None:
        self.lexical = HashingEmbedder(64)

    def _embed(self, texts):
        out = []
        for t in texts:
            lower = t.lower()
            topic = np.zeros(len(TOPICS), dtype=np.float32)
            for i, (word, synonym) in enumerate(TOPICS.items()):
                if word in lower or synonym in lower:
                    topic[i] = 3.0
            v = np.concatenate([topic, self.lexical.embed_documents([t])[0]])
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.asarray(out, dtype=np.float32)

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_queries(self, texts):
        return self._embed(texts)


def write_fixture(root: Path, candidates: list[dict[str, Any]], **manifest: Any) -> Path:
    docs, queries = [], []
    for topic, synonym in TOPICS.items():
        for plan in PLANS:
            doc_id = f"{topic}-{plan}"
            text = (f"{topic.title()} policy for plan {plan}.\n\n{FILLER}. " * 2).strip()
            docs.append({"doc_id": doc_id, "text": text, "lang": "en"})
            if plan in PLANS[:4]:
                queries.append(
                    {
                        "query_id": f"q-{doc_id}",
                        "text": f"{synonym} on plan {plan}",
                        "relevant": {doc_id: 2},
                        "tags": [topic],
                    }
                )
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    (root / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    path = root / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "name": "fixture",
                "corpus": "corpus.jsonl",
                "queries": "queries.jsonl",
                "k": 3,
                "candidates": candidates,
                **manifest,
            }
        )
    )
    return path


LEXICAL = {
    "name": "lexical",
    "embedder": {"type": "hashing", "dim": 256},
    "chunking": {"method": "document"},
}
SEMANTIC = {
    "name": "semantic",
    "embedder": {"type": "custom", "model": "synonyms"},
    "chunking": {"method": "document"},
}


def test_semantic_candidate_wins(tmp_path) -> None:
    manifest = load_manifest(write_fixture(tmp_path, [LEXICAL, SEMANTIC]))
    summary, results = run_experiment(manifest, embedders={"synonyms": SynonymEmbedder()})
    assert summary["format"] == "vectraudit.experiment"
    assert summary["corpus"]["documents"] == 30
    by = {r.spec.name: r for r in results}
    lexical = by["lexical"].run["summary"]["metrics"]
    semantic = by["semantic"].run["summary"]["metrics"]
    assert semantic["recall"] == 1.0
    assert semantic["mrr"] > 0.9
    assert lexical["mrr"] < 0.4  # matches the plan name in every topic's document
    # Runs are ordinary run files over the same dataset: compare_runs works directly.
    for r in results:
        check_run(json.loads(json.dumps(r.run)))
        assert all(
            set(q["ids"]) <= {f"{t}-{p}" for t in TOPICS for p in PLANS} for q in r.run["queries"]
        )
    verdict = compare_runs(by["semantic"].run, by["lexical"].run)
    assert verdict["verdict"] == "fail"
    assert any(d["path"] == "backend.config.embedder.label" for d in verdict["config_diff"])
    stats = by["semantic"].stats
    assert stats["chunks"] == 30
    assert stats["index_bytes"] > 0
    assert stats["embed_documents_s"] is not None


def test_chunking_is_judged_on_documents(tmp_path) -> None:
    small = {**SEMANTIC, "name": "semantic-small", "chunking": {"method": "words", "size": 6}}
    paras = {
        **SEMANTIC,
        "name": "semantic-paragraphs",
        "chunking": {"method": "paragraphs", "size": 40},
    }
    manifest = load_manifest(write_fixture(tmp_path, [SEMANTIC, small, paras]))
    _, results = run_experiment(manifest, embedders={"synonyms": SynonymEmbedder()})
    chunk_counts = [r.stats["chunks"] for r in results]
    assert chunk_counts[0] == 30
    assert chunk_counts[1] > 30
    for r in results:
        run = r.run
        assert run["summary"]["metrics"]["recall"] == 1.0
        # k distinct documents even when several chunks of one document rank first.
        assert all(len(q["ids"]) == 3 for q in run["queries"])
        assert all(len(set(q["ids"])) == len(q["ids"]) for q in run["queries"])


def test_approximate_index_has_an_exact_reference(tmp_path) -> None:
    ivf = {
        **SEMANTIC,
        "name": "semantic-ivf",
        "index": {"factory": "IVF4,Flat", "params": {"nprobe": 1}},
    }
    manifest = load_manifest(write_fixture(tmp_path, [SEMANTIC, ivf]))
    _, results = run_experiment(manifest, embedders={"synonyms": SynonymEmbedder()})
    exact, approx = results
    assert exact.run["search"]["exact_reference"] is False
    assert exact.run["summary"]["ann_recall"] is None
    assert approx.run["search"]["exact_reference"] is True
    assert approx.run["search"]["params"] == {"nprobe": 1}
    assert 0 <= approx.run["summary"]["ann_recall"] <= 1


def test_span_labels_give_chunk_level_metrics(tmp_path) -> None:
    path = write_fixture(tmp_path, [{**SEMANTIC, "chunking": {"method": "words", "size": 6}}])
    rows = [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text().splitlines()]
    for row in rows:
        doc_id = next(iter(row["relevant"]))
        row["relevant_spans"] = [{"doc_id": doc_id, "start": 0, "end": 30}]
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    _, results = run_experiment(load_manifest(path), embedders={"synonyms": SynonymEmbedder()})
    span = results[0].span_summary
    assert span is not None
    assert span["n"] == len(rows)
    assert 0 < span["metrics"]["recall"] <= 1


def test_embedding_cache_reused(tmp_path) -> None:
    manifest = load_manifest(write_fixture(tmp_path, [LEXICAL]))
    first, _ = run_experiment(manifest, cache_root=tmp_path / "cache")
    second, _ = run_experiment(manifest, cache_root=tmp_path / "cache")
    assert first["candidates"][0]["stats"]["embedding_cached"] is False
    assert second["candidates"][0]["stats"]["embedding_cached"] is True
    assert second["candidates"][0]["stats"]["embed_documents_s"] is None


def test_input_errors(tmp_path) -> None:
    path = write_fixture(tmp_path, [LEXICAL])
    rows = [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text().splitlines()]
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({**rows[0], "relevant": {"refund-alpha#0": 1}})
    )
    with pytest.raises(ExperimentError, match="not chunk ids"):
        run_experiment(load_manifest(path))
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"query_id": "v", "vector": [0.1, 0.2], "relevant": {"refund-alpha": 1}})
    )
    with pytest.raises(ExperimentError, match="no text"):
        run_experiment(load_manifest(path))
    bad_factory = write_fixture(tmp_path, [{**LEXICAL, "index": {"factory": "Nonsense9"}}])
    with pytest.raises(ExperimentError, match="invalid FAISS factory"):
        run_experiment(load_manifest(bad_factory))


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "must be an object"),
        ({"corpus": "c", "queries": "q", "candidates": []}, "non-empty list"),
        ({"corpus": "c", "queries": "q", "candidates": [LEXICAL, LEXICAL]}, "unique"),
        ({"corpus": "c", "queries": "q", "k": 0, "candidates": [LEXICAL]}, "k must"),
        (
            {"corpus": "c", "queries": "q", "candidates": [{**LEXICAL, "name": "bad name"}]},
            "name must",
        ),
        (
            {
                "corpus": "c",
                "queries": "q",
                "candidates": [{**LEXICAL, "chunking": {"method": "lines"}}],
            },
            "chunking method",
        ),
        (
            {
                "corpus": "c",
                "queries": "q",
                "candidates": [{**LEXICAL, "embedder": {"type": "hashing", "colour": 1}}],
            },
            "unexpected keyword",
        ),
        (
            {
                "corpus": "c",
                "queries": "q",
                "candidates": [{**LEXICAL, "index": {"params": {"nprobe": 0}}}],
            },
            "positive",
        ),
        (
            {
                "corpus": "c",
                "queries": "q",
                "candidates": [{**LEXICAL, "index": {"metric": "cosine"}}],
            },
            "ip or l2",
        ),
        ({"corpus": "c", "queries": "q", "candidates": [LEXICAL], "extra": 1}, "unknown"),
    ],
)
def test_manifest_errors(raw, message) -> None:
    with pytest.raises(ExperimentError, match=message):
        parse_manifest(raw)


def test_manifest_paths_and_errors(tmp_path) -> None:
    path = write_fixture(tmp_path, [LEXICAL])
    m = load_manifest(path)
    assert m.corpus_path == tmp_path / "corpus.jsonl"
    assert m.k == 3
    with pytest.raises(ExperimentError, match="not found"):
        load_manifest(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ExperimentError, match="not valid JSON"):
        load_manifest(tmp_path / "bad.json")
