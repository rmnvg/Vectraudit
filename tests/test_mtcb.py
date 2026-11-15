"""MTCB datasets import as exact spans, and MTCB's containment recall is scored beside ours."""

from __future__ import annotations

import itertools
import json
import math

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import DatasetError, load_dataset
from vectraudit.evaluation.dataset_formats import import_mtcb
from vectraudit.experiments.chunking import ChunkingSpec, chunk_document
from vectraudit.experiments.corpus import Document
from vectraudit.experiments.labels import containing_rank, containment_metrics
from vectraudit.experiments.manifest import load_manifest
from vectraudit.experiments.runner import run_experiment

DOCS = [
    "Refunds are issued within thirty days of purchase. Annual plans are refunded pro rata "
    "after the first month. Contact billing to start a refund.",
    "Passwords must be twelve characters long. Reset links expire after one hour. "
    "Contact billing to start a refund.",
    "Shipping takes five business days in the mainland. Islands take ten business days.",
]
QUESTIONS = [
    ("How long do refunds take?", "Refunds are issued within thirty days of purchase."),
    ("How are annual plans refunded?", "Annual plans are refunded pro rata after the first month."),
    ("When do reset links expire?", "Reset links expire after one hour."),
    ("How do I start a refund?", "Contact billing to start a refund."),  # in two documents
    ("How long is shipping to islands?", "Islands take ten business days."),
    ("What about returns?", "Returns are free within sixty days."),  # nowhere in the corpus
]


def write_mtcb(root, *, corpus_key="text", passage_key="chunk-must-contain", questions="questions"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "corpus.jsonl").write_text("\n".join(json.dumps({corpus_key: d}) for d in DOCS))
    (root / f"{questions}.jsonl").write_text(
        "\n".join(json.dumps({"question": q, passage_key: p}) for q, p in QUESTIONS)
    )
    return root


def test_every_passage_round_trips_into_exact_spans(tmp_path):
    imported = import_mtcb(write_mtcb(tmp_path / "gacha"))
    texts = {d["doc_id"]: d["text"] for d in imported.corpus}
    assert list(texts) == ["doc-0", "doc-1", "doc-2"]
    passages = dict(QUESTIONS)
    assert len(imported.rows) == 5
    for row in imported.rows:
        for span in row["relevant_spans"]:
            assert texts[span["doc_id"]][span["start"] : span["end"]] == passages[row["text"]]
    by_question = {r["text"]: r["relevant_spans"] for r in imported.rows}
    # A passage found in two documents is a span in each: MTCB accepts a chunk from either.
    assert [s["doc_id"] for s in by_question["How do I start a refund?"]] == ["doc-0", "doc-1"]
    assert dict(imported.skipped) == {"passage not found exactly in any document": 1}
    assert imported.provenance["format"] == "mtcb"
    assert len(imported.provenance["corpus_sha256"]) == 64


def test_the_generator_spelling_and_parquet_layout_are_read(tmp_path):
    alt = write_mtcb(
        tmp_path / "gen", corpus_key="document", passage_key="chunk_must_contain", questions="qa"
    )
    assert len(import_mtcb(alt).rows) == 5
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    hub = tmp_path / "hub"
    (hub / "corpus").mkdir(parents=True)
    (hub / "questions").mkdir()
    pq.write_table(pa.table({"text": DOCS}), hub / "corpus" / "train-00000-of-00001.parquet")
    pq.write_table(
        pa.table(
            {"question": [q for q, _ in QUESTIONS], "chunk-must-contain": [p for _, p in QUESTIONS]}
        ),
        hub / "questions" / "train-00000-of-00001.parquet",
    )
    assert import_mtcb(hub).rows == import_mtcb(write_mtcb(tmp_path / "plain")).rows


def test_bad_directories_are_refused(tmp_path):
    with pytest.raises(DatasetError, match="not a directory"):
        import_mtcb(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError, match=r"no corpus: expected corpus\.jsonl"):
        import_mtcb(empty)
    (empty / "corpus.jsonl").write_text(json.dumps({"title": "no text"}))
    (empty / "questions.jsonl").write_text(json.dumps({"question": "q", "chunk-must-contain": "p"}))
    with pytest.raises(DatasetError, match="has no text"):
        import_mtcb(empty)
    (empty / "corpus.jsonl").write_text(json.dumps({"text": "unrelated"}))
    with pytest.raises(DatasetError, match="No labelled queries"):
        import_mtcb(empty)


# --- MTCB's metric ----------------------------------------------------------------------------


def test_containment_metrics_match_hand_computed_values():
    assert containment_metrics(0, 5) == {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "precision": 0.0}
    third = containment_metrics(3, 5)
    assert third == {"recall": 1.0, "mrr": 1 / 3, "ndcg": 1 / math.log2(4), "precision": 0.2}
    assert containment_metrics(1, 10)["ndcg"] == 1.0


@pytest.mark.parametrize("size", [3, 8, 200])
def test_a_chunk_contains_a_span_exactly_when_it_contains_the_passage_text(tmp_path, size):
    """MTCB's rule is ``passage in chunk.text``; ours works on offsets. They must agree."""
    imported = import_mtcb(write_mtcb(tmp_path / "d"))
    from vectraudit.evaluation.dataset import parse_dataset
    from vectraudit.evaluation.dataset_tools import render_dataset

    ds = parse_dataset(render_dataset(imported.rows))
    passages = dict(QUESTIONS)
    chunks = [
        c
        for d in imported.corpus
        for c in chunk_document(Document(d["doc_id"], d["text"]), ChunkingSpec("words", size))
    ]
    for q in ds.queries:
        passage = passages[q.text]
        for order in itertools.islice(itertools.permutations(chunks, 3), 200):
            expected = next((i for i, c in enumerate(order, 1) if passage in c.text), 0)
            assert containing_rank(q.relevant_spans, list(order), 3) == expected
        assert containing_rank(q.relevant_spans, chunks, 0) == 0


def experiment(tmp_path, chunking, k):
    runner = CliRunner()
    src = write_mtcb(tmp_path / "src")
    args = [
        "dataset",
        "import",
        str(src),
        "--format",
        "mtcb",
        "--out",
        str(tmp_path / "queries.jsonl"),
    ]
    refused = runner.invoke(app, args)
    assert refused.exit_code != 0
    assert "--corpus-out" in " ".join(refused.output.split())
    done = runner.invoke(app, [*args, "--corpus-out", str(tmp_path / "corpus.jsonl")])
    assert done.exit_code == 0, done.output
    assert "Wrote 3 documents" in done.output
    assert len(load_dataset(tmp_path / "queries.jsonl")) == 5
    manifest = {
        "name": "mtcb",
        "corpus": "corpus.jsonl",
        "queries": "queries.jsonl",
        "k": k,
        "candidates": [
            {"name": "c", "embedder": {"type": "hashing", "dim": 256}, "chunking": chunking}
        ],
    }
    (tmp_path / "experiment.json").write_text(json.dumps(manifest))
    _, results = run_experiment(load_manifest(tmp_path / "experiment.json"))
    return results[0].span_summary


def test_imported_dataset_scores_hand_computed_containment_recall(tmp_path):
    # Whole documents as chunks and k = all three: every passage is inside a retrieved chunk.
    whole = experiment(tmp_path / "whole", {"method": "document"}, 3)
    assert whole["passage_containment"]["metrics"]["recall"] == 1.0
    assert whole["passage_containment"]["n"] == 5
    # Three-word chunks cannot hold any of these passages (five words or more): MTCB recall is
    # 0, although the chunks still overlap the spans, which the overlap metric rewards.
    tiny = experiment(tmp_path / "tiny", {"method": "words", "size": 3}, 3)
    assert tiny["passage_containment"]["metrics"] == {
        "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "precision": 0.0,
    }  # fmt: skip
    assert tiny["metrics"]["recall"] > 0
