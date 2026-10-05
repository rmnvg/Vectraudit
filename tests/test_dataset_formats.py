"""Importers for labelled datasets kept in TREC, BEIR, Ragas and DeepEval formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.evaluation.dataset_formats import (
    CorpusIndex,
    import_beir,
    import_context_testset,
    import_format,
    import_qrels,
    load_corpus,
    read_qrels,
    read_query_texts,
)
from vectraudit.evaluation.dataset_tools import render_dataset

REFUND = "Refunds are issued within 14 days of a return being received at our warehouse."
RESET = "To reset your password, open Settings, choose Security and follow the email link."
SHIPPING = "Standard shipping takes three to five business days inside the country."


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _corpus(tmp_path: Path) -> Path:
    return _jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"_id": "doc-refund", "title": "Refunds", "text": REFUND},
            {"_id": "doc-reset", "text": "Account help.\n" + RESET + " Contact support if stuck."},
            {"_id": "doc-ship", "text": SHIPPING},
            {"_id": "doc-ship-copy", "text": SHIPPING},
        ],
    )


def _valid(out) -> None:
    """The imported rows load as a dataset with the provenance in its header."""
    ds = parse_dataset(render_dataset(out.rows, {"provenance": out.provenance}))
    assert ds.n_labelled == len(out.rows)


def test_read_qrels_trec_and_beir(tmp_path) -> None:
    trec = tmp_path / "qrels.txt"
    trec.write_text("q1 0 d1 2\nq1 0 d2 0\nq2 0 d3 1\nq2 0 d9 -2\n\nq1 0 d1 2\n")
    qrels, negative = read_qrels(trec)
    assert qrels == {"q1": {"d1": 2.0, "d2": 0.0}, "q2": {"d3": 1.0}}
    assert negative == 1
    assert list(qrels) == ["q1", "q2"]

    beir = tmp_path / "test.tsv"
    beir.write_text("query-id\tcorpus-id\tscore\nq1\tdoc with space\t1\n")
    assert read_qrels(beir) == ({"q1": {"doc with space": 1.0}}, 0)

    for content, match in [
        ("q1 0 d1 2\nq1 0 d1 1\n", "judged twice"),
        ("q1 d1\n", "got 2 fields"),
        ("q1 0 d1 2\nq2 0 d2 high\n", "not a number"),
        ("query-id\tcorpus-id\tscore\n", "no relevance judgments"),
    ]:
        trec.write_text(content)
        with pytest.raises(DatasetError, match=match):
            read_qrels(trec)
    with pytest.raises(DatasetError, match="not found"):
        read_qrels(tmp_path / "missing.txt")


def test_read_query_texts(tmp_path) -> None:
    tsv = tmp_path / "queries.tsv"
    tsv.write_text("q1\tHow do refunds work?\nq2\t Reset password \n")
    assert read_query_texts(tsv) == {"q1": "How do refunds work?", "q2": "Reset password"}
    tsv.write_text("q1 no tab here\n")
    with pytest.raises(DatasetError, match="qid<TAB>text"):
        read_query_texts(tsv)
    tsv.write_text("q1\ta\nq1\tb\n")
    with pytest.raises(DatasetError, match="duplicate query id"):
        read_query_texts(tsv)
    jl = _jsonl(tmp_path / "q.jsonl", [{"_id": 7, "text": "seven", "metadata": {}}])
    assert read_query_texts(jl) == {"7": "seven"}
    _jsonl(jl, [{"text": "no id"}])
    with pytest.raises(DatasetError, match="needs an id"):
        read_query_texts(jl)


def test_import_trec(tmp_path) -> None:
    (tmp_path / "topics.tsv").write_text("q1\tRefund timing\nq2\tPassword\nq3\tUnjudged\n")
    (tmp_path / "qrels.txt").write_text(
        "q1 0 doc-refund 2\nq1 0 doc-ship 0\nq2 0 doc-reset 0\nq4 0 doc-x 1\nq1 0 junk -1\n"
    )
    out = import_qrels(tmp_path / "topics.tsv", tmp_path / "qrels.txt")
    assert out.rows == [
        {
            "query_id": "q1",
            "text": "Refund timing",
            "relevant": {"doc-refund": 2.0, "doc-ship": 0.0},
        }
    ]
    assert out.skipped == {"no relevant document": 1, "no query text": 1}
    assert out.provenance["source_format"] == "trec"
    assert len(out.provenance["qrels_sha256"]) == 64
    assert out.provenance["skipped"] == {"no query text": 1, "no relevant document": 1}
    assert any("negative" in w for w in out.warnings)
    assert any("1 queries in topics.tsv have no judgments" in w for w in out.warnings)
    _valid(out)

    (tmp_path / "qrels.txt").write_text("q2 0 doc-reset 0\n")
    with pytest.raises(DatasetError, match=r"No labelled queries.*no relevant document \(1\)"):
        import_qrels(tmp_path / "topics.tsv", tmp_path / "qrels.txt")


def test_import_beir(tmp_path) -> None:
    root = tmp_path / "fiqa"
    (root / "qrels").mkdir(parents=True)
    _jsonl(
        root / "queries.jsonl", [{"_id": "1", "text": "Refund timing"}, {"_id": "2", "text": "x"}]
    )
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n1\tdoc-refund\t1\n")
    (root / "qrels" / "dev.tsv").write_text("query-id\tcorpus-id\tscore\n2\tdoc-ship\t1\n")
    out = import_beir(root)
    assert out.rows == [{"query_id": "1", "text": "Refund timing", "relevant": {"doc-refund": 1.0}}]
    assert out.provenance["source_format"] == "beir"
    assert out.provenance["split"] == "test"
    assert import_beir(root, split="dev").rows[0]["query_id"] == "2"
    with pytest.raises(DatasetError, match="Available splits: dev, test"):
        import_beir(root, split="train")
    with pytest.raises(DatasetError, match="not a directory"):
        import_beir(root / "queries.jsonl")
    _valid(out)


def test_corpus_index_locates_passages(tmp_path) -> None:
    index = load_corpus(_corpus(tmp_path))
    assert len(index) == 4
    assert index.find(REFUND, limit=5) == ["doc-refund"]
    # Contained in a longer document, with different whitespace.
    assert index.find(RESET.replace(" ", "\n  ", 3), limit=5) == ["doc-reset"]
    assert index.find(SHIPPING, limit=5) == ["doc-ship", "doc-ship-copy"]
    assert index.find(SHIPPING, limit=1) == ["doc-ship", "doc-ship-copy"]  # limit + 1: ambiguous
    assert index.find("Nothing like this appears anywhere.", limit=5) == []
    # A passage never spans two documents.
    assert index.find(REFUND[-20:] + " " + "Account help.", limit=5) == []

    ids = CorpusIndex([("a", "x " * 50), ("b", "y")])
    assert ids.find("x x x", limit=5) == ["a"]
    with pytest.raises(DatasetError, match="no documents"):
        CorpusIndex([])

    _jsonl(tmp_path / "c.jsonl", [{"pk": 1, "body": REFUND}])
    assert load_corpus(tmp_path / "c.jsonl", id_field="pk", text_field="body").find(
        REFUND, limit=5
    ) == ["1"]
    with pytest.raises(DatasetError, match="--corpus-id-field"):
        load_corpus(tmp_path / "c.jsonl")
    with pytest.raises(DatasetError, match="--corpus-text-field"):
        load_corpus(tmp_path / "c.jsonl", id_field="pk")


def test_import_ragas_by_text_and_id(tmp_path) -> None:
    index = load_corpus(_corpus(tmp_path))
    testset = _jsonl(
        tmp_path / "ragas.jsonl",
        [
            {
                "user_input": "How long do refunds take?",
                "reference_contexts": [REFUND],
                "reference": "14 days",
                "synthesizer_name": "single_hop_specific_query_synthesizer",
            },
            {
                "user_input": "Refund and password?",
                "reference_contexts": [
                    REFUND,
                    RESET,
                    "A passage that is not in the corpus at all.",
                ],
                "synthesizer_name": "multi_hop_abstract_query_synthesizer",
            },
            {"user_input": "Shipping time?", "reference_contexts": [SHIPPING]},
            {"user_input": "Short", "reference_contexts": ["Yes."]},
            {"user_input": "By id", "reference_context_ids": ["doc-7", 8]},
            {"user_input": "No context at all"},
            {"user_input": "How long do refunds take?", "reference_contexts": [REFUND]},
        ],
    )
    out = import_context_testset(testset, source_format="ragas", corpus=index, max_matches=1)
    by_text = {r["text"]: r for r in out.rows}
    assert by_text["How long do refunds take?"]["relevant"] == {"doc-refund": 1.0}
    assert by_text["How long do refunds take?"]["tags"] == ["single_hop_specific_query_synthesizer"]
    assert by_text["Refund and password?"]["relevant"] == {"doc-refund": 1.0, "doc-reset": 1.0}
    assert by_text["By id"]["relevant"] == {"doc-7": 1.0, "8": 1.0}
    assert by_text["By id"]["query_id"].startswith("q-")
    assert len(out.rows) == 3
    assert out.skipped == {
        "no reference passage found in the corpus": 2,  # ambiguous shipping, too-short passage
        "no reference context": 1,
        "duplicate question": 1,
    }
    assert out.provenance["passage_matching"] == {
        "ambiguous": 1,
        "located": 4,
        "not found": 1,
        "too short": 1,
    }
    assert any("3 of 7 reference passages could not be located" in w for w in out.warnings)
    _valid(out)

    # Ids are content-derived, so they do not depend on the file's order.
    again = import_context_testset(testset, source_format="ragas", corpus=index)
    assert {r["query_id"] for r in out.rows} <= {r["query_id"] for r in again.rows}

    with pytest.raises(DatasetError, match="Pass --corpus"):
        import_context_testset(testset, source_format="ragas")
    _jsonl(testset, [{"input": "deepeval shape"}])
    with pytest.raises(DatasetError, match="Is this a ragas file"):
        import_context_testset(testset, source_format="ragas")
    with pytest.raises(DatasetError, match="must be one of"):
        import_context_testset(testset, source_format="phoenix")


def test_import_ragas_csv_with_python_lists(tmp_path) -> None:
    # pandas DataFrame.to_csv writes list columns as Python reprs.
    path = tmp_path / "testset.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_input", "reference_contexts", "reference"])
        w.writerow(["Refunds?", repr([REFUND]), "14 days"])
        w.writerow(["Reset?", json.dumps([RESET]), "Settings"])
    out = import_context_testset(path, source_format="ragas", corpus=load_corpus(_corpus(tmp_path)))
    assert [r["relevant"] for r in out.rows] == [{"doc-refund": 1.0}, {"doc-reset": 1.0}]


def test_import_deepeval_json_and_csv(tmp_path) -> None:
    index = load_corpus(_corpus(tmp_path))
    goldens = tmp_path / "goldens.json"
    goldens.write_text(
        json.dumps(
            [
                {
                    "input": "How long do refunds take?",
                    "expected_output": "14 days",
                    "context": [REFUND],
                    "retrieval_context": [RESET],
                },
                {"id": "g-2", "input": "Reset?", "context": [RESET]},
            ]
        )
    )
    out = import_context_testset(goldens, source_format="deepeval", corpus=index)
    # retrieval_context is what a system retrieved, never a label.
    assert out.rows[0]["relevant"] == {"doc-refund": 1.0}
    assert out.rows[1]["query_id"] == "g-2"
    _valid(out)

    path = tmp_path / "goldens.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "expected_output", "context"])
        w.writerow(["Both?", "", f"{REFUND}|{RESET}"])
    out = import_context_testset(path, source_format="deepeval", corpus=index)
    assert out.rows[0]["relevant"] == {"doc-refund": 1.0, "doc-reset": 1.0}


def test_import_format_dispatch(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    testset = _jsonl(
        tmp_path / "ragas.jsonl", [{"user_input": "Refunds?", "reference_contexts": [REFUND]}]
    )
    out = import_format("ragas", testset, corpus=corpus)
    assert out.provenance["corpus_file"] == "corpus.jsonl"
    assert len(out.provenance["corpus_sha256"]) == 64
    (tmp_path / "topics.tsv").write_text("q1\tRefunds\n")
    with pytest.raises(DatasetError, match="needs --qrels"):
        import_format("trec", tmp_path / "topics.tsv")
    with pytest.raises(DatasetError, match="Unknown format 'phoenix'"):
        import_format("phoenix", testset)
