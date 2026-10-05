"""Corpus loading and chunking with source offsets."""

from __future__ import annotations

import json

import pytest

from vectraudit.experiments.chunking import ChunkingSpec, chunk_corpus, chunk_document
from vectraudit.experiments.corpus import CorpusError, Document, load_corpus, make_corpus

TEXT = " ".join(f"w{i}" for i in range(10))  # w0 ... w9


def test_load_corpus(tmp_path) -> None:
    path = tmp_path / "c.jsonl"
    rows = [{"id": 1, "body": "alpha beta", "lang": "en"}, {"id": "b", "body": "gamma"}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    corpus = load_corpus(path, id_field="id", text_field="body")
    assert [d.doc_id for d in corpus.documents] == ["1", "b"]
    assert corpus.documents[0].meta == {"lang": "en"}
    assert corpus.get("b").text == "gamma"
    reordered = make_corpus(list(reversed(corpus.documents)))
    assert reordered.sha256 == corpus.sha256
    edited = make_corpus([Document("1", "alpha beta!"), corpus.documents[1]])
    assert edited.sha256 != corpus.sha256


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"doc_id": "a", "text": ""}], "non-empty text"),
        ([{"doc_id": "", "text": "x"}], "doc_id"),
        ([{"doc_id": True, "text": "x"}], "doc_id"),
        ([{"doc_id": "a", "text": "x"}, {"doc_id": "a", "text": "y"}], "duplicate"),
        ([], "no documents"),
    ],
)
def test_corpus_errors(tmp_path, rows, message) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(CorpusError, match=message):
        load_corpus(path)


def test_word_windows_with_overlap() -> None:
    doc = Document("d", TEXT)
    chunks = chunk_document(doc, ChunkingSpec("words", size=4, overlap=1))
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert [c.chunk_id for c in chunks] == ["d#0", "d#1", "d#2"]
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text
    no_overlap = chunk_document(doc, ChunkingSpec("words", size=4))
    assert [c.text for c in no_overlap] == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
    whole = chunk_document(doc, ChunkingSpec("words", size=50))
    assert [c.text for c in whole] == [TEXT]


def test_paragraphs_merge_and_split() -> None:
    text = "one two\n\nthree four five\n\n\nsix\n\n" + " ".join(f"x{i}" for i in range(7))
    doc = Document("p", text)
    chunks = chunk_document(doc, ChunkingSpec("paragraphs", size=6))
    # 2 + 3 + 1 words fit the 6-word limit together; the 7-word paragraph is split.
    assert [c.text.split() for c in chunks] == [
        ["one", "two", "three", "four", "five", "six"],
        ["x0", "x1", "x2", "x3", "x4", "x5"],
        ["x6"],
    ]
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text


def test_document_chunking_and_labels() -> None:
    corpus = make_corpus([Document("a", "  padded text  "), Document("b", TEXT)])
    chunks = chunk_corpus(corpus, ChunkingSpec("document"))
    assert [(c.chunk_id, c.text) for c in chunks] == [("a#0", "padded text"), ("b#0", TEXT)]
    assert ChunkingSpec("words", 150, 20).label == "words-150-overlap20"
    assert ChunkingSpec("words", 150).label == "words-150"
    assert ChunkingSpec("paragraphs", 300).label == "paragraphs-300"


@pytest.mark.parametrize(
    "kwargs", [{"method": "sentences"}, {"size": 0}, {"size": 5, "overlap": 5}, {"overlap": -1}]
)
def test_invalid_specs(kwargs) -> None:
    with pytest.raises(ValueError, match=r"method|size|overlap"):
        ChunkingSpec(**kwargs)
