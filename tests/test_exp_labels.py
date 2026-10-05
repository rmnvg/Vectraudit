"""Labels on documents and spans, mapped onto chunkings."""

from __future__ import annotations

import json

import pytest

from vectraudit.evaluation.dataset import DatasetError, parse_dataset
from vectraudit.experiments.chunking import ChunkingSpec, chunk_document
from vectraudit.experiments.corpus import Document
from vectraudit.experiments.labels import chunk_labels, document_labels, documents_in_order

DOC = Document("d1", "Refunds take five days. Passwords reset by email. Invoices are monthly.")


def test_spans_in_datasets() -> None:
    row = {
        "query_id": "q",
        "text": "how long do refunds take",
        "relevant": {"d9": 1},
        "relevant_spans": [{"doc_id": "d1", "start": 0, "end": 23, "grade": 2}],
    }
    ds = parse_dataset(json.dumps(row))
    q = ds.queries[0]
    assert q.labelled
    assert q.relevant_spans[0].grade == 2.0
    assert document_labels(q) == {"d9": 1.0, "d1": 2.0}
    without = parse_dataset(json.dumps({k: v for k, v in row.items() if k != "relevant_spans"}))
    assert without.sha256 != ds.sha256
    spans_only = parse_dataset(
        json.dumps(
            {
                "query_id": "q",
                "text": "x",
                "relevant_spans": [{"doc_id": "d1", "start": 5, "end": 9}],
            }
        )
    )
    assert spans_only.queries[0].labelled
    assert spans_only.n_labelled == 1


@pytest.mark.parametrize(
    ("spans", "message"),
    [
        ("d1", "must be a list"),
        ([{"doc_id": "d1", "start": 5, "end": 5}], "start < end"),
        ([{"doc_id": "d1", "start": -1, "end": 5}], "start < end"),
        ([{"doc_id": "d1", "start": 0, "end": 5, "grade": 0}], "positive"),
        ([{"doc_id": "d1", "start": 0, "end": 5, "page": 3}], "doc_id, start, end"),
        ([{"start": 0, "end": 5}], "doc_id"),
    ],
)
def test_invalid_spans(spans, message) -> None:
    with pytest.raises(DatasetError, match=message):
        parse_dataset(json.dumps({"query_id": "q", "text": "x", "relevant_spans": spans}))


def test_documents_in_order() -> None:
    doc_of = {"a#0": "a", "a#1": "a", "b#0": "b", "c#3": "c"}
    assert documents_in_order(["a#1", "b#0", "a#0", "c#3"], doc_of) == ["a", "b", "c"]


def test_chunk_labels_follow_the_span_across_chunkings() -> None:
    q = parse_dataset(
        json.dumps(
            {
                "query_id": "q",
                "text": "refunds",
                "relevant_spans": [
                    {"doc_id": "d1", "start": 0, "end": 23, "grade": 2},
                    {"doc_id": "d1", "start": 50, "end": 72, "grade": 1},
                ],
            }
        )
    ).queries[0]
    small = chunk_document(DOC, ChunkingSpec("words", size=4))
    labels = chunk_labels(q.relevant_spans, small)
    texts = {c.chunk_id: c.text for c in small}
    assert texts["d1#0"].startswith("Refunds")
    assert labels["d1#0"] == 2.0
    assert "d1#1" not in labels or labels["d1#1"] in (1.0, 2.0)
    whole = chunk_document(DOC, ChunkingSpec("document"))
    assert chunk_labels(q.relevant_spans, whole) == {"d1#0": 2.0}
    other = Document("d2", DOC.text)
    assert chunk_labels(q.relevant_spans, chunk_document(other, ChunkingSpec("document"))) == {}
