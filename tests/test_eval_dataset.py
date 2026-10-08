"""Evaluation dataset parsing, validation and versioning."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vectraudit.evaluation.dataset import DatasetError, load_dataset, parse_dataset


def _jsonl(*rows: dict[str, object]) -> str:
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def test_parse_full_query() -> None:
    ds = parse_dataset(
        _jsonl(
            {
                "query_id": "q1",
                "vector": [0.5, 1, -2.0],
                "text": "refund policy",
                "relevant": {"doc-1": 2, "doc-2": 0},
                "filter": {"lang": "en", "year": 2024, "meta.public": True},
                "tags": ["faq", "billing", "faq"],
            },
            {"query_id": "q2", "vector": [0, 0, 1]},
        )
    )
    q = ds.queries[0]
    assert len(ds) == 2
    assert ds.dim == 3
    assert q.vector is not None
    assert q.vector.dtype == np.float32
    assert q.relevant == {"doc-1": 2.0, "doc-2": 0.0}
    assert q.filter == {"lang": "en", "year": 2024, "meta.public": True}
    assert q.tags == ("billing", "faq")
    assert q.labelled
    assert not ds.queries[1].labelled
    assert ds.n_labelled == 1
    assert not ds.needs_embedder
    assert ds.summary() == {
        "sha256": ds.sha256,
        "n_queries": 2,
        "n_families": 2,
        "n_labelled": 1,
        "n_filtered": 1,
        "dim": 3,
        "tags": ["billing", "faq"],
        "meta": {},
    }


def test_text_only_queries_need_an_embedder() -> None:
    ds = parse_dataset(_jsonl({"query_id": "q", "text": "hello"}))
    assert ds.dim is None
    assert ds.needs_embedder


def test_hash_ignores_order_and_formatting_but_not_content() -> None:
    a = {"query_id": "a", "vector": [1.0, 2.0], "relevant": {"d": 1}}
    b = {"query_id": "b", "vector": [3.0, 4.0]}
    base = parse_dataset(_jsonl(a, b)).sha256
    assert parse_dataset(_jsonl(b, a)).sha256 == base
    assert parse_dataset(json.dumps(a, indent=None) + "\n\n" + json.dumps(b)).sha256 == base
    assert parse_dataset(_jsonl({**a, "relevant": {"d": 2}}, b)).sha256 != base
    assert parse_dataset(_jsonl({**a, "vector": [1.0, 2.5]}, b)).sha256 != base
    assert parse_dataset(_jsonl({**a, "tags": ["x"]}, b)).sha256 != base


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([1, 2], "JSON object"),
        ({"vector": [1.0]}, "query_id"),
        ({"query_id": "", "vector": [1.0]}, "query_id"),
        ({"query_id": "q"}, "vector, a text"),
        ({"query_id": "q", "vector": []}, "non-empty list"),
        ({"query_id": "q", "vector": [1, "2"]}, "only numbers"),
        ({"query_id": "q", "vector": [True]}, "only numbers"),
        ({"query_id": "q", "vector": [1e300]}, "finite float32"),
        ({"query_id": "q", "text": "  "}, "text"),
        ({"query_id": "q", "vector": [1.0], "relevant": ["d"]}, "map document ids"),
        ({"query_id": "q", "vector": [1.0], "relevant": {"d": -1}}, ">= 0"),
        ({"query_id": "q", "vector": [1.0], "relevant": {"d": True}}, ">= 0"),
        ({"query_id": "q", "vector": [1.0], "relevant": {"d": 0}}, "positive grade"),
        ({"query_id": "q", "vector": [1.0], "filter": {"a b": 1}}, "dotted identifier"),
        ({"query_id": "q", "vector": [1.0], "filter": {"a": [1]}}, "equality only"),
        ({"query_id": "q", "vector": [1.0], "filter": {"a": None}}, "equality only"),
        ({"query_id": "q", "vector": [1.0], "tags": "faq"}, "tags"),
        ({"query_id": "q", "vector": [1.0], "relevance": {"d": 1}}, "unknown field"),
    ],
)
def test_invalid_rows_name_the_line(row: object, message: str) -> None:
    text = _jsonl({"query_id": "ok", "vector": [1.0]}) + json.dumps(row)
    with pytest.raises(DatasetError, match=rf"^Line 2: .*{message}"):
        parse_dataset(text)


def test_dataset_level_errors(tmp_path) -> None:
    with pytest.raises(DatasetError, match="duplicate query_id"):
        parse_dataset(_jsonl({"query_id": "q", "vector": [1.0]}, {"query_id": "q", "text": "x"}))
    with pytest.raises(DatasetError, match="dimension 2, expected 1"):
        parse_dataset(
            _jsonl({"query_id": "a", "vector": [1.0]}, {"query_id": "b", "vector": [1, 2]})
        )
    with pytest.raises(DatasetError, match="no queries"):
        parse_dataset("\n\n")
    with pytest.raises(DatasetError, match=r"Line 1: .*NaN"):
        parse_dataset('{"query_id": "q", "vector": [NaN]}')
    with pytest.raises(DatasetError, match="Line 1"):
        parse_dataset("{not json")
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")
    (tmp_path / "bin.jsonl").write_bytes(b"\xff\xfe")
    with pytest.raises(DatasetError, match="UTF-8"):
        load_dataset(tmp_path / "bin.jsonl")


def test_load_dataset(tmp_path) -> None:
    path = tmp_path / "ds.jsonl"
    path.write_text(_jsonl({"query_id": "q", "vector": [1.0, 2.0], "relevant": {"d": 1}}))
    ds = load_dataset(path)
    assert ds.sha256 == parse_dataset(path.read_text()).sha256
    assert len(ds.sha256) == 64


def test_header_metadata_is_part_of_the_version() -> None:
    row = _jsonl({"query_id": "q", "text": "hi"})
    header = {"dataset": {"name": "support", "embedding_model": "e5", "normalized": True}}
    ds = parse_dataset(json.dumps(header) + "\n" + row)
    assert ds.meta == {"name": "support", "embedding_model": "e5", "normalized": True}
    assert ds.summary()["meta"]["embedding_model"] == "e5"
    assert ds.sha256 != parse_dataset(row).sha256
    other = {"dataset": {"name": "support", "embedding_model": "e5-large", "normalized": True}}
    assert parse_dataset(json.dumps(other) + "\n" + row).sha256 != ds.sha256


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ({"dataset": {"colour": "red"}}, "unknown header field"),
        ({"dataset": {"name": 3}}, "name must be a string"),
        ({"dataset": {"normalized": "yes"}}, "true or false"),
        ({"dataset": [], "x": 1}, "nothing else"),
    ],
)
def test_invalid_headers(header, message) -> None:
    with pytest.raises(DatasetError, match=message):
        parse_dataset(json.dumps(header) + "\n" + _jsonl({"query_id": "q", "text": "x"}))


def test_header_must_come_first() -> None:
    text = _jsonl({"query_id": "q", "text": "x"}) + json.dumps({"dataset": {"name": "n"}})
    with pytest.raises(DatasetError, match="Line 2: the dataset header must be the first line"):
        parse_dataset(text)
