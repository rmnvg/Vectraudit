"""Qdrant connector, against qdrant-client's embedded local mode (no server needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_eval_runner import make_dataset
from vectraudit.connectors import ConnectorError, Distance, ErrorCode, SearchRequest
from vectraudit.connectors.config import open_connector, parse_connection
from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.evaluation.runner import RunOptions, execute_run

qdrant_client = pytest.importorskip("qdrant_client")
models = qdrant_client.models
N = 300


def build(path: Path, vectors: np.ndarray, *, named: list[str] | None = None) -> Path:
    """A local Qdrant collection "docs" holding ``vectors`` with ids 0..n-1 and payloads."""
    client = qdrant_client.QdrantClient(path=str(path))
    params = models.VectorParams(size=vectors.shape[1], distance=models.Distance.EUCLID)
    config = {name: params for name in named} if named else params
    client.create_collection("docs", vectors_config=config)
    points = []
    for i, v in enumerate(vectors):
        vector = {name: v.tolist() for name in named} if named else v.tolist()
        payload = {"lang": "en" if i % 2 == 0 else "de", "score": float(i % 3), "public": i < 100}
        points.append(models.PointStruct(id=i, vector=vector, payload=payload))
    client.upsert("docs", points=points)
    client.close()
    return path


@pytest.fixture
def store(synthetic, tmp_path) -> Path:
    return build(tmp_path / "qdrant", np.load(synthetic["vectors"])[:N])


@pytest.fixture(scope="module")
def query(synthetic) -> np.ndarray:
    return np.load(synthetic["queries"])[0]


def connect(path: Path, **extra) -> object:
    return open_connector(
        parse_connection({"backend": "qdrant", "path": str(path), "collection": "docs", **extra})
    )


def test_describe_and_capabilities(store) -> None:
    with connect(store) as c:
        info = c.describe()
        caps = c.capabilities()
    assert info.backend == "qdrant"
    assert info.distance is Distance.L2
    assert info.dim == 32
    assert info.count == N
    assert info.target == {"collection": "docs", "vector": None, "mode": "local"}
    assert info.snapshot is None
    assert info.client_version
    assert "hnsw" in info.config
    assert caps.filters
    assert caps.exact_reference
    assert caps.tuning_params == ("hnsw_ef",)


def test_search_matches_exact_faiss(synthetic, store, tmp_path, query) -> None:
    import faiss

    flat = faiss.IndexFlatL2(32)
    flat.add(np.load(synthetic["vectors"])[:N])
    faiss.write_index(flat, str(tmp_path / "flat.index"))
    expected = FaissConnector(tmp_path / "flat.index").search(SearchRequest(query, 10))
    with connect(store) as c:
        assert c.search(SearchRequest(query, 10, params={"hnsw_ef": 64})) == expected
        assert c.search(SearchRequest(query, 10, exact=True)) == expected


def test_filters(store, query) -> None:
    with connect(store) as c:
        en = c.search(SearchRequest(query, 20, filter={"lang": "en"}))
        both = c.search(SearchRequest(query, 20, filter={"lang": "en", "public": True}))
        floats = c.search(SearchRequest(query, 20, filter={"score": 2.0}))
    assert len(en) == 20
    assert all(int(i) % 2 == 0 for i in en)
    assert all(int(i) % 2 == 0 and int(i) < 100 for i in both)
    assert all(int(i) % 3 == 2 for i in floats)


def test_named_vectors(synthetic, tmp_path, query) -> None:
    path = build(tmp_path / "named", np.load(synthetic["vectors"])[:50], named=["dense", "alt"])
    with pytest.raises(ConnectorError, match="several named vectors") as e:
        connect(path)
    assert "alt, dense" in (e.value.hint or "")
    with pytest.raises(ConnectorError, match="no dense vector 'nope'"):
        connect(path, vector_name="nope")
    with connect(path, vector_name="alt") as c:
        assert c.describe().target["vector"] == "alt"
        assert len(c.search(SearchRequest(query, 5))) == 5


def test_unnamed_collection_rejects_vector_name(store) -> None:
    with pytest.raises(ConnectorError, match="single unnamed vector"):
        connect(store, vector_name="dense")


def test_config_errors(store, tmp_path) -> None:
    with pytest.raises(ConnectorError, match="not found") as e:
        connect(store, collection="missing")
    assert e.value.code is ErrorCode.CONFIG
    cases = [
        ({"backend": "qdrant", "collection": "docs"}, "either"),
        ({"backend": "qdrant", "collection": "docs", "url": "ftp://x"}, "Invalid Qdrant url"),
        ({"backend": "qdrant", "collection": "docs", "url": "http://u:p@h:6333"}, "credentials"),
        ({"backend": "qdrant", "url": "http://h:6333"}, 'needs "collection"'),
        ({"backend": "qdrant", "collection": "d", "url": "http://h", "timeout_s": 0}, ">= 1"),
        ({"backend": "qdrant", "collection": "d", "url": "http://h", "index": "x"}, "Unknown"),
    ]
    for raw, message in cases:
        with pytest.raises(ConnectorError, match=message):
            open_connector(parse_connection(raw, tmp_path))


def test_unreachable_server_is_a_connection_error() -> None:
    raw = {"backend": "qdrant", "url": "http://127.0.0.1:9", "collection": "d", "timeout_s": 1}
    with pytest.raises(ConnectorError) as e:
        open_connector(parse_connection(raw))
    assert e.value.code is ErrorCode.CONNECTION


def test_api_key_comes_from_the_environment(monkeypatch) -> None:
    raw = {
        "backend": "qdrant",
        "url": "http://127.0.0.1:9",
        "collection": "d",
        "api_key_env": "VA_QDRANT_KEY",
    }
    with pytest.raises(ConnectorError, match="VA_QDRANT_KEY is not set"):
        open_connector(parse_connection(raw))


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, ErrorCode.AUTH),
        (403, ErrorCode.AUTH),
        (404, ErrorCode.CONFIG),
        (408, ErrorCode.TIMEOUT),
        (429, ErrorCode.CONNECTION),
        (503, ErrorCode.CONNECTION),
        (400, ErrorCode.QUERY),
    ],
)
def test_http_errors_are_classified(status, code) -> None:
    from vectraudit.connectors.qdrant import _http_error

    assert _http_error(status, b"{}", "docs").code is code


def test_bad_tuning_params(store, query) -> None:
    with connect(store) as c:
        for params in ({"hnsw_ef": 0}, {"hnsw_ef": 1.5}, {"nprobe": 4}):
            with pytest.raises(ConnectorError) as e:
                c.search(SearchRequest(query, 5, params=params))
            assert e.value.code is ErrorCode.CONFIG


def test_run_with_filters_on_qdrant(synthetic, store) -> None:
    ds = make_dataset(synthetic, n=10, filter={"lang": "en"})
    with connect(store) as c:
        run = execute_run(ds, c, RunOptions(k=5, params={"hnsw_ef": 32}))
    assert run["status"] == "complete"
    assert run["summary"]["ann_recall"] == 1.0
    assert run["backend"]["distance"] == "l2"
    assert all(int(i) % 2 == 0 for q in run["queries"] for i in q["ids"])
    json.dumps(run, allow_nan=False)
