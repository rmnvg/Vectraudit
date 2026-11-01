"""Chroma connector over the v2 REST API.

Unit tests use a fake shaped like the responses a real Chroma 1.5.9 server returned. The
live tests need ``VECTRAUDIT_TEST_CHROMA_URL`` (e.g. http://localhost:8000; CI runs the
``chromadb/chroma:1.5.9`` image) and seed their own collections through the REST API.
"""

from __future__ import annotations

import contextlib
import json
import os
import urllib.error
import urllib.request
import uuid

import numpy as np
import pytest

from vectraudit.connectors import chroma as chroma_module
from vectraudit.connectors.base import ConnectorError, ErrorCode, SearchRequest
from vectraudit.connectors.chroma import ChromaConnector, exact_ranking, where
from vectraudit.connectors.config import open_connector, parse_connection

URL = os.environ.get("VECTRAUDIT_TEST_CHROMA_URL")
COLLECTION_ID = "842aaa5a-6041-401e-b507-94062aafd8d2"


def collection(space="cosine", dimension=8):
    """As GET .../collections/{name} returned it from a real server (trimmed)."""
    return {
        "id": COLLECTION_ID, "name": "kb", "tenant": "default_tenant",
        "database": "default_database", "dimension": dimension, "log_position": 0,
        "version": 0, "metadata": None,
        "configuration_json": {
            "hnsw": {"space": space, "ef_construction": 100, "ef_search": 50,
                     "max_neighbors": 16, "resize_factor": 1.2, "sync_threshold": 1000},
            "spann": None,
            "embedding_function": {"type": "known", "name": "default", "config": {}},
        },
    }  # fmt: skip


class Fake:
    """Answers the endpoints the connector uses; records every request."""

    def __init__(self, responses=None, records=None):
        self.calls = []
        self.responses = responses or {}
        self.records = records or []  # (id, vector, metadata)

    def __call__(self, method, url, headers, body, timeout):
        payload = json.loads(body) if body else None
        self.calls.append((method, url, dict(headers), payload))
        path = url.split("127.0.0.1:8000", 1)[1]
        for suffix, response in self.responses.items():
            if path.endswith(suffix):
                return (
                    response
                    if isinstance(response, tuple)
                    else (200, json.dumps(response).encode())
                )
        if path.endswith("/version"):
            return 200, b'"1.0.0"'
        if path.endswith("/count"):
            return 200, str(len(self.records)).encode()
        if path.endswith("/get"):
            rows = [r for r in self.records if _matches(r[2], payload.get("where"))]
            page = rows[payload["offset"] : payload["offset"] + payload["limit"]]
            out = {"ids": [r[0] for r in page], "include": payload["include"],
                   "embeddings": [r[1] for r in page]}  # fmt: skip
            if "metadatas" in payload["include"]:
                out["metadatas"] = [r[2] for r in page]
            return 200, json.dumps(out).encode()
        if path.endswith("/query"):
            metadatas = [[{"doc_id": "B"}, {"doc_id": "A"}, None]]
            out = {"ids": [["b", "a", "b"]], "include": [], "metadatas": metadatas}
            return 200, json.dumps(out).encode()
        return 200, json.dumps(collection()).encode()


def _matches(meta, clause):
    if clause is None:
        return True
    parts = clause.get("$and", [clause])
    return all(meta.get(k) == v for part in parts for k, v in part.items())


def connect(fake, **kw):
    return ChromaConnector(url="http://127.0.0.1:8000", collection="kb", send=fake, **kw)


def test_describe_reports_space_size_and_index_settings():
    fake = Fake(records=[("a", [0.0] * 8, {})] * 3)
    info = connect(fake).describe()
    assert (info.backend, info.distance.value, info.dim, info.count) == ("chroma", "cosine", 8, 3)
    assert info.config == {"index": "hnsw", "space": "cosine", "ef_search": 50,
                           "ef_construction": 100, "max_neighbors": 16}  # fmt: skip
    assert info.snapshot is None
    assert info.target == {"tenant": "default_tenant", "database": "default_database",
                           "collection": "kb"}  # fmt: skip
    assert fake.calls[1][1].endswith(
        "/api/v2/tenants/default_tenant/databases/default_database/collections/kb"
    )
    assert fake.calls[2][1].endswith(f"/collections/{COLLECTION_ID}/count")


def test_a_query_sends_the_vector_k_and_a_where_clause():
    fake = Fake()
    ids = connect(fake).search(SearchRequest(vector=np.ones(8, np.float32), k=2,
                                             filter={"lang": "en", "public": True}))  # fmt: skip
    assert ids == ["b", "a"]  # duplicates removed, cut at k by the base class
    method, url, _, body = fake.calls[-1]
    assert (method, url.rsplit("/", 1)[1]) == ("POST", "query")
    assert body == {"query_embeddings": [[1.0] * 8], "n_results": 2, "include": [],
                    "where": {"$and": [{"lang": "en"}, {"public": True}]}}  # fmt: skip
    assert where({}) is None
    assert where({"lang": "en"}) == {"lang": "en"}


def test_id_field_reads_ids_from_metadata():
    fake = Fake()
    ids = connect(fake, id_field="doc_id").search(SearchRequest(vector=np.ones(8, np.float32), k=3))
    assert ids == ["B", "A"]  # the record without the field contributes nothing
    assert fake.calls[-1][3]["include"] == ["metadatas"]


@pytest.mark.parametrize("space", ["l2", "cosine", "ip"])
def test_the_exact_ranking_uses_chromas_distances(space):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 6))
    query = rng.normal(size=6)
    expected = {
        "l2": ((vectors - query) ** 2).sum(axis=1),
        "ip": 1 - vectors @ query,
        "cosine": 1 - vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)),
    }[space]
    ids = [f"r{i}" for i in range(50)]
    assert exact_ranking(ids, vectors, query, space, 5) == [
        ids[i] for i in np.argsort(expected)[:5]
    ]


def test_the_exact_reference_pages_through_the_collections_vectors_with_the_filter(monkeypatch):
    monkeypatch.setattr(chroma_module, "PAGE", 4)
    rng = np.random.default_rng(1)
    records = [(f"r{i}", rng.normal(size=8).tolist(), {"lang": "en" if i % 2 else "de"})
               for i in range(10)]  # fmt: skip
    fake = Fake(records=records)
    connector = connect(fake)
    query = np.asarray(records[3][1], dtype=np.float32)
    assert connector.search(
        SearchRequest(vector=query, k=1, exact=True, filter={"lang": "en"})
    ) == ["r3"]
    gets = [c for c in fake.calls if c[1].endswith("/get")]
    assert [g[3]["offset"] for g in gets] == [0, 4]  # five records in pages of four
    assert all(g[3]["where"] == {"lang": "en"} for g in gets)
    connector.search(SearchRequest(vector=query, k=1, exact=True, filter={"lang": "en"}))
    assert len([c for c in fake.calls if c[1].endswith("/get")]) == 2  # cached per filter


def test_the_exact_reference_is_bounded():
    fake = Fake(records=[(f"r{i}", [0.0] * 8, {}) for i in range(6)])
    with pytest.raises(ConnectorError, match="limited to 5 records") as info:
        connect(fake, exact_limit=5).search(
            SearchRequest(vector=np.zeros(8, np.float32), k=1, exact=True)
        )
    assert info.value.code == ErrorCode.UNSUPPORTED


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, ErrorCode.AUTH), (403, ErrorCode.AUTH), (404, ErrorCode.CONFIG),
     (408, ErrorCode.TIMEOUT), (429, ErrorCode.CONNECTION), (503, ErrorCode.CONNECTION),
     (400, ErrorCode.QUERY)],
)  # fmt: skip
def test_http_failures_are_typed(status, code):
    fake = Fake(responses={"/collections/kb": (status, b'{"message": "nope"}')})
    with pytest.raises(ConnectorError) as info:
        connect(fake)
    assert info.value.code == code


def test_a_token_is_sent_in_both_headers_and_only_over_https_or_locally():
    fake = Fake()
    connect(fake, token="t0k")
    headers = fake.calls[0][2]
    assert (headers["Authorization"], headers["x-chroma-token"]) == ("Bearer t0k", "t0k")
    with pytest.raises(ConnectorError, match="Refusing to send the Chroma token"):
        ChromaConnector(url="http://chroma.example", collection="kb", token="t", send=Fake())
    with pytest.raises(ConnectorError, match="must not contain credentials"):
        ChromaConnector(url="https://u:p@chroma.example", collection="kb", send=Fake())


def test_unusable_collections_are_refused():
    empty = Fake(responses={"/collections/kb": collection(dimension=None)})
    with pytest.raises(ConnectorError, match="no vector dimension"):
        connect(empty).describe()
    odd = Fake(responses={"/collections/kb": collection(space="hamming")})
    with pytest.raises(ConnectorError, match="not supported") as info:
        connect(odd).describe()
    assert info.value.code == ErrorCode.UNSUPPORTED
    with pytest.raises(ConnectorError, match="isn't JSON"):
        connect(Fake(responses={"/collections/kb": (200, b"<html>")}))


def test_the_connection_file_is_validated(monkeypatch):
    monkeypatch.setenv("CHROMA_TOKEN", "t")
    cfg = parse_connection(
        {"backend": "chroma", "url": "http://127.0.0.1:9", "collection": "kb",
         "token_env": "CHROMA_TOKEN", "timeout_s": 1}
    )  # fmt: skip
    with pytest.raises(ConnectorError) as info:
        open_connector(cfg)  # nothing listens on port 9
    assert info.value.code == ErrorCode.CONNECTION
    with pytest.raises(ConnectorError, match="collection"):
        open_connector(parse_connection({"backend": "chroma", "url": "http://x"}))
    with pytest.raises(ConnectorError, match="nprobe"):
        open_connector(parse_connection(
            {"backend": "chroma", "url": "http://x", "collection": "kb", "nprobe": 4}))  # fmt: skip


# --- live, against a real Chroma server -----------------------------------------------


def _rest(method, path, body=None):
    request = urllib.request.Request(
        URL.rstrip("/") + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )  # fmt: skip
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read() or b"null")


@pytest.fixture(scope="module")
def live():
    if not URL:
        pytest.skip("Set VECTRAUDIT_TEST_CHROMA_URL to run Chroma live tests.")
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(300, 16)).astype(np.float32)
    base = "/api/v2/tenants/default_tenant/databases/default_database/collections"
    names = {}
    for space in ("l2", "cosine", "ip"):
        name = f"va-{space}-{uuid.uuid4().hex[:8]}"
        created = _rest("POST", base, {"name": name, "configuration": {"hnsw": {"space": space}}})
        _rest("POST", f"{base}/{created['id']}/add", {
            "ids": [f"r{i}" for i in range(300)],
            "embeddings": vectors.tolist(),
            "metadatas": [{"doc_id": f"doc-{i}", "lang": "en" if i % 3 else "de"}
                          for i in range(300)],
        })  # fmt: skip
        names[space] = name
    yield vectors, names
    for name in names.values():
        with contextlib.suppress(urllib.error.URLError):
            _rest("DELETE", f"{base}/{name}")


def _truth(vectors, query, space, mask=None):
    index = np.arange(len(vectors)) if mask is None else np.flatnonzero(mask)
    m, q = vectors[index].astype(np.float64), query.astype(np.float64)
    distances = {
        "l2": ((m - q) ** 2).sum(axis=1),
        "ip": 1 - m @ q,
        "cosine": 1 - m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q)),
    }[space]
    return [f"r{i}" for i in index[np.argsort(distances, kind="stable")[:10]]]


@pytest.mark.parametrize("space", ["l2", "cosine", "ip"])
def test_live_search_exact_reference_and_filters(live, space):
    vectors, names = live
    connector = ChromaConnector(url=URL, collection=names[space])
    info = connector.describe()
    assert (info.distance.value, info.dim, info.count) == (space, 16, 300)
    queries = np.random.default_rng(9).normal(size=(10, 16)).astype(np.float32)
    english = np.array([i % 3 != 0 for i in range(300)])
    for query in queries:
        assert connector.search(SearchRequest(vector=query, k=10, exact=True)) == _truth(
            vectors, query, space
        )
        assert connector.search(
            SearchRequest(vector=query, k=10, exact=True, filter={"lang": "en"})
        ) == _truth(vectors, query, space, english)
        found = connector.search(SearchRequest(vector=query, k=10, filter={"lang": "en"}))
        assert found
        assert all(int(i[1:]) % 3 != 0 for i in found)
    by_metadata = ChromaConnector(url=URL, collection=names[space], id_field="doc_id")
    assert all(
        i.startswith("doc-") for i in by_metadata.search(SearchRequest(vector=queries[0], k=5))
    )
