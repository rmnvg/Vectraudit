"""Pinecone connector: request shapes and errors (fake transport), then pinecone-local.

Integration tests need ``VECTRAUDIT_TEST_PINECONE_URL`` (the pinecone-local control plane,
e.g. ``http://localhost:5080``) and are skipped without it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid

import numpy as np
import pytest

from tests.test_eval_runner import make_dataset
from vectraudit.connectors import ConnectorError, Distance, ErrorCode, SearchRequest
from vectraudit.connectors.config import open_connector, parse_connection
from vectraudit.connectors.pinecone import API_VERSION, PineconeConnector
from vectraudit.evaluation.runner import RunOptions, execute_run

INDEX = {
    "name": "docs",
    "dimension": 3,
    "metric": "dotproduct",
    "host": "docs-abc.svc.pinecone.io",
    "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
}


class Fake:
    """Records requests; answers from a table of (method, path suffix) -> (status, body)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), json.loads(body) if body else None))
        for (m, suffix), (status, payload) in self.routes.items():
            if m == method and url.endswith(suffix):
                if isinstance(payload, Exception):
                    raise payload
                return status, json.dumps(payload).encode()
        return 404, b'{"error": {"message": "nope"}}'


def connect(routes, **kwargs) -> tuple[PineconeConnector, Fake]:
    fake = Fake({("GET", "/indexes/docs"): (200, INDEX), **routes})
    return PineconeConnector(api_key="pk-secret", index="docs", send=fake, **kwargs), fake


def test_query_shape_and_results() -> None:
    c, fake = connect(
        {
            ("POST", "/query"): (
                200,
                {
                    "matches": [
                        {"id": "b", "score": 0.9},
                        {"id": "a", "score": 0.5},
                        {"id": "b", "score": 0.4},
                    ]
                },
            )
        },
        namespace="prod",
    )
    ids = c.search(
        SearchRequest(np.array([1, 0, 0], dtype=np.float32), 5, filter={"lang": "en", "year": 2024})
    )
    assert ids == ["b", "a"]
    _, url, headers, body = fake.calls[-1]
    assert url == "https://docs-abc.svc.pinecone.io/query"
    assert headers["Api-Key"] == "pk-secret"
    assert headers["X-Pinecone-API-Version"] == API_VERSION
    assert body == {
        "vector": [1.0, 0.0, 0.0],
        "topK": 5,
        "namespace": "prod",
        "includeValues": False,
        "includeMetadata": False,
        "filter": {"lang": {"$eq": "en"}, "year": {"$eq": 2024}},
    }
    assert fake.calls[0][1] == "https://api.pinecone.io/indexes/docs"


def test_describe_and_capabilities() -> None:
    stats = {
        "namespaces": {"prod": {"vectorCount": 7}, "dev": {"vectorCount": 2}},
        "dimension": 3,
        "indexFullness": 0.1,
        "totalVectorCount": 9,
    }
    c, _ = connect({("POST", "/describe_index_stats"): (200, stats)}, namespace="prod")
    info = c.describe()
    assert info.backend == "pinecone"
    assert info.distance is Distance.IP
    assert info.dim == 3
    assert info.count == 7
    assert info.target == {"index": "docs", "namespace": "prod"}
    assert info.snapshot is None
    assert info.config["namespaces"] == 2
    caps = c.capabilities()
    assert caps.filters
    assert not caps.exact_reference
    assert caps.tuning_params == ()
    whole, _ = connect({("POST", "/describe_index_stats"): (200, stats)})
    assert whole.describe().count == 9
    with pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(np.zeros(3, dtype=np.float32), 3, exact=True))
    assert e.value.code is ErrorCode.UNSUPPORTED


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, ErrorCode.AUTH),
        (403, ErrorCode.AUTH),
        (404, ErrorCode.CONFIG),
        (400, ErrorCode.QUERY),
        (429, ErrorCode.CONNECTION),
        (503, ErrorCode.CONNECTION),
        (504, ErrorCode.TIMEOUT),
    ],
)
def test_http_errors(status, code) -> None:
    c, _ = connect({("POST", "/query"): (status, {"message": "bad"})})
    with pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(np.zeros(3, dtype=np.float32), 3))
    assert e.value.code is code


def test_transport_errors_and_missing_index() -> None:
    for exc, code in [
        (urllib.error.URLError("refused"), ErrorCode.CONNECTION),
        (TimeoutError(), ErrorCode.TIMEOUT),
        (urllib.error.URLError(TimeoutError()), ErrorCode.TIMEOUT),
    ]:
        c, _ = connect({("POST", "/query"): (0, exc)})
        with pytest.raises(ConnectorError) as e:
            c.search(SearchRequest(np.zeros(3, dtype=np.float32), 3))
        assert e.value.code is code
    with pytest.raises(ConnectorError) as e:
        PineconeConnector(api_key="k", index="missing", send=Fake({}))
    assert e.value.code is ErrorCode.CONFIG
    assert "index 'missing' not found" in e.value.message


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_plane_url": "http://pinecone.example"}, "Refusing to send the API key"),
        ({"host": "http://data.example"}, "Refusing to send the API key"),
        ({"control_plane_url": "https://u:p@api.pinecone.io"}, "must not contain credentials"),
        ({"control_plane_url": "ftp://x"}, "Invalid Pinecone"),
    ],
)
def test_url_safety(kwargs, message) -> None:
    with pytest.raises(ConnectorError, match=message):
        PineconeConnector(
            api_key="k", index="docs", send=Fake({("GET", "/indexes/docs"): (200, INDEX)}), **kwargs
        )


def test_connection_file(monkeypatch) -> None:
    with pytest.raises(ConnectorError, match="Secrets must not"):
        parse_connection({"backend": "pinecone", "index": "docs", "api_key": "x"})
    with pytest.raises(ConnectorError, match='Missing "api_key_env"'):
        open_connector(parse_connection({"backend": "pinecone", "index": "docs"}))
    monkeypatch.setenv("VA_PC", "k")
    with pytest.raises(ConnectorError, match="Unknown field"):
        open_connector(
            parse_connection(
                {"backend": "pinecone", "index": "d", "api_key_env": "VA_PC", "collection": "x"}
            )
        )


# --- against pinecone-local --------------------------------------------------------------

LOCAL = os.environ.get("VECTRAUDIT_TEST_PINECONE_URL")


def _http(method: str, url: str, body=None):
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={
            "Api-Key": "pclocal",
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": API_VERSION,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


@pytest.fixture
def local_index(synthetic):
    """A cosine index of the synthetic vectors (normalised) with metadata, in namespace ns."""
    if not LOCAL:
        pytest.skip("Set VECTRAUDIT_TEST_PINECONE_URL to run Pinecone integration tests.")
    name = f"va-{uuid.uuid4().hex[:10]}"
    vectors = np.load(synthetic["vectors_normalized"])[:300]
    info = _http(
        "POST",
        f"{LOCAL}/indexes",
        {
            "name": name,
            "dimension": 32,
            "metric": "cosine",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
        },
    )
    host = "http://" + info["host"]
    for start in range(0, len(vectors), 100):
        _http(
            "POST",
            f"{host}/vectors/upsert",
            {
                "namespace": "ns",
                "vectors": [
                    {
                        "id": str(i),
                        "values": v.tolist(),
                        "metadata": {"lang": "en" if i % 2 == 0 else "de", "public": i < 100},
                    }
                    for i, v in enumerate(vectors[start : start + 100], start)
                ],
            },
        )
    yield name, vectors
    _http("DELETE", f"{LOCAL}/indexes/{name}")


def test_matches_exact_search(local_index, monkeypatch) -> None:
    import faiss

    name, vectors = local_index
    monkeypatch.setenv("VA_PC_KEY", "pclocal")
    cfg = parse_connection(
        {
            "backend": "pinecone",
            "index": name,
            "namespace": "ns",
            "api_key_env": "VA_PC_KEY",
            "control_plane_url": LOCAL,
        }
    )
    flat = faiss.IndexFlatIP(32)
    flat.add(vectors)
    query = vectors[7] * 0.9 + vectors[8] * 0.1
    query /= np.linalg.norm(query)
    _, expected = flat.search(query.reshape(1, -1).astype(np.float32), 10)
    with open_connector(cfg) as c:
        info = c.describe()
        assert info.distance is Distance.COSINE
        assert info.count == 300
        found = c.search(SearchRequest(query.astype(np.float32), 10))
        assert found == [str(i) for i in expected[0]]
        en = c.search(SearchRequest(query.astype(np.float32), 20, filter={"lang": "en"}))
        both = c.search(
            SearchRequest(query.astype(np.float32), 20, filter={"lang": "en", "public": True})
        )
    assert all(int(i) % 2 == 0 for i in en)
    assert all(int(i) % 2 == 0 and int(i) < 100 for i in both)


def test_run_and_errors_on_pinecone_local(local_index, synthetic) -> None:
    name, _ = local_index
    c = PineconeConnector(api_key="pclocal", index=name, namespace="ns", control_plane_url=LOCAL)
    ds = make_dataset(synthetic, n=10)
    run = execute_run(ds, c, RunOptions(k=5))
    assert run["status"] == "complete"
    assert run["search"]["exact_reference"] is False
    assert run["summary"]["metrics"]["recall"] > 0
    with pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(np.zeros(3, dtype=np.float32), 3))
    assert e.value.code is ErrorCode.QUERY
    with pytest.raises(ConnectorError) as e:
        PineconeConnector(api_key="pclocal", index="no-such-index", control_plane_url=LOCAL)
    assert e.value.code is ErrorCode.CONFIG
