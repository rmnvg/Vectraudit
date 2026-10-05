"""Weaviate connector: configuration and error mapping, then a real Weaviate server.

Integration tests need ``VECTRAUDIT_TEST_WEAVIATE_URL`` (HTTP, e.g. http://localhost:8080)
and ``VECTRAUDIT_TEST_WEAVIATE_GRPC`` (e.g. http://localhost:50051); skipped without them.
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_eval_runner import make_dataset
from vectraudit.connectors import ConnectorError, Distance, ErrorCode, SearchRequest
from vectraudit.connectors.config import open_connector, parse_connection
from vectraudit.connectors.weaviate import WeaviateConnector, _error
from vectraudit.evaluation.runner import RunOptions, execute_run

weaviate = pytest.importorskip("weaviate")
URL = os.environ.get("VECTRAUDIT_TEST_WEAVIATE_URL")
GRPC = os.environ.get("VECTRAUDIT_TEST_WEAVIATE_GRPC")


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (type("AuthenticationFailedError", (Exception,), {})("no"), ErrorCode.AUTH),
        (type("InsufficientPermissionsError", (Exception,), {})("no"), ErrorCode.AUTH),
        (type("WeaviateConnectionError", (Exception,), {})("refused"), ErrorCode.CONNECTION),
        (Exception("Deadline Exceeded"), ErrorCode.TIMEOUT),
        (Exception("could not find class Missing in schema"), ErrorCode.CONFIG),
        (Exception("tenant acme does not exist"), ErrorCode.CONFIG),
        (Exception("vector lengths don't match"), ErrorCode.QUERY),
        (SimpleNamespace(status_code=403, __str__=lambda self: "x"), ErrorCode.AUTH),
    ],
)
def test_error_mapping(exc, code) -> None:
    assert _error(exc, "Docs").code is code  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"url": "ftp://x"}, "Invalid Weaviate url"),
        ({"url": "https://u:p@w.example"}, "must not contain credentials"),
        ({"url": "http://w.example", "api_key": "k"}, "Refusing to send the API key"),
        ({"url": "https://abc.weaviate.cloud"}, "needs an API key"),
    ],
)
def test_configuration_errors(kwargs, message) -> None:
    with pytest.raises(ConnectorError, match=message):
        WeaviateConnector(collection="Docs", **kwargs)


def test_unreachable_server() -> None:
    with pytest.raises(ConnectorError) as e:
        WeaviateConnector(
            url="http://127.0.0.1:9", grpc_url="http://127.0.0.1:10", collection="Docs", timeout_s=2
        )
    assert e.value.code is ErrorCode.CONNECTION


def test_connection_file(monkeypatch) -> None:
    with pytest.raises(ConnectorError, match="Secrets must not"):
        parse_connection(
            {"backend": "weaviate", "url": "https://x", "collection": "D", "api_key": "k"}
        )
    with pytest.raises(ConnectorError, match="Unknown field"):
        open_connector(
            parse_connection(
                {"backend": "weaviate", "url": "https://x", "collection": "D", "namespace": "x"}
            )
        )
    with pytest.raises(ConnectorError, match='needs "collection"'):
        open_connector(parse_connection({"backend": "weaviate", "url": "https://x"}))


# --- against Weaviate --------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(synthetic):
    """Collections: HNSW (cosine), flat, two named vectors, and multi-tenant."""
    if not URL or not GRPC:
        pytest.skip("Set VECTRAUDIT_TEST_WEAVIATE_URL and _GRPC to run Weaviate tests.")
    from urllib.parse import urlsplit

    from weaviate.classes.config import Configure, DataType, Property, VectorDistances
    from weaviate.classes.tenants import Tenant

    http, grpc = urlsplit(URL), urlsplit(GRPC)
    client = weaviate.connect_to_local(host=http.hostname, port=http.port, grpc_port=grpc.port)
    vectors = np.load(synthetic["vectors_normalized"])[:300]
    suffix = uuid.uuid4().hex[:8]
    names = {k: f"Va{k}{suffix}" for k in ("Hnsw", "Flat", "Named", "Tenants")}
    props = [
        Property(name="doc_id", data_type=DataType.TEXT),
        Property(name="lang", data_type=DataType.TEXT),
        Property(name="public", data_type=DataType.BOOL),
    ]

    def fill(coll, vector_of):
        with coll.batch.fixed_size(100) as batch:
            for i, v in enumerate(vectors):
                batch.add_object(
                    properties={
                        "doc_id": str(i),
                        "lang": "en" if i % 2 == 0 else "de",
                        "public": i < 100,
                    },
                    vector=vector_of(v),
                )
        assert not coll.batch.failed_objects

    hnsw = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
    flat = Configure.VectorIndex.flat(distance_metric=VectorDistances.COSINE)
    fill(
        client.collections.create(
            names["Hnsw"],
            properties=props,
            vector_config=Configure.Vectors.self_provided(vector_index_config=hnsw),
        ),
        lambda v: v.tolist(),
    )
    fill(
        client.collections.create(
            names["Flat"],
            properties=props,
            vector_config=Configure.Vectors.self_provided(vector_index_config=flat),
        ),
        lambda v: v.tolist(),
    )
    fill(
        client.collections.create(
            names["Named"],
            properties=props,
            vector_config=[
                Configure.Vectors.self_provided(name="body", vector_index_config=hnsw),
                Configure.Vectors.self_provided(name="title", vector_index_config=hnsw),
            ],
        ),
        lambda v: {"body": v.tolist(), "title": v[::-1].tolist()},
    )
    tenants = client.collections.create(
        names["Tenants"],
        properties=props,
        vector_config=Configure.Vectors.self_provided(vector_index_config=hnsw),
        multi_tenancy_config=Configure.multi_tenancy(enabled=True),
    )
    tenants.tenants.create([Tenant(name="acme"), Tenant(name="globex")])
    fill(tenants.with_tenant("acme"), lambda v: v.tolist())
    yield names, vectors
    for name in names.values():
        client.collections.delete(name)
    client.close()


def connect(collection: str, **kwargs) -> WeaviateConnector:
    return WeaviateConnector(url=URL, grpc_url=GRPC, collection=collection, **kwargs)


def _exact(vectors, query, k):
    import faiss

    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    return [str(i) for i in flat.search(query.reshape(1, -1), k)[1][0]]


def test_describe_and_search(server) -> None:
    names, vectors = server
    query = (vectors[3] * 0.8 + vectors[4] * 0.2).astype(np.float32)
    query /= np.linalg.norm(query)
    with connect(names["Hnsw"], id_property="doc_id") as c:
        info = c.describe()
        assert info.backend == "weaviate"
        assert info.distance is Distance.COSINE
        assert info.dim == 32
        assert info.count == 300
        assert info.config["index_type"] == "hnsw"
        assert info.backend_version.startswith("Weaviate 1.")
        assert not c.capabilities().exact_reference
        found = c.search(SearchRequest(query, 10))
        assert len(set(found) & set(_exact(vectors, query, 10))) >= 9
        en = c.search(SearchRequest(query, 20, filter={"lang": "en"}))
        both = c.search(SearchRequest(query, 20, filter={"lang": "en", "public": True}))
        with pytest.raises(ConnectorError) as e:
            c.search(SearchRequest(query, 5, exact=True))
        assert e.value.code is ErrorCode.UNSUPPORTED
    assert all(int(i) % 2 == 0 for i in en)
    assert all(int(i) % 2 == 0 and int(i) < 100 for i in both)
    with connect(names["Hnsw"]) as c:
        assert all(len(i) == 36 for i in c.search(SearchRequest(query, 3)))  # object UUIDs


def test_flat_index_is_an_exact_reference(server) -> None:
    names, vectors = server
    query = vectors[11]
    with connect(names["Flat"], id_property="doc_id") as c:
        assert c.capabilities().exact_reference
        assert c.describe().config["index_type"] == "flat"
        assert c.search(SearchRequest(query, 10, exact=True)) == _exact(vectors, query, 10)


def test_named_vectors_and_tenants(server) -> None:
    names, vectors = server
    with pytest.raises(ConnectorError, match="several named vectors") as e:
        connect(names["Named"])
    assert "body, title" in (e.value.hint or "")
    with pytest.raises(ConnectorError, match="no vector 'nope'"):
        connect(names["Named"], vector_name="nope")
    with connect(names["Named"], vector_name="title", id_property="doc_id") as c:
        assert c.describe().target["vector"] == "title"
        assert c.search(SearchRequest(vectors[5][::-1].copy(), 1)) == ["5"]
    with connect(names["Tenants"], tenant="acme", id_property="doc_id") as c:
        assert c.describe().count == 300
        assert len(c.search(SearchRequest(vectors[0], 5))) == 5
    with connect(names["Tenants"], tenant="globex", id_property="doc_id") as c:
        assert c.search(SearchRequest(vectors[0], 5)) == []
    with connect(names["Tenants"], tenant="initech") as c, pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(vectors[0], 5))
    assert e.value.code in (ErrorCode.CONFIG, ErrorCode.QUERY)


def test_config_errors_on_server(server) -> None:
    names, _ = server
    with pytest.raises(ConnectorError, match="not found") as e:
        connect("NoSuchCollection")
    assert e.value.code is ErrorCode.CONFIG
    with pytest.raises(ConnectorError, match="no property 'docid'"):
        connect(names["Hnsw"], id_property="docid")
    with connect(names["Hnsw"]) as c, pytest.raises(ConnectorError) as e:
        c.search(SearchRequest(np.zeros(3, dtype=np.float32), 3))
    assert e.value.code is ErrorCode.QUERY


def test_run_on_weaviate(server, synthetic, monkeypatch) -> None:
    names, _ = server
    cfg = parse_connection(
        {
            "backend": "weaviate",
            "url": URL,
            "grpc_url": GRPC,
            "collection": names["Flat"],
            "id_property": "doc_id",
        }
    )
    ds = make_dataset(synthetic, n=10)
    with open_connector(cfg) as c:
        run = execute_run(ds, c, RunOptions(k=5))
    assert run["status"] == "complete"
    assert run["search"]["exact_reference"] is True
    assert run["summary"]["ann_recall"] == 1.0
