"""Elasticsearch/OpenSearch connector: request shapes, errors and a full run on a fake
cluster that really searches; then real clusters when CI provides them.

Integration tests need ``VECTRAUDIT_TEST_ELASTICSEARCH_URL`` and/or
``VECTRAUDIT_TEST_OPENSEARCH_URL`` (e.g. http://localhost:9200) and are skipped without them.
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
from vectraudit.connectors.elasticsearch import ElasticsearchConnector
from vectraudit.evaluation.runner import RunOptions, execute_run

VECTORS = {
    "a": ([1.0, 0.0, 0.0], {"tenant": "acme", "doc_id": "doc-a"}),
    "b": ([0.9, 0.1, 0.0], {"tenant": "beta", "doc_id": "doc-b"}),
    "c": ([0.0, 1.0, 0.0], {"tenant": "acme", "doc_id": "doc-c"}),
}


def _score(metric: str, q: np.ndarray, v: np.ndarray) -> float:
    if metric in ("cosine", "cosinesimil"):
        return float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)))
    if metric in ("l2_norm", "l2"):
        return -float(np.linalg.norm(q - v))
    return float(q @ v)


class FakeCluster:
    """Answers like Elasticsearch 8 or OpenSearch 2, scoring documents by brute force."""

    def __init__(self, flavor="elasticsearch", metric="cosine", docs=None, mapping=None):
        self.flavor, self.metric = flavor, metric
        self.docs = docs if docs is not None else VECTORS
        self.calls: list = []
        dim = len(next(iter(self.docs.values()))[0])
        field = (
            {
                "type": "knn_vector",
                "dimension": dim,
                "method": {"name": "hnsw", "space_type": metric},
            }
            if flavor == "opensearch"
            else {"type": "dense_vector", "dims": dim, "similarity": metric, "index": True}
        )
        self.mapping = mapping or {"embedding": field, "tenant": {"type": "keyword"}}
        self.fail: tuple[int, dict] | None = None

    def __call__(self, method, url, headers, body, timeout):
        payload = json.loads(body) if body else None
        self.calls.append((method, url, dict(headers), payload))
        path = url.split("://", 1)[1].split("/", 1)[1] if url.count("/") > 2 else ""
        if self.fail and path.endswith("_search"):
            status, error = self.fail
            return status, json.dumps(error).encode()
        if path == "":
            version = {"number": "2.17.0", "distribution": "opensearch"}
            return 200, json.dumps(
                {"version": version if self.flavor == "opensearch" else {"number": "8.15.0"}}
            ).encode()
        if path == "docs/_mapping":
            return 200, json.dumps({"docs-v2": {"mappings": {"properties": self.mapping}}}).encode()
        if path == "docs/_count":
            return 200, json.dumps({"count": len(self.docs)}).encode()
        if path == "docs/_search":
            return 200, json.dumps(self._search(payload)).encode()
        return 404, json.dumps({"error": {"reason": "no such index"}}).encode()

    def _search(self, body):
        if "knn" in body:  # Elasticsearch approximate
            query, k = np.array(body["knn"]["query_vector"]), body["knn"]["k"]
            filters = body["knn"].get("filter", {}).get("bool", {}).get("filter", [])
        elif "knn" in body.get("query", {}):  # OpenSearch approximate
            clause = body["query"]["knn"]["embedding"]
            query, k = np.array(clause["vector"]), clause["k"]
            filters = clause.get("filter", {}).get("bool", {}).get("filter", [])
        else:  # exact script_score
            script = body["query"]["script_score"]
            params = script["script"]["params"]
            query = np.array(params.get("query_vector") or params.get("query_value"))
            k = body["size"]
            filters = script["query"].get("bool", {}).get("filter", [])
        wanted = {key: value for f in filters for key, value in f["term"].items()}
        hits = [
            (
                _score(self.metric, query, np.array(vec)),
                doc_id,
                meta,
            )
            for doc_id, (vec, meta) in self.docs.items()
            if all(meta.get(key) == value for key, value in wanted.items())
        ]
        hits.sort(key=lambda h: -h[0])
        source = body.get("_source")
        return {
            "hits": {
                "hits": [
                    {"_id": d, "_score": s, **({"_source": m} if source else {})}
                    for s, d, m in hits[: min(k, body["size"])]
                ]
            }
        }


def connect(cluster=None, **kwargs):
    cluster = cluster or FakeCluster()
    options = {"url": "https://es.example.com:9200", "index": "docs", "vector_field": "embedding"}
    return ElasticsearchConnector(**{**options, **kwargs}, send=cluster), cluster


def query(*values, k=2, **kwargs):
    return SearchRequest(np.array(values, dtype=np.float32), k, **kwargs)


def test_elasticsearch_knn_request_and_results():
    c, cluster = connect(api_key="es-secret")
    assert c.search(query(1, 0, 0, params={"num_candidates": 1})) == ["a", "b"]
    method, url, headers, body = cluster.calls[-1]
    assert (method, url) == ("POST", "https://es.example.com:9200/docs/_search")
    assert headers["Authorization"] == "ApiKey es-secret"
    assert body["knn"]["num_candidates"] == 2  # never below k
    assert body["_source"] is False
    assert c.search(query(1, 0, 0, filter={"tenant": "acme"})) == ["a", "c"]
    assert cluster.calls[-1][3]["knn"]["filter"] == {
        "bool": {"filter": [{"term": {"tenant": "acme"}}]}
    }


def test_exact_reference_scores_every_document_with_the_index_metric():
    c, cluster = connect(FakeCluster(metric="dot_product"))
    assert c.search(query(1, 0, 0, exact=True)) == ["a", "b"]
    script = cluster.calls[-1][3]["query"]["script_score"]["script"]
    assert "dotProduct(params.query_vector, 'embedding')" in script["source"]
    assert c.describe().distance is Distance.IP


def test_opensearch_is_detected_and_uses_the_knn_query():
    c, cluster = connect(FakeCluster("opensearch", "l2"), basic_auth="admin:pw")
    info = c.describe()
    assert (info.backend, info.distance, info.dim, info.count) == ("opensearch", Distance.L2, 3, 3)
    assert c.capabilities().tuning_params == ("ef_search",)
    assert c.search(query(0.9, 0.1, 0, params={"ef_search": 64}, filter={"tenant": "acme"})) == [
        "a",
        "c",
    ]
    clause = cluster.calls[-1][3]["query"]["knn"]["embedding"]
    assert clause["method_parameters"] == {"ef_search": 64}
    assert cluster.calls[-1][2]["Authorization"].startswith("Basic ")
    c.search(query(1, 0, 0, exact=True))
    params = cluster.calls[-1][3]["query"]["script_score"]["script"]["params"]
    assert (params["space_type"], params["field"]) == ("l2", "embedding")


def test_ids_can_come_from_a_source_field():
    c, cluster = connect(id_field="doc_id")
    assert c.search(query(1, 0, 0)) == ["doc-a", "doc-b"]
    assert cluster.calls[-1][3]["_source"] == ["doc_id"]


def test_a_run_measures_recall_against_the_exact_reference(synthetic):
    vectors = np.load(synthetic["vectors"])
    docs = {str(i): (v.tolist(), {}) for i, v in enumerate(vectors)}
    c, _ = connect(FakeCluster(metric="l2_norm", docs=docs))
    run = execute_run(make_dataset(synthetic, n=10), c, RunOptions(k=5))
    assert run["status"] == "complete"
    assert run["search"]["exact_reference"] is True
    assert run["summary"]["ann_recall"] == 1.0  # the fake's ANN is exact
    assert run["summary"]["metrics"]["recall"] == 1.0


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"embedding": {"type": "keyword"}}, "not a dense_vector"),
        ({"other": {"type": "dense_vector", "dims": 3}}, "not a dense_vector"),
    ],
)
def test_a_field_that_is_not_a_vector_is_refused(mapping, message):
    with pytest.raises(ConnectorError, match=message) as e:
        connect(FakeCluster(mapping=mapping))
    assert e.value.code is ErrorCode.CONFIG


def test_an_unindexed_field_allows_only_the_exact_reference():
    field = {"type": "dense_vector", "dims": 3, "similarity": "cosine", "index": False}
    c, _ = connect(FakeCluster(mapping={"embedding": field}))
    assert c.search(query(1, 0, 0, exact=True)) == ["a", "b"]
    with pytest.raises(ConnectorError, match="not indexed") as e:
        c.search(query(1, 0, 0))
    assert e.value.code is ErrorCode.UNSUPPORTED


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, ErrorCode.AUTH), (404, ErrorCode.CONFIG), (429, ErrorCode.CONNECTION),
     (503, ErrorCode.CONNECTION), (400, ErrorCode.QUERY)],
)  # fmt: skip
def test_http_errors_map_to_typed_codes(status, code):
    c, cluster = connect()
    cluster.fail = (status, {"error": {"reason": "shard failure"}})
    with pytest.raises(ConnectorError) as e:
        c.search(query(1, 0, 0))
    assert e.value.code is code


def test_transport_errors():
    def down(*args):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    with pytest.raises(ConnectorError) as e:
        ElasticsearchConnector(url="http://localhost:9200", index="d", vector_field="v", send=down)
    assert e.value.code is ErrorCode.CONNECTION

    def slow(*args):
        raise TimeoutError

    with pytest.raises(ConnectorError) as e:
        ElasticsearchConnector(url="http://localhost:9200", index="d", vector_field="v", send=slow)
    assert e.value.code is ErrorCode.TIMEOUT


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"url": "http://es.example.com:9200", "api_key": "k"}, "over http://"),
        ({"url": "https://user:pw@es.example.com"}, "must not contain credentials"),
        ({"api_key": "k", "basic_auth": "u:p"}, "not both"),
        ({"basic_auth": "no-colon"}, "user:password"),
        ({"vector_field": "embedding'); drop"}, "Invalid vector_field"),
    ],
)
def test_unsafe_configuration_is_refused(kwargs, message):
    with pytest.raises(ConnectorError, match=message):
        connect(**kwargs)


def test_connection_files_under_both_names(monkeypatch):
    monkeypatch.setenv("ES_KEY", "secret-from-env")
    cluster = FakeCluster()
    import vectraudit.connectors.elasticsearch as module

    monkeypatch.setattr(module, "_urllib_send", cluster)
    for backend in ("elasticsearch", "opensearch"):
        cfg = parse_connection(
            {
                "backend": backend,
                "url": "https://es.example.com:9200",
                "index": "docs",
                "vector_field": "embedding",
                "api_key_env": "ES_KEY",
            }
        )
        with open_connector(cfg) as c:
            assert c.search(query(1, 0, 0)) == ["a", "b"]
    assert cluster.calls[-1][2]["Authorization"] == "ApiKey secret-from-env"
    with pytest.raises(ConnectorError, match="Unknown field"):
        open_connector(
            parse_connection(
                {
                    "backend": "elasticsearch",
                    "url": "https://x",
                    "index": "d",
                    "vector_field": "v",
                    "collection": "x",
                }
            )
        )


# --- real clusters (CI) ------------------------------------------------------------------


def _request(base, method, path, body=None, content_type="application/json"):
    data = (
        None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    )
    req = urllib.request.Request(
        base + path, data=data, method=method, headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read() or b"{}")


REAL = [
    ("elasticsearch", "l2_norm", Distance.L2),
    ("elasticsearch", "cosine", Distance.COSINE),
    ("elasticsearch", "max_inner_product", Distance.IP),
    ("opensearch", "l2", Distance.L2),
    ("opensearch", "cosinesimil", Distance.COSINE),
    ("opensearch", "innerproduct", Distance.IP),
]


@pytest.mark.parametrize(("flavor", "metric", "distance"), REAL)
def test_run_on_a_real_cluster(flavor, metric, distance, synthetic):
    base = os.environ.get(f"VECTRAUDIT_TEST_{flavor.upper()}_URL")
    if not base:
        pytest.skip(f"Set VECTRAUDIT_TEST_{flavor.upper()}_URL to run {flavor} tests.")
    vectors = np.load(synthetic["vectors"])
    index = f"va-{uuid.uuid4().hex[:8]}"
    if flavor == "opensearch":
        field = {
            "type": "knn_vector",
            "dimension": vectors.shape[1],
            "method": {"name": "hnsw", "space_type": metric, "engine": "lucene"},
        }
        settings = {"settings": {"index": {"knn": True}}}
    else:
        field = {
            "type": "dense_vector",
            "dims": vectors.shape[1],
            "index": True,
            "similarity": metric,
        }
        settings = {}
    mapping = {"mappings": {"properties": {"embedding": field, "tenant": {"type": "keyword"}}}}
    _request(base, "PUT", f"/{index}", {**settings, **mapping})
    lines = []
    for i, v in enumerate(vectors):
        lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
        lines.append(json.dumps({"embedding": v.tolist(), "tenant": "acme" if i % 2 else "beta"}))
    _request(
        base,
        "POST",
        "/_bulk?refresh=true",
        ("\n".join(lines) + "\n").encode(),
        "application/x-ndjson",
    )
    try:
        c = ElasticsearchConnector(url=base, index=index, vector_field="embedding")
        info = c.describe()
        assert (info.backend, info.distance, info.count) == (flavor, distance, len(vectors))
        run = execute_run(make_dataset(synthetic, n=10), c, RunOptions(k=5))
        assert run["status"] == "complete"
        assert run["summary"]["ann_recall"] > 0.8
        filtered = c.search(SearchRequest(vectors[0], 5, filter={"tenant": "beta"}))
        assert filtered
        assert all(int(i) % 2 == 0 for i in filtered)
        # The exact reference agrees with a brute-force answer under the same distance.
        probe = vectors[3]
        scores = [_score(metric, probe, v) for v in vectors]
        expected = [str(i) for i in np.argsort(scores)[::-1][:3]]
        assert c.search(SearchRequest(probe, 3, exact=True)) == expected
    finally:
        _request(base, "DELETE", f"/{index}")
