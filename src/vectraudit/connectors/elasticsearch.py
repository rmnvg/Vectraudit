"""Elasticsearch and OpenSearch connector, over their REST APIs (no client library).

Connection fields::

    {"backend": "elasticsearch", "url": "https://search.example.com:9200", "index": "docs",
     "vector_field": "embedding", "api_key_env": "ES_API_KEY", "timeout_s": 10}

``id_field`` reads document IDs from a ``_source`` field instead of ``_id``. Authentication
is ``api_key_env`` (an Elasticsearch API key, sent as ``ApiKey``) or ``basic_auth_env`` (a
variable holding ``user:password``, common for OpenSearch), or neither for an open cluster.
Credentials are sent only over HTTPS, or to a local server.

Which engine it is comes from ``GET /``: OpenSearch reports ``distribution: opensearch``.

- **Elasticsearch 8** (``dense_vector``): approximate search with the ``knn`` option;
  tuning parameter ``num_candidates``.
- **OpenSearch 2** (``knn_vector``): the ``knn`` query; tuning parameter ``ef_search``
  (``method_parameters``, OpenSearch 2.16 and later).

Both offer an **exact reference**: a brute-force ``script_score`` over the same documents and
filters, so ANN recall is measured against the backend's own vectors and distance. Filters
are portable equality conditions, sent as ``term`` clauses. There is no immutable snapshot
ID: the index can change between runs, which the run records say.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, urlsplit

from vectraudit import __version__
from vectraudit.connectors.base import (
    BackendInfo,
    Capabilities,
    Connector,
    ConnectorError,
    Distance,
    ErrorCode,
    FilterValue,
    SearchRequest,
)

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
FIELD = re.compile(r"^[A-Za-z0-9_@][A-Za-z0-9_.@-]*$")
ES_DISTANCES = {
    "cosine": Distance.COSINE,
    "dot_product": Distance.IP,
    "max_inner_product": Distance.IP,
    "l2_norm": Distance.L2,
}
OS_DISTANCES = {"cosinesimil": Distance.COSINE, "innerproduct": Distance.IP, "l2": Distance.L2}
# Exact scoring scripts. script_score needs non-negative scores, so inner products are
# mapped through an order-preserving transform; rankings, not scores, are compared.
ES_SCRIPTS = {
    "cosine": "cosineSimilarity(params.query_vector, '{f}') + 1.0",
    "l2_norm": "1 / (1 + l2norm(params.query_vector, '{f}'))",
    "dot_product": (
        "double s = dotProduct(params.query_vector, '{f}'); return s < 0 ? 1 / (1 - s) : s + 1;"
    ),
}
ES_SCRIPTS["max_inner_product"] = ES_SCRIPTS["dot_product"]
Send = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


def _urllib_send(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


def _base_url(url: str, *, credentials: bool) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConnectorError(ErrorCode.CONFIG, f"Invalid search cluster url {url!r}.")
    if parts.username or parts.password:
        raise ConnectorError(
            ErrorCode.CONFIG,
            "The cluster url must not contain credentials.",
            "Put them in the variable named by api_key_env or basic_auth_env.",
        )
    if credentials and parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
        raise ConnectorError(
            ErrorCode.CONFIG,
            f"Refusing to send credentials over {parts.scheme}:// to {parts.hostname}.",
            "Use the https:// address (plain http is allowed only for a local server).",
        )
    return url.rstrip("/")


def _field(properties: Mapping[str, Any], dotted: str) -> Mapping[str, Any] | None:
    node: Any = {"properties": properties}
    for part in dotted.split("."):
        node = (node.get("properties") or {}).get(part) if isinstance(node, Mapping) else None
        if node is None:
            return None
    return node if isinstance(node, Mapping) else None


def _source_value(source: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(source, Mapping):
            return None
        source = source.get(part)
    return source


def _filter(conditions: Mapping[str, FilterValue]) -> list[dict[str, Any]]:
    return [{"term": {key: value}} for key, value in conditions.items()]


class ElasticsearchConnector(Connector):
    """Query one Elasticsearch or OpenSearch index by dense vector."""

    name = "Elasticsearch/OpenSearch"

    def __init__(
        self,
        *,
        url: str,
        index: str,
        vector_field: str,
        id_field: str | None = None,
        api_key: str | None = None,
        basic_auth: str | None = None,
        timeout_s: int = 10,
        send: Send | None = None,
    ) -> None:
        for what, value in (("vector_field", vector_field), ("id_field", id_field)):
            if value is not None and not FIELD.match(value):
                raise ConnectorError(ErrorCode.CONFIG, f"Invalid {what} {value!r}.")
        if api_key and basic_auth:
            raise ConnectorError(ErrorCode.CONFIG, "Give api_key_env or basic_auth_env, not both.")
        if basic_auth is not None and ":" not in basic_auth:
            raise ConnectorError(
                ErrorCode.CONFIG, "The basic_auth_env variable must hold user:password."
            )
        self.url = _base_url(url, credentials=bool(api_key or basic_auth))
        self.index = index
        self.vector_field = vector_field
        self.id_field = id_field
        self.timeout_s = timeout_s
        self._send = send or _urllib_send
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"vectraudit/{__version__}",
        }
        if api_key:
            self._headers["Authorization"] = f"ApiKey {api_key}"
        elif basic_auth:
            token = base64.b64encode(basic_auth.encode()).decode()
            self._headers["Authorization"] = f"Basic {token}"
        root = self._call("GET", "/", None, what="cluster")
        version = root.get("version") or {}
        self.flavor = (
            "opensearch" if version.get("distribution") == "opensearch" else "elasticsearch"
        )
        self.version = str(version.get("number") or "")
        mappings = self._call("GET", f"/{quote(index)}/_mapping", None, what=f"index {index!r}")
        properties: dict[str, Any] = {}
        for value in mappings.values():  # an alias resolves to one or more concrete indices
            properties.update(((value or {}).get("mappings") or {}).get("properties") or {})
        spec = _field(properties, vector_field)
        expected = "knn_vector" if self.flavor == "opensearch" else "dense_vector"
        if spec is None or spec.get("type") != expected:
            raise ConnectorError(
                ErrorCode.CONFIG,
                f"{vector_field!r} in {index!r} is not a {expected} field.",
                "Set vector_field to the field holding the document embeddings.",
            )
        self._spec = dict(spec)

    # --- HTTP ------------------------------------------------------------------------------

    def _call(self, method: str, path: str, body: Any, *, what: str = "request") -> Any:
        data = None if body is None else json.dumps(body).encode()
        try:
            status, raw = self._send(
                method, self.url + path, self._headers, data, float(self.timeout_s)
            )
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        except TimeoutError as e:
            raise ConnectorError(
                ErrorCode.TIMEOUT, f"The cluster did not answer within {self.timeout_s} s."
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ConnectorError(
                    ErrorCode.TIMEOUT, f"The cluster did not answer within {self.timeout_s} s."
                ) from e
            raise ConnectorError(
                ErrorCode.CONNECTION,
                f"Cannot reach the search cluster: {e.reason}",
                "Check network access from the runner and the url.",
            ) from e
        if status >= 400:
            raise _http_error(status, raw, what)
        try:
            return json.loads(raw) if raw else {}
        except ValueError as e:
            raise ConnectorError(
                ErrorCode.QUERY, "The cluster returned a response that isn't JSON."
            ) from e

    # --- contract --------------------------------------------------------------------------

    def _metric(self) -> str:
        if self.flavor == "opensearch":
            method = self._spec.get("method") or {}
            return str(self._spec.get("space_type") or method.get("space_type") or "l2")
        return str(self._spec.get("similarity") or "cosine")

    def capabilities(self) -> Capabilities:
        indexed = self.flavor == "opensearch" or self._spec.get("index", True) is not False
        return Capabilities(
            search=indexed,
            filters=True,
            exact_reference=True,
            config_inspection=True,
            tuning_params=("ef_search",) if self.flavor == "opensearch" else ("num_candidates",),
        )

    def describe(self) -> BackendInfo:
        metric = self._metric()
        distance = (OS_DISTANCES if self.flavor == "opensearch" else ES_DISTANCES).get(metric)
        if distance is None:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED, f"Vector similarity {metric!r} is not supported."
            )
        dim = self._spec.get("dimension") if self.flavor == "opensearch" else self._spec.get("dims")
        if not isinstance(dim, int):
            raise ConnectorError(
                ErrorCode.UNSUPPORTED, f"{self.vector_field!r} declares no vector dimension."
            )
        count = self._call("GET", f"/{quote(self.index)}/_count", None, what="document count")
        config = {
            "engine": self.flavor,
            "vector_field": self.vector_field,
            "metric": metric,
            "index_options": self._spec.get("index_options") or self._spec.get("method"),
        }
        return BackendInfo(
            backend=self.flavor,
            distance=distance,
            dim=dim,
            count=count.get("count") if isinstance(count.get("count"), int) else None,
            target={"index": self.index, "vector_field": self.vector_field},
            backend_version=f"{self.flavor} {self.version}".strip(),
            client_version="REST",
            config=config,
            snapshot=None,
        )

    def _body(self, request: SearchRequest) -> dict[str, Any]:
        vector = [float(x) for x in request.vector]
        filters = _filter(request.filter)
        body: dict[str, Any] = {
            "size": request.k,
            "_source": [self.id_field] if self.id_field else False,
        }
        if request.exact:
            base: dict[str, Any] = {"bool": {"filter": filters}} if filters else {"match_all": {}}
            if self.flavor == "opensearch":
                script = {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": self.vector_field,
                        "query_value": vector,
                        "space_type": self._metric(),
                    },
                }
            else:
                source = ES_SCRIPTS[self._metric()].format(f=self.vector_field)
                script = {"source": source, "params": {"query_vector": vector}}
            body["query"] = {"script_score": {"query": base, "script": script}}
            return body
        if self.flavor == "opensearch":
            clause: dict[str, Any] = {"vector": vector, "k": request.k}
            if filters:
                clause["filter"] = {"bool": {"filter": filters}}
            if "ef_search" in request.params:
                clause["method_parameters"] = {"ef_search": int(request.params["ef_search"])}
            body["query"] = {"knn": {self.vector_field: clause}}
            return body
        knn: dict[str, Any] = {"field": self.vector_field, "query_vector": vector, "k": request.k}
        if "num_candidates" in request.params:
            knn["num_candidates"] = max(int(request.params["num_candidates"]), request.k)
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        body["knn"] = knn
        return body

    def _search(self, request: SearchRequest) -> list[str]:
        if not request.exact and not self.capabilities().search:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED,
                f"{self.vector_field!r} is not indexed for approximate search (index: false).",
                "Only the exact reference can run on this field.",
            )
        response = self._call(
            "POST", f"/{quote(self.index)}/_search", self._body(request), what="search"
        )
        ids = []
        for hit in (response.get("hits") or {}).get("hits") or []:
            value = (
                _source_value(hit.get("_source"), self.id_field)
                if self.id_field
                else hit.get("_id")
            )
            if value is not None:
                ids.append(str(value))
        return ids


def _http_error(status: int, raw: bytes, what: str) -> ConnectorError:
    try:
        payload = json.loads(raw)
        error = payload.get("error")
        detail = str((error or {}).get("reason") if isinstance(error, dict) else error or "")
    except (ValueError, AttributeError):
        detail = raw[:200].decode("utf-8", "replace")
    if status in (401, 403):
        return ConnectorError(
            ErrorCode.AUTH,
            f"The cluster rejected the credentials (HTTP {status}).",
            "Check the variable named by api_key_env or basic_auth_env and its index privileges.",
        )
    if status == 404:
        return ConnectorError(ErrorCode.CONFIG, f"The {what} was not found.")
    if status in (408, 504):
        return ConnectorError(ErrorCode.TIMEOUT, f"The cluster timed out (HTTP {status}).")
    if status == 429 or status >= 500:
        return ConnectorError(
            ErrorCode.CONNECTION,
            f"The cluster is rate limiting or unavailable (HTTP {status}).",
            "Retry later, or lower concurrency; the run is incomplete, not regressed.",
        )
    return ConnectorError(
        ErrorCode.QUERY, f"The cluster rejected the {what} (HTTP {status}): {detail}"
    )
