"""Chroma connector, over Chroma's v2 REST API (no client library).

Connection fields::

    {"backend": "chroma", "url": "http://localhost:8000", "collection": "support-kb",
     "tenant": "default_tenant", "database": "default_database", "token_env": "CHROMA_TOKEN"}

``id_field`` reads document IDs from a metadata field instead of Chroma's record IDs. A token
(``token_env``) is sent as ``Authorization: Bearer`` and ``x-chroma-token``, the two headers
Chroma's token authentication accepts, and only over HTTPS or to a local server.

- **Search:** ``POST .../collections/{id}/query`` with the query embedding and ``n_results``.
  Filters are portable equality conditions, sent as a Chroma ``where`` clause. Chroma sets
  ``ef_search`` per collection, not per query, so there are no tuning parameters; the value
  in use is recorded with the configuration.
- **Distance** comes from the collection's ``hnsw`` (or ``spann``) ``space``: ``l2`` (squared
  Euclidean), ``cosine`` or ``ip``.
- **Exact reference:** Chroma has no brute-force query. The connector fetches the
  collection's own vectors through ``.../get`` with the same filter (cached per filter, at
  most ``exact_limit`` records) and ranks them with Chroma's own distance definitions, so ANN
  recall is measured against the backend's stored vectors.
- There is no immutable snapshot ID (a local server's ``log_position`` stays at 0 across
  writes, so it cannot mark a version): the collection can change between runs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, urlsplit

import numpy as np

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
DISTANCES = {"l2": Distance.L2, "cosine": Distance.COSINE, "ip": Distance.IP}
PAGE = 1000
EXACT_LIMIT = 200_000
Send = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


def _urllib_send(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


def where(conditions: Mapping[str, FilterValue]) -> dict[str, Any] | None:
    """Equality conditions as a Chroma ``where`` clause (``$and`` for several)."""
    clauses = [{key: value} for key, value in sorted(conditions.items())]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def exact_ranking(
    ids: list[str], vectors: np.ndarray[Any, Any], query: np.ndarray[Any, Any], space: str, k: int
) -> list[str]:
    """The ``k`` nearest records by Chroma's distance for ``space`` (smaller is nearer).

    ``l2`` is the squared Euclidean distance, ``ip`` is ``1 - dot`` and ``cosine`` is
    ``1 - cosine similarity``, as Chroma computes them. Ties keep the fetched order.
    """
    if not ids:
        return []
    q = query.astype(np.float64)
    m = vectors.astype(np.float64)
    if space == "l2":
        distances = np.sum((m - q) ** 2, axis=1)
    elif space == "ip":
        distances = 1.0 - m @ q
    else:
        norms = np.linalg.norm(m, axis=1) * max(float(np.linalg.norm(q)), 1e-30)
        distances = 1.0 - (m @ q) / np.maximum(norms, 1e-30)
    order = np.argsort(distances, kind="stable")[:k]
    return [ids[i] for i in order]


class ChromaConnector(Connector):
    """Query one Chroma collection by vector."""

    name = "Chroma"

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        tenant: str = "default_tenant",
        database: str = "default_database",
        id_field: str | None = None,
        token: str | None = None,
        timeout_s: int = 10,
        exact_limit: int = EXACT_LIMIT,
        send: Send | None = None,
    ) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConnectorError(ErrorCode.CONFIG, f"Invalid Chroma url {url!r}.")
        if parts.username or parts.password:
            raise ConnectorError(
                ErrorCode.CONFIG,
                "The Chroma url must not contain credentials.",
                "Put the token in the variable named by token_env.",
            )
        if token and parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
            raise ConnectorError(
                ErrorCode.CONFIG,
                f"Refusing to send the Chroma token over {parts.scheme}:// to {parts.hostname}.",
                "Use the https:// address (plain http is allowed only for a local server).",
            )
        self.url = url.rstrip("/")
        self.tenant, self.database = tenant, database
        self.collection_name = collection
        self.id_field = id_field
        self.timeout_s = timeout_s
        self.exact_limit = exact_limit
        self._send = send or _urllib_send
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"vectraudit/{__version__}",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
            self._headers["x-chroma-token"] = token
        self.version = str(self._call("GET", "/api/v2/version", None, what="server"))
        info = self._call("GET", self._base(), None, what=f"collection {collection!r}")
        if not isinstance(info, dict) or not info.get("id"):
            raise ConnectorError(ErrorCode.QUERY, "Chroma returned an unexpected collection.")
        self.collection_id = str(info["id"])
        config = info.get("configuration_json") or {}
        self._index = config.get("hnsw") or config.get("spann") or {}
        self._index_kind = (
            "hnsw" if config.get("hnsw") else ("spann" if config.get("spann") else None)
        )
        self._dim = info.get("dimension")
        self._exact: dict[tuple[tuple[str, FilterValue], ...], tuple[list[str], Any]] = {}

    # --- HTTP ------------------------------------------------------------------------------

    def _base(self) -> str:
        return (
            f"/api/v2/tenants/{quote(self.tenant, safe='')}/databases/"
            f"{quote(self.database, safe='')}/collections/{quote(self.collection_name, safe='')}"
        )

    def _records(self) -> str:
        return (
            f"/api/v2/tenants/{quote(self.tenant, safe='')}/databases/"
            f"{quote(self.database, safe='')}/collections/{quote(self.collection_id, safe='')}"
        )

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
                ErrorCode.TIMEOUT, f"Chroma did not answer within {self.timeout_s} s."
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ConnectorError(
                    ErrorCode.TIMEOUT, f"Chroma did not answer within {self.timeout_s} s."
                ) from e
            raise ConnectorError(
                ErrorCode.CONNECTION,
                f"Cannot reach Chroma: {e.reason}",
                "Check network access from the runner and the url.",
            ) from e
        if status >= 400:
            raise _http_error(status, raw, what)
        try:
            return json.loads(raw) if raw else {}
        except ValueError as e:
            raise ConnectorError(
                ErrorCode.QUERY, "Chroma returned a response that isn't JSON."
            ) from e

    # --- contract --------------------------------------------------------------------------

    def _space(self) -> str:
        return str(self._index.get("space") or "l2")

    def capabilities(self) -> Capabilities:
        return Capabilities(
            search=True,
            filters=True,
            exact_reference=True,
            exact_reference_note=(
                "Computed by Vectraudit from the collection's stored vectors (fetched with the "
                f"same filter, at most {self.exact_limit} records) with Chroma's distance."
            ),
            config_inspection=True,
            tuning_params=(),
        )

    def describe(self) -> BackendInfo:
        distance = DISTANCES.get(self._space())
        if distance is None:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED, f"Chroma distance space {self._space()!r} is not supported."
            )
        if not isinstance(self._dim, int):
            raise ConnectorError(
                ErrorCode.UNSUPPORTED,
                f"Collection {self.collection_name!r} reports no vector dimension (is it empty?).",
            )
        count = self._call("GET", self._records() + "/count", None, what="record count")
        return BackendInfo(
            backend="chroma",
            distance=distance,
            dim=self._dim,
            count=count if isinstance(count, int) and not isinstance(count, bool) else None,
            target={
                "tenant": self.tenant,
                "database": self.database,
                "collection": self.collection_name,
            },
            backend_version=f"chroma api {self.version}",
            client_version="REST",
            config={
                "index": self._index_kind,
                "space": self._space(),
                **{
                    key: self._index.get(key)
                    for key in ("ef_search", "ef_construction", "max_neighbors", "search_nprobe")
                    if key in self._index
                },
            },
            snapshot=None,
        )

    def _ids(self, ids: list[str], metadatas: Any) -> list[str]:
        if not self.id_field:
            return [str(i) for i in ids]
        out = []
        for meta in metadatas or []:
            value = meta.get(self.id_field) if isinstance(meta, dict) else None
            if value is not None:
                out.append(str(value))
        return out

    def _search(self, request: SearchRequest) -> list[str]:
        if request.exact:
            return self._exact_search(request)
        body: dict[str, Any] = {
            "query_embeddings": [[float(x) for x in request.vector]],
            "n_results": request.k,
            "include": ["metadatas"] if self.id_field else [],
        }
        clause = where(request.filter)
        if clause is not None:
            body["where"] = clause
        response = self._call("POST", self._records() + "/query", body, what="query")
        ids = (response.get("ids") or [[]])[0] if isinstance(response, dict) else []
        metadatas = (response.get("metadatas") or [None])[0] if self.id_field else None
        return self._ids(list(ids or []), metadatas)

    def _exact_search(self, request: SearchRequest) -> list[str]:
        key = tuple(sorted(request.filter.items()))
        if key not in self._exact:
            self._exact[key] = self._fetch_vectors(request.filter)
        ids, vectors = self._exact[key]
        query = np.asarray(request.vector, dtype=np.float64)
        return exact_ranking(ids, vectors, query, self._space(), request.k)

    def _fetch_vectors(self, conditions: Mapping[str, FilterValue]) -> tuple[list[str], Any]:
        clause = where(conditions)
        ids: list[str] = []
        rows: list[list[float]] = []
        offset = 0
        include = ["embeddings", "metadatas"] if self.id_field else ["embeddings"]
        while True:
            body: dict[str, Any] = {"limit": PAGE, "offset": offset, "include": include}
            if clause is not None:
                body["where"] = clause
            page = self._call("POST", self._records() + "/get", body, what="records")
            page_ids = list(page.get("ids") or [])
            embeddings = page.get("embeddings") or []
            if len(embeddings) != len(page_ids):
                raise ConnectorError(ErrorCode.QUERY, "Chroma returned records without vectors.")
            ids += self._ids(page_ids, page.get("metadatas")) if self.id_field else page_ids
            rows += embeddings
            if len(rows) > self.exact_limit:
                raise ConnectorError(
                    ErrorCode.UNSUPPORTED,
                    f"The exact reference is limited to {self.exact_limit} records.",
                    "Measure ANN recall on a filtered subset, or raise exact_limit.",
                )
            if len(page_ids) < PAGE:
                break
            offset += len(page_ids)
        if self.id_field and len(ids) != len(rows):
            raise ConnectorError(
                ErrorCode.QUERY, f"Some records have no {self.id_field!r} metadata field."
            )
        return ids, np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)


def _http_error(status: int, raw: bytes, what: str) -> ConnectorError:
    try:
        payload = json.loads(raw)
        detail = str(payload.get("message") or payload.get("error") or "")[:200]
    except (ValueError, AttributeError):
        detail = raw[:200].decode("utf-8", "replace")
    if status in (401, 403):
        return ConnectorError(
            ErrorCode.AUTH,
            f"Chroma rejected the credentials (HTTP {status}).",
            "Check the variable named by token_env and the tenant and database.",
        )
    if status == 404:
        return ConnectorError(ErrorCode.CONFIG, f"The {what} was not found.")
    if status in (408, 504):
        return ConnectorError(ErrorCode.TIMEOUT, f"Chroma timed out (HTTP {status}).")
    if status == 429 or status >= 500:
        return ConnectorError(
            ErrorCode.CONNECTION,
            f"Chroma is rate limiting or unavailable (HTTP {status}).",
            "Retry later, or lower concurrency; the run is incomplete, not regressed.",
        )
    return ConnectorError(ErrorCode.QUERY, f"Chroma rejected the {what} (HTTP {status}): {detail}")
