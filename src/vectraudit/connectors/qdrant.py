"""Qdrant connector (``vectraudit[qdrant]``): dense search on one collection, read-only.

Connection fields::

    {"backend": "qdrant", "url": "https://xyz.cloud.qdrant.io:6333",
     "api_key_env": "QDRANT_API_KEY", "collection": "docs", "vector_name": "dense",
     "timeout_s": 10, "params": {"hnsw_ef": 128}}

``path`` instead of ``url`` opens an embedded local Qdrant (always exact search; for
development and tests). Only read calls are made: collection info and point queries.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit

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

INSTALL_HINT = 'Install the Qdrant client: `pip install "vectraudit[qdrant]"`.'
_DISTANCES = {"Cosine": Distance.COSINE, "Euclid": Distance.L2, "Dot": Distance.IP}


class QdrantConnector(Connector):
    """Search one Qdrant collection by dense vector."""

    name = "Qdrant"

    def __init__(
        self,
        collection: str,
        *,
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
        vector_name: str | None = None,
        timeout_s: int = 10,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ConnectorError(
                ErrorCode.DEPENDENCY, "qdrant-client is not installed.", INSTALL_HINT
            ) from e
        if (url is None) == (path is None):
            raise ConnectorError(
                ErrorCode.CONFIG, 'Give the Qdrant connection either "url" or "path".'
            )
        if url is not None:
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise ConnectorError(ErrorCode.CONFIG, f"Invalid Qdrant url {url!r}.")
            if parts.username or parts.password:
                raise ConnectorError(
                    ErrorCode.CONFIG,
                    "The Qdrant url must not contain credentials.",
                    'Use "api_key_env" instead.',
                )
        self.collection = collection
        self.url = url
        self.local = path is not None
        # describe() reports the server version itself; skip the client's warning-only probe.
        self._client: Any = QdrantClient(
            url=url, path=path, api_key=api_key, timeout=timeout_s, check_compatibility=False
        )
        self._timeout = timeout_s
        try:
            self._info: Any = self._call(lambda: self._client.get_collection(collection))
            self.vector_name, self._params = self._resolve_vector(vector_name)
        except BaseException:
            self.close()  # release connections (and the local-mode storage lock)
            raise

    def _resolve_vector(self, requested: str | None) -> tuple[str | None, Any]:
        vectors = self._info.config.params.vectors
        if not isinstance(vectors, Mapping):
            if requested is not None:
                raise ConnectorError(
                    ErrorCode.CONFIG,
                    f"Collection {self.collection!r} has a single unnamed vector; "
                    f"remove vector_name {requested!r}.",
                )
            return None, vectors
        if requested is None:
            if len(vectors) != 1:
                raise ConnectorError(
                    ErrorCode.CONFIG,
                    f"Collection {self.collection!r} has several named vectors.",
                    f"Set vector_name to one of: {', '.join(sorted(vectors))}.",
                )
            requested = next(iter(vectors))
        if requested not in vectors:
            raise ConnectorError(
                ErrorCode.CONFIG,
                f"Collection {self.collection!r} has no dense vector {requested!r}.",
                f"Available: {', '.join(sorted(vectors)) or 'none'}.",
            )
        return requested, vectors[requested]

    def _call(self, fn: Any) -> Any:
        """Run a client call, translating client exceptions into ConnectorErrors."""
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

        try:
            with warnings.catch_warnings():
                # Local mode warns that search_params are ignored (it is always exact).
                warnings.simplefilter("ignore", UserWarning)
                return fn()
        except UnexpectedResponse as e:
            raise _http_error(e.status_code, e.content, self.collection) from e
        except ResponseHandlingException as e:
            source = getattr(e, "source", e)
            if "Timeout" in type(source).__name__:
                raise ConnectorError(
                    ErrorCode.TIMEOUT, f"Qdrant did not answer within {self._timeout} s."
                ) from e
            raise ConnectorError(
                ErrorCode.CONNECTION,
                f"Cannot reach Qdrant at {self.url}: {source}",
                "Check the url, network access from the runner and that Qdrant is running.",
            ) from e
        except ValueError as e:  # local mode reports problems as ValueError
            message = str(e)
            code = ErrorCode.CONFIG if "not found" in message else ErrorCode.QUERY
            raise ConnectorError(code, f"Qdrant: {message}") from e

    def capabilities(self) -> Capabilities:
        return Capabilities(
            filters=True,
            exact_reference=True,
            config_inspection=True,
            tuning_params=("hnsw_ef",),
        )

    def describe(self) -> BackendInfo:
        params = self._params
        distance = _DISTANCES.get(_enum_name(params.distance))
        if distance is None:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED,
                f"Qdrant distance {_enum_name(params.distance)} is not supported.",
            )
        server_version = None
        # Version is informational; a server that hides it must not fail the run.
        with contextlib.suppress(ConnectorError, AttributeError):
            server_version = self._call(lambda: self._client.info()).version
        info = self._info
        hnsw = info.config.hnsw_config
        vector_hnsw = getattr(params, "hnsw_config", None)
        return BackendInfo(
            backend="qdrant",
            distance=distance,
            dim=int(params.size),
            count=info.points_count,
            target={
                "collection": self.collection,
                "vector": self.vector_name,
                "mode": "local" if self.local else "server",
            },
            backend_version=server_version,
            client_version=_client_version(),
            config={
                "status": _enum_name(info.status),
                "indexed_vectors": info.indexed_vectors_count,
                "segments": info.segments_count,
                "hnsw": _model_dict(vector_hnsw or hnsw),
                "quantization": _model_dict(
                    getattr(params, "quantization_config", None) or info.config.quantization_config
                ),
                "on_disk": getattr(params, "on_disk", None),
                "datatype": _enum_name(getattr(params, "datatype", None)),
            },
            snapshot=None,
        )

    def _search(self, request: SearchRequest) -> list[str]:
        from qdrant_client import models

        hnsw_ef = request.params.get("hnsw_ef")
        if hnsw_ef is not None and (type(hnsw_ef) is not int or hnsw_ef < 1):
            raise ConnectorError(ErrorCode.CONFIG, "hnsw_ef must be a positive integer.")
        response = self._call(
            lambda: self._client.query_points(
                self.collection,
                query=request.vector.tolist(),
                using=self.vector_name,
                query_filter=_filter(request.filter) if request.filter else None,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef, exact=request.exact),
                limit=request.k,
                with_payload=False,
                with_vectors=False,
                timeout=self._timeout,
            )
        )
        return [str(p.id) for p in response.points]

    def close(self) -> None:
        client, self._client = getattr(self, "_client", None), None
        if client is not None:
            client.close()


def _filter(conditions: Mapping[str, FilterValue]) -> Any:
    """Portable equality filter -> Qdrant ``must`` conditions."""
    from qdrant_client import models

    must: list[Any] = []
    for key, value in conditions.items():
        if isinstance(value, float):
            must.append(models.FieldCondition(key=key, range=models.Range(gte=value, lte=value)))
        else:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must)


def _http_error(status: int | None, content: bytes, collection: str) -> ConnectorError:
    detail = content[:300].decode("utf-8", "replace") if content else ""
    if status in (401, 403):
        return ConnectorError(
            ErrorCode.AUTH,
            f"Qdrant rejected the credentials (HTTP {status}).",
            "Check the API key in the variable named by api_key_env, and that it can read "
            f"{collection!r}.",
        )
    if status == 404:
        return ConnectorError(ErrorCode.CONFIG, f"Qdrant collection {collection!r} not found.")
    if status in (408, 504):
        return ConnectorError(ErrorCode.TIMEOUT, f"Qdrant timed out (HTTP {status}).")
    if status == 429 or (status is not None and status >= 500):
        return ConnectorError(ErrorCode.CONNECTION, f"Qdrant is unavailable (HTTP {status}).")
    return ConnectorError(ErrorCode.QUERY, f"Qdrant rejected the query (HTTP {status}): {detail}")


def _enum_name(value: Any) -> Any:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw) if raw is not None else None


def _model_dict(model: Any) -> Any:
    if model is None:
        return None
    dump = getattr(model, "model_dump", None)
    return dump(mode="json", exclude_none=True) if dump is not None else str(model)


def _client_version() -> str | None:
    try:
        return version("qdrant-client")
    except PackageNotFoundError:
        return None
