"""Weaviate connector (``vectraudit[weaviate]``): near-vector search on one collection.

Connection fields::

    {"backend": "weaviate", "url": "https://abc.c0.europe-west3.gcp.weaviate.cloud",
     "api_key_env": "WEAVIATE_API_KEY", "collection": "Docs", "id_property": "doc_id",
     "vector_name": "body", "tenant": "acme", "timeout_s": 10}

``id_property`` names the property holding your document id (default: the object UUID).
``vector_name`` is required when the collection has several named vectors. Self-hosted
Weaviate also takes ``grpc_url`` (default: the same host on port 50051). Exact reference
search is only available for collections with a flat index: HNSW has no exact mode.
Only reads are made: collection configuration, counts and near-vector queries.
"""

from __future__ import annotations

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
    SearchRequest,
)

INSTALL_HINT = 'Install the Weaviate client: `pip install "vectraudit[weaviate]"`.'
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
CLOUD_SUFFIXES = (".weaviate.cloud", ".weaviate.network")
_DISTANCES = {"cosine": Distance.COSINE, "dot": Distance.IP, "l2-squared": Distance.L2}


def _value(obj: Any) -> str:
    return str(getattr(obj, "value", obj))


class WeaviateConnector(Connector):
    """Search one Weaviate collection (and tenant) by vector."""

    name = "Weaviate"

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str | None = None,
        id_property: str | None = None,
        vector_name: str | None = None,
        tenant: str | None = None,
        grpc_url: str | None = None,
        timeout_s: int = 10,
    ) -> None:
        try:
            import weaviate
            from weaviate.classes.init import AdditionalConfig, Auth, Timeout
        except ImportError as e:
            raise ConnectorError(
                ErrorCode.DEPENDENCY, "weaviate-client is not installed.", INSTALL_HINT
            ) from e
        parts = urlsplit(url if "://" in url else f"https://{url}")
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConnectorError(ErrorCode.CONFIG, f"Invalid Weaviate url {url!r}.")
        if parts.username or parts.password:
            raise ConnectorError(
                ErrorCode.CONFIG,
                "The Weaviate url must not contain credentials.",
                'Use "api_key_env" instead.',
            )
        if api_key and parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
            raise ConnectorError(
                ErrorCode.CONFIG,
                f"Refusing to send the API key over http:// to {parts.hostname}.",
                "Use the https:// address.",
            )
        self.collection_name = collection
        self.id_property = id_property
        self.vector_name = vector_name
        self.tenant = tenant
        self.url = url
        auth = Auth.api_key(api_key) if api_key else None
        extra = AdditionalConfig(timeout=Timeout(init=timeout_s, query=timeout_s, insert=timeout_s))
        self._client: Any = None
        try:
            if parts.hostname.endswith(CLOUD_SUFFIXES):
                if auth is None:
                    raise ConnectorError(
                        ErrorCode.CONFIG,
                        "Weaviate Cloud needs an API key.",
                        'Set "api_key_env" to the variable holding a read-only key.',
                    )
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url, auth_credentials=auth, additional_config=extra
                )
            else:
                grpc = urlsplit(grpc_url or f"{parts.scheme}://{parts.hostname}:50051")
                self._client = weaviate.connect_to_custom(
                    http_host=parts.hostname,
                    http_port=parts.port or (443 if parts.scheme == "https" else 80),
                    http_secure=parts.scheme == "https",
                    grpc_host=grpc.hostname or parts.hostname,
                    grpc_port=grpc.port or 50051,
                    grpc_secure=grpc.scheme == "https",
                    auth_credentials=auth,
                    additional_config=extra,
                )
            self._check_collection()
        except ConnectorError:
            self.close()
            raise
        except Exception as e:
            self.close()
            raise _error(e, collection) from e

    def _check_collection(self) -> None:
        if not self._client.collections.exists(self.collection_name):
            raise ConnectorError(
                ErrorCode.CONFIG, f"Weaviate collection {self.collection_name!r} not found."
            )
        coll = self._client.collections.get(self.collection_name)
        self._collection: Any = coll.with_tenant(self.tenant) if self.tenant else coll
        self._config: Any = coll.config.get()
        vectors: Mapping[str, Any] = self._config.vector_config or {}
        if vectors:
            if self.vector_name is None:
                if len(vectors) != 1:
                    raise ConnectorError(
                        ErrorCode.CONFIG,
                        f"Collection {self.collection_name!r} has several named vectors.",
                        f"Set vector_name to one of: {', '.join(sorted(vectors))}.",
                    )
                self.vector_name = next(iter(vectors))
            elif self.vector_name not in vectors:
                raise ConnectorError(
                    ErrorCode.CONFIG,
                    f"Collection {self.collection_name!r} has no vector {self.vector_name!r}.",
                    f"Available: {', '.join(sorted(vectors))}.",
                )
            self._index: Any = vectors[self.vector_name].vector_index_config
            self._target: str | None = self.vector_name if len(vectors) > 1 else None
        else:
            if self.vector_name is not None:
                raise ConnectorError(
                    ErrorCode.CONFIG,
                    f"Collection {self.collection_name!r} has a single unnamed vector; "
                    f"remove vector_name {self.vector_name!r}.",
                )
            self._index = self._config.vector_index_config
            self._target = None
        if self.id_property is not None:
            names = {p.name for p in self._config.properties}
            if self.id_property not in names:
                raise ConnectorError(
                    ErrorCode.CONFIG,
                    f"Collection {self.collection_name!r} has no property {self.id_property!r}.",
                    f"Properties: {', '.join(sorted(names)) or 'none'}.",
                )

    @property
    def _index_type(self) -> str:
        return type(self._index).__name__.removeprefix("_VectorIndexConfig").lower()

    def capabilities(self) -> Capabilities:
        flat = self._index_type == "flat"
        return Capabilities(
            filters=True,
            exact_reference=flat,
            exact_reference_note=None,
            config_inspection=True,
        )

    def _call(self, fn: Any) -> Any:
        try:
            return fn()
        except ConnectorError:
            raise
        except Exception as e:
            raise _error(e, self.collection_name) from e

    def describe(self) -> BackendInfo:
        metric = _value(self._index.distance_metric)
        distance = _DISTANCES.get(metric)
        if distance is None:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED, f"Weaviate distance {metric!r} is not supported."
            )
        sample = self._call(
            lambda: self._collection.query.fetch_objects(
                limit=1, include_vector=[self.vector_name] if self.vector_name else True
            )
        )
        if not sample.objects:
            raise ConnectorError(ErrorCode.CONFIG, f"Collection {self.collection_name!r} is empty.")
        vector = sample.objects[0].vector
        if isinstance(vector, Mapping):
            vector = vector.get(self.vector_name or "default") or next(iter(vector.values()), [])
        count = self._call(lambda: self._collection.aggregate.over_all(total_count=True))
        meta = self._call(self._client.get_meta)
        index_config = {
            k: _value(v) if hasattr(v, "value") else v
            for k, v in vars(self._index).items()
            if isinstance(v, (int, float, str, bool)) or hasattr(v, "value")
        }
        return BackendInfo(
            backend="weaviate",
            distance=distance,
            dim=len(vector),
            count=count.total_count,
            target={
                "collection": self.collection_name,
                "vector": self.vector_name,
                "tenant": self.tenant,
                "id_property": self.id_property,
            },
            backend_version=f"Weaviate {meta.get('version')}",
            client_version=_client_version(),
            config={
                "index_type": self._index_type,
                "index": index_config,
                "multi_tenancy": bool(getattr(self._config.multi_tenancy_config, "enabled", False)),
            },
            snapshot=None,
        )

    def _search(self, request: SearchRequest) -> list[str]:
        from weaviate.classes.query import Filter

        filters = None
        for key, value in request.filter.items():
            condition = Filter.by_property(key).equal(value)
            filters = condition if filters is None else filters & condition
        response = self._call(
            lambda: self._collection.query.near_vector(
                near_vector=[float(x) for x in request.vector],
                limit=request.k,
                target_vector=self._target,
                filters=filters,
                return_properties=[self.id_property] if self.id_property else [],
            )
        )
        if self.id_property:
            return [str(o.properties.get(self.id_property)) for o in response.objects]
        return [str(o.uuid) for o in response.objects]

    def close(self) -> None:
        client, self._client = getattr(self, "_client", None), None
        if client is not None:
            client.close()


def _error(e: Exception, collection: str) -> ConnectorError:
    name = type(e).__name__
    message = str(e).strip().splitlines()[0] if str(e).strip() else name
    status = getattr(e, "status_code", None)
    lowered = message.lower()
    if "AuthenticationFailed" in name or "InsufficientPermissions" in name or status in (401, 403):
        return ConnectorError(
            ErrorCode.AUTH,
            f"Weaviate rejected the credentials: {message}",
            "Check the key in the variable named by api_key_env and its read permissions.",
        )
    if "deadline" in lowered or "timed out" in lowered or "timeout" in name.lower():
        return ConnectorError(ErrorCode.TIMEOUT, f"Weaviate timed out: {message}")
    if "WeaviateConnectionError" in name or "unavailable" in lowered:
        return ConnectorError(
            ErrorCode.CONNECTION,
            f"Cannot reach Weaviate: {message}",
            "Check the url (and grpc_url), network access and that Weaviate is running.",
        )
    if "could not find class" in lowered or "not found" in lowered:
        return ConnectorError(ErrorCode.CONFIG, f"Weaviate collection {collection!r}: {message}")
    if "tenant" in lowered:
        return ConnectorError(ErrorCode.CONFIG, f"Weaviate tenant problem: {message}")
    return ConnectorError(ErrorCode.QUERY, f"Weaviate: {message}")


def _client_version() -> str | None:
    try:
        return f"weaviate-client {version('weaviate-client')}"
    except PackageNotFoundError:
        return None
