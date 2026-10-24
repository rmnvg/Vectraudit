"""Connection files: which backend to evaluate, and how to reach it, as JSON.

::

    {"backend": "faiss", "index": "ivf_pq.index", "vectors": "vectors.npy",
     "params": {"nprobe": 16}}

Relative paths resolve against the connection file's directory. Secrets never appear in
the file: a field such as ``api_key`` or ``dsn`` is given as ``api_key_env`` / ``dsn_env``,
naming an environment variable the runner reads at connect time. ``params`` are the
default search parameters for every query; ``name`` is an optional display label.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from vectraudit.connectors.base import Connector, ConnectorError, ErrorCode, ParamValue

SECRET_FIELDS = frozenset({"api_key", "dsn", "password", "token", "url_with_credentials"})
COMMON_FIELDS = frozenset({"backend", "name", "params"})


@dataclass(frozen=True)
class ConnectionConfig:
    """A parsed connection file. ``options`` are the backend-specific fields."""

    backend: str
    options: dict[str, Any]
    params: dict[str, ParamValue] = field(default_factory=dict)
    name: str | None = None
    base_dir: Path = Path()

    def public(self) -> dict[str, Any]:
        """The configuration as recorded in run files: env var names, never their values."""
        return {"backend": self.backend, "name": self.name, **self.options, "params": self.params}


def load_connection(path: str | Path) -> ConnectionConfig:
    """Read and validate a connection file."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConnectorError(ErrorCode.CONFIG, f"Connection file not found: {p}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ConnectorError(ErrorCode.CONFIG, f"{p} is not valid JSON: {e}") from e
    return parse_connection(raw, base_dir=p.parent)


def parse_connection(raw: Any, base_dir: Path = Path()) -> ConnectionConfig:
    """Validate a connection mapping (see the module docstring)."""
    if not isinstance(raw, dict):
        raise ConnectorError(ErrorCode.CONFIG, "A connection must be a JSON object.")
    backend = raw.get("backend")
    if backend not in BACKENDS:
        raise ConnectorError(
            ErrorCode.CONFIG,
            f"Unknown backend {backend!r}.",
            f"Use one of: {', '.join(sorted(BACKENDS))}.",
        )
    literal = SECRET_FIELDS & set(raw)
    if literal:
        key = sorted(literal)[0]
        raise ConnectorError(
            ErrorCode.CONFIG,
            f"Secrets must not be written in connection files ({key!r}).",
            f'Put the value in an environment variable and use "{key}_env": "VAR_NAME".',
        )
    params = raw.get("params", {})
    if not isinstance(params, dict) or any(
        not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float))
        for k, v in params.items()
    ):
        raise ConnectorError(ErrorCode.CONFIG, "params must map parameter names to numbers.")
    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        raise ConnectorError(ErrorCode.CONFIG, "name must be a string.")
    options = {k: v for k, v in raw.items() if k not in COMMON_FIELDS}
    return ConnectionConfig(backend, options, dict(params), name, base_dir)


def secret(options: Mapping[str, Any], key: str, *, required: bool = True) -> str | None:
    """Read the secret named by ``options[f"{key}_env"]`` from the environment."""
    var = options.get(f"{key}_env")
    if var is None:
        if required:
            raise ConnectorError(
                ErrorCode.CONFIG, f'Missing "{key}_env": the environment variable holding {key}.'
            )
        return None
    if not isinstance(var, str) or not var:
        raise ConnectorError(ErrorCode.CONFIG, f"{key}_env must name an environment variable.")
    value = os.environ.get(var)
    if not value:
        raise ConnectorError(
            ErrorCode.CONFIG,
            f"Environment variable {var} is not set.",
            f"Export {var} (in CI, from a secret) before running.",
        )
    return value


def option(options: Mapping[str, Any], key: str, kind: type, default: Any = None) -> Any:
    """A typed backend option: ``default`` when absent, ``ConnectorError`` when mistyped."""
    value = options.get(key)
    if value is None:
        return default
    if (kind is int and isinstance(value, bool)) or not isinstance(value, kind):
        raise ConnectorError(ErrorCode.CONFIG, f"{key} must be a {kind.__name__}.")
    return value


def require(options: Mapping[str, Any], key: str, kind: type) -> Any:
    """A required typed backend option."""
    if options.get(key) is None:
        raise ConnectorError(ErrorCode.CONFIG, f'The connection needs "{key}".')
    return option(options, key, kind)


def resolve_path(cfg: ConnectionConfig, value: str) -> Path:
    """A path from the connection file, relative to the file's directory."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else cfg.base_dir / p


def _optional_path(cfg: ConnectionConfig, key: str) -> Path | None:
    value = option(cfg.options, key, str)
    return None if value is None else resolve_path(cfg, value)


# Shared by offline doctor and runtime opens: structural checks need no backend client.
_OPTION_TYPES: dict[str, dict[str, type]] = {
    "faiss": {"index": str, "vectors": str, "ids": str, "reconstructed_reference": bool},
    "qdrant": {
        "url": str,
        "path": str,
        "api_key_env": str,
        "collection": str,
        "vector_name": str,
        "timeout_s": int,
    },
    "pgvector": {
        "dsn_env": str,
        "table": str,
        "id_column": str,
        "vector_column": str,
        "distance": str,
        "filter_columns": list,
        "statement_timeout_ms": int,
        "connect_timeout_s": int,
    },
    "pinecone": {
        "index": str,
        "namespace": str,
        "host": str,
        "api_key_env": str,
        "control_plane_url": str,
        "timeout_s": int,
    },
    "elasticsearch": {
        "url": str,
        "index": str,
        "vector_field": str,
        "id_field": str,
        "api_key_env": str,
        "basic_auth_env": str,
        "timeout_s": int,
    },
    "weaviate": {
        "url": str,
        "collection": str,
        "api_key_env": str,
        "id_property": str,
        "vector_name": str,
        "tenant": str,
        "grpc_url": str,
        "timeout_s": int,
    },
}
_REQUIRED = {
    "faiss": ("index",),
    "qdrant": ("collection",),
    "pgvector": ("dsn_env", "table"),
    "pinecone": ("api_key_env", "index"),
    "weaviate": ("url", "collection"),
    "elasticsearch": ("url", "index", "vector_field"),
}
_OPTION_TYPES["opensearch"] = _OPTION_TYPES["elasticsearch"]
_REQUIRED["opensearch"] = _REQUIRED["elasticsearch"]


def validate_connection(cfg: ConnectionConfig) -> None:
    """Validate backend options offline, without accessing files, secrets or servers."""
    if cfg.backend not in _OPTION_TYPES:
        raise ConnectorError(ErrorCode.CONFIG, "Unknown connection backend.")
    known = _OPTION_TYPES[cfg.backend]
    _reject_unknown(cfg, set(known))
    for key in _REQUIRED[cfg.backend]:
        if key.endswith("_env") and cfg.options.get(key) is None:
            raise ConnectorError(
                ErrorCode.CONFIG, f'Missing "{key}": the environment variable holding the secret.'
            )
    for key, kind in known.items():
        value = (
            require(cfg.options, key, kind)
            if key in _REQUIRED[cfg.backend]
            else option(cfg.options, key, kind)
        )
        if value is None:
            continue
        if kind is str and key != "namespace" and not value.strip():
            raise ConnectorError(ErrorCode.CONFIG, f"{key} must not be empty.")
        if kind is int and value < 1:
            raise ConnectorError(ErrorCode.CONFIG, f"{key} must be >= 1.")
        if kind is list and any(not isinstance(item, str) for item in value):
            raise ConnectorError(ErrorCode.CONFIG, f"{key} must be a list of column names.")
    if cfg.backend == "faiss" and cfg.options.get("ids") and not cfg.options.get("vectors"):
        raise ConnectorError(ErrorCode.CONFIG, "ids needs vectors.")
    if cfg.backend == "qdrant" and bool(cfg.options.get("url")) == bool(cfg.options.get("path")):
        raise ConnectorError(
            ErrorCode.CONFIG, 'Give the Qdrant connection either "url" or "path" (exactly one).'
        )

    for key in ("url", "grpc_url", "host", "control_plane_url"):
        value = cfg.options.get(key)
        if value is None:
            continue
        if cfg.backend in ("pinecone", "weaviate") and "://" not in value:
            value = "https://" + value
        try:
            parts = urlsplit(value)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise ValueError("Invalid address")
            if parts.username is not None or parts.password is not None:
                raise ConnectorError(
                    ErrorCode.CONFIG,
                    f"The {cfg.backend.title()} {key} must not contain credentials.",
                )
            _ = parts.port
        except ValueError as exc:
            raise ConnectorError(ErrorCode.CONFIG, f"Invalid {cfg.backend.title()} {key}.") from exc


def _open_faiss(cfg: ConnectionConfig) -> Connector:
    from vectraudit.connectors.faiss_index import FaissConnector

    _reject_unknown(cfg, {"index", "vectors", "ids", "reconstructed_reference"})
    return FaissConnector(
        resolve_path(cfg, require(cfg.options, "index", str)),
        vectors=_optional_path(cfg, "vectors"),
        ids=_optional_path(cfg, "ids"),
        reconstructed_reference=option(cfg.options, "reconstructed_reference", bool, False),
    )


def _open_qdrant(cfg: ConnectionConfig) -> Connector:
    from vectraudit.connectors.qdrant import QdrantConnector

    _reject_unknown(cfg, {"url", "path", "api_key_env", "collection", "vector_name", "timeout_s"})
    path = _optional_path(cfg, "path")
    return QdrantConnector(
        require(cfg.options, "collection", str),
        url=option(cfg.options, "url", str),
        path=None if path is None else str(path),
        api_key=secret(cfg.options, "api_key", required=False),
        vector_name=option(cfg.options, "vector_name", str),
        timeout_s=_positive_int(cfg, "timeout_s", 10),
    )


def _open_pinecone(cfg: ConnectionConfig) -> Connector:
    from vectraudit.connectors.pinecone import CONTROL_PLANE, PineconeConnector

    _reject_unknown(
        cfg, {"index", "namespace", "host", "api_key_env", "control_plane_url", "timeout_s"}
    )
    api_key = secret(cfg.options, "api_key")
    assert api_key is not None
    return PineconeConnector(
        api_key=api_key,
        index=require(cfg.options, "index", str),
        namespace=option(cfg.options, "namespace", str, ""),
        host=option(cfg.options, "host", str),
        control_plane_url=option(cfg.options, "control_plane_url", str, CONTROL_PLANE),
        timeout_s=_positive_int(cfg, "timeout_s", 10),
    )


def _open_weaviate(cfg: ConnectionConfig) -> Connector:
    from vectraudit.connectors.weaviate import WeaviateConnector

    _reject_unknown(
        cfg,
        {
            "url",
            "grpc_url",
            "api_key_env",
            "collection",
            "id_property",
            "vector_name",
            "tenant",
            "timeout_s",
        },
    )
    return WeaviateConnector(
        url=require(cfg.options, "url", str),
        collection=require(cfg.options, "collection", str),
        api_key=secret(cfg.options, "api_key", required=False),
        id_property=option(cfg.options, "id_property", str),
        vector_name=option(cfg.options, "vector_name", str),
        tenant=option(cfg.options, "tenant", str),
        grpc_url=option(cfg.options, "grpc_url", str),
        timeout_s=_positive_int(cfg, "timeout_s", 10),
    )


def _open_elasticsearch(cfg: ConnectionConfig) -> Connector:
    from vectraudit.connectors.elasticsearch import ElasticsearchConnector

    _reject_unknown(cfg, set(_OPTION_TYPES["elasticsearch"]))
    return ElasticsearchConnector(
        url=require(cfg.options, "url", str),
        index=require(cfg.options, "index", str),
        vector_field=require(cfg.options, "vector_field", str),
        id_field=option(cfg.options, "id_field", str),
        api_key=secret(cfg.options, "api_key", required=False),
        basic_auth=secret(cfg.options, "basic_auth", required=False),
        timeout_s=_positive_int(cfg, "timeout_s", 10),
    )


def _open_pgvector(cfg: ConnectionConfig) -> Connector:
    from vectraudit.connectors.pgvector import PgvectorConnector

    _reject_unknown(
        cfg,
        {
            "dsn_env",
            "table",
            "id_column",
            "vector_column",
            "distance",
            "filter_columns",
            "statement_timeout_ms",
            "connect_timeout_s",
        },
    )
    filter_columns = option(cfg.options, "filter_columns", list, [])
    if any(not isinstance(c, str) for c in filter_columns):
        raise ConnectorError(ErrorCode.CONFIG, "filter_columns must be a list of column names.")
    dsn = secret(cfg.options, "dsn")
    assert dsn is not None
    return PgvectorConnector(
        dsn,
        table=require(cfg.options, "table", str),
        id_column=option(cfg.options, "id_column", str, "id"),
        vector_column=option(cfg.options, "vector_column", str, "embedding"),
        distance=option(cfg.options, "distance", str, "cosine"),
        filter_columns=filter_columns,
        statement_timeout_ms=_positive_int(cfg, "statement_timeout_ms", 10_000),
        connect_timeout_s=_positive_int(cfg, "connect_timeout_s", 10),
    )


def _positive_int(cfg: ConnectionConfig, key: str, default: int) -> int:
    value: int = option(cfg.options, key, int, default)
    if value < 1:
        raise ConnectorError(ErrorCode.CONFIG, f"{key} must be >= 1.")
    return value


def _reject_unknown(cfg: ConnectionConfig, known: set[str]) -> None:
    unknown = set(cfg.options) - known
    if unknown:
        raise ConnectorError(
            ErrorCode.CONFIG,
            f"Unknown field(s) for {cfg.backend}: {', '.join(sorted(unknown))}.",
            f"Allowed: {', '.join(sorted(known | COMMON_FIELDS))}.",
        )


BACKENDS: dict[str, Callable[[ConnectionConfig], Connector]] = {
    "faiss": _open_faiss,
    "qdrant": _open_qdrant,
    "pgvector": _open_pgvector,
    "pinecone": _open_pinecone,
    "weaviate": _open_weaviate,
    "elasticsearch": _open_elasticsearch,
    "opensearch": _open_elasticsearch,
}


def open_connector(cfg: ConnectionConfig) -> Connector:
    """Connect to the configured backend."""
    validate_connection(cfg)
    return BACKENDS[cfg.backend](cfg)
