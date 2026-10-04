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
}


def open_connector(cfg: ConnectionConfig) -> Connector:
    """Connect to the configured backend."""
    return BACKENDS[cfg.backend](cfg)
