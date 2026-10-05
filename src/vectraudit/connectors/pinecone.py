"""Pinecone connector, over Pinecone's REST API with a pinned API version.

Connection fields::

    {"backend": "pinecone", "index": "docs", "api_key_env": "PINECONE_API_KEY",
     "namespace": "prod", "timeout_s": 10}

``host`` overrides the data-plane address (private endpoints); ``control_plane_url``
overrides ``https://api.pinecone.io`` (e.g. ``http://localhost:5080`` for pinecone-local).
Using REST with a pinned ``X-Pinecone-API-Version`` instead of the SDK keeps responses
stable across SDK releases and needs no extra dependency. Pinecone is a black box here: no
exact reference, no per-query tuning parameters and no snapshot id; results are measured,
not explained. The API key is only ever sent over HTTPS (or to a local emulator).
"""

from __future__ import annotations

import json
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

API_VERSION = "2025-04"
CONTROL_PLANE = "https://api.pinecone.io"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DISTANCES = {"cosine": Distance.COSINE, "euclidean": Distance.L2, "dotproduct": Distance.IP}
Send = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


def _urllib_send(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


def _base_url(url: str, what: str) -> str:
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConnectorError(ErrorCode.CONFIG, f"Invalid Pinecone {what} {url!r}.")
    if parts.username or parts.password:
        raise ConnectorError(ErrorCode.CONFIG, f"The Pinecone {what} must not contain credentials.")
    if parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
        raise ConnectorError(
            ErrorCode.CONFIG,
            f"Refusing to send the API key over {parts.scheme}:// to {parts.hostname}.",
            "Use the https:// address (plain http is allowed only for a local emulator).",
        )
    return url.rstrip("/")


def _pinecone_filter(conditions: Mapping[str, FilterValue]) -> dict[str, Any]:
    return {key: {"$eq": value} for key, value in conditions.items()}


class PineconeConnector(Connector):
    """Query one Pinecone index (and namespace) by dense vector."""

    name = "Pinecone"

    def __init__(
        self,
        *,
        api_key: str,
        index: str,
        namespace: str = "",
        host: str | None = None,
        control_plane_url: str = CONTROL_PLANE,
        timeout_s: int = 10,
        send: Send | None = None,
    ) -> None:
        self.index = index
        self.namespace = namespace
        self.timeout_s = timeout_s
        self._send = send or _urllib_send
        self._headers = {
            "Api-Key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": API_VERSION,
            "User-Agent": f"vectraudit/{__version__}",
        }
        control = _base_url(control_plane_url, "control_plane_url")
        self._index_info: dict[str, Any] = self._call(
            "GET", f"{control}/indexes/{quote(index)}", None, what=f"index {index!r}"
        )
        resolved = host or self._index_info.get("host")
        if not resolved:
            raise ConnectorError(ErrorCode.CONFIG, f"Pinecone index {index!r} has no host yet.")
        local = urlsplit(control).hostname in LOCAL_HOSTS and "://" not in str(resolved)
        self.host = _base_url(("http://" if local else "") + str(resolved), "host")

    def _call(self, method: str, url: str, body: Any, *, what: str = "request") -> Any:
        data = None if body is None else json.dumps(body).encode()
        try:
            status, raw = self._send(method, url, self._headers, data, float(self.timeout_s))
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        except TimeoutError as e:
            raise ConnectorError(
                ErrorCode.TIMEOUT, f"Pinecone did not answer within {self.timeout_s} s."
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ConnectorError(
                    ErrorCode.TIMEOUT, f"Pinecone did not answer within {self.timeout_s} s."
                ) from e
            raise ConnectorError(
                ErrorCode.CONNECTION,
                f"Cannot reach Pinecone: {e.reason}",
                "Check network access from the runner and the index host.",
            ) from e
        if status >= 400:
            raise _http_error(status, raw, what)
        try:
            return json.loads(raw) if raw else {}
        except ValueError as e:
            raise ConnectorError(
                ErrorCode.QUERY, "Pinecone returned a response that isn't JSON."
            ) from e

    def capabilities(self) -> Capabilities:
        return Capabilities(filters=True, exact_reference=False, config_inspection=True)

    def describe(self) -> BackendInfo:
        stats = self._call("POST", f"{self.host}/describe_index_stats", {}, what="index stats")
        namespaces = stats.get("namespaces") or {}
        if self.namespace:
            count = (namespaces.get(self.namespace) or {}).get("vectorCount", 0)
        else:
            count = stats.get("totalVectorCount")
        metric = str(self._index_info.get("metric") or stats.get("metric") or "")
        distance = _DISTANCES.get(metric)
        if distance is None:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED, f"Pinecone metric {metric!r} is not supported."
            )
        dim = self._index_info.get("dimension") or stats.get("dimension")
        if not isinstance(dim, int):
            raise ConnectorError(
                ErrorCode.UNSUPPORTED, "This Pinecone index has no dense dimension."
            )
        return BackendInfo(
            backend="pinecone",
            distance=distance,
            dim=dim,
            count=count if isinstance(count, int) else None,
            target={"index": self.index, "namespace": self.namespace or None},
            backend_version=f"Pinecone API {API_VERSION}",
            client_version="REST",
            config={
                "spec": self._index_info.get("spec"),
                "namespaces": len(namespaces),
                "index_fullness": stats.get("indexFullness"),
                "deletion_protection": self._index_info.get("deletion_protection"),
            },
            snapshot=None,
        )

    def _search(self, request: SearchRequest) -> list[str]:
        body: dict[str, Any] = {
            "vector": [float(x) for x in request.vector],
            "topK": request.k,
            "namespace": self.namespace,
            "includeValues": False,
            "includeMetadata": False,
        }
        if request.filter:
            body["filter"] = _pinecone_filter(request.filter)
        response = self._call("POST", f"{self.host}/query", body, what="query")
        return [str(m["id"]) for m in response.get("matches") or [] if "id" in m]


def _http_error(status: int, raw: bytes, what: str) -> ConnectorError:
    try:
        payload = json.loads(raw)
        detail = str(payload.get("message") or (payload.get("error") or {}).get("message") or "")
    except (ValueError, AttributeError):
        detail = raw[:200].decode("utf-8", "replace")
    if status in (401, 403):
        return ConnectorError(
            ErrorCode.AUTH,
            f"Pinecone rejected the API key (HTTP {status}).",
            "Check the key in the variable named by api_key_env and its project access.",
        )
    if status == 404:
        return ConnectorError(ErrorCode.CONFIG, f"Pinecone {what} not found.")
    if status in (408, 504):
        return ConnectorError(ErrorCode.TIMEOUT, f"Pinecone timed out (HTTP {status}).")
    if status == 429 or status >= 500:
        return ConnectorError(
            ErrorCode.CONNECTION,
            f"Pinecone is rate limiting or unavailable (HTTP {status}).",
            "Retry later, or lower concurrency; the run is incomplete, not regressed.",
        )
    return ConnectorError(
        ErrorCode.QUERY, f"Pinecone rejected the {what} (HTTP {status}): {detail}"
    )
