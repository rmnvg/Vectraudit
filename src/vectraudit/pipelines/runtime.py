"""Adapters run only from trusted local Python/CLI, never arbitrary browser code."""

from __future__ import annotations

import email.utils
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import ValidationError

from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.models import Hit, PipelineRequest, PipelineResult, StageTrace


class Pipeline(Protocol):
    def __call__(self, request: PipelineRequest) -> PipelineResult: ...


@dataclass(frozen=True)
class Stage:
    """An instrumented stage. Parents explicitly describe branches and fusion inputs."""

    name: str
    kind: Any
    run: Callable[[PipelineRequest, dict[str, list[Hit]]], list[Hit]]
    parents: tuple[str, ...] = ()


class PythonPipeline:
    """Execute a topologically ordered set of stages and measure every call."""

    def __init__(self, stages: list[Stage]) -> None:
        if not stages or len(stages) > 64:
            raise ValueError("a pipeline needs 1-64 stages")
        seen: set[str] = set()
        for stage in stages:
            StageTrace(name=stage.name, kind=stage.kind, hits=[], latency_ms=0)
            if stage.name in seen or any(p not in seen for p in stage.parents):
                raise ValueError("stage names must be unique and parents must precede children")
            seen.add(stage.name)
        self.stages = stages

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        outputs: dict[str, list[Hit]] = {}
        traces = []
        for stage in self.stages:
            start = time.perf_counter()
            hits = stage.run(
                request, {p: [h.model_copy(deep=True) for h in outputs[p]] for p in stage.parents}
            )
            elapsed = (time.perf_counter() - start) * 1000
            trace = StageTrace(
                name=stage.name,
                kind=stage.kind,
                parents=list(stage.parents),
                hits=hits,
                latency_ms=elapsed,
                source="measured",
            )
            outputs[stage.name] = trace.hits
            traces.append(trace)
        hits = traces[-1].hits
        if len(hits) > request.k:
            raise ValueError("pipeline must explicitly truncate delivered results to k")
        return PipelineResult(hits=hits, stages=traces)


def retry_after(value: str | None) -> float | None:
    """Parse ``Retry-After`` (delay seconds or an HTTP date); ``None`` when absent or invalid."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class HTTPPipeline:
    """POST the request contract to an explicitly configured endpoint.

    Local/private endpoints are supported for customer deployments. Redirects and
    ambient proxy configuration are disabled; tokens never enter run artifacts.
    """

    def __init__(self, url: str, *, token: str | None = None, timeout: float = 30) -> None:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("endpoint must be an HTTP(S) URL")
        if parts.username or parts.password or parts.fragment or parts.query:
            raise ValueError("endpoint must not contain credentials, query strings or fragments")
        if not 0 < timeout <= 300:
            raise ValueError("timeout must be in (0, 300] seconds")
        if (
            token
            and parts.scheme != "https"
            and parts.hostname not in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("authenticated remote endpoints require HTTPS")
        self.url, self.token, self.timeout = url, token, timeout

    def __call__(self, request: PipelineRequest) -> PipelineResult:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            self.url, data=request.model_dump_json().encode(), headers=headers, method="POST"
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(req, timeout=self.timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            code = {
                401: "http_auth",
                403: "http_auth",
                404: "http_not_found",
                429: "http_rate_limit",
                502: "http_unavailable",
                503: "http_unavailable",
                504: "http_unavailable",
            }.get(exc.code, "http_error")
            wait = retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            exc.close()
            raise PipelineCallError(
                code, "pipeline endpoint returned an HTTP error", retry_after=wait
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            # HTTP errors can contain signed URLs, headers or customer text.
            cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            code = (
                "timeout"
                if isinstance(cause, TimeoutError | socket.timeout)
                else "connection_failed"
            )
            raise PipelineCallError(code, "pipeline endpoint failed or timed out") from exc
        if len(raw) > 8 * 1024 * 1024:
            raise PipelineCallError("response_too_large", "pipeline response exceeds 8 MiB")
        try:
            result = PipelineResult.model_validate(json.loads(raw))
        except (ValueError, UnicodeError, ValidationError) as exc:
            raise PipelineCallError("invalid_response", "pipeline response is not valid") from exc
        # Remote timings are reported, never independently measured by this runner.
        for stage in result.stages:
            stage.source = "reported"
        return result
