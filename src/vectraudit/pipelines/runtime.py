"""Adapters run only from trusted local Python/CLI, never arbitrary browser code."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

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
        except (OSError, urllib.error.URLError) as exc:
            # HTTP errors can contain signed URLs, headers or customer text.
            raise ValueError("pipeline endpoint failed or timed out") from exc
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("pipeline response exceeds 8 MiB")
        result = PipelineResult.model_validate(json.loads(raw))
        # Remote timings are reported, never independently measured by this runner.
        for stage in result.stages:
            stage.source = "reported"
        return result
