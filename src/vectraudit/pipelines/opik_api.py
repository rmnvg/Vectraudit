"""Pull traces, spans and feedback scores from the Opik REST API.

The Opik counterpart of :mod:`vectraudit.pipelines.langfuse_api` and
:mod:`vectraudit.pipelines.langsmith_api`. Requests follow Opik's published OpenAPI
specification (``sdks/code_generation/fern/openapi/openapi.yaml``) and the Opik Python SDK
for authentication:

- ``GET /v1/private/traces`` and ``GET /v1/private/spans`` by ``project_name``, page-numbered
  (``page``, ``size``), filtered by ``from_time``/``to_time`` on **creation** time.
- Opik Cloud authenticates with an ``Authorization`` header holding the API key and a
  ``Comet-Workspace`` header; a self-hosted Opik may need neither.

Traces are taken by the window; spans are read over the window plus a grace period (spans
can be created after their trace) and kept only for the traces taken, so a trace is
complete unless its spans arrived later than the grace. A trace missing spans imports as a
gap, never as a result.

Opik keeps the question on the trace, not on a span, so each trace is written as its
trace's root span, with the trace's top-level spans parented to it. The file then imports
through the same reader as every other format.

Feedback scores come inline on traces and spans. Scores on a span count for its trace.
Reasons (Opik's free-text explanations) are never written out. Nothing is written to Opik.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode, urlsplit

from vectraudit import __version__
from vectraudit.pipelines.langfuse_api import LOCAL_HOSTS, MAX_RECORDS, VERDICTS, _iso

Send = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]
OPIK_CLOUD = "https://www.comet.com/opik/api"
PAGE = 100
SPAN_FIELDS = (
    "id", "trace_id", "parent_span_id", "name", "type", "start_time", "end_time",
    "input", "output", "metadata",
)  # fmt: skip


class OpikAPIError(ValueError):
    """The Opik API could not be read; the message says why and what to change."""


def _urllib_send(url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET", headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


@dataclass
class OpikSource:
    """One Opik deployment (Cloud or self-hosted) and workspace."""

    url: str
    api_key: str | None = None
    workspace: str | None = None
    timeout_s: float = 30.0
    send: Send | None = None
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise OpikAPIError(f"Invalid Opik URL {self.url!r}.")
        if parts.username or parts.password or parts.query:
            raise OpikAPIError("The Opik URL must not contain credentials or a query.")
        if self.api_key and parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
            raise OpikAPIError(
                f"Refusing to send the Opik API key over http:// to {parts.hostname}; use https://."
            )
        if not self.api_key and parts.hostname == urlsplit(OPIK_CLOUD).hostname:
            raise OpikAPIError("Opik Cloud needs an API key: set OPIK_API_KEY.")
        self.url = self.url.rstrip("/")

    def get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """One GET returning a JSON object."""
        headers = {"Accept": "application/json", "User-Agent": f"vectraudit/{__version__}"}
        if self.api_key:
            headers["Authorization"] = self.api_key
        if self.workspace:
            headers["Comet-Workspace"] = self.workspace
        url = f"{self.url}{path}?{urlencode({k: v for k, v in params.items() if v is not None})}"
        self.calls += 1
        try:
            status, raw = (self.send or _urllib_send)(url, headers, self.timeout_s)
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, b""
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise OpikAPIError(f"Cannot reach Opik at {self.url}: {reason}") from exc
        if status in (401, 403):
            raise OpikAPIError(
                f"Opik refused the request (HTTP {status}); check OPIK_API_KEY and OPIK_WORKSPACE."
            )
        if status == 404:
            raise OpikAPIError(
                f"Opik answered HTTP 404 for {path}; check OPIK_URL_OVERRIDE (it ends in /api) "
                "and the project name."
            )
        if status == 429:
            raise OpikAPIError("Opik rate limit reached (HTTP 429); pull a shorter window.")
        if status != 200:  # error bodies are not echoed: they can repeat request details
            raise OpikAPIError(f"Opik answered HTTP {status} for {path}.")
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise OpikAPIError(f"Opik returned non-JSON for {path}.") from exc
        if not isinstance(body, dict) or not isinstance(body.get("content"), list):
            raise OpikAPIError(f"Unexpected Opik response shape for {path}.")
        return body


def _pages(
    source: OpikSource, path: str, params: Mapping[str, Any], max_records: int
) -> Iterator[dict[str, Any]]:
    """Every record of a page-numbered listing.

    The server may return smaller pages than asked for (the specification sets no maximum),
    so paging follows the reported ``total`` and only falls back to a short page when no
    total is given.
    """
    count = 0
    page = 1
    while True:
        body = source.get(path, {**params, "page": page, "size": PAGE})
        content = body["content"]
        for record in content:
            if not isinstance(record, dict):
                raise OpikAPIError(f"Unexpected record in the {path} response.")
            count += 1
            if count > max_records:
                raise OpikAPIError(
                    f"More than {max_records} records in the window; pull a shorter window."
                )
            yield record
        total = body.get("total")
        if not content:
            return
        if isinstance(total, int) and not isinstance(total, bool):
            if count >= total:
                return
        elif len(content) < PAGE:
            return
        page += 1


def fetch(
    source: OpikSource,
    project: str,
    *,
    start: str,
    end: str,
    grace_minutes: int = 60,
    max_records: int = MAX_RECORDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(traces, spans)``: traces created in ``[start, end)`` and their spans."""
    since, until = _iso(start, "--since"), _iso(end, "--until")
    if not 0 <= grace_minutes <= 24 * 60:
        raise OpikAPIError("--grace-minutes must be between 0 and 1440.")
    late = datetime.fromisoformat(until.replace("Z", "+00:00")) + timedelta(minutes=grace_minutes)
    window = {"project_name": project, "from_time": since, "to_time": until}
    traces: dict[str, dict[str, Any]] = {}
    for trace in _pages(source, "/v1/private/traces", window, max_records):
        if trace.get("id"):
            traces.setdefault(str(trace["id"]), trace)
    spans: dict[str, dict[str, Any]] = {}
    if traces:
        span_window = {**window, "to_time": late.isoformat().replace("+00:00", "Z")}
        for span in _pages(source, "/v1/private/spans", span_window, max_records):
            if span.get("id") and str(span.get("trace_id")) in traces:
                spans.setdefault(str(span["id"]), span)
    return list(traces.values()), list(spans.values())


def records(
    traces: Sequence[Mapping[str, Any]], spans: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Span records for ``import-traces``: each trace as its root span, then its spans."""
    out: list[dict[str, Any]] = []
    for trace in traces:
        out.append(
            {
                "id": trace["id"],
                "trace_id": trace["id"],
                "parent_span_id": None,
                "name": trace.get("name"),
                "type": "general",
                "start_time": trace.get("start_time"),
                "end_time": trace.get("end_time"),
                "input": trace.get("input"),
                "output": trace.get("output"),
                "metadata": trace.get("metadata"),
            }
        )
    for span in spans:
        record = {k: span.get(k) for k in SPAN_FIELDS}
        if not record["parent_span_id"]:
            record["parent_span_id"] = span["trace_id"]  # top-level spans hang off the trace
        out.append(record)
    return sorted(out, key=lambda r: (str(r["trace_id"]), str(r.get("start_time")), str(r["id"])))


def scores_to_judgments(
    traces: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    *,
    human_sources: Sequence[str] = ("ui",),
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Judgment lines for ``attach-answers``, keyed by trace ID, and why others were skipped.

    - A ``category_name`` of ``correct``, ``partial`` or ``incorrect`` (any case) is that
      verdict; any other category is skipped, never guessed from its numeric value.
    - Otherwise the numeric ``value`` stays a score that needs a declared threshold.

    ``ui`` scores (the Opik interface and annotation queues) are human review; ``sdk`` and
    ``online_scoring`` scores are model judgments. Reasons are never copied. The traces'
    ``span_feedback_scores`` are averages and are not used; each span's own scores are.
    """
    humans = {s.lower() for s in human_sources}
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    owned = [(str(t["id"]), t.get("feedback_scores")) for t in traces if t.get("id")]
    owned += [(str(s.get("trace_id")), s.get("feedback_scores")) for s in spans]
    for trace_id, scores in owned:
        for score in scores or []:
            if not isinstance(score, Mapping):
                skipped["invalid_value"] += 1
                continue
            name = score.get("name")
            if not isinstance(name, str) or not name.strip() or len(name) > 200:
                skipped["unnamed"] += 1
                continue
            row: dict[str, Any] = {
                "trace_id": trace_id,
                "evaluator": name,
                "kind": "human" if str(score.get("source") or "").lower() in humans else "model",
            }
            category, value = score.get("category_name"), score.get("value")
            if isinstance(category, str) and category.strip():
                if category.strip().lower() not in VERDICTS:
                    skipped["category_not_a_verdict"] += 1
                    continue
                row["verdict"] = category.strip().lower()
            elif (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                row["score"] = float(value)
            else:
                skipped["invalid_value"] += 1
                continue
            rows.append(row)
    return rows, skipped
