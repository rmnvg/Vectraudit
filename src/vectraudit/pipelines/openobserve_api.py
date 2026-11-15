"""Pull traces from OpenObserve and rebuild them as OTLP JSON.

Teams that send their LLM application's OpenTelemetry traces to OpenObserve can import
them without an exporter of their own. Requests follow OpenObserve's published API
documentation:

- ``GET /api/{org}/{stream}/traces/latest?start_time&end_time&from&size`` lists the traces
  in a window (times in microseconds; trace start times in nanoseconds).
- ``POST /api/{org}/_search?type=traces`` with ``{"query": {"sql", "start_time",
  "end_time", "from", "size"}}`` returns the spans of the chosen traces.
- HTTP Basic authentication (the user and password, or the token the UI shows).

Traces are taken by their start time, then every span of each one is read over the window
plus a grace period, so a trace is not cut at the window's end.

**How OpenObserve stores spans, and what is rebuilt.** Each span is a row: ``trace_id``,
``span_id``, ``reference_parent_span_id``, ``operation_name``, ``start_time`` and
``end_time`` (nanoseconds), and the span's attributes flattened into top-level columns.
Flattening lower-cases attribute keys and replaces every other character (including the
dot) with an underscore, and prefixes ``attr_`` to a key that would collide with a span
column. That is not reversible in general (``a.b_c`` and ``a_b.c`` look alike), so only the
attribute keys the importer reads are restored, each matched by an explicit pattern:
OpenInference retrieval, reranker, LLM message, prompt and session keys, and the
OpenLLMetry keys :mod:`vectraudit.pipelines.openllmetry` translates. Other columns are kept
under their stored names, where nothing reads them. Events are stored as a JSON string with
their original keys and are rebuilt as recorded.

The rebuilt file is OTLP JSON, so ``import-traces``, ``intake``, ``drift`` and
``score-drift`` read it unchanged, including OpenLLMetry translation. Nothing is written to
OpenObserve. OpenObserve is AGPL-3.0; this client only calls its HTTP API and contains none
of its code.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from vectraudit import __version__
from vectraudit.pipelines.langfuse_api import LOCAL_HOSTS, MAX_RECORDS, _iso

Send = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]
PAGE = 500
TRACE_BATCH = 100  # trace IDs per span query
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
SPAN_COLUMNS = {
    "_timestamp", "_o2_ingest_ts", "trace_id", "span_id", "reference_parent_span_id",
    "reference_ref_type", "reference_parent_trace_id", "flags", "span_status", "span_kind",
    "operation_name", "start_time", "end_time", "duration", "service_name", "events",
    "links", "_original", "_all_values",
}  # fmt: skip
_KNOWN_KEYS = (
    "openinference.span.kind", "input.value", "input.mime_type", "output.value",
    "output.mime_type", "metadata", "session.id", "user.id", "prompt.id", "prompt.vendor",
    "prompt.url", "llm.prompt_template.version", "reranker.query", "reranker.model_name",
    "reranker.top_k", "gen_ai.system", "gen_ai.input.messages", "llm.request.type",
    "db.system", "db.operation", "db.vector.query.top_k", "db.vector.query.embeddings_count",
    "db.chroma.query.n_results", "db.chroma.query.texts_count",
    "db.chroma.query.embeddings_count", "traceloop.span.kind", "traceloop.entity.input",
    "traceloop.entity.output", "traceloop.entity.name", "traceloop.workflow.name",
    "traceloop.prompt.key", "traceloop.prompt.version", "traceloop.prompt.version_hash",
    "traceloop.prompt.managed",
)  # fmt: skip
_DOC_FIELD = r"(id|content|score|metadata)"
_PATTERNS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
    (
        re.compile(rf"^retrieval_documents_(\d+)_document_{_DOC_FIELD}$"),
        lambda m: f"retrieval.documents.{m[1]}.document.{m[2]}",
    ),
    (
        re.compile(rf"^reranker_(input|output)_documents_(\d+)_document_{_DOC_FIELD}$"),
        lambda m: f"reranker.{m[1]}_documents.{m[2]}.document.{m[3]}",
    ),
    (
        re.compile(r"^llm_input_messages_(\d+)_message_(content|role)$"),
        lambda m: f"llm.input_messages.{m[1]}.message.{m[2]}",
    ),
    (
        re.compile(r"^llm_input_messages_(\d+)_message_contents_(\d+)_message_content_text$"),
        lambda m: f"llm.input_messages.{m[1]}.message.contents.{m[2]}.message_content.text",
    ),
    (
        re.compile(r"^gen_ai_prompt_(\d+)_(content|role)$"),
        lambda m: f"gen_ai.prompt.{m[1]}.{m[2]}",
    ),
    (
        re.compile(r"^traceloop_association_properties_(.+)$"),
        lambda m: f"traceloop.association.properties.{m[1]}",
    ),
)


def stored_key(key: str) -> str:
    """An attribute key as OpenObserve stores it: lower case, other characters as ``_``."""
    return "".join(c.lower() if c.isalnum() else "_" for c in key)


_RESTORE = {stored_key(k): k for k in _KNOWN_KEYS}


def restore_key(column: str) -> str:
    """The original attribute key for a stored column, when it is one the importer reads."""
    if column.startswith("attr_") and column[5:] in SPAN_COLUMNS:
        column = column[5:]
    if column in _RESTORE:
        return _RESTORE[column]
    for pattern, original in _PATTERNS:
        match = pattern.match(column)
        if match:
            return original(match)
    return column


class OpenObserveAPIError(ValueError):
    """OpenObserve could not be read; the message says why and what to change."""


def _urllib_send(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


@dataclass
class OpenObserveSource:
    """One OpenObserve organisation and trace stream."""

    url: str
    org: str
    stream: str = "default"
    user: str | None = None
    password: str | None = None
    token: str | None = None
    timeout_s: float = 60.0
    send: Send | None = None
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise OpenObserveAPIError(f"Invalid OpenObserve URL {self.url!r}.")
        if parts.username or parts.password or parts.query:
            raise OpenObserveAPIError(
                "The OpenObserve URL must not contain credentials or a query."
            )
        for what, value in (("organisation", self.org), ("stream", self.stream)):
            if not _NAME.match(value):
                raise OpenObserveAPIError(
                    f"Invalid {what} {value!r}: letters, digits, '_' and '-' only."
                )
        if self.token and (self.user or self.password):
            raise OpenObserveAPIError("Give a token or a user and password, not both.")
        if not self.token and not (self.user and self.password):
            raise OpenObserveAPIError(
                "OpenObserve credentials are missing: set OPENOBSERVE_TOKEN, or "
                "OPENOBSERVE_USER and OPENOBSERVE_PASSWORD."
            )
        if parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
            raise OpenObserveAPIError(
                f"Refusing to send OpenObserve credentials over http:// to {parts.hostname}; "
                "use https://."
            )
        self.url = self.url.rstrip("/")

    def _auth(self) -> str:
        token = self.token or base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        return f"Basic {token}"

    def request(
        self, method: str, path: str, *, params: Mapping[str, Any], body: Any = None
    ) -> dict[str, Any]:
        """One authenticated request returning a JSON object with ``hits``."""
        headers = {
            "Accept": "application/json",
            "Authorization": self._auth(),
            "User-Agent": f"vectraudit/{__version__}",
        }
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        url = f"{self.url}{path}?{urlencode(params)}"
        self.calls += 1
        try:
            status, raw = (self.send or _urllib_send)(method, url, headers, data, self.timeout_s)
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, b""
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise OpenObserveAPIError(f"Cannot reach OpenObserve at {self.url}: {reason}") from exc
        if status in (401, 403):
            raise OpenObserveAPIError(
                f"OpenObserve refused the credentials (HTTP {status}); check the user and "
                "password or token, and that they can read this organisation."
            )
        if status == 404:
            raise OpenObserveAPIError(
                f"OpenObserve answered HTTP 404 for {path}; check the organisation and stream."
            )
        if status == 429:
            raise OpenObserveAPIError(
                "OpenObserve rate limit reached (HTTP 429); pull a shorter window."
            )
        if status != 200:  # error bodies are not echoed: they can repeat the query
            raise OpenObserveAPIError(f"OpenObserve answered HTTP {status} for {path}.")
        try:
            out = json.loads(raw)
        except ValueError as exc:
            raise OpenObserveAPIError(f"OpenObserve returned non-JSON for {path}.") from exc
        if not isinstance(out, dict) or not isinstance(out.get("hits"), list):
            raise OpenObserveAPIError(f"Unexpected OpenObserve response shape for {path}.")
        return out


def _micros(iso: str) -> int:
    moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(moment.timestamp() * 1_000_000)


def _paged(fetch: Callable[[int], dict[str, Any]], max_records: int) -> Iterator[dict[str, Any]]:
    offset = count = 0
    while True:
        page = fetch(offset)
        hits = page["hits"]
        for hit in hits:
            if not isinstance(hit, dict):
                raise OpenObserveAPIError("Unexpected record in the OpenObserve response.")
            count += 1
            if count > max_records:
                raise OpenObserveAPIError(
                    f"More than {max_records} records in the window; pull a shorter window."
                )
            yield hit
        total = page.get("total")
        offset += len(hits)
        if not hits or (isinstance(total, int) and offset >= total) or len(hits) < PAGE:
            return


def fetch(
    source: OpenObserveSource,
    *,
    start: str,
    end: str,
    grace_minutes: int = 60,
    max_records: int = MAX_RECORDS,
) -> tuple[list[str], list[dict[str, Any]]]:
    """``(trace IDs, span rows)`` for the traces that started in ``[start, end)``."""
    since, until = _iso(start, "--since"), _iso(end, "--until")
    if not 0 <= grace_minutes <= 24 * 60:
        raise OpenObserveAPIError("--grace-minutes must be between 0 and 1440.")
    t0, t1 = _micros(since), _micros(until)
    late = t1 + int(timedelta(minutes=grace_minutes).total_seconds() * 1_000_000)
    stream = quote(source.stream, safe="")

    def traces_page(offset: int) -> dict[str, Any]:
        return source.request(
            "GET",
            f"/api/{source.org}/{stream}/traces/latest",
            params={"start_time": t0, "end_time": t1, "from": offset, "size": PAGE, "filter": ""},
        )

    trace_ids: list[str] = []
    seen: set[str] = set()
    for hit in _paged(traces_page, max_records):
        trace_id = hit.get("trace_id")
        if isinstance(trace_id, str) and _TRACE_ID.match(trace_id) and trace_id not in seen:
            seen.add(trace_id)
            trace_ids.append(trace_id)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for i in range(0, len(trace_ids), TRACE_BATCH):
        batch = trace_ids[i : i + TRACE_BATCH]
        # IDs are validated as 32 hex characters, so quoting them cannot alter the query.
        sql = (
            f'SELECT * FROM "{source.stream}" WHERE trace_id IN ('
            + ", ".join(f"'{t}'" for t in batch)
            + ") ORDER BY start_time ASC, span_id ASC"
        )

        def spans_page(offset: int, sql: str = sql) -> dict[str, Any]:
            return source.request(
                "POST",
                f"/api/{source.org}/_search",
                params={"type": "traces"},
                body={
                    "query": {
                        "sql": sql,
                        "start_time": t0,
                        "end_time": late,
                        "from": offset,
                        "size": PAGE,
                    }
                },
            )

        for row in _paged(spans_page, max_records):
            key = (str(row.get("trace_id")), str(row.get("span_id")))
            if key[0] in seen and key[1]:
                rows.setdefault(key, row)
                if len(rows) > max_records:
                    raise OpenObserveAPIError(
                        f"More than {max_records} spans in the window; pull a shorter window."
                    )
    return trace_ids, list(rows.values())


def _otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value, default=str)}


def _attributes(pairs: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _otlp_value(v)} for k, v in pairs.items() if v is not None]


def _events(raw: Any) -> list[dict[str, Any]]:
    try:
        events = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    out = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict) or not isinstance(event.get("name"), str):
            continue
        attributes = {k: v for k, v in event.items() if k not in ("name", "_timestamp")}
        out.append(
            {
                "name": event["name"],
                "timeUnixNano": str(int(event.get("_timestamp") or 0) * 1000),
                "attributes": _attributes(attributes),
            }
        )
    return out


def _order(span: Mapping[str, Any]) -> tuple[str, int, str]:
    return (str(span["traceId"]), int(str(span["startTimeUnixNano"])), str(span["spanId"]))


def to_otlp(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stored span rows as one OTLP JSON object, attribute keys restored where known."""
    spans = []
    for row in rows:
        attributes: dict[str, Any] = {}
        for column, value in row.items():
            if column in SPAN_COLUMNS or column.startswith(("service_", "o2_", "_")):
                continue
            original = restore_key(column)
            attributes.setdefault(original, value)
        spans.append(
            {
                "traceId": row.get("trace_id"),
                "spanId": row.get("span_id"),
                "parentSpanId": row.get("reference_parent_span_id") or "",
                "name": row.get("operation_name"),
                "startTimeUnixNano": str(row.get("start_time") or 0),
                "endTimeUnixNano": str(row.get("end_time") or row.get("start_time") or 0),
                "attributes": _attributes(attributes),
                "events": _events(row.get("events")),
            }
        )
    spans.sort(key=_order)
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
