"""Pull spans and annotations from the Arize Phoenix REST API.

The Phoenix counterpart of :mod:`vectraudit.pipelines.langfuse_api` and
:mod:`vectraudit.pipelines.opik_api`. Requests follow Phoenix's published OpenAPI document
(``schemas/openapi.json``) and were checked against a running Phoenix 21 server:

- ``GET /v1/projects/{project}/spans`` returns span records in exactly the shape
  ``import-traces`` reads as ``spans`` (``context.trace_id``/``span_id``, ``parent_id``,
  ``span_kind`` and flat OpenInference ``attributes``), so they are written unchanged.
  Paging is by ``next_cursor``.
- The window selects **root spans** (``parent_id=null``) that started in it; every span of
  those traces is then read by trace ID, whenever it started, so a trace is never cut by
  the window's end.
- ``GET .../trace_annotations`` and ``.../span_annotations`` by trace and span IDs.
- Phoenix Cloud and secured servers take ``Authorization: Bearer PHOENIX_API_KEY``; a local
  server may need none.

Annotations become answer judgments: ``annotator_kind`` ``HUMAN`` is human review, any
other kind (``LLM``, ``CODE`` ...) a model judgment. Notes (Phoenix stores them as
annotations named ``note``) are not judgments and are skipped. Explanations and note text
are never written out. Nothing is written to Phoenix.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from vectraudit import __version__
from vectraudit.pipelines.langfuse_api import LOCAL_HOSTS, MAX_RECORDS, VERDICTS, _iso

Send = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]
PostSend = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]
PHOENIX_CLOUD_HOST = "app.phoenix.arize.com"
SPAN_PAGE = 1000
ANNOTATION_PAGE = 1000
ID_BATCH = 50
"""IDs per request; keeps query strings well under common URL limits."""
NOTE_NAME = "note"
NOTE_PREFIX = "px-"
HUMAN_KINDS = ("HUMAN",)


class PhoenixAPIError(ValueError):
    """The Phoenix API could not be read; the message says why and what to change."""


def _urllib_send(url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET", headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


def _urllib_post(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method="POST", headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


@dataclass
class PhoenixSource:
    """One Phoenix server (self-hosted, local or Phoenix Cloud space)."""

    url: str
    api_key: str | None = None
    timeout_s: float = 30.0
    send: Send | None = None
    post_send: PostSend | None = None
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise PhoenixAPIError(f"Invalid Phoenix URL {self.url!r}.")
        if parts.username or parts.password or parts.query:
            raise PhoenixAPIError("The Phoenix URL must not contain credentials or a query.")
        if self.api_key and parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
            raise PhoenixAPIError(
                f"Refusing to send the Phoenix API key over http:// to {parts.hostname}; "
                "use https://."
            )
        if not self.api_key and parts.hostname == PHOENIX_CLOUD_HOST:
            raise PhoenixAPIError("Phoenix Cloud needs an API key: set PHOENIX_API_KEY.")
        self.url = self.url.rstrip("/")
        if self.url.endswith("/v1/traces"):  # the collector endpoint, as OTLP exporters take it
            self.url = self.url[: -len("/v1/traces")]

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": f"vectraudit/{__version__}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def post(self, path: str, params: Mapping[str, Any], payload: Any) -> int:
        """One POST of a JSON body; returns the status (200 or 404), raising on others.

        404 is returned, not raised: Phoenix answers it when a batch names traces it does
        not have, and the caller decides what to do. Response bodies are not echoed.
        """
        headers = {**self._headers(), "Content-Type": "application/json"}
        url = f"{self.url}{path}?{urlencode(params)}"
        self.calls += 1
        try:
            status, _ = (self.post_send or _urllib_post)(
                url, headers, json.dumps(payload).encode(), self.timeout_s
            )
        except urllib.error.HTTPError as exc:
            status = exc.code
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise PhoenixAPIError(f"Cannot reach Phoenix at {self.url}: {reason}") from exc
        if status in (200, 404):
            return status
        if status in (401, 403):
            raise PhoenixAPIError(
                f"Phoenix refused the request (HTTP {status}); check PHOENIX_API_KEY "
                "(writing annotations needs a key with write access)."
            )
        if status == 429:
            raise PhoenixAPIError("Phoenix rate limit reached (HTTP 429); try again later.")
        raise PhoenixAPIError(f"Phoenix answered HTTP {status} for {path}.")

    def get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """One GET returning a ``{"data": [...], "next_cursor": ...}`` object."""
        headers = self._headers()
        query = urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
        self.calls += 1
        try:
            status, raw = (self.send or _urllib_send)(
                f"{self.url}{path}?{query}", headers, self.timeout_s
            )
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, b""
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise PhoenixAPIError(f"Cannot reach Phoenix at {self.url}: {reason}") from exc
        if status in (401, 403):
            raise PhoenixAPIError(
                f"Phoenix refused the request (HTTP {status}); check PHOENIX_API_KEY."
            )
        if status == 404:
            raise PhoenixAPIError(
                f"Phoenix answered HTTP 404 for {path}; check the project name and "
                "PHOENIX_COLLECTOR_ENDPOINT (for Phoenix Cloud it ends in /s/SPACE)."
            )
        if status == 429:
            raise PhoenixAPIError("Phoenix rate limit reached (HTTP 429); pull a shorter window.")
        if status != 200:  # error bodies are not echoed: they can repeat request details
            raise PhoenixAPIError(f"Phoenix answered HTTP {status} for {path}.")
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise PhoenixAPIError(f"Phoenix returned non-JSON for {path}.") from exc
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise PhoenixAPIError(f"Unexpected Phoenix response shape for {path}.")
        return body


def _pages(
    source: PhoenixSource,
    path: str,
    params: Mapping[str, Any],
    page: int,
    max_records: int,
    count: list[int],
) -> Iterator[dict[str, Any]]:
    """Every record of a cursor-paged listing; ``count`` is shared across calls."""
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        body = source.get(path, {**params, "limit": page, "cursor": cursor})
        for record in body["data"]:
            if not isinstance(record, dict):
                raise PhoenixAPIError(f"Unexpected record in the {path} response.")
            count[0] += 1
            if count[0] > max_records:
                raise PhoenixAPIError(
                    f"More than {max_records} records in the window; pull a shorter window."
                )
            yield record
        cursor = body.get("next_cursor")
        if not isinstance(cursor, str) or not cursor or not body["data"] or cursor in seen:
            return
        seen.add(cursor)


def _project_path(project: str, tail: str) -> str:
    return f"/v1/projects/{quote(project, safe='')}/{tail}"


def _batches(ids: Sequence[str]) -> Iterator[list[str]]:
    for i in range(0, len(ids), ID_BATCH):
        yield list(ids[i : i + ID_BATCH])


def _span_ids(span: Mapping[str, Any]) -> tuple[str, str] | None:
    context = span.get("context")
    if not isinstance(context, Mapping):
        return None
    trace_id, span_id = context.get("trace_id"), context.get("span_id")
    if isinstance(trace_id, str) and trace_id and isinstance(span_id, str) and span_id:
        return trace_id, span_id
    return None


def fetch_spans(
    source: PhoenixSource,
    project: str,
    *,
    start: str,
    end: str,
    max_records: int = MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Every span of the traces whose root span started in ``[start, end)``."""
    since, until = _iso(start, "--since"), _iso(end, "--until")
    path = _project_path(project, "spans")
    count = [0]
    roots = {"start_time": since, "end_time": until, "parent_id": "null"}
    trace_ids: dict[str, None] = {}
    for span in _pages(source, path, roots, SPAN_PAGE, max_records, count):
        ids = _span_ids(span)
        if ids:
            trace_ids.setdefault(ids[0])
    spans: dict[str, dict[str, Any]] = {}
    for batch in _batches(list(trace_ids)):
        for span in _pages(source, path, {"trace_id": batch}, SPAN_PAGE, max_records, count):
            ids = _span_ids(span)
            if ids and ids[0] in trace_ids:
                spans.setdefault(ids[1], span)
    return sorted(
        spans.values(),
        key=lambda s: (s["context"]["trace_id"], str(s.get("start_time")), s["context"]["span_id"]),
    )


def fetch_annotations(
    source: PhoenixSource,
    project: str,
    spans: Sequence[Mapping[str, Any]],
    *,
    max_records: int = MAX_RECORDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(trace_annotations, span_annotations)`` for the pulled spans."""
    pairs = [ids for ids in (_span_ids(s) for s in spans) if ids]
    trace_ids = list(dict.fromkeys(t for t, _ in pairs))
    span_ids = list(dict.fromkeys(s for _, s in pairs))
    count = [0]
    traces: list[dict[str, Any]] = []
    path = _project_path(project, "trace_annotations")
    for batch in _batches(trace_ids):
        traces.extend(
            _pages(source, path, {"trace_ids": batch}, ANNOTATION_PAGE, max_records, count)
        )
    on_spans: list[dict[str, Any]] = []
    path = _project_path(project, "span_annotations")
    for batch in _batches(span_ids):
        on_spans.extend(
            _pages(source, path, {"span_ids": batch}, ANNOTATION_PAGE, max_records, count)
        )
    return traces, on_spans


def is_note(annotation: Mapping[str, Any]) -> bool:
    """Whether an annotation is a Phoenix note (free text), not a judgment."""
    identifier = annotation.get("identifier")
    return annotation.get("name") == NOTE_NAME or (
        isinstance(identifier, str) and identifier.startswith(NOTE_PREFIX)
    )


def annotations_to_judgments(
    trace_annotations: Sequence[Mapping[str, Any]],
    span_annotations: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Judgment lines for ``attach-answers``, keyed by trace ID, and why others were skipped.

    - A ``label`` of ``correct``, ``partial`` or ``incorrect`` (any case) is that verdict,
      with the score kept beside it; any other label is skipped, never guessed from its
      score (Phoenix evaluators score ``hallucinated`` and ``factual`` in either direction).
    - Without a label, a numeric ``score`` stays a score that needs a declared threshold.
    - Span annotations count for the span's trace. ``names`` keeps only those annotations.
    """
    trace_of = {ids[1]: ids[0] for ids in (_span_ids(s) for s in spans) if ids}
    wanted = set(names)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    owned: list[tuple[str | None, Mapping[str, Any]]] = []
    for a in trace_annotations:
        trace = a.get("trace_id")
        owned.append((trace if isinstance(trace, str) else None, a))
    for a in span_annotations:
        span = a.get("span_id")
        owned.append((trace_of.get(span) if isinstance(span, str) else None, a))
    for trace_id, annotation in owned:
        if not isinstance(annotation, Mapping):
            skipped["invalid_value"] += 1
            continue
        if is_note(annotation):
            skipped["notes"] += 1
            continue
        name = annotation.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            skipped["unnamed"] += 1
            continue
        if wanted and name not in wanted:
            skipped["not_chosen"] += 1
            continue
        if trace_id is None:
            skipped["span_not_pulled"] += 1
            continue
        kind = annotation.get("annotator_kind")
        row: dict[str, Any] = {
            "trace_id": trace_id,
            "evaluator": name,
            "kind": "human" if kind in HUMAN_KINDS else "model",
        }
        result = annotation.get("result")
        result = result if isinstance(result, Mapping) else {}
        label, score = result.get("label"), result.get("score")
        number = (
            float(score)
            if isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            else None
        )
        if isinstance(label, str) and label.strip():
            if label.strip().lower() not in VERDICTS:
                skipped["label_not_a_verdict"] += 1
                continue
            row["verdict"] = label.strip().lower()
        elif number is None:
            skipped["no_label_or_score"] += 1
            continue
        if number is not None:
            row["score"] = number
        rows.append(row)
    return rows, skipped
