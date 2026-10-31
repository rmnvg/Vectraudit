"""Pull observations and scores from the Langfuse public API.

The trace importer and production intake read Langfuse observations from files (Step 64).
This module fetches the same records straight from a Langfuse project, so a scheduled
``intake`` or ``import-traces`` no longer needs someone to export a file first, and it turns
Langfuse scores into the judgment lines ``attach-answers`` reads.

Endpoints (from Langfuse's published OpenAPI specification):

- ``GET /api/public/v2/observations``: cursor pagination, at most 1,000 per page, filtered by
  ``fromStartTime`` (inclusive) and ``toStartTime``. The ``io`` and ``metadata`` field groups
  are requested because retriever documents and prompts live in input and output.
- ``GET /api/public/v3/scores``: cursor pagination, at most 100 per page, filtered by
  ``fromTimestamp`` (inclusive) and ``toTimestamp`` (exclusive), with the ``subject`` group so
  each score names its trace.

Authentication is HTTP Basic with the project's public and secret keys, the same variables
the Langfuse SDKs read. Nothing is written back to Langfuse. Score comments are never
requested, so reviewers' free text does not reach Vectraudit.

Langfuse notes that endpoints other than v2 observations can lag by about ten minutes, so a
score window that ends "now" may miss the newest scores; pull them again later.
"""

from __future__ import annotations

import base64
import json
import math
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from vectraudit import __version__

Send = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
OBSERVATION_FIELDS = "core,basic,io,metadata"
OBSERVATION_PAGE = 1000
SCORE_PAGE = 100
MAX_RECORDS = 200_000
VERDICTS = ("correct", "partial", "incorrect")


class LangfuseAPIError(ValueError):
    """The Langfuse API could not be read; the message says why and what to change."""


def _urllib_send(url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET", headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


def _iso(value: str, what: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LangfuseAPIError(f"{what} must be an ISO 8601 time, got {value!r}") from exc
    if parsed.tzinfo is None:
        raise LangfuseAPIError(f"{what} needs a time zone, for example {value}Z")
    return parsed.isoformat().replace("+00:00", "Z")


@dataclass
class LangfuseSource:
    """One Langfuse project, addressed by its host and API keys."""

    host: str
    public_key: str
    secret_key: str
    timeout_s: float = 30.0
    send: Send | None = None
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        parts = urlsplit(self.host)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise LangfuseAPIError(f"Invalid Langfuse host {self.host!r}.")
        if parts.username or parts.password or parts.query:
            raise LangfuseAPIError("The Langfuse host must not contain credentials or a query.")
        if parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
            raise LangfuseAPIError(
                f"Refusing to send Langfuse keys over http:// to {parts.hostname}; use https://."
            )
        if not self.public_key or not self.secret_key:
            raise LangfuseAPIError(
                "Langfuse keys are missing: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
            )
        self.host = self.host.rstrip("/")

    def get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """One authenticated GET returning a JSON object."""
        query = urlencode({k: v for k, v in params.items() if v is not None})
        token = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {token}",
            "User-Agent": f"vectraudit/{__version__}",
        }
        self.calls += 1
        try:
            status, raw = (self.send or _urllib_send)(
                f"{self.host}{path}?{query}", headers, self.timeout_s
            )
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, b""
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise LangfuseAPIError(f"Cannot reach Langfuse at {self.host}: {reason}") from exc
        if status in (401, 403):
            raise LangfuseAPIError(
                f"Langfuse refused the keys (HTTP {status}); check the project's public and "
                "secret keys and the host's region."
            )
        if status == 429:
            raise LangfuseAPIError(
                "Langfuse rate limit reached (HTTP 429); pull a shorter window or try later."
            )
        if status != 200:  # error bodies are not echoed: they can repeat request details
            raise LangfuseAPIError(f"Langfuse answered HTTP {status} for {path}.")
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise LangfuseAPIError(f"Langfuse returned non-JSON for {path}.") from exc
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise LangfuseAPIError(f"Unexpected Langfuse response shape for {path}.")
        return body


def _pages(
    source: LangfuseSource, path: str, params: Mapping[str, Any], max_records: int
) -> Iterator[dict[str, Any]]:
    cursor: str | None = None
    seen: set[str] = set()
    count = 0
    while True:
        body = source.get(path, {**params, "cursor": cursor})
        for record in body["data"]:
            if not isinstance(record, dict):
                raise LangfuseAPIError(f"Unexpected record in the {path} response.")
            count += 1
            if count > max_records:
                raise LangfuseAPIError(
                    f"More than {max_records} records in the window; pull a shorter window."
                )
            yield record
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        cursor = meta.get("cursor") if isinstance(meta, dict) else None
        if not cursor:
            return
        if cursor in seen:
            raise LangfuseAPIError(f"Langfuse repeated a pagination cursor for {path}.")
        seen.add(cursor)


def fetch_observations(
    source: LangfuseSource,
    *,
    start: str,
    end: str,
    environments: Sequence[str] = (),
    max_records: int = MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Observations that started in ``[start, end)``, ordered by trace and start time.

    Input and output arrive as raw strings, as in a Langfuse export; the trace importer
    parses them. Duplicate observation IDs (a record repeated across pages) are kept once.
    """
    params: dict[str, Any] = {
        "fields": OBSERVATION_FIELDS,
        "limit": OBSERVATION_PAGE,
        "fromStartTime": _iso(start, "--since"),
        "toStartTime": _iso(end, "--until"),
    }
    if environments:
        params["environment"] = list(environments)
    unique: dict[str, dict[str, Any]] = {}
    for record in _pages(source, "/api/public/v2/observations", params, max_records):
        if not record.get("traceId") or not record.get("id"):
            continue  # an observation outside any trace cannot be imported
        unique.setdefault(str(record["id"]), record)
    return sorted(
        unique.values(), key=lambda r: (str(r["traceId"]), str(r.get("startTime")), str(r["id"]))
    )


def fetch_scores(
    source: LangfuseSource,
    *,
    start: str,
    end: str,
    names: Sequence[str] = (),
    max_records: int = MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Scores timestamped in ``[start, end)``, with their subject and without comments."""
    params: dict[str, Any] = {
        "fields": "subject",
        "limit": SCORE_PAGE,
        "fromTimestamp": _iso(start, "--since"),
        "toTimestamp": _iso(end, "--until"),
    }
    if names:
        params["name"] = ",".join(names)
    return list(_pages(source, "/api/public/v3/scores", params, max_records))


def scores_to_judgments(
    scores: Sequence[Mapping[str, Any]], *, human_sources: Sequence[str] = ("ANNOTATION",)
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Judgment lines for ``attach-answers``, keyed by trace ID, and why others were skipped.

    - ``BOOLEAN``: true is ``correct``, false is ``incorrect``.
    - ``NUMERIC``: kept as a score; ``attach-answers`` needs a declared threshold for it.
    - ``CATEGORICAL``: used only when the category is ``correct``, ``partial`` or
      ``incorrect`` (any case); other categories are skipped, never guessed.
    - ``TEXT`` and ``CORRECTION`` scores carry free text and are skipped.
    - Session and experiment scores name no trace and are skipped.

    Annotation scores (``human_sources``) are human judgments; API and evaluator scores are
    model judgments. The score name is the evaluator and its config ID, when present, the
    rubric.
    """
    humans = {s.upper() for s in human_sources}
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for score in scores:
        subject = score.get("subject") if isinstance(score.get("subject"), Mapping) else {}
        assert isinstance(subject, Mapping)
        kind = subject.get("kind")
        trace_id = (
            subject.get("id")
            if kind == "trace"
            else subject.get("traceId")
            if kind == "observation"
            else None
        )
        if not isinstance(trace_id, str) or not trace_id:
            skipped[f"{kind or 'unknown'}_subject"] += 1
            continue
        name = score.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            skipped["unnamed"] += 1
            continue
        data_type = str(score.get("dataType") or "").upper()
        value = score.get("value")
        row: dict[str, Any] = {
            "trace_id": trace_id,
            "evaluator": name,
            "kind": "human" if str(score.get("source") or "").upper() in humans else "model",
        }
        config = score.get("configId")
        if isinstance(config, str) and config and len(config) <= 200:
            row["rubric"] = config
        if data_type == "BOOLEAN" and isinstance(value, bool):
            row["verdict"] = "correct" if value else "incorrect"
        elif (
            data_type == "NUMERIC" and type(value) in (int, float) and math.isfinite(value)  # type: ignore[arg-type]
        ):
            row["score"] = float(value)  # type: ignore[arg-type]
        elif data_type == "CATEGORICAL" and isinstance(value, str):
            if value.strip().lower() not in VERDICTS:
                skipped["category_not_a_verdict"] += 1
                continue
            row["verdict"] = value.strip().lower()
        elif data_type in ("TEXT", "CORRECTION"):
            skipped["free_text"] += 1
            continue
        else:
            skipped["invalid_value"] += 1
            continue
        rows.append(row)
    return rows, skipped


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """One JSON object per line, written atomically."""
    from vectraudit.labelling.task import atomic_write

    atomic_write(
        path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    )
