"""Pull runs and feedback from the LangSmith API.

The LangSmith counterpart of :mod:`vectraudit.pipelines.langfuse_api`: it fetches the run
records ``import-traces`` and ``intake`` already read from exports (Step 64), and turns
LangSmith feedback into the judgment lines ``attach-answers`` reads.

Requests follow the official LangSmith SDK (``langsmith.Client``):

- ``GET /sessions?name=PROJECT`` resolves the tracing project to its ID.
- ``POST /runs/query`` with ``session``, ``trace_filter``, ``select`` and ``limit``,
  cursor-paged through ``cursors.next``. The time window applies to each trace's **root**
  run (``trace_filter``), so every run of a trace that started in the window is returned,
  including children that started after it ends: a trace is never split between pulls.
- ``GET /feedback?run=ID…``, offset-paged, for the pulled runs.

Authentication is the ``x-api-key`` header. Nothing is written to LangSmith. Feedback
comments and corrections arrive in the response but are never written anywhere.
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
from urllib.parse import urlencode, urlsplit

from vectraudit import __version__
from vectraudit.pipelines.langfuse_api import LOCAL_HOSTS, MAX_RECORDS, VERDICTS, _iso

Send = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]
RUN_FIELDS = (
    "id", "trace_id", "parent_run_id", "name", "run_type", "start_time", "end_time",
    "inputs", "outputs", "extra", "error",
)  # fmt: skip
RUN_PAGE = 100
FEEDBACK_PAGE = 100
FEEDBACK_BATCH = 50  # run IDs per feedback request, to keep URLs short


class LangSmithAPIError(ValueError):
    """The LangSmith API could not be read; the message says why and what to change."""


def _urllib_send(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read()


@dataclass
class LangSmithSource:
    """One LangSmith workspace, addressed by its API URL and key."""

    api_url: str
    api_key: str
    timeout_s: float = 30.0
    send: Send | None = None
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        parts = urlsplit(self.api_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise LangSmithAPIError(f"Invalid LangSmith endpoint {self.api_url!r}.")
        if parts.username or parts.password or parts.query:
            raise LangSmithAPIError(
                "The LangSmith endpoint must not contain credentials or a query."
            )
        if parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
            raise LangSmithAPIError(
                f"Refusing to send the LangSmith key over http:// to {parts.hostname}; "
                "use https://."
            )
        if not self.api_key:
            raise LangSmithAPIError("The LangSmith key is missing: set LANGSMITH_API_KEY.")
        self.api_url = self.api_url.rstrip("/")

    def request(self, method: str, path: str, *, params: Any = None, body: Any = None) -> Any:
        """One authenticated request returning parsed JSON."""
        url = self.api_url + path + (f"?{urlencode(params, doseq=True)}" if params else "")
        headers = {
            "Accept": "application/json",
            "x-api-key": self.api_key,
            "User-Agent": f"vectraudit/{__version__}",
        }
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        self.calls += 1
        try:
            status, raw = (self.send or _urllib_send)(method, url, headers, data, self.timeout_s)
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, b""
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise LangSmithAPIError(f"Cannot reach LangSmith at {self.api_url}: {reason}") from exc
        if status in (401, 403):
            raise LangSmithAPIError(
                f"LangSmith refused the key (HTTP {status}); check LANGSMITH_API_KEY and, "
                "outside the US region, LANGSMITH_ENDPOINT."
            )
        if status == 429:
            raise LangSmithAPIError(
                "LangSmith rate limit reached (HTTP 429); pull a shorter window or try later."
            )
        if status != 200:  # error bodies are not echoed: they can repeat request details
            raise LangSmithAPIError(f"LangSmith answered HTTP {status} for {path}.")
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise LangSmithAPIError(f"LangSmith returned non-JSON for {path}.") from exc


def project_id(source: LangSmithSource, name: str) -> str:
    """The ID of the tracing project named exactly ``name``."""
    found = source.request("GET", "/sessions", params={"name": name, "limit": 1})
    if not isinstance(found, list):
        raise LangSmithAPIError("Unexpected LangSmith response shape for /sessions.")
    for project in found:
        if isinstance(project, Mapping) and project.get("name") == name and project.get("id"):
            return str(project["id"])
    raise LangSmithAPIError(f"No LangSmith project named {name!r} for this key.")


def _quoted(value: str) -> str:
    return json.dumps(value)  # LangSmith filter strings use double-quoted literals


def fetch_runs(
    source: LangSmithSource,
    project: str,
    *,
    start: str,
    end: str,
    max_records: int = MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Every run of the traces whose root run started in ``[start, end)``.

    Runs come back as a LangSmith export holds them; repeated runs are kept once.
    """
    since, until = _iso(start, "--since"), _iso(end, "--until")
    body: dict[str, Any] = {
        "session": [project_id(source, project)],
        "trace_filter": f"and(gte(start_time, {_quoted(since)}), lt(start_time, {_quoted(until)}))",
        "select": list(RUN_FIELDS),
        "limit": RUN_PAGE,
    }
    unique: dict[str, dict[str, Any]] = {}
    for run in _cursor_pages(source, body, max_records):
        if run.get("id") and run.get("trace_id"):
            unique.setdefault(str(run["id"]), run)
    return sorted(
        unique.values(),
        key=lambda r: (str(r["trace_id"]), str(r.get("start_time")), str(r["id"])),
    )


def _cursor_pages(
    source: LangSmithSource, body: dict[str, Any], max_records: int
) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    count = 0
    while True:
        page = source.request("POST", "/runs/query", body=body)
        runs = page.get("runs") if isinstance(page, Mapping) else None
        if not isinstance(runs, list):
            raise LangSmithAPIError("Unexpected LangSmith response shape for /runs/query.")
        for run in runs:
            if not isinstance(run, dict):
                raise LangSmithAPIError("Unexpected record in the /runs/query response.")
            count += 1
            if count > max_records:
                raise LangSmithAPIError(
                    f"More than {max_records} runs in the window; pull a shorter window."
                )
            yield run
        cursors = page.get("cursors") if isinstance(page.get("cursors"), Mapping) else {}
        cursor = cursors.get("next") if isinstance(cursors, Mapping) else None
        if not runs or not cursor:
            return
        if cursor in seen:
            raise LangSmithAPIError("LangSmith repeated a pagination cursor for /runs/query.")
        seen.add(cursor)
        body = {**body, "cursor": cursor}


def fetch_feedback(
    source: LangSmithSource, run_ids: Sequence[str], *, max_records: int = MAX_RECORDS
) -> list[dict[str, Any]]:
    """Feedback on the given runs, without duplicates."""
    unique: dict[str, dict[str, Any]] = {}
    ids = sorted(set(run_ids))
    for i in range(0, len(ids), FEEDBACK_BATCH):
        batch = ids[i : i + FEEDBACK_BATCH]
        offset = 0
        while True:
            page = source.request(
                "GET",
                "/feedback",
                params={"run": batch, "limit": FEEDBACK_PAGE, "offset": offset},
            )
            if not isinstance(page, list):
                raise LangSmithAPIError("Unexpected LangSmith response shape for /feedback.")
            for item in page:
                if not isinstance(item, dict):
                    raise LangSmithAPIError("Unexpected record in the /feedback response.")
                unique.setdefault(str(item.get("id")), item)
                if len(unique) > max_records:
                    raise LangSmithAPIError(
                        f"More than {max_records} feedback records; pull a shorter window."
                    )
            if len(page) < FEEDBACK_PAGE:
                break
            offset += len(page)
    return list(unique.values())


def feedback_to_judgments(
    feedback: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    *,
    human_sources: Sequence[str] = ("app",),
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Judgment lines for ``attach-answers``, keyed by trace ID, and why others were skipped.

    - A ``value`` of ``correct``, ``partial`` or ``incorrect`` (any case) is that verdict.
    - Otherwise a boolean ``score`` is ``correct`` or ``incorrect``, and a numeric
      ``score`` stays a score that needs a declared threshold.
    - Anything else (free-text or other categorical values, no score) is skipped, never
      guessed. So is feedback on a run outside the pulled traces.

    Feedback whose source type is in ``human_sources`` (default ``app``, the LangSmith
    interface and annotation queues) is human review; ``api`` and ``model`` feedback are
    model judgments. The feedback key is the evaluator.
    """
    humans = {s.lower() for s in human_sources}
    trace_of = {str(r["id"]): str(r["trace_id"]) for r in runs if r.get("id")}
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for item in feedback:
        trace = trace_of.get(str(item.get("run_id")))
        if trace is None:
            skipped["run_not_pulled"] += 1
            continue
        key = item.get("key")
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            skipped["unnamed"] += 1
            continue
        source = item.get("feedback_source")
        kind = source.get("type") if isinstance(source, Mapping) else None
        row: dict[str, Any] = {
            "trace_id": trace,
            "evaluator": key,
            "kind": "human" if str(kind or "").lower() in humans else "model",
        }
        value, score = item.get("value"), item.get("score")
        if isinstance(value, str) and value.strip().lower() in VERDICTS:
            row["verdict"] = value.strip().lower()
        elif isinstance(score, bool):
            row["verdict"] = "correct" if score else "incorrect"
        elif isinstance(score, (int, float)) and math.isfinite(score):  # bool handled above
            row["score"] = float(score)
        elif isinstance(value, str) and value.strip():
            skipped["value_not_a_verdict"] += 1
            continue
        else:
            skipped["no_score"] += 1
            continue
        rows.append(row)
    return rows, skipped
