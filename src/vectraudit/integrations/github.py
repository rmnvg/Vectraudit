"""Publish check results to GitHub from inside a GitHub Actions job.

Uses the job's ``GITHUB_TOKEN`` (permissions ``checks: write`` for check runs,
``pull-requests: write`` for comments) and the REST API; no GitHub App and no hosted
service. Pull requests from forks get a read-only token: publishing then fails with 403,
which callers report as a warning, never as a failed check.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vectraudit import __version__

COMMENT_MARKER = "<!-- vectraudit:retrieval-check -->"
MAX_OUTPUT = 65_000  # GitHub caps check-run summary and text at 65,535 characters
TIMEOUT_S = 15
_CONCLUSIONS = {
    "pass": "success",
    "fail": "failure",
    "error": "failure",
    "not_comparable": "failure",
}
_TITLES = {"pass": "Pass", "fail": "Fail", "error": "Error", "not_comparable": "Not comparable"}


class GitHubError(RuntimeError):
    """A GitHub API call failed; ``status`` is the HTTP status when there was one."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def forbidden(self) -> bool:
        return self.status in (401, 403, 404)


@dataclass(frozen=True)
class GitHubContext:
    """Where to publish: repository, commit, pull request and credentials."""

    token: str
    repository: str
    head_sha: str
    pull_request: int | None
    api_url: str = "https://api.github.com"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GitHubContext:
        """Read the Actions environment; the head SHA of a pull request, not the merge SHA."""
        env = os.environ if env is None else env
        token = env.get("GITHUB_TOKEN") or env.get("GH_TOKEN")
        repository = env.get("GITHUB_REPOSITORY")
        if not token:
            raise GitHubError("GITHUB_TOKEN is not set: pass it to the step's env.")
        if not repository:
            raise GitHubError("GITHUB_REPOSITORY is not set: run inside GitHub Actions.")
        event: dict[str, Any] = {}
        path = env.get("GITHUB_EVENT_PATH")
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    event = json.load(f)
            except (OSError, ValueError):
                event = {}
        pr = event.get("pull_request") or {}
        sha = (pr.get("head") or {}).get("sha") or env.get("GITHUB_SHA")
        if not sha:
            raise GitHubError("Cannot tell which commit to publish to (GITHUB_SHA is not set).")
        number = pr.get("number")
        return cls(
            token=token,
            repository=repository,
            head_sha=sha,
            pull_request=int(number) if isinstance(number, int) else None,
            api_url=env.get("GITHUB_API_URL", "https://api.github.com").rstrip("/"),
        )


def _request(ctx: GitHubContext, method: str, path: str, body: Any = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{ctx.api_url}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {ctx.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"vectraudit/{__version__}",
            **({"Content-Type": "application/json"} if data is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        raise GitHubError(f"GitHub {method} {path} failed (HTTP {e.code}): {detail}", e.code) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise GitHubError(f"Cannot reach the GitHub API: {e}") from e
    return json.loads(raw) if raw else None


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[: MAX_OUTPUT - 60] + "\n\n…truncated; see the full report.\n"


def check_run_payload(
    result: Mapping[str, Any],
    markdown: str,
    *,
    head_sha: str,
    name: str = "Retrieval check",
    details_url: str | None = None,
) -> dict[str, Any]:
    """The REST body for a completed check run (exposed for tests and dry runs)."""
    verdict = result["verdict"]
    reasons = [r["message"] for r in result["reasons"]]
    title = _TITLES[verdict]
    if reasons:
        title += f": {reasons[0]}"
    payload: dict[str, Any] = {
        "name": name,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": _CONCLUSIONS[verdict],
        "output": {
            "title": title[:255],
            "summary": _clip(markdown),
        },
    }
    if details_url:
        payload["details_url"] = details_url
    return payload


def publish_check_run(
    ctx: GitHubContext,
    result: Mapping[str, Any],
    markdown: str,
    *,
    name: str = "Retrieval check",
    details_url: str | None = None,
) -> str | None:
    """Create a completed check run on the head commit; returns its URL."""
    payload = check_run_payload(
        result, markdown, head_sha=ctx.head_sha, name=name, details_url=details_url
    )
    response = _request(ctx, "POST", f"/repos/{ctx.repository}/check-runs", payload)
    return (response or {}).get("html_url")


def upsert_pr_comment(ctx: GitHubContext, markdown: str) -> str | None:
    """Create or update the single vectraudit comment on the pull request; returns its URL."""
    if ctx.pull_request is None:
        raise GitHubError("Not a pull request event: there is no pull request to comment on.")
    body = f"{COMMENT_MARKER}\n{_clip(markdown)}"
    base = f"/repos/{ctx.repository}/issues"
    existing = None
    for page in range(1, 11):  # at most 1,000 comments scanned
        comments = _request(
            ctx, "GET", f"{base}/{ctx.pull_request}/comments?per_page=100&page={page}"
        )
        existing = next(
            (c for c in comments or [] if COMMENT_MARKER in (c.get("body") or "")), None
        )
        if existing is not None or not comments or len(comments) < 100:
            break
    if existing is not None:
        response = _request(ctx, "PATCH", f"{base}/comments/{existing['id']}", {"body": body})
    else:
        response = _request(ctx, "POST", f"{base}/{ctx.pull_request}/comments", {"body": body})
    return (response or {}).get("html_url")
