"""GitHub check runs and PR comments, against a local mock of the REST API."""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit.cli import app
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.runfile import save_run
from vectraudit.integrations.github import (
    COMMENT_MARKER,
    MAX_OUTPUT,
    GitHubContext,
    GitHubError,
    check_run_payload,
    publish_check_run,
    upsert_pr_comment,
)


class MockGitHub(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict[str, Any]]] = []
    comments: ClassVar[list[dict[str, Any]]] = []
    status: ClassVar[int] = 200

    def _reply(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        self.requests.append(
            {
                "method": method,
                "path": self.path,
                "body": body,
                "auth": self.headers.get("Authorization"),
            }
        )
        if self.status != 200:
            self._reply(self.status, {"message": "Resource not accessible by integration"})
        elif self.path.endswith("/check-runs"):
            self._reply(201, {"html_url": "https://github.test/checks/1"})
        elif method == "GET" and "/comments" in self.path:
            self._reply(200, self.comments)
        elif method == "POST" and self.path.endswith("/comments"):
            comment = {
                "id": len(self.comments) + 1,
                "body": body["body"],
                "html_url": "https://github.test/c/new",
            }
            self.comments.append(comment)
            self._reply(201, comment)
        elif method == "PATCH":
            cid = int(self.path.rsplit("/", 1)[1])
            self.comments[cid - 1]["body"] = body["body"]
            self._reply(200, {"html_url": f"https://github.test/c/{cid}"})
        else:
            self._reply(404, {"message": "Not Found"})

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_PATCH(self) -> None:
        self._handle("PATCH")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def github(tmp_path, monkeypatch) -> Iterator[type[MockGitHub]]:
    MockGitHub.requests, MockGitHub.comments, MockGitHub.status = [], [], 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockGitHub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7, "head": {"sha": "headsha"}}}))
    for key, value in {
        "GITHUB_TOKEN": "ghs_test",
        "GITHUB_REPOSITORY": "acme/rag",
        "GITHUB_SHA": "mergesha",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_API_URL": f"http://127.0.0.1:{server.server_port}",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    # The code also accepts the gh CLI's GH_TOKEN; a developer's or runner's own must not
    # leak into tests that remove GITHUB_TOKEN to check the "not set" path.
    monkeypatch.delenv("GH_TOKEN", raising=False)
    yield MockGitHub
    server.shutdown()


def _result(verdict_values: list[float]) -> dict[str, Any]:
    return compare_runs(fake_run([1.0] * 40), fake_run(verdict_values))


def test_context_from_environment(github, tmp_path) -> None:
    ctx = GitHubContext.from_env()
    assert ctx.head_sha == "headsha"
    assert ctx.pull_request == 7
    assert ctx.repository == "acme/rag"
    push = GitHubContext.from_env(
        {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "a/b", "GITHUB_SHA": "pushsha"}
    )
    assert push.head_sha == "pushsha"
    assert push.pull_request is None
    assert push.api_url == "https://api.github.com"
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        GitHubContext.from_env({"GITHUB_REPOSITORY": "a/b"})
    with pytest.raises(GitHubError, match="GITHUB_REPOSITORY"):
        GitHubContext.from_env({"GITHUB_TOKEN": "t"})


def test_check_run_payload() -> None:
    passing = check_run_payload(_result([1.0] * 40), "summary", head_sha="s", details_url="u")
    assert passing["conclusion"] == "success"
    assert passing["output"]["title"] == "Pass"
    assert passing["details_url"] == "u"
    failing = check_run_payload(_result([0.0] * 40), "x" * (MAX_OUTPUT + 10), head_sha="s")
    assert failing["conclusion"] == "failure"
    assert failing["output"]["title"].startswith("Fail: ndcg dropped")
    assert len(failing["output"]["summary"]) <= MAX_OUTPUT
    assert "truncated" in failing["output"]["summary"]
    run = fake_run([1.0] * 5)
    broken = copy.deepcopy(run)
    broken["status"] = "failed"
    payload = check_run_payload(compare_runs(run, broken), "s", head_sha="s")
    assert payload["conclusion"] == "failure"
    assert payload["output"]["title"].startswith("Error")


def test_publish_check_run_and_comment(github) -> None:
    ctx = GitHubContext.from_env()
    result = _result([0.0] * 40)
    url = publish_check_run(ctx, result, "## Fail", name="Retrieval check (qdrant)")
    assert url == "https://github.test/checks/1"
    req = github.requests[0]
    assert req["path"] == "/repos/acme/rag/check-runs"
    assert req["auth"] == "Bearer ghs_test"
    assert req["body"]["head_sha"] == "headsha"
    assert req["body"]["name"] == "Retrieval check (qdrant)"

    assert upsert_pr_comment(ctx, "first") == "https://github.test/c/new"
    assert upsert_pr_comment(ctx, "second") == "https://github.test/c/1"
    assert len(github.comments) == 1
    assert github.comments[0]["body"] == f"{COMMENT_MARKER}\nsecond"
    methods = [r["method"] for r in github.requests[1:]]
    assert methods == ["GET", "POST", "GET", "PATCH"]


def test_forbidden_and_non_pr(github) -> None:
    github.status = 403
    ctx = GitHubContext.from_env()
    with pytest.raises(GitHubError) as e:
        publish_check_run(ctx, _result([1.0] * 40), "s")
    assert e.value.forbidden
    push = GitHubContext("t", "a/b", "sha", None, ctx.api_url)
    with pytest.raises(GitHubError, match="Not a pull request"):
        upsert_pr_comment(push, "x")


def _runs(tmp_path: Path, cand_values: list[float]) -> tuple[Path, Path]:
    base = save_run(fake_run([1.0] * 40), tmp_path / "base.json")
    cand = save_run(fake_run(cand_values), tmp_path / "cand.json")
    return base, cand


def test_cli_publishes(github, tmp_path) -> None:
    base, cand = _runs(tmp_path, [0.0] * 40)
    result = CliRunner().invoke(
        app,
        [
            "check",
            str(base),
            str(cand),
            "--github-check",
            "--github-comment",
            "--report-url",
            "https://ci.test/report",
        ],
    )
    assert result.exit_code == 3, result.output
    assert "Published check run: https://github.test/checks/1" in result.output
    assert "Published pull request comment" in result.output
    check = github.requests[0]["body"]
    assert check["details_url"] == "https://ci.test/report"
    assert "[Full report](https://ci.test/report)" in check["output"]["summary"]


def test_cli_publishing_failures_are_warnings(github, tmp_path, monkeypatch) -> None:
    base, cand = _runs(tmp_path, [1.0] * 40)
    github.status = 403
    result = CliRunner().invoke(app, ["check", str(base), str(cand), "--github-check"])
    assert result.exit_code == 0, result.output
    assert "forks can't publish" in " ".join(result.output.split())
    monkeypatch.delenv("GITHUB_TOKEN")
    result = CliRunner().invoke(app, ["check", str(base), str(cand), "--github-comment"])
    assert result.exit_code == 0
    assert "not published to GitHub: GITHUB_TOKEN is not set" in " ".join(result.output.split())
