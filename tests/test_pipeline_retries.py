"""Automatic retries of transient pipeline failures, within an explicit budget."""

import json
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vectraudit.core.jobs import JobCancelledError
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.runfile import check_run
from vectraudit.pipelines.errors import RETRYABLE, PipelineCallError
from vectraudit.pipelines.models import Hit, PipelineRequest, PipelineResult
from vectraudit.pipelines.runner import (
    NO_RETRY,
    PipelineOptions,
    RetryPolicy,
    execute_pipeline,
)
from vectraudit.pipelines.runtime import HTTPPipeline, retry_after


@pytest.fixture
def clock(monkeypatch):
    """A fake clock: backoff waits advance time instantly and are recorded."""

    class Clock:
        def __init__(self):
            self.now = 0.0
            self.slept: list[float] = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.slept.append(seconds)
            self.now += seconds

    fake = Clock()
    monkeypatch.setattr("vectraudit.pipelines.runner.time.monotonic", fake.monotonic)
    monkeypatch.setattr("vectraudit.pipelines.runner.time.sleep", fake.sleep)
    return fake


def dataset(n=1, **fields):
    return parse_dataset(
        "\n".join(
            json.dumps({"query_id": str(i), "text": "invoice", "relevant": {"a": 1}, **fields})
            for i in range(n)
        )
    )


class Flaky:
    """Fails each query's first ``failures`` calls with ``code``, then answers."""

    def __init__(self, code="timeout", failures=1, retry_after=None):
        self.code, self.failures, self.retry_after = code, failures, retry_after
        self.calls: list[str] = []

    def __call__(self, request):
        self.calls.append(request.query_id)
        if self.calls.count(request.query_id) <= self.failures:
            if self.code == "exception":
                raise RuntimeError("private")
            raise PipelineCallError(self.code, "private", retry_after=self.retry_after)
        return PipelineResult(hits=[Hit(id="a")])


def options(**retry):
    return PipelineOptions(warmup=0, retry=RetryPolicy(**retry))


@pytest.mark.parametrize("code", sorted(RETRYABLE))
def test_transient_failures_recover_and_are_recorded(clock, code):
    pipe = Flaky(code, failures=2)
    run = check_run(execute_pipeline(dataset(), pipe, options()))
    assert run["status"] == "complete"
    assert pipe.calls == ["0", "0", "0"]
    query = run["queries"][0]
    assert query["error"] is None
    assert query["retries"] == [
        {"call": "repeat 1", "code": code, "delay_s": 1.0},
        {"call": "repeat 1", "code": code, "delay_s": 2.0},
    ]
    assert run["summary"]["retries_by_code"] == {code: 2}
    assert run["summary"]["errors_by_code"] == {}
    assert sum(clock.slept) == pytest.approx(3.0)
    assert run["timing"]["retry"]["max_attempts"] == 3
    assert "private" not in json.dumps(run)
    assert compare_runs(run, run)["verdict"] == "pass"


@pytest.mark.parametrize("code", ["http_auth", "http_not_found", "invalid_response", "exception"])
def test_non_transient_failures_are_not_retried(clock, code):
    pipe = Flaky(code, failures=1)
    run = execute_pipeline(dataset(), pipe, options())
    assert pipe.calls == ["0"]
    assert run["status"] == "partial"
    assert "retries" not in run["queries"][0]
    assert run["summary"]["retries_by_code"] == {}
    assert clock.slept == []


def test_exhausted_attempts_keep_the_query_failed_with_its_cause(clock):
    pipe = Flaky("http_unavailable", failures=5)
    run = execute_pipeline(dataset(), pipe, options(max_attempts=3))
    assert pipe.calls == ["0"] * 3
    query = run["queries"][0]
    assert query["error"]["code"] == "http_unavailable"
    assert len(query["retries"]) == 2
    assert run["status"] == "partial"
    assert compare_runs(run, run)["verdict"] == "error"


def test_backoff_is_exponential_and_capped(clock):
    policy = options(max_attempts=6, base_delay_s=2, max_delay_s=10)
    run = execute_pipeline(dataset(), Flaky("timeout", failures=5), policy)
    assert [r["delay_s"] for r in run["queries"][0]["retries"]] == [2.0, 4.0, 8.0, 10.0, 10.0]
    assert sum(clock.slept) == pytest.approx(34.0)
    assert run["status"] == "complete"


def test_retry_after_is_honoured_and_over_long_requests_fail(clock):
    honoured = execute_pipeline(dataset(), Flaky("http_rate_limit", retry_after=7), options())
    assert honoured["queries"][0]["retries"][0]["delay_s"] == 7.0
    assert honoured["status"] == "complete"
    pipe = Flaky("http_rate_limit", retry_after=120)
    refused = execute_pipeline(dataset(), pipe, options(max_delay_s=30))
    assert pipe.calls == ["0"]
    assert refused["queries"][0]["error"]["code"] == "http_rate_limit"
    assert "retries" not in refused["queries"][0]
    assert sum(clock.slept) == pytest.approx(7.0)


def test_run_budget_bounds_retries_across_queries(clock):
    pipe = Flaky("connection_failed", failures=1)
    run = execute_pipeline(dataset(3), pipe, options(max_retries=1))
    assert run["queries"][0]["error"] is None
    assert [q["error"]["code"] for q in run["queries"][1:]] == ["connection_failed"] * 2
    assert run["summary"]["retries_by_code"] == {"connection_failed": 1}
    assert pipe.calls == ["0", "0", "1", "2"]


def test_no_retry_policy_and_plain_run_options_disable_retries(clock):
    pipe = Flaky("timeout")
    run = execute_pipeline(dataset(), pipe, PipelineOptions(warmup=0, retry=NO_RETRY))
    assert pipe.calls == ["0"]
    assert run["status"] == "partial"
    assert run["timing"]["retry"]["max_attempts"] == 1


def test_repeats_and_warmup_retry_each_call(clock):
    calls = []

    def pipe(request):
        calls.append(request.query_id)
        if len(calls) in (1, 3):
            raise PipelineCallError("timeout", "private")
        return PipelineResult(hits=[Hit(id="a")])

    run = execute_pipeline(
        dataset(), pipe, PipelineOptions(warmup=1, repeats=2, retry=RetryPolicy())
    )
    query = run["queries"][0]
    assert [r["call"] for r in query["retries"]] == ["warmup", "repeat 1"]
    assert query["repeat_ids"] == [["a"], ["a"]]
    assert len(query["latency_samples_ms"]) == 2


def test_cancellation_is_noticed_during_backoff(clock):
    def progress(done, total):
        if clock.now > 0:
            raise JobCancelledError("cancelled")

    run = execute_pipeline(
        dataset(), Flaky("timeout", failures=5), options(base_delay_s=30), progress=progress
    )
    assert run["status"] == "cancelled"
    assert clock.now < 30


def test_retries_survive_checkpoint_resume(clock, tmp_path):
    path = tmp_path / "run.sqlite"
    common = {"config": {"pipeline_version": "v1"}, "corpus_version": "v1", "checkpoint": path}
    first = execute_pipeline(dataset(), Flaky("timeout"), options(), **common)
    again = execute_pipeline(dataset(), Flaky("timeout"), options(), resume=True, **common)
    assert first["queries"][0]["retries"] == again["queries"][0]["retries"]
    assert again["summary"]["retries_by_code"] == {"timeout": 1}


def test_policy_validation():
    for bad in (
        {"max_attempts": 0},
        {"max_attempts": 11},
        {"base_delay_s": 5, "max_delay_s": 1},
        {"max_delay_s": 301},
        {"max_retries": -1},
    ):
        with pytest.raises(ValueError, match=r"max_attempts|delay|max_retries"):
            RetryPolicy(**bad)


def test_retry_after_parsing():
    assert retry_after(None) is None
    assert retry_after("") is None
    assert retry_after(" 12 ") == 12.0
    assert retry_after("soon") is None
    future = datetime.now(timezone.utc) + timedelta(seconds=60)
    assert 50 <= retry_after(format_datetime(future, usegmt=True)) <= 61
    past = datetime.now(timezone.utc) - timedelta(seconds=60)
    assert retry_after(format_datetime(past, usegmt=True)) == 0.0


def test_http_503_with_retry_after_recovers(clock):
    statuses = [503, 200]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            status = statuses.pop(0)
            self.send_response(status)
            if status == 503:
                self.send_header("Retry-After", "4")
            self.end_headers()
            if status == 200:
                self.wfile.write(json.dumps({"hits": [{"id": "a"}]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        adapter = HTTPPipeline(f"http://127.0.0.1:{server.server_port}/search")
        with pytest.raises(PipelineCallError) as failure:
            HTTPPipeline(f"http://127.0.0.1:{server.server_port}/search")(
                PipelineRequest(query_id="q", text="x")
            )
        assert failure.value.code == "http_unavailable"
        assert failure.value.retry_after == 4.0
        statuses[:] = [503, 200]
        run = execute_pipeline(dataset(), adapter, options())
        assert run["status"] == "complete"
        assert run["queries"][0]["retries"] == [
            {"call": "repeat 1", "code": "http_unavailable", "delay_s": 4.0}
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cli_retry_options_are_recorded_and_frozen_for_resume(tmp_path):
    from typer.testing import CliRunner

    from vectraudit.cli import app

    (tmp_path / "queries.jsonl").write_text('{"query_id":"q","text":"hello","no_answer":true}\n')
    (tmp_path / "corpus.jsonl").write_text('{"doc_id":"a","text":"unrelated"}\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "queries.jsonl",
                "candidates": [
                    {"name": "local", "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"}}
                ],
            }
        )
    )
    out = tmp_path / "runs"
    args = ["workbench", "run", str(manifest), "--out", str(out)]
    cli = CliRunner()
    result = cli.invoke(app, [*args, "--max-attempts", "1", "--retry-budget", "0"])
    assert result.exit_code == 0, result.output
    run = json.loads(next(out.glob("*.run.json")).read_text())
    assert run["timing"]["retry"]["max_attempts"] == 1
    assert run["timing"]["retry"]["max_retries"] == 0
    journal = next((out / ".checkpoints").iterdir())
    changed = cli.invoke(app, [*args, "--resume", str(journal)])
    assert changed.exit_code == 2
    assert "changed" in changed.output
    assert cli.invoke(app, [*args, "--max-attempts", "0"]).exit_code != 0
