"""Bounded concurrent query execution, rate limits and cancellation."""

import json
import sqlite3
import threading
import time

import pytest

from tests.fake_time import patch_time
from vectraudit.core.jobs import JobCancelledError
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.runfile import check_run
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.manifest import Candidate, candidate_config
from vectraudit.pipelines.models import Hit, PipelineResult
from vectraudit.pipelines.runner import PipelineOptions, RetryPolicy, execute_pipeline

PINNED = {"config": {"pipeline_version": "v1"}, "corpus_version": "v1"}


def dataset(n=20):
    rows = [
        {"query_id": f"q{i:02d}", "text": f"question {i}", "relevant": {f"d{i % 3}": 1}}
        for i in range(n)
    ]
    if n > 4:
        rows[3]["forbidden"] = ["d9"]
        rows[4]["no_answer"] = True
        rows[4].pop("relevant")
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


class Tracked:
    """Thread-safe adapter that records how many calls overlap."""

    def __init__(self, delay=0.01, fail=()):
        self.delay, self.fail = delay, set(fail)
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.calls: list[str] = []

    def __call__(self, request):
        with self.lock:
            self.calls.append(request.query_id)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if request.query_id in self.fail:
                raise PipelineCallError("http_auth", "private")
            number = int(request.query_id[1:])
            return PipelineResult(hits=[Hit(id=f"d{number % 3}"), Hit(id="d9")])
        finally:
            with self.lock:
                self.active -= 1


def comparable(run):
    """Everything except timing, which legitimately differs between executions."""
    keep = ("query_id", "ids", "repeat_ids", "metrics", "forbidden_ids", "no_answer_correct")
    return [{k: q.get(k) for k in keep} | {"error": q["error"]} for q in run["queries"]]


def test_concurrent_results_match_serial_in_dataset_order():
    serial = execute_pipeline(dataset(), Tracked(), PipelineOptions(warmup=0, repeats=2))
    pipe = Tracked()
    concurrent = check_run(
        execute_pipeline(dataset(), pipe, PipelineOptions(warmup=0, repeats=2, concurrency=6))
    )
    assert comparable(concurrent) == comparable(serial)
    assert [q["query_id"] for q in concurrent["queries"]] == [f"q{i:02d}" for i in range(20)]
    assert concurrent["summary"]["metrics"] == serial["summary"]["metrics"]
    assert 1 < pipe.peak <= 6
    assert concurrent["timing"]["concurrency"] == 6
    assert "6 concurrent queries" in concurrent["timing"]["scope"]
    assert serial["timing"]["concurrency"] == 1
    # Safety assertions are still judged per query and per repeat.
    assert concurrent["queries"][3]["forbidden_ids"] == ["d9"]
    assert concurrent["queries"][4]["no_answer_correct"] is False


def test_in_flight_queries_never_exceed_the_limit():
    pipe = Tracked(delay=0.02)
    execute_pipeline(dataset(30), pipe, PipelineOptions(warmup=0, concurrency=3))
    assert pipe.peak == 3
    assert sorted(pipe.calls) == [f"q{i:02d}" for i in range(30)]


def test_failures_keep_typed_causes_under_concurrency():
    pipe = Tracked(fail={"q01", "q07"})
    run = execute_pipeline(dataset(), pipe, PipelineOptions(warmup=0, concurrency=4))
    assert run["status"] == "partial"
    assert run["summary"]["errors_by_code"] == {"http_auth": 2}
    assert [q["query_id"] for q in run["queries"] if q["error"]] == ["q01", "q07"]
    assert "private" not in json.dumps(run)


def test_adapters_can_refuse_threads():
    class Unsafe(Tracked):
        thread_safe = False

    with pytest.raises(ValueError, match="thread_safe"):
        execute_pipeline(dataset(), Unsafe(), PipelineOptions(warmup=0, concurrency=2))
    assert execute_pipeline(dataset(2), Unsafe(), PipelineOptions(warmup=0))["status"] == (
        "complete"
    )


@pytest.mark.parametrize("workers", [1, 4])
def test_rate_limit_spaces_calls(workers):
    pipe = Tracked(delay=0)
    started = time.monotonic()
    run = execute_pipeline(
        dataset(11), pipe, PipelineOptions(warmup=0, concurrency=workers, max_rps=50)
    )
    # 11 calls at 50/s need at least 10 intervals of 20 ms.
    assert time.monotonic() - started >= 0.19
    assert run["timing"]["max_rps"] == 50


def test_retry_budget_is_shared_across_workers():
    seen: set[str] = set()
    lock = threading.Lock()

    def flaky(request):
        with lock:
            first = request.query_id not in seen
            seen.add(request.query_id)
        if first:
            raise PipelineCallError("connection_failed", "private")
        return PipelineResult(hits=[Hit(id="d0")])

    options = PipelineOptions(
        warmup=0, concurrency=4, retry=RetryPolicy(base_delay_s=0, max_retries=3)
    )
    run = execute_pipeline(dataset(10), flaky, options)
    assert run["summary"]["retries_by_code"] == {"connection_failed": 3}
    assert run["summary"]["errors_by_code"] == {"connection_failed": 7}


def test_concurrent_cancellation_keeps_finished_work_and_resumes(tmp_path):
    path = tmp_path / "run.sqlite"
    options = PipelineOptions(warmup=0, concurrency=4)

    def cancel_after_five(done, total):
        if done >= 5:
            raise JobCancelledError("cancelled")

    first_pipe = Tracked(delay=0.01)
    partial = execute_pipeline(
        dataset(), first_pipe, options, checkpoint=path, progress=cancel_after_five, **PINNED
    )
    assert partial["status"] == "cancelled"
    finished = {q["query_id"] for q in partial["queries"]}
    assert 5 <= len(finished) < 20
    assert not any(q["error"] for q in partial["queries"])
    second_pipe = Tracked(delay=0)
    resumed = execute_pipeline(
        dataset(), second_pipe, options, checkpoint=path, resume=True, **PINNED
    )
    assert resumed["status"] == "complete"
    assert not finished & set(second_pipe.calls)
    assert sorted(set(first_pipe.calls) | set(second_pipe.calls)) == sorted(
        q["query_id"] for q in resumed["queries"]
    )


def test_changing_concurrency_cannot_resume_a_checkpoint(tmp_path):
    path = tmp_path / "run.sqlite"
    execute_pipeline(dataset(3), Tracked(), PipelineOptions(warmup=0), checkpoint=path, **PINNED)
    with pytest.raises(ValueError, match="changed"):
        execute_pipeline(
            dataset(3),
            Tracked(),
            PipelineOptions(warmup=0, concurrency=2),
            checkpoint=path,
            resume=True,
            **PINNED,
        )


def test_cancel_during_backoff_leaves_the_query_interrupted_not_failed(tmp_path, monkeypatch):
    path = tmp_path / "run.sqlite"
    clock = {"now": 0.0}
    patch_time(
        monkeypatch,
        "vectraudit.pipelines.runner",
        monotonic=lambda: clock["now"],
        sleep=lambda s: clock.__setitem__("now", clock["now"] + s),
    )

    def down(request):
        raise PipelineCallError("timeout", "private")

    def progress(done, total):
        if clock["now"] > 0:
            raise JobCancelledError("cancelled")

    run = execute_pipeline(
        dataset(2), down, PipelineOptions(warmup=0), checkpoint=path, progress=progress, **PINNED
    )
    assert run["status"] == "cancelled"
    assert run["queries"] == []
    # The begun attempt has no result: resume treats it as interrupted, not as a failure.
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT query_id, result, error FROM attempts").fetchall()
    assert rows == [("q00", None, None)]


def test_latency_is_not_compared_across_different_concurrency():
    base = execute_pipeline(dataset(), Tracked(), PipelineOptions(warmup=0))
    same = execute_pipeline(dataset(), Tracked(), PipelineOptions(warmup=0))
    loaded = execute_pipeline(dataset(), Tracked(), PipelineOptions(warmup=0, concurrency=4))
    assert compare_runs(base, same)["latency"]["compared"] is True
    check = compare_runs(base, loaded)
    assert check["latency"]["compared"] is False
    assert any("different concurrency" in w for w in check["warnings"])


def test_options_validation():
    for bad in ({"concurrency": 0}, {"concurrency": 33}, {"max_rps": 0}, {"max_rps": 1001}):
        with pytest.raises(ValueError, match=r"concurrency|max_rps"):
            PipelineOptions(**bad)


def test_execution_settings_stay_out_of_candidate_identity(tmp_path):
    base = {"name": "api", "adapter": {"kind": "http", "target": "https://example.test/x"}}
    plain = candidate_config(Candidate.model_validate(base), tmp_path)
    tuned = candidate_config(
        Candidate.model_validate({**base, "concurrency": 8, "max_rps": 20}), tmp_path
    )
    assert plain == tuned


def test_cli_runs_a_manifest_candidate_concurrently(tmp_path):
    from typer.testing import CliRunner

    from vectraudit.cli import app

    queries = [
        {"query_id": f"q{i}", "text": f"invoice {i}", "relevant": {"a": 1}} for i in range(8)
    ]
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    (tmp_path / "corpus.jsonl").write_text(
        '{"doc_id":"a","text":"invoice payment"}\n{"doc_id":"b","text":"refund policy"}\n'
    )
    candidate = {"name": "local", "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"}}
    out = {}
    for workers in (1, 4):
        manifest = tmp_path / f"manifest{workers}.json"
        manifest.write_text(
            json.dumps(
                {
                    "dataset": "queries.jsonl",
                    "repeats": 1,
                    "candidates": [{**candidate, "concurrency": workers, "max_rps": 500}],
                }
            )
        )
        target = tmp_path / f"runs{workers}"
        result = CliRunner().invoke(app, ["workbench", "run", str(manifest), "--out", str(target)])
        assert result.exit_code == 0, result.output
        out[workers] = json.loads(next(target.glob("*.run.json")).read_text())
    assert out[4]["timing"]["concurrency"] == 4
    assert out[4]["backend"]["config"] == out[1]["backend"]["config"]
    assert comparable(out[4]) == comparable(out[1])
