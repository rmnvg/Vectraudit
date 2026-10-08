"""Recovery guarantees: crash boundaries, frozen identity, retries and release safety."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.core.jobs import JobCancelledError
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.policy import CriticalQueryRule, Policy
from vectraudit.evaluation.runfile import check_run
from vectraudit.labelling.storage import TaskBusy, task_lock
from vectraudit.pipelines.checkpoint import checkpoint_state
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.models import PipelineResult
from vectraudit.pipelines.runner import PipelineOptions
from vectraudit.pipelines.runner import execute_pipeline as run_pipeline


def execute_pipeline(*args, **kwargs):
    kwargs.setdefault("config", {"pipeline_version": "v1"})
    kwargs.setdefault("corpus_version", "v1")
    return run_pipeline(*args, **kwargs)


def dataset(n=3):
    return parse_dataset(
        "\n".join(
            json.dumps({"query_id": str(i), "text": "question", "no_answer": True})
            for i in range(n)
        )
    )


class Adapter:
    def __init__(self):
        self.calls = []
        self.fail = set()

    def __call__(self, request):
        self.calls.append(request.query_id)
        if request.query_id in self.fail:
            raise RuntimeError("secret must not appear in the journal")
        return PipelineResult(hits=[])


def test_cancel_resume_and_release_safety(tmp_path):
    pipe = Adapter()
    path = tmp_path / "run.sqlite"

    def cancel(done, total):
        if done == 2:
            raise JobCancelledError()

    partial = execute_pipeline(dataset(), pipe, checkpoint=path, progress=cancel)
    assert partial["status"] == "cancelled"
    assert pipe.calls == ["0", "1"]
    assert checkpoint_state(path)["finished"] == 2
    resumed = execute_pipeline(dataset(), pipe, checkpoint=path, resume=True)
    assert pipe.calls == ["0", "1", "2"]
    assert resumed["run_id"] == partial["run_id"]
    assert check_run(resumed)["status"] == "complete"
    assert all(q["attempt_count"] == 1 for q in resumed["queries"])
    assert compare_runs(resumed, partial, Policy(metrics=()))["verdict"] != "pass"
    again = execute_pipeline(dataset(), pipe, checkpoint=path, resume=True)
    assert again["queries"] == resumed["queries"]
    assert pipe.calls == ["0", "1", "2"]


@pytest.mark.parametrize("code", [None, "http_rate_limit", "timeout"])
def test_failed_queries_need_explicit_retry_and_preserve_evidence(tmp_path, code):
    class FailingAdapter(Adapter):
        def __call__(self, request):
            if code and request.query_id in self.fail:
                self.calls.append(request.query_id)
                raise PipelineCallError(code, "secret must not appear in the journal")
            return super().__call__(request)

    pipe = FailingAdapter()
    pipe.fail = {"1"}
    path = tmp_path / "run.sqlite"
    first = execute_pipeline(dataset(), pipe, checkpoint=path)
    assert first["status"] == "partial"
    assert first["summary"]["errors_by_code"] == {code or "pipeline_error": 1}
    pipe.fail.clear()
    same = execute_pipeline(dataset(), pipe, checkpoint=path, resume=True)
    assert same["status"] == "partial"
    assert same["summary"]["errors_by_code"] == first["summary"]["errors_by_code"]
    assert pipe.calls == ["0", "1", "2"]
    recovered = execute_pipeline(dataset(), pipe, checkpoint=path, resume=True, retry_errors=True)
    assert pipe.calls == ["0", "1", "2", "1"]
    query = recovered["queries"][1]
    assert query["attempt_count"] == 2
    assert [a["status"] for a in query["attempts"]] == ["failed", "complete"]
    assert query["attempts"][0]["result"]["error"]["code"] == (code or "pipeline_error")
    assert query["attempts"][0]["result"]["elapsed_ms"] >= 0
    assert "secret" not in json.dumps(recovered)
    assert recovered["status"] == "complete"
    assert recovered["summary"]["errors_by_code"] == {}


@pytest.mark.parametrize(
    "changed", ["dataset", "config", "corpus", "k", "repeats", "warmup", "evidence"]
)
def test_changed_inputs_rejected_before_query_calls(tmp_path, changed):
    pipe = Adapter()
    path = tmp_path / "run.sqlite"
    args = {
        "dataset": dataset(),
        "pipeline": pipe,
        "options": PipelineOptions(warmup=0),
        "config": {"adapter": "hybrid", "depth": 5},
        "corpus_version": "v1",
        "checkpoint": path,
    }
    execute_pipeline(**args)
    if changed == "dataset":
        args["dataset"] = dataset(4)
    elif changed == "config":
        args["config"] = {"adapter": "hybrid", "depth": 6}
    elif changed == "corpus":
        args["corpus_version"] = "v2"
    else:
        args["options"] = PipelineOptions(
            **{
                "warmup": 0,
                changed: {"k": 2, "repeats": 2, "warmup": 1, "evidence": False}[changed],
            }
        )
    with pytest.raises(ValueError, match="changed"):
        execute_pipeline(**args, resume=True)
    assert pipe.calls == ["0", "1", "2"]


def test_lock_missing_corrupt_and_symlink_checkpoints(tmp_path):
    pipe = Adapter()
    path = tmp_path / "run.sqlite"
    with pytest.raises(ValueError, match="does not exist"):
        execute_pipeline(dataset(), pipe, checkpoint=path, resume=True)
    with task_lock(path), pytest.raises(TaskBusy):
        execute_pipeline(dataset(), pipe, checkpoint=path)
    execute_pipeline(dataset(), pipe, checkpoint=path)
    with pytest.raises(ValueError, match="exists"):
        execute_pipeline(dataset(), pipe, checkpoint=path)
    link = tmp_path / "link.sqlite"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlinks"):
        execute_pipeline(dataset(), pipe, checkpoint=link, resume=True)
    import sqlite3

    path.write_bytes(b"corrupt")
    with pytest.raises(sqlite3.DatabaseError):
        execute_pipeline(dataset(), pipe, checkpoint=path, resume=True)
    assert pipe.calls == ["0", "1", "2"]


@pytest.mark.parametrize(
    ("config", "corpus"),
    [({"adapter": "http"}, "v1"), ({"adapter": "python", "pipeline_version": "v1"}, None)],
)
def test_external_resume_requires_explicit_versions(tmp_path, config, corpus):
    pipe = Adapter()
    path = tmp_path / "run.sqlite"
    execute_pipeline(dataset(), pipe, config=config, corpus_version=corpus, checkpoint=path)
    with pytest.raises(ValueError, match="requires corpus_version and pipeline_version"):
        execute_pipeline(
            dataset(), pipe, config=config, corpus_version=corpus, checkpoint=path, resume=True
        )
    assert len(pipe.calls) == 3


def test_process_crash_recovers_1000_queries_without_replaying_saved_results(tmp_path):
    # Kill the actual interpreter inside query 500, not a simulated caught exception.
    script = tmp_path / "worker.py"
    script.write_text("""
import json, os, sys
from pathlib import Path
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.pipelines.models import PipelineResult
from vectraudit.pipelines.runner import execute_pipeline
root = Path(sys.argv[1])
def pipeline(request):
    with (root / 'calls').open('a') as f:
        f.write(request.query_id + '\\n')
        f.flush()
        os.fsync(f.fileno())
    if request.query_id == '500' and not (root / 'restart').exists():
        os._exit(17)
    return PipelineResult(hits=[])
rows = ({'query_id':str(i),'text':'q','no_answer':True} for i in range(1000))
ds = parse_dataset('\\n'.join(map(json.dumps, rows)))
run = execute_pipeline(ds, pipeline, checkpoint=root / 'run.sqlite',
    config={'pipeline_version':'v1'}, corpus_version='v1', resume=(root / 'restart').exists())
(root / 'result.json').write_text(json.dumps(run))
""")
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    first = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], env=env, capture_output=True, timeout=60
    )
    assert first.returncode == 17, first.stderr.decode()
    state = checkpoint_state(tmp_path / "run.sqlite")
    assert state["finished"] == 500
    assert state["status"] != "complete"
    (tmp_path / "restart").touch()
    second = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], env=env, capture_output=True, timeout=60
    )
    assert second.returncode == 0, second.stderr.decode()
    run = check_run(json.loads((tmp_path / "result.json").read_text()))
    assert run["status"] == "complete"
    assert len(run["queries"]) == 1000
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls == [str(i) for i in range(501)] + [str(i) for i in range(500, 1000)]
    assert run["queries"][500]["attempt_count"] == 2
    assert run["queries"][500]["attempts"][0]["status"] == "interrupted"


def test_cli_batch_resume_and_changed_configuration(tmp_path):
    (tmp_path / "queries.jsonl").write_text('{"query_id":"q","text":"hello","no_answer":true}\n')
    (tmp_path / "corpus.jsonl").write_text('{"doc_id":"a","text":"unrelated"}\n')
    manifest = tmp_path / "manifest.json"
    spec = {
        "dataset": "queries.jsonl",
        "repeats": 1,
        "candidates": [{"name": "local", "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"}}],
    }
    manifest.write_text(json.dumps(spec))
    out = tmp_path / "runs"
    cli = CliRunner()
    args = ["workbench", "run", str(manifest), "--out", str(out)]
    first = cli.invoke(app, args)
    assert first.exit_code == 0, first.output
    journal_dir = next((out / ".checkpoints").iterdir())
    initial = json.loads(next(out.glob("*.run.json")).read_text())
    resumed = cli.invoke(app, [*args, "--resume", str(journal_dir)])
    assert resumed.exit_code == 0, resumed.output
    assert len(list(out.glob("*.run.json"))) == 1
    assert json.loads(next(out.glob("*.run.json")).read_text())["queries"] == initial["queries"]
    spec["k"] = 2
    manifest.write_text(json.dumps(spec))
    rejected = cli.invoke(app, [*args, "--resume", str(journal_dir)])
    assert rejected.exit_code == 2
    assert "changed" in rejected.output


def test_changed_callable_is_rejected_even_in_same_module(tmp_path):
    path = tmp_path / "run.sqlite"

    def first(request):
        return PipelineResult(hits=[])

    def changed(request):
        pytest.fail("Changed adapter must not execute")

    execute_pipeline(dataset(), first, checkpoint=path)
    with pytest.raises(ValueError, match="changed"):
        execute_pipeline(dataset(), changed, checkpoint=path, resume=True)


def test_missing_started_cli_checkpoint_fails_closed(tmp_path):
    (tmp_path / "queries.jsonl").write_text('{"query_id":"q","text":"hello","no_answer":true}\n')
    (tmp_path / "corpus.jsonl").write_text('{"doc_id":"a","text":"unrelated"}\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "queries.jsonl",
                "repeats": 1,
                "candidates": [
                    {"name": "local", "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"}}
                ],
            }
        )
    )
    out = tmp_path / "runs"
    cli = CliRunner()
    args = ["workbench", "run", str(manifest), "--out", str(out)]
    assert cli.invoke(app, args).exit_code == 0
    directory = next((out / ".checkpoints").iterdir())
    next(directory.glob("*.sqlite")).unlink()
    result = cli.invoke(app, [*args, "--resume", str(directory)])
    assert result.exit_code == 2
    assert "checkpoint is missing" in result.output
    assert len(list(out.glob("*.run.json"))) == 1


class Flaky:
    """Returns ``guide`` on the first call per query, then drops it; optionally fails once."""

    def __init__(self, fail_after_first=False):
        self.calls = {}
        self.fail_after_first = fail_after_first

    def __call__(self, request):
        n = self.calls[request.query_id] = self.calls.get(request.query_id, 0) + 1
        if self.fail_after_first and n == 2:
            raise RuntimeError("transient")
        ids = ["intro", "guide"] if n == 1 else ["intro", "other"]
        return PipelineResult(hits=[{"id": i} for i in ids])


def critical(max_rank=2):
    return Policy(
        metrics=(),
        critical_queries=(CriticalQueryRule("0", "guide", max_rank),),
    )


def test_critical_documents_are_checked_on_every_repeat_through_resume(tmp_path):
    path = tmp_path / "run.sqlite"
    options = PipelineOptions(warmup=0, repeats=2, k=2)

    def cancel(done, total):
        if done == 1:
            raise JobCancelledError()

    partial = execute_pipeline(dataset(2), Flaky(), options, checkpoint=path, progress=cancel)
    assert partial["status"] == "cancelled"
    run = execute_pipeline(dataset(2), Flaky(), options, checkpoint=path, resume=True)
    query = check_run(run)["queries"][0]
    assert query["repeat_ids"] == [["intro", "guide"], ["intro", "other"]]
    result = compare_runs(run, run, critical())
    assert result["verdict"] == "fail"
    assert result["critical_queries"][0]["failures"] == [
        {"attempt": None, "repeat": 2, "ids": ["intro", "other"], "rank": None}
    ]


def test_retry_cannot_hide_a_ranking_observed_before_the_failure(tmp_path):
    path = tmp_path / "run.sqlite"
    options = PipelineOptions(warmup=0, repeats=2, k=2)
    pipe = Flaky(fail_after_first=True)
    first = execute_pipeline(dataset(1), pipe, options, checkpoint=path)
    assert first["queries"][0]["error"]
    assert first["queries"][0]["repeat_ids"] == [["intro", "guide"]]
    pipe.calls.clear()
    pipe.fail_after_first = False
    retried = execute_pipeline(
        dataset(1), pipe, options, checkpoint=path, resume=True, retry_errors=True
    )
    query = check_run(retried)["queries"][0]
    assert query["attempt_count"] == 2
    assert query["attempts"][0]["result"]["repeat_ids"] == [["intro", "guide"]]
    # The rule holds on the failed attempt's ranking but not on the retry's second repeat.
    rule = compare_runs(retried, retried, critical())["critical_queries"][0]
    assert rule["executions"] == 3
    assert [(f["attempt"], f["repeat"]) for f in rule["failures"]] == [(2, 2)]
    # A ranking lost on the failed attempt keeps failing after a clean retry.
    query["attempts"][0]["result"]["repeat_ids"] = [["intro", "other"]]
    query["repeat_ids"] = [["intro", "guide"]] * 2
    query["ids"] = ["intro", "guide"]
    rule = compare_runs(retried, retried, critical())["critical_queries"][0]
    assert [(f["attempt"], f["repeat"]) for f in rule["failures"]] == [(1, 1)]
