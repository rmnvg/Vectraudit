"""Large runs are listed, summarised, validated and tabulated without loading their questions."""

from __future__ import annotations

import csv
import json
import math
import tracemalloc
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_answer_judgments import ALL_WRONG, judged, write
from tests.test_answer_judgments import run as planted_run
from tests.test_components import broken_run
from vectraudit.cli import app
from vectraudit.evaluation import runfile
from vectraudit.evaluation.answers import attach_file
from vectraudit.evaluation.metrics import RANK_METRICS, SEMANTIC_METRICS
from vectraudit.evaluation.run_table import NAMES, RunTableError, question_rows, write_table
from vectraudit.evaluation.runfile import (
    RunFileError,
    load_run,
    read_run_header,
    run_header_report,
)
from vectraudit.pipelines import Hit, PipelineOptions, PipelineResult, execute_pipeline

METRICS = (*SEMANTIC_METRICS, *RANK_METRICS)


def repeated_run():
    """Three repeats, a nondeterministic pipeline and one question that fails."""
    from tests.test_components import rows

    class Flaky:
        def __init__(self):
            self.calls = 0

        def __call__(self, request):
            if request.query_id == "q3":
                raise RuntimeError("down")
            self.calls += 1
            first = ["a", "x"] if self.calls % 2 else ["x", "b"]
            return PipelineResult(hits=[Hit(id=d) for d in first])

    return execute_pipeline(rows(), Flaky(), PipelineOptions(k=2, warmup=0, repeats=3))


def runs(tmp_path):
    answered = attach_file(planted_run(), write(tmp_path, judged(ALL_WRONG)))
    return {
        "broken": broken_run({"multi_hop": ["retrieve", "rerank"]}),
        "answered": answered,
        "repeated": repeated_run(),
    }


def save(tmp_path, name, run):
    path = tmp_path / f"{name}.run.json"
    path.write_text(json.dumps(run, indent=2))
    return path


def write_big(path: Path, target_bytes: int) -> int:
    """A valid complete run of about ``target_bytes``, written without holding it in memory."""
    run = broken_run()
    template = next(q for q in run["queries"] if not q.get("error"))
    n = target_bytes // (len(json.dumps(template)) + 2)
    header = {k: v for k, v in run.items() if k != "queries"}
    header["dataset"] = {**header["dataset"], "n_queries": n}
    before, after = json.dumps({**header, "queries": "@@"}).split('"@@"')
    with path.open("w") as f:
        f.write(before + "[")
        for i in range(n):
            f.write(("," if i else "") + json.dumps({**template, "query_id": f"q{i}"}))
        f.write("]" + after)
    return n


# --- reading the header -----------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 4096, runfile.CHUNK_BYTES])
def test_the_header_is_the_run_without_its_questions(tmp_path, monkeypatch, chunk):
    monkeypatch.setattr(runfile, "CHUNK_BYTES", chunk)
    for name, run in runs(tmp_path).items():
        path = save(tmp_path, name, run)
        header = read_run_header(path)
        full = load_run(path)
        assert header.pop("queries_omitted") == len(full["queries"]), name
        assert header == {k: v for k, v in full.items() if k != "queries"}, name


def test_each_question_is_handed_over_once_with_the_header_read_so_far(tmp_path):
    path = save(tmp_path, "broken", broken_run())
    seen = []
    read_run_header(path, on_query=lambda header, q: seen.append((header["run_id"], q["query_id"])))
    full = load_run(path)
    assert seen == [(full["run_id"], q["query_id"]) for q in full["queries"]]


def test_questions_before_the_header_are_still_read_correctly(tmp_path):
    run = broken_run()
    path = tmp_path / "odd.run.json"
    path.write_text(
        json.dumps({"queries": run["queries"], **{k: v for k, v in run.items() if k != "queries"}})
    )
    header = read_run_header(path)
    assert header["queries_omitted"] == len(run["queries"])
    assert header["run_id"] == run["run_id"]


def broken(run, change):
    copy = json.loads(json.dumps(run))
    change(copy)
    return copy


INVALID = {
    "format": lambda r: r.update(format="vectraudit.benchmark"),
    "duplicate": lambda r: r["queries"].append(r["queries"][0]),
    "bad metric": lambda r: r["queries"][0]["metrics"].update(ndcg=2.0),
    "incomplete": lambda r: r["queries"].pop(),
    "no queries": lambda r: r.pop("queries"),
    "queries not a list": lambda r: r.update(queries={}),
}


@pytest.mark.parametrize("change", INVALID.values(), ids=INVALID.keys())
def test_invalid_runs_fail_exactly_as_loading_them_fails(tmp_path, change):
    path = save(tmp_path, "bad", broken(broken_run(), change))
    with pytest.raises(RunFileError) as loaded:
        load_run(path)
    with pytest.raises(RunFileError) as streamed:
        read_run_header(path)
    assert str(streamed.value) == str(loaded.value)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"format": "vectraudit.run", "x": NaN}', "NaN"),
        ('{"format": "vectraudit.run"} trailing', "data after the run object"),
        ('{"format": "vectraudit.run", "queries": [{"query_id"', "not valid JSON"),
        ("[]", "a run must be a JSON object"),
    ],
)
def test_malformed_files_are_refused(tmp_path, text, message):
    path = tmp_path / "bad.run.json"
    path.write_text(text)
    with pytest.raises(RunFileError, match=message):
        read_run_header(path)
    path.write_bytes(b'{"format": "\xff"}')
    with pytest.raises(RunFileError, match="UTF-8"):
        read_run_header(path)
    with pytest.raises(RunFileError, match="not found"):
        read_run_header(tmp_path / "missing.run.json")


def peak_mib(call):
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()


def test_a_large_run_is_read_in_bounded_memory(tmp_path):
    path = tmp_path / "big.run.json"
    n = write_big(path, 20 * 2**20)
    peak = peak_mib(lambda: read_run_header(path))
    assert peak < 16, peak
    assert read_run_header(path)["queries_omitted"] == n


@pytest.mark.slow
def test_a_200_mib_run_is_listed_and_summarised_without_loading_it(tmp_path):
    """The phase acceptance: a 200 MiB run, listed by the workbench, under a stated bound."""
    from fastapi.testclient import TestClient

    from vectraudit.pipelines.web import create_workbench

    path = tmp_path / "huge.run.json"
    n = write_big(path, 200 * 2**20)
    assert path.stat().st_size > 190 * 2**20
    report = {}
    peak = peak_mib(lambda: report.update(run_header_report(path)))
    assert peak < 32, peak  # the documented bound: under 32 MiB of Python allocations
    assert report["questions"] == n
    client = TestClient(create_workbench(tmp_path))
    listed = client.get("/api/workbench").json()["artifacts"]
    assert {"name": "huge.run.json", "bytes": path.stat().st_size, "too_large": True} in listed
    opened = client.get("/api/workbench/huge.run.json").json()
    assert (opened["format"], opened["questions"]) == ("vectraudit.run_header", n)


# --- the workbench ------------------------------------------------------------------------------


def test_the_workbench_lists_and_summarises_a_run_too_large_to_open(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from vectraudit.pipelines import guided, web
    from vectraudit.pipelines.guided import GuidedWorkspace
    from vectraudit.pipelines.web import create_workbench

    run_id = "a" * 32  # workbench runs are named by opaque IDs
    path = save(tmp_path, run_id, broken_run())
    big = {"format": "vectraudit.benchmark", "pad": "x" * 2000}
    (tmp_path / "other.json").write_text(json.dumps(big))
    monkeypatch.setattr(web, "MAX_RUN_BYTES", 1000)
    monkeypatch.setattr(guided, "MAX_RUN_BYTES", 1000)
    client = TestClient(create_workbench(tmp_path))
    names = {a["name"]: a for a in client.get("/api/workbench").json()["artifacts"]}
    assert names[f"{run_id}.run.json"]["too_large"] is True
    assert "other.json" not in names  # other large artifacts are still not served
    opened = client.get(f"/api/workbench/{run_id}.run.json").json()
    assert opened["format"] == "vectraudit.run_header"
    assert opened["questions"] == len(broken_run()["queries"])
    assert opened["run"]["summary"]["n_queries"] == opened["questions"]
    assert "queries" not in opened["run"]
    summary = GuidedWorkspace(tmp_path).run_summary(run_id)
    assert summary["too_large"] is True
    assert summary["queries_omitted"] == opened["questions"]
    assert path.stat().st_size > 1000


# --- the table ---------------------------------------------------------------------------------


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def number(text):
    return None if text == "" else float(text)


def test_the_table_round_trips_every_metric_per_question_and_repeat(tmp_path):
    paths = [save(tmp_path, name, run) for name, run in runs(tmp_path).items()]
    out = tmp_path / "runs.csv"
    done = write_table(paths, out)
    rows = read_csv(out)
    assert list(rows[0]) == list(NAMES)
    expected = 0
    by_key = {(r["run_file"], r["query_id"], r["repeat"]): r for r in rows}
    for path in paths:
        run = load_run(path)
        for q in run["queries"]:
            samples = q.get("repeat_metrics")
            for i, sample in enumerate(samples or [None]):
                row = by_key[(path.name, q["query_id"], str(i + 1))]
                expected += 1
                for name in METRICS:
                    value = (sample or {}).get(name)
                    assert number(row[f"metric_{name}"]) == value, (path.name, q["query_id"], name)
    assert done == {"runs": 3, "rows": expected} == {"runs": 3, "rows": len(rows)}


def test_repeats_errors_and_flags_have_their_own_columns(tmp_path):
    run = repeated_run()
    rows = [r for q in run["queries"] for r in question_rows(run, q, "x.run.json")]
    q0 = [r for r in rows if r["query_id"] == "q0"]
    assert [r["repeat"] for r in q0] == [1, 2, 3]
    assert {r["repeats"] for r in q0} == {3}
    assert [r["returned"] for r in q0] == [2, 2, 2]
    assert [r["latency_ms"] is not None for r in q0] == [True, True, True]
    assert q0[0]["nondeterministic"] is True
    assert q0[0]["component"] == "kept"  # repeat 1 delivered a relevant document
    assert q0[1]["component"] is None  # it describes the first execution only
    (failed,) = [r for r in rows if r["query_id"] == "q3"]
    assert failed["error_code"] is not None
    assert failed["metric_ndcg"] is None
    none = [r for r in rows if r["query_id"] == "none"]
    # Documents came back for an out-of-scope question, on every repeat.
    assert [r["no_answer_correct"] for r in none] == [False, False, False]
    assert json.loads(q0[0]["tags"]) == []


def test_the_table_holds_no_text_and_parquet_matches_csv(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    paths = [save(tmp_path, name, run) for name, run in runs(tmp_path).items()]
    write_table(paths, tmp_path / "runs.csv")
    write_table(paths, tmp_path / "runs.parquet")
    text = (tmp_path / "runs.csv").read_text()
    assert "question 1" not in text  # the benchmark's question text
    assert "PLANTED-PRIVATE" not in text
    table = pq.read_table(tmp_path / "runs.parquet").to_pylist()
    rows = read_csv(tmp_path / "runs.csv")
    assert len(table) == len(rows)
    for parquet_row, csv_row in zip(table, rows, strict=True):
        for name in METRICS:
            left, right = parquet_row[f"metric_{name}"], number(csv_row[f"metric_{name}"])
            assert left == right or (left is None and right is None)
        assert parquet_row["repeat"] == int(csv_row["repeat"])


def test_table_errors(tmp_path):
    with pytest.raises(RunTableError, match=r"\.csv or \.parquet"):
        write_table([tmp_path / "a.run.json"], tmp_path / "out.xlsx")
    with pytest.raises(RunTableError, match="at least one"):
        write_table([], tmp_path / "out.csv")


def test_an_imported_run_without_latency_tabulates(tmp_path):
    from tests.test_trec import fixture_dataset, write_trec
    from vectraudit.evaluation.trec import import_run

    ds = fixture_dataset()
    imported = import_run(write_trec(tmp_path / "a.trec"), ds, k=10)
    rows = [r for q in imported["queries"] for r in question_rows(imported, q, "t.run.json")]
    assert all(r["timing_source"] == "external:trec" for r in rows)
    assert all(r["query_latency_ms"] is None and r["latency_ms"] is None for r in rows)
    assert all(not math.isnan(r["metric_ndcg"]) for r in rows if r["metric_ndcg"] is not None)


# --- the commands ------------------------------------------------------------------------------


def test_runs_summary_and_table_commands(tmp_path):
    good = save(tmp_path, "broken", broken_run())
    bad = save(tmp_path, "bad", broken(broken_run(), INVALID["duplicate"]))
    cli = CliRunner()
    out = tmp_path / "summary.json"
    result = cli.invoke(app, ["runs", "summary", str(good), "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "broken.run.json: complete, 12 questions" in result.output
    (report,) = json.loads(out.read_text())
    assert report["format"] == "vectraudit.run_header"
    failed = cli.invoke(app, ["runs", "summary", str(good), str(bad)])
    assert failed.exit_code == 1
    assert "duplicate query_id" in failed.output
    table = tmp_path / "runs.csv"
    result = cli.invoke(app, ["runs", "table", str(good), "--out", str(table)])
    assert result.exit_code == 0, result.output
    assert "Wrote 12 rows from 1 runs" in result.output
    wrong = cli.invoke(app, ["runs", "table", str(good), "--out", str(tmp_path / "x.txt")])
    assert wrong.exit_code != 0
