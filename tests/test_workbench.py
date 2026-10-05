"""Bounded optimization, benchmark maintenance, monitoring and artifact privacy."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.runfile import RunFileError, check_run
from vectraudit.pipelines import Hit, PipelineOptions, PipelineResult, execute_pipeline
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.monitor import monitor_check
from vectraudit.pipelines.optimize import optimize
from vectraudit.pipelines.web import create_workbench


def benchmark():
    return parse_dataset(
        "\n".join(
            json.dumps(row)
            for row in [
                {"query_id": "t", "text": "test", "relevant": {"a": 1}},
                {"query_id": "h", "text": "holdout", "relevant": {"b": 1}, "split": "heldout"},
            ]
        )
    )


def test_only_selected_candidate_sees_heldout():
    calls = []

    def good(q):
        calls.append(("good", q.query_id))
        return PipelineResult(hits=[Hit(id="a" if q.query_id == "t" else "b")])

    def bad(q):
        calls.append(("bad", q.query_id))
        return PipelineResult(hits=[Hit(id="z")])

    report = optimize(benchmark(), [("good", good, {}), ("bad", bad, {})], repeats=1)
    assert report["selected"] == "good"
    assert report["eligible"]
    assert calls == [("good", "t"), ("bad", "t"), ("good", "h")]
    assert report["requires_review"]


def test_heldout_failure_is_not_retuned():
    report = optimize(
        benchmark(),
        [("candidate", lambda q: PipelineResult(hits=[Hit(id="a")]), {})],
        max_cost_usd=0,
        repeats=1,
    )
    assert report["selected"] is None
    assert not report["eligible"]
    assert report["candidates"][0]["cost_usd"] is None


def test_budget_and_split_leakage_rejected():
    with pytest.raises(ValueError, match="budget"):
        optimize(
            benchmark(),
            [(str(i), lambda q: PipelineResult(), {}) for i in range(3)],
            max_candidates=2,
        )
    ds = parse_dataset(
        '{"query_id":"a","text":"SAME"}\n{"query_id":"b","text":"same","split":"heldout"}'
    )
    with pytest.raises(ValueError, match="boundary"):
        optimize(ds, [("a", lambda q: PipelineResult(), {})])


def test_hybrid_filters_tenant_and_records_losses():
    pipeline = hybrid_pipeline(
        [
            {
                "doc_id": "secret",
                "text": "test exact",
                "vector": [1, 0],
                "metadata": {"tenant": "b"},
            },
            {
                "doc_id": "allowed",
                "text": "test",
                "vector": [0.8, 0.2],
                "metadata": {"tenant": "a"},
            },
        ]
    )
    ds = parse_dataset(
        '{"query_id":"q","text":"test","vector":[1,0],"context":{"tenant":"a"},"forbidden":["secret"],"relevant":{"allowed":1}}'
    )
    run = execute_pipeline(
        ds, pipeline, PipelineOptions(k=1, warmup=0, params={"vector_weight": 0.5})
    )
    assert run["queries"][0]["ids"] == ["allowed"]
    assert run["summary"]["permissions"]["violations"] == 0
    assert [s["kind"] for s in run["queries"][0]["evidence"]["stages"]] == [
        "keyword",
        "vector",
        "fusion",
        "filter",
        "rerank",
        "truncate",
    ]


def test_forbidden_repeat_is_not_hidden_by_safe_first_result():
    ds = parse_dataset('{"query_id":"q","text":"test","no_answer":true,"forbidden":["secret"]}')
    calls = []

    def flaky(q):
        calls.append(q)
        return PipelineResult(hits=[] if len(calls) == 1 else [Hit(id="secret")])

    run = execute_pipeline(ds, flaky, PipelineOptions(warmup=0, repeats=2))
    assert run["queries"][0]["ids"] == []
    assert run["queries"][0]["forbidden_ids"] == ["secret"]
    assert run["queries"][0]["no_answer_correct"] is False


def test_audit_keeps_heldout_out_of_judgment_queue():
    ds = benchmark()
    run = execute_pipeline(ds, lambda q: PipelineResult(hits=[Hit(id="new")]))
    audit = audit_benchmark(ds, [run])
    assert {r["query_id"] for r in audit["judgment_queue"]} == {"t"}
    assert "multilingual" in audit["missing_categories"]


def test_monitor_reports_corpus_and_config_separately():
    run = execute_pipeline(
        benchmark(),
        lambda q: PipelineResult(hits=[Hit(id="a")]),
        config={"depth": 10},
        corpus_version="v1",
    )
    changed = json.loads(json.dumps(run))
    changed["backend"]["corpus_version"] = "v2"
    result = monitor_check(run, changed)
    assert result["drift"]["corpus"] == "changed"
    assert result["drift"]["configuration"] == "unchanged"
    assert not result["drift"]["results"]


def test_local_web_rejects_symlinks_and_other_formats(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text('{"token":"secret"}')
    (root / "link.json").symlink_to(secret)
    (root / "other.json").write_text("{}")
    c = TestClient(create_workbench(root))
    assert c.get("/api/workbench/link.json").status_code == 404
    assert c.get("/api/workbench/other.json").status_code == 422
    assert c.get("/api/workbench", headers={"host": "evil.test"}).status_code == 400


def test_native_cli_workflow(tmp_path):
    manifest = Path(__file__).parents[1] / "examples/workbench/manifest.json"
    result = CliRunner().invoke(app, ["workbench", "run", str(manifest), "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("*.run.json"))) == 2
    result = CliRunner().invoke(
        app, ["workbench", "optimize", str(manifest), "--out", str(tmp_path / "optimization.json")]
    )
    assert result.exit_code == 0, result.output


def test_invalid_safety_fields_cannot_load():
    run = execute_pipeline(benchmark(), lambda q: PipelineResult())
    run["queries"][0]["no_answer_correct"] = "false"
    with pytest.raises(RunFileError, match="boolean"):
        check_run(run)


def test_no_answer_counts_are_separate_from_positive_labels():
    from vectraudit.evaluation.dataset_tools import dataset_stats

    ds = parse_dataset('{"query_id":"q","text":"missing","no_answer":true}')
    stats = dataset_stats(ds)
    assert stats["n_labelled"] == stats["n_no_answer"] == 1
    assert stats["n_positive"] == 0


def test_legacy_annotation_cannot_silently_discard_safety_fields():
    from vectraudit.experiments.corpus import Document, make_corpus
    from vectraudit.labelling.pooling import pool

    ds = parse_dataset('{"query_id":"q","text":"missing","no_answer":true}')
    with pytest.raises(ValueError, match="source benchmark"):
        pool(ds, make_corpus([Document("d", "text")]))
