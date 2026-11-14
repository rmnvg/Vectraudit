"""Batch mode: a throughput measured beside single-query latency, never mixed with it."""

from __future__ import annotations

import json

import faiss
import pytest
from typer.testing import CliRunner

from tests.test_sweep import K, ivf_setup  # noqa: F401  (the fixture)
from vectraudit.cli import app
from vectraudit.core import runs
from vectraudit.core import sweep as W


def test_batch_mode_measures_throughput_and_leaves_latency_and_choices_alone(ivf_setup):  # noqa: F811
    li, _, _, qs, truth = ivf_setup
    before = faiss.omp_get_max_threads()
    single = W.sweep(li, qs, truth, values=[1, 4], k=K)
    batched = W.sweep(li, qs, truth, values=[1, 4], k=K, batch=True)
    assert single.batch_threads is None
    assert all(p.batch_qps is None for p in single.points)
    assert batched.batch_threads == before  # FAISS's usual threads for the batch call
    assert faiss.omp_get_max_threads() == before  # restored afterwards
    for one, many in zip(single.points, batched.points, strict=True):
        assert many.batch_qps > 0
        assert many.batch_qps == pytest.approx(len(qs) / many.batch_seconds)
        assert many.batch_recall == pytest.approx(many.recall)  # same results, all at once
        assert one.recall == many.recall
    assert single.recommend(0.5).value == batched.recommend(0.5).value


def cli(synthetic, *extra):
    args = ["sweep", str(synthetic["ivf_flat"]), "--vectors", str(synthetic["vectors"])]
    return CliRunner().invoke(app, [*args, "--values", "1,4", "--n-queries", "40", *extra])


def test_the_table_and_json_label_each_mode(synthetic, tmp_path):
    table = cli(synthetic, "--batch", "--target", "0.1")
    assert table.exit_code == 0, table.output
    text = " ".join(table.output.split())
    assert "latency: one query per call, 1 thread · batch q/s: every query in one call" in text
    assert "batch q/s" in text
    assert "Batch q/s is a throughput, not a latency" in text
    as_json = cli(synthetic, "--batch", "--target", "0.1", "--json")
    data = json.loads(as_json.output)
    assert data["batch_threads"] >= 1
    assert "not a latency" in data["modes"]["batch_qps"]
    assert all(p["batch_qps"] > 0 for p in data["points"])
    plain = json.loads(cli(synthetic, "--target", "0.1", "--json").output)
    assert plain["batch_threads"] is None
    assert set(plain["modes"]) == {"latency_mean_ms, latency_p95_ms"}
    refused = cli(synthetic, "--batch", "--pass-rates", "0.5")
    assert refused.exit_code != 0
    assert "unfiltered sweep" in refused.output


def test_saved_runs_keep_batch_results_and_still_diff(synthetic, tmp_path):
    base, cur = tmp_path / "base.json", tmp_path / "cur.json"
    assert cli(synthetic, "--target", "0.1", "--save", str(base)).exit_code == 0
    assert cli(synthetic, "--batch", "--target", "0.1", "--save", str(cur)).exit_code == 0
    saved = runs.load_run(cur)
    assert saved["settings"]["batch_threads"] >= 1
    assert all(p["batch_qps"] > 0 for p in saved["points"])
    assert "batch_qps" not in runs.load_run(base)["points"][0]
    diff = runs.compare_runs(runs.load_run(base), saved)
    assert diff.comparable  # batch numbers are not part of the regression check
