"""Tuning a search-time setting on any backend: bounded, read-only, and honest about recall."""

from __future__ import annotations

import json
from itertools import pairwise

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.conftest import SMALL
from vectraudit.cli import app
from vectraudit.connectors.config import open_connector, parse_connection
from vectraudit.core import search as S
from vectraudit.core import sweep as W
from vectraudit.core import vectors as V
from vectraudit.core.loader import load_index
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import render_dataset
from vectraudit.evaluation.tune import (
    TunePlan,
    TuneRefused,
    describe_plan,
    missed_neighbours,
    render,
    tune,
)

K = 10
NQ = 40
VALUES = (1, 2, 4, 8, SMALL["nlist"])


def dataset(synthetic, n=NQ):
    queries = np.load(synthetic["queries"])[:n]
    return parse_dataset(
        render_dataset([{"query_id": f"q{i}", "vector": v.tolist()} for i, v in enumerate(queries)])
    )


def faiss_connection(synthetic, index="ivf_flat"):
    return parse_connection(
        {"backend": "faiss", "index": str(synthetic[index]), "vectors": str(synthetic["vectors"])}
    )


def run(synthetic, plan=None, **kwargs):
    cfg = faiss_connection(synthetic)
    plan = plan or TunePlan("nprobe", VALUES, k=K, repeats=1, max_rps=100_000)
    with open_connector(cfg) as c:
        target = c.describe().target["index"]
        kwargs.setdefault("confirm_target", target)
        return tune(dataset(synthetic), c, plan, **kwargs)


def test_recall_rises_with_the_setting_and_the_smallest_good_value_is_recommended(synthetic):
    report = run(synthetic)
    assert (report["format"], report["status"]) == ("vectraudit.tune", "complete")
    recalls = [p["recall"] for p in report["points"]]
    assert recalls == sorted(recalls)
    assert recalls[-1] == 1.0  # every list probed is the exact search
    rec = report["recommendation"]
    meeting = [p["value"] for p in report["points"] if p["recall"] >= 0.95]
    assert (rec["status"], rec["value"]) == ("ok", min(meeting))
    assert report["reference"]["source"] == "backend exact search"
    assert "round trip included" in report["timing"]["scope"]
    assert report["searches_sent"] == report["searches_planned"]


def test_tune_agrees_with_the_faiss_sweep_on_recall(synthetic):
    report = run(synthetic)
    li = load_index(synthetic["ivf_flat"])
    src = V.from_arrays(li, np.load(synthetic["vectors"]))
    qs = W.given_queries(li, np.load(synthetic["queries"])[:NQ])
    truth = W.ground_truth_ids(S.GroundTruth(src, li.metric), qs, K)
    swept = W.sweep(li, qs, truth, values=list(VALUES), k=K, repeats=1)
    for ours, theirs in zip(report["points"], swept.points, strict=True):
        assert ours["value"] == theirs.value
        assert ours["recall"] == pytest.approx(theirs.recall)


def test_missed_neighbours_say_from_which_value_they_return_and_give_no_cause(synthetic):
    report = run(synthetic, TunePlan("nprobe", VALUES, k=K, repeats=1, max_rps=1e5, target=0.5))
    assert report["missed"], "nprobe=1 misses neighbours on this index"
    at = report["missed_at"]
    for row in report["missed"]:
        for m in row["missed"]:
            assert m["found_from"] is None or m["found_from"] > at
            assert set(m) == {"doc_id", "found_from"}  # no cause field
    text = render(report)
    assert "does not expose why a neighbour was missed" in text
    assert "<- recommended" in text


def test_missed_neighbours_by_hand():
    ds = parse_dataset(render_dataset([{"query_id": "q0", "vector": [0.0]}]))
    rankings = {1: [["a"]], 2: [["a", "b"]], 4: [["a", "b"]]}
    (row,) = missed_neighbours(ds, [["a", "b", "c"]], rankings, 1)
    assert row["recall"] == pytest.approx(1 / 3)
    assert row["missed"] == [{"doc_id": "b", "found_from": 2}, {"doc_id": "c", "found_from": None}]


def test_a_watched_question_keeps_its_ranking_at_every_value(synthetic):
    report = run(synthetic, watch="q3")
    assert list(report["watched"]["by_value"]) == [str(v) for v in VALUES]
    assert "Question q3 (10 exact neighbours):" in render(report)
    with pytest.raises(TuneRefused, match="no question"):
        run(synthetic, watch="nope")


def test_the_sweep_is_refused_before_any_search(synthetic):
    cases = [
        (TunePlan("hnsw_ef", VALUES), {}, "no search-time parameter 'hnsw_ef'"),
        (TunePlan("nprobe", VALUES, max_searches=10), {}, "above the ceiling of 10"),
        (TunePlan("nprobe", VALUES), {"confirm_target": None}, "--confirm-target"),
        (TunePlan("nprobe", VALUES), {"confirm_target": "other"}, "does not match"),
    ]
    for plan, kwargs, message in cases:
        with pytest.raises(TuneRefused, match=message):
            run(synthetic, plan, **kwargs)
    for bad in ({"values": ()}, {"values": (1, 1)}, {"values": (0,)}, {"target": 1.5}):
        with pytest.raises(TuneRefused):
            TunePlan(**{"param": "nprobe", "values": VALUES, **bad})


def test_a_backend_without_exact_search_needs_a_reference(synthetic, monkeypatch):
    from dataclasses import replace

    from vectraudit.connectors.faiss_index import FaissConnector

    real = FaissConnector.capabilities
    monkeypatch.setattr(
        FaissConnector, "capabilities", lambda self: replace(real(self), exact_reference=False)
    )
    with pytest.raises(TuneRefused, match="--reference"):
        run(synthetic)


def test_the_plan_states_every_search(synthetic):
    plan = TunePlan("nprobe", (1, 2), k=K, repeats=3, max_rps=10)
    assert plan.searches(40, exact=True) == 2 * (40 * 3 + 5) + 40
    lines = describe_plan(plan, "faiss", "x.index", 40, exact=True)
    assert lines[0].startswith("290 read-only searches to faiss target x.index")
    assert "nothing is written" in lines[2]


def test_qdrant_accepts_its_parameter(synthetic, tmp_path):
    pytest.importorskip("qdrant_client")
    from tests.test_connector_qdrant import build, connect

    store = build(tmp_path / "qdrant", np.load(synthetic["vectors"])[:300])
    with connect(store) as c:
        report = tune(
            dataset(synthetic, 10), c, TunePlan("hnsw_ef", (16, 64), k=5, repeats=1, max_rps=1e5),
            confirm_target="docs",
        )  # fmt: skip
    assert report["status"] == "complete", report["error"]
    assert [p["recall"] for p in report["points"]] == [1.0, 1.0]  # embedded Qdrant is exact
    assert report["recommendation"]["value"] == 16
    assert report["missed"] == []


def test_pgvector_hnsw_ef_search_reaches_exact_recall(synthetic):
    psycopg = pytest.importorskip("psycopg")
    from tests import test_connector_pgvector as P

    if not P.ADMIN_DSN:
        pytest.skip("Set VECTRAUDIT_TEST_PG_DSN to run pgvector integration tests.")
    del psycopg
    pg = P.pg.__wrapped__(synthetic)
    with P.connect(pg["reader"]) as c:
        report = tune(
            dataset(synthetic, 20), c,
            TunePlan("hnsw.ef_search", (10, 40, 400), k=10, repeats=1, max_rps=1e5),
            confirm_target=c.describe().target["table"],
        )  # fmt: skip
    assert report["status"] == "complete", report["error"]
    points = report["points"]
    assert points[-1]["recall"] == 1.0
    assert all(a["recall"] <= b["recall_ci_high"] + 1e-9 for a, b in pairwise(points))
    meeting = [p["value"] for p in points if p["recall"] >= 0.95]
    assert report["recommendation"]["value"] == min(meeting)


def test_tune_command(synthetic, tmp_path):
    data = tmp_path / "queries.jsonl"
    queries = np.load(synthetic["queries"])[:NQ]
    data.write_text(
        render_dataset([{"query_id": f"q{i}", "vector": v.tolist()} for i, v in enumerate(queries)])
    )
    conn = tmp_path / "connection.json"
    conn.write_text(
        json.dumps(
            {
                "backend": "faiss",
                "index": str(synthetic["ivf_flat"]),
                "vectors": str(synthetic["vectors"]),
            }
        )
    )
    out = tmp_path / "tune.json"
    args = ["tune", "--connection", str(conn), "--dataset", str(data), "--param", "nprobe"]
    args += ["--values", "1,4,16", "--out", str(out), "--repeats", "1", "--max-rps", "1000"]
    cli = CliRunner()
    refused = cli.invoke(app, args)
    assert refused.exit_code == 1
    assert "--confirm-target" in " ".join(refused.output.split())
    assert not out.exists()
    target = synthetic["ivf_flat"].name
    done = cli.invoke(app, [*args, "--confirm-target", target, "--query", "q1"])
    assert done.exit_code == 0, done.output
    assert "read-only searches to faiss" in done.output
    assert "Recommended nprobe=" in done.output
    report = json.loads(out.read_text())
    assert report["status"] == "complete"
    assert report["connection"]["backend"] == "faiss"
    unreachable = cli.invoke(
        app, [*args, "--confirm-target", target, "--target", "1.0", "--values", "1"]
    )
    assert unreachable.exit_code == 2


# --- Phase 43: filters, batch and comparing two sweeps ----------------------------------------


def qdrant_report(synthetic, tmp_path, plan):
    pytest.importorskip("qdrant_client")
    from tests.test_connector_qdrant import build, connect

    store = build(tmp_path / "qdrant", np.load(synthetic["vectors"])[:300])
    with connect(store) as c:
        return tune(dataset(synthetic, 10), c, plan, confirm_target="docs")


def test_each_filter_gets_its_own_curve_exact_search_and_pass_rate(synthetic, tmp_path):
    plan = TunePlan(
        "hnsw_ef", (16, 64), k=5, repeats=1, max_rps=1e5,
        filters=({"lang": "de"}, {"public": True}),
    )  # fmt: skip
    report = qdrant_report(synthetic, tmp_path, plan)
    assert report["status"] == "complete", report["error"]
    by_filter = {json.dumps(c["filter"]): c for c in report["filtered"]}
    assert by_filter['{"lang": "de"}']["pass_rate"] == pytest.approx(0.5)
    assert by_filter['{"public": true}']["pass_rate"] == pytest.approx(100 / 300)
    for curve in report["filtered"]:
        assert [p["value"] for p in curve["points"]] == [16, 64]
        assert curve["recommendation"]["value"] == 16
    # one unfiltered curve and two filtered ones, each with its own exact searches
    assert report["searches_sent"] == report["searches_planned"] == 3 * (2 * (10 + 5) + 10)
    assert 'Filter {"lang": "de"} (pass rate 50.00%): recommended hnsw_ef=16' in render(report)


def test_filters_are_refused_where_they_cannot_be_measured(synthetic):
    plan = TunePlan("nprobe", (1, 4), filters=({"lang": "de"},))
    with pytest.raises(TuneRefused, match="does not support filters"):
        run(synthetic, plan)  # this FAISS connection has no metadata


def test_batch_throughput_is_reported_apart_from_latency(synthetic, tmp_path):
    report = run(synthetic, TunePlan("nprobe", (1, 8), k=K, repeats=2, max_rps=1e6, batch=True))
    assert report["batch"]["measured"] is True
    assert "never a latency" in report["batch"]["scope"]
    for p in report["points"]:
        assert p["batch_qps"] > 0
        assert p["batch_recall"] == pytest.approx(p["recall"])  # the same results, all at once
        assert set(p) >= {"latency_mean_ms", "latency_p95_ms"}
    assert report["searches_sent"] == report["searches_planned"]
    assert "Batch q/s is a throughput" in render(report)
    plain = run(synthetic)
    assert all("batch_qps" not in p for p in plain["points"])
    qdrant = qdrant_report(
        synthetic, tmp_path, TunePlan("hnsw_ef", (16,), k=5, repeats=1, max_rps=1e5, batch=True)
    )
    assert qdrant["points"][0]["batch_recall"] == 1.0


def test_batch_is_refused_where_the_backend_has_no_batch_call(synthetic, monkeypatch):
    from vectraudit.connectors.faiss_index import FaissConnector

    monkeypatch.setattr(FaissConnector, "batch_search", False)
    with pytest.raises(TuneRefused, match="no batch search call"):
        run(synthetic, TunePlan("nprobe", (1,), batch=True))


def other_index(synthetic, index):
    plan = TunePlan("nprobe", VALUES, k=K, repeats=1, max_rps=1e6)
    with open_connector(faiss_connection(synthetic, index)) as c:
        return tune(dataset(synthetic), c, plan, confirm_target=c.describe().target["index"])


def test_two_setups_compare_at_matched_values(synthetic):
    from vectraudit.evaluation.tune import compare_tunes, render_comparison

    flat, pq = run(synthetic), other_index(synthetic, "ivf_pq")
    result = compare_tunes(flat, pq)
    assert result["comparable"] is True
    assert [m["value"] for m in result["matched"]] == list(VALUES)
    assert all(m["p95_ms_change"] is not None for m in result["matched"])  # same backend
    assert result["matched"][-1]["recall_b"] <= result["matched"][-1]["recall_a"]  # PQ loses
    text = render_comparison(result)
    assert text.startswith("A: faiss target")
    assert "Questions missing neighbours:" in text
    # Different questions, k or references: not compared at all.
    assert not compare_tunes(flat, {**pq, "k": 5})["comparable"]
    other = {**pq, "dataset": {"sha256": "x", "n_queries": 1}}
    assert "different question sets" in compare_tunes(flat, other)["reasons"][0]
    external = {**pq, "reference": {"source": "external:annb", "sha256": "abc"}}
    assert "different exact references" in compare_tunes(flat, external)["reasons"][0]
    assert not compare_tunes(flat, {**pq, "status": "cancelled"})["comparable"]
    # Another backend: recall is compared, latency is not subtracted.
    elsewhere = {**pq, "backend": {**pq["backend"], "backend": "qdrant"}}
    crossed = compare_tunes(flat, elsewhere)
    assert all(m["p95_ms_change"] is None for m in crossed["matched"])
    assert "not subtracted" in crossed["latency_note"]


def test_tune_compare_command(synthetic, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(run(synthetic)))
    b.write_text(json.dumps(other_index(synthetic, "ivf_pq")))
    cli = CliRunner()
    out = tmp_path / "cmp.json"
    done = cli.invoke(app, ["tune-compare", str(a), str(b), "--json", str(out)])
    assert done.exit_code == 0, done.output
    assert "nprobe=1: recall" in done.output
    assert json.loads(out.read_text())["format"] == "vectraudit.tune_comparison"
    b.write_text(json.dumps({**json.loads(b.read_text()), "k": 3}))
    assert cli.invoke(app, ["tune-compare", str(a), str(b)]).exit_code == 4
    b.write_text("{}")
    assert cli.invoke(app, ["tune-compare", str(a), str(b)]).exit_code not in (0, 4)


# --- Phase 44.2: handing the recommendation to release checks ----------------------------------


def test_a_tuned_connection_carries_the_recommendation_and_keeps_paths_working(tmp_path):
    from vectraudit.evaluation.tune import tuned_connection

    report = {
        "format": "vectraudit.tune",
        "status": "complete",
        "backend": {"backend": "faiss"},
        "param": "nprobe",
        "recommendation": {"value": 8},
    }
    raw = {
        "backend": "faiss",
        "index": "data/ivf.index",
        "vectors": "/abs/vectors.npy",
        "params": {"nprobe": 1, "k_factor": 2},
    }
    source, out = tmp_path / "conf", tmp_path / "conf" / "tuned"
    tuned = tuned_connection(raw, report, source, out)
    assert tuned["params"] == {"nprobe": 8, "k_factor": 2}
    assert (out / tuned["index"]).resolve() == (source / "data/ivf.index").resolve()
    assert tuned["vectors"] == "/abs/vectors.npy"  # absolute paths are kept
    assert raw["params"] == {"nprobe": 1, "k_factor": 2}  # the input is not changed
    for bad, message in [
        ({**report, "status": "cancelled"}, "recommended no value"),
        ({**report, "recommendation": {"value": None}}, "recommended no value"),
        ({**report, "backend": {"backend": "qdrant"}}, "ran on qdrant"),
        ({**report, "format": "vectraudit.run"}, "Not a vectraudit.tune"),
    ]:
        with pytest.raises(TuneRefused, match=message):
            tuned_connection(raw, bad, source, out)


def test_a_tuned_connection_passes_a_release_check_against_the_untuned_one(synthetic, tmp_path):
    """Phase 44 acceptance: tune, write the tuned copy, record both, check."""
    data = tmp_path / "queries.jsonl"
    queries = np.load(synthetic["queries"])[:NQ]
    data.write_text(
        render_dataset([{"query_id": f"q{i}", "vector": v.tolist()} for i, v in enumerate(queries)])
    )
    # Relative paths, so the copy in another folder must rewrite them to keep working.
    folder = tmp_path / "retrieval"
    folder.mkdir()
    for key in ("ivf_flat", "vectors"):
        (folder / synthetic[key].name).symlink_to(synthetic[key])
    conn = folder / "connection.json"
    untuned = {
        "backend": "faiss",
        "index": synthetic["ivf_flat"].name,
        "vectors": synthetic["vectors"].name,
        "params": {"nprobe": 1},
    }
    conn.write_text(json.dumps(untuned))
    tuned_path = tmp_path / "tuned" / "connection.json"
    cli = CliRunner()
    args = ["tune", "--connection", str(conn), "--dataset", str(data), "--param", "nprobe"]
    args += ["--values", ",".join(map(str, VALUES)), "--out", str(tmp_path / "tune.json")]
    args += ["--repeats", "1", "--max-rps", "1000", "--target", "1.0"]
    args += ["--confirm-target", synthetic["ivf_flat"].name]
    same = cli.invoke(app, [*args, "--write-connection", str(conn)])
    assert same.exit_code == 1
    assert "names the connection file itself" in " ".join(same.output.split())
    done = cli.invoke(app, [*args, "--write-connection", str(tuned_path)])
    assert done.exit_code == 0, done.output
    recommended = json.loads((tmp_path / "tune.json").read_text())["recommendation"]["value"]
    assert recommended > 1, done.output  # nprobe 1 misses neighbours on this data
    assert f"with nprobe={recommended}" in " ".join(done.output.split())
    assert json.loads(conn.read_text()) == untuned  # the original is untouched
    assert json.loads(tuned_path.read_text())["params"] == {"nprobe": recommended}

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"metrics": {"ann_recall": {"min": 1.0}}, "min_queries": 10}))
    runs = {}
    for name, connection in (("base", conn), ("tuned", tuned_path)):
        runs[name] = tmp_path / f"{name}.run.json"
        record = ["record", "--connection", str(connection), "--dataset", str(data)]
        result = cli.invoke(app, [*record, "--out", str(runs[name]), "--k", str(K)])
        assert result.exit_code == 0, result.output
    result_json = tmp_path / "result.json"
    check = ["check", str(runs["base"]), str(runs["tuned"]), "--policy", str(policy)]
    result = cli.invoke(app, [*check, "--json", str(result_json), "--no-github-summary"])
    assert result.exit_code == 0, result.output
    verdict = json.loads(result_json.read_text())
    assert verdict["verdict"] == "pass"
    assert verdict["metrics"]["ann_recall"]["candidate"] == 1.0
    assert verdict["latency"]["compared"] is True  # the latency change is reported beside it
    assert isinstance(verdict["latency"]["p95_change_pct"], float)
    # The untuned setting is what the policy would have blocked.
    swapped = ["check", str(runs["tuned"]), str(runs["base"]), "--policy", str(policy)]
    assert cli.invoke(app, [*swapped, "--no-github-summary"]).exit_code == 3


def test_no_tuned_connection_is_written_without_a_recommendation(synthetic, tmp_path):
    data = tmp_path / "queries.jsonl"
    queries = np.load(synthetic["queries"])[:NQ]
    data.write_text(
        render_dataset([{"query_id": f"q{i}", "vector": v.tolist()} for i, v in enumerate(queries)])
    )
    conn = tmp_path / "connection.json"
    index, vectors = str(synthetic["ivf_flat"]), str(synthetic["vectors"])
    conn.write_text(json.dumps({"backend": "faiss", "index": index, "vectors": vectors}))
    out = tmp_path / "tuned.json"
    args = ["tune", "--connection", str(conn), "--dataset", str(data), "--param", "nprobe"]
    args += ["--values", "1", "--out", str(tmp_path / "tune.json"), "--repeats", "1"]
    args += ["--max-rps", "1000", "--target", "1.0", "--write-connection", str(out)]
    result = CliRunner().invoke(app, [*args, "--confirm-target", synthetic["ivf_flat"].name])
    assert result.exit_code == 2
    assert "Not written" in result.output
    assert not out.exists()
