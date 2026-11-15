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
