"""Metrics, run execution and run files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vectraudit.connectors.base import (
    BackendInfo,
    Capabilities,
    Connector,
    ConnectorError,
    Distance,
    ErrorCode,
    SearchRequest,
)
from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.evaluation.dataset import Dataset, parse_dataset
from vectraudit.evaluation.metrics import (
    METRIC_CONTRACT_VERSION,
    ann_recall,
    latency_summary,
    metric_contract,
    semantic_metrics,
)
from vectraudit.evaluation.runfile import RunFileError, check_run, load_run, save_run
from vectraudit.evaluation.runner import RunOptions, execute_run, provenance


def make_dataset(synthetic: dict[str, Path], n: int = 20, **extra: Any) -> Dataset:
    """Labelled queries: each query's 3 exact neighbours are relevant (grades 3, 2, 1)."""
    import faiss

    vectors = np.load(synthetic["vectors"])
    queries = np.load(synthetic["queries"])[:n]
    _, nn = faiss.knn(queries, vectors, 3)
    rows = []
    for i, (q, ids) in enumerate(zip(queries, nn, strict=True)):
        row: dict[str, Any] = {
            "query_id": f"q{i}",
            "vector": q.tolist(),
            "relevant": {str(d): 3 - r for r, d in enumerate(ids)},
            "tags": ["even" if i % 2 == 0 else "odd"],
        }
        rows.append({**row, **extra})
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


# --- metrics -----------------------------------------------------------------------------


def test_semantic_metrics() -> None:
    labels = {"a": 2.0, "b": 1.0, "z": 0.0}
    m = semantic_metrics(["x", "b", "a", "b"], labels, 4)
    assert m["recall"] == 1.0
    assert m["precision"] == 0.5
    assert m["mrr"] == 0.5
    assert 0 < m["ndcg"] < 1
    assert semantic_metrics(["a", "b"], labels, 2)["ndcg"] == pytest.approx(1.0)
    assert semantic_metrics([], labels, 3) == {"recall": 0, "precision": 0, "mrr": 0, "ndcg": 0}


def test_ann_recall_and_latency() -> None:
    assert ann_recall(["a", "b", "c"], ["a", "c", "d", "e"]) == 0.5
    assert ann_recall(["a"], []) is None
    assert latency_summary([]) is None
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary is not None
    assert summary["p50"] == 2.5
    assert summary["max"] == 4.0
    assert metric_contract(5)["k"] == 5
    assert metric_contract(5)["version"] == METRIC_CONTRACT_VERSION


# --- runner ------------------------------------------------------------------------------


def test_complete_run_on_faiss(synthetic) -> None:
    ds = make_dataset(synthetic)
    conn = FaissConnector(synthetic["ivf_flat"], vectors=synthetic["vectors"])
    full = execute_run(ds, conn, RunOptions(k=5, params={"nprobe": 16}, repeats=2))
    pq = FaissConnector(synthetic["ivf_pq"], vectors=synthetic["vectors"])
    low = execute_run(ds, pq, RunOptions(k=5, params={"nprobe": 1}))

    assert full["status"] == "complete"
    assert full["error"] is None
    assert full["dataset"]["sha256"] == ds.sha256
    assert full["backend"]["backend"] == "faiss"
    assert full["search"] == {
        "k": 5,
        "params": {"nprobe": 16},
        "embedder": None,
        "evidence": True,
        "exact_reference": True,
        "exact_reference_note": None,
    }
    s = full["summary"]
    assert s["n_ok"] == 20
    assert s["n_errors"] == 0
    assert s["n_labelled"] == 20
    assert s["ann_recall"] == 1.0
    assert s["metrics"]["recall"] == 1.0
    assert s["latency_ms"]["p95"] >= s["latency_ms"]["p50"] >= 0
    assert low["summary"]["ann_recall"] < 1.0
    q = full["queries"][0]
    assert q["query_id"] == "q0"
    assert q["tags"] == ["even"]
    assert len(q["ids"]) == 5
    assert q["ids"] == q["exact_ids"]
    assert not q["nondeterministic"]
    json.dumps(full, allow_nan=False)


def test_unlabelled_queries_and_no_exact(synthetic) -> None:
    ds = parse_dataset(json.dumps({"query_id": "u", "vector": [0.0] * 32}))
    run = execute_run(ds, FaissConnector(synthetic["hnsw_flat"]), RunOptions(k=3, warmup=0))
    assert run["status"] == "complete"
    assert run["queries"][0]["metrics"] is None
    assert run["summary"]["metrics"] is None
    assert run["summary"]["ann_recall"] is None
    assert run["search"]["exact_reference"] is False


def test_filtered_queries_on_faiss_make_a_partial_run(synthetic) -> None:
    ds = make_dataset(synthetic, n=4, filter={"lang": "en"})
    run = execute_run(ds, FaissConnector(synthetic["flat_l2"]), RunOptions(k=3))
    assert run["status"] == "partial"
    assert run["summary"]["n_errors"] == 4
    assert run["summary"]["errors_by_code"] == {"unsupported": 4}
    assert run["summary"]["error_rate"] == 1.0
    assert run["queries"][0]["error"]["code"] == "unsupported"
    assert run["summary"]["metrics"] is None


def test_requested_exact_reference_must_exist(synthetic) -> None:
    run = execute_run(
        make_dataset(synthetic, n=2), FaissConnector(synthetic["hnsw_flat"]), RunOptions(exact=True)
    )
    assert run["status"] == "failed"
    assert run["error"]["code"] == "unsupported"
    assert run["queries"] == []


def test_dimension_mismatch_fails_before_searching(synthetic) -> None:
    ds = parse_dataset(json.dumps({"query_id": "q", "vector": [1.0, 2.0]}))
    run = execute_run(ds, FaissConnector(synthetic["flat_l2"]))
    assert run["status"] == "failed"
    assert "dimension 2" in run["error"]["message"]


def test_text_queries_are_embedded(synthetic) -> None:
    ds = parse_dataset(
        json.dumps({"query_id": "t", "text": "hello"})
        + "\n"
        + json.dumps({"query_id": "v", "vector": [0.0] * 32})
    )
    conn = FaissConnector(synthetic["flat_l2"])
    missing = execute_run(ds, conn)
    assert missing["status"] == "failed"
    assert "--embedder" in missing["error"]["hint"]

    seen: list[list[str]] = []

    def embed(texts):
        seen.append(list(texts))
        return np.ones((len(texts), 32), dtype=np.float32)

    run = execute_run(ds, conn, RunOptions(k=2), embed=embed, embedder_name="fake")
    assert run["status"] == "complete"
    assert seen == [["hello"]]
    assert run["search"]["embedder"] == "fake"
    assert run["timing"]["embedding_s"] >= 0

    bad = execute_run(ds, conn, embed=lambda texts: np.ones((3, 32)))
    assert bad["status"] == "failed"
    assert "wrong shape" in bad["error"]["message"]


class Scripted(Connector):
    """A connector whose search behaviour is a function of the call number."""

    name = "scripted"

    def __init__(self, behaviour) -> None:
        self.calls = 0
        self.behaviour = behaviour

    def describe(self) -> BackendInfo:
        return BackendInfo("scripted", Distance.COSINE, 2, 10)

    def capabilities(self) -> Capabilities:
        return Capabilities(exact_reference=True)

    def _search(self, request: SearchRequest) -> list[str]:
        self.calls += 1
        return self.behaviour(self.calls, request)


def _two_dim(n: int) -> Dataset:
    rows = [{"query_id": f"q{i}", "vector": [1.0, 0.0], "relevant": {"a": 1}} for i in range(n)]
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


def test_connection_loss_fails_the_run_and_keeps_finished_queries() -> None:
    def behaviour(call: int, request: SearchRequest) -> list[str]:
        if call > 4:
            raise ConnectorError(ErrorCode.CONNECTION, "connection reset")
        return ["a", "b"]

    run = execute_run(_two_dim(5), Scripted(behaviour), RunOptions(k=2, warmup=0))
    assert run["status"] == "failed"
    assert run["error"]["code"] == "connection"
    assert len(run["queries"]) == 2
    assert run["summary"]["n_executed"] == 2
    assert run["summary"]["n_queries"] == 5


def test_cancel_and_internal_errors() -> None:
    def interrupt(call: int, request: SearchRequest) -> list[str]:
        if call == 3:
            raise KeyboardInterrupt
        return ["a"]

    run = execute_run(_two_dim(4), Scripted(interrupt), RunOptions(warmup=0))
    assert run["status"] == "cancelled"
    assert len(run["queries"]) == 1

    def crash(call: int, request: SearchRequest) -> list[str]:
        raise RuntimeError("boom")

    run = execute_run(_two_dim(2), Scripted(crash))
    assert run["status"] == "failed"
    assert run["error"]["code"] == "internal"
    assert "boom" in run["error"]["message"]


def test_nondeterministic_results_are_flagged() -> None:
    def flaky(call: int, request: SearchRequest) -> list[str]:
        return ["a", "b"] if call % 2 else ["b", "a"]

    run = execute_run(_two_dim(1), Scripted(flaky), RunOptions(k=2, warmup=0, repeats=2))
    assert run["queries"][0]["nondeterministic"]
    assert run["summary"]["n_nondeterministic"] == 1


def test_run_options_validation() -> None:
    with pytest.raises(ValueError, match="k must"):
        RunOptions(k=0)
    with pytest.raises(ValueError, match="repeats"):
        RunOptions(repeats=0)


def test_provenance_reads_github_actions(monkeypatch) -> None:
    for key, value in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_SHA": "abc123",
        "GITHUB_REPOSITORY": "acme/rag",
        "GITHUB_RUN_ID": "42",
        "GITHUB_REF": "refs/pull/7/merge",
    }.items():
        monkeypatch.setenv(key, value)
    p = provenance({"embedding_model": "e5-large"})
    assert p["commit_sha"] == "abc123"
    assert p["ci"]["run_url"] == "https://github.com/acme/rag/actions/runs/42"
    assert p["labels"] == {"embedding_model": "e5-large"}
    assert p["runner"]["python"]


# --- run files ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_record(synthetic) -> dict[str, Any]:
    conn = FaissConnector(synthetic["ivf_flat"], vectors=synthetic["vectors"])
    return execute_run(make_dataset(synthetic, n=5), conn, RunOptions(k=3))


def test_save_and_load_round_trip(run_record, tmp_path) -> None:
    path = save_run(run_record, tmp_path / "runs" / "base.json")
    assert load_run(path) == run_record


def test_failed_runs_are_valid_files() -> None:
    def crash(call: int, request: SearchRequest) -> list[str]:
        raise ConnectorError(ErrorCode.AUTH, "bad key")

    record = execute_run(_two_dim(1), Scripted(crash))
    assert check_run(json.loads(json.dumps(record)))["status"] == "failed"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(format="other"), "not a vectraudit run"),
        (lambda r: r.update(format_version=99), "version 99"),
        (lambda r: r.update(status="done"), "status"),
        (lambda r: r["dataset"].pop("sha256"), "dataset.sha256"),
        (lambda r: r["search"].update(k=0), "search.k"),
        (lambda r: r["queries"][0].update(ids=[1, 2]), r"queries\[0\]: ids"),
        (lambda r: r["queries"][0].update(ids=["a"] * 9), "more ids than k"),
        (lambda r: r["queries"][0].update(latency_ms=-1), "latency_ms"),
        (lambda r: r["queries"][0]["metrics"].update(recall=1.5), "metrics.recall"),
        (lambda r: r["queries"][0].update(ann_recall=2), "ann_recall"),
        (lambda r: r["queries"].append(dict(r["queries"][0])), "duplicate query_id"),
        (lambda r: r.update(backend=None), "missing backend"),
    ],
)
def test_check_run_rejects_bad_records(run_record, mutate, message) -> None:
    record = json.loads(json.dumps(run_record))
    mutate(record)
    with pytest.raises(RunFileError, match=message):
        check_run(record)


def test_load_run_errors(tmp_path) -> None:
    with pytest.raises(RunFileError, match="not found"):
        load_run(tmp_path / "missing.json")
    (tmp_path / "nan.json").write_text('{"x": NaN}')
    with pytest.raises(RunFileError, match="not valid JSON"):
        load_run(tmp_path / "nan.json")
    (tmp_path / "list.json").write_text("[]")
    with pytest.raises(RunFileError, match=r"list\.json: a run must be"):
        load_run(tmp_path / "list.json")
    with pytest.raises(ValueError, match="Out of range float"):
        save_run({"x": float("nan")}, tmp_path / "out.json")


# --- evidence ----------------------------------------------------------------------------


def test_faiss_ivf_evidence_explains_misses(uniform_ivf) -> None:
    index, vectors, ds = uniform_ivf
    conn = FaissConnector(index, vectors=vectors)
    low = execute_run(ds, conn, RunOptions(k=10, params={"nprobe": 1}))
    full = execute_run(ds, conn, RunOptions(k=10, params={"nprobe": 16}))
    ev = [q["evidence"] for q in low["queries"]]
    assert all(e["kind"] == "faiss_ivf" for e in ev)
    assert all(e["source"].startswith("replayed") for e in ev)
    assert all(e["nprobe"] == 1 for e in ev)
    # IVF-Flat codes are exact: every miss at nprobe=1 is an unprobed list.
    missed = [e for e in ev if e["missed"]]
    assert missed, "nprobe=1 should miss some neighbours"
    for e, q in zip(ev, low["queries"], strict=True):
        assert set(e["missed"]) == set(q["exact_ids"]) - set(q["ids"])
        assert set(e["missed_by_reason"]) <= {"CELL_NOT_PROBED"}
        assert 1 <= e["min_nprobe_for_all"] <= 16
        if e["missed"]:
            assert e["min_nprobe_for_all"] > 1
    assert all(not q["evidence"]["missed"] for q in full["queries"])


def test_pq_misses_in_probed_lists_are_quantization(synthetic) -> None:
    ds = make_dataset(synthetic, n=20)
    conn = FaissConnector(synthetic["ivf_pq"], vectors=synthetic["vectors"])
    run = execute_run(ds, conn, RunOptions(k=10, params={"nprobe": 16}))
    reasons = {r for q in run["queries"] for r in q["evidence"]["missed_by_reason"]}
    assert reasons == {"QUANTIZATION"}


def test_evidence_is_optional(synthetic) -> None:
    ds = make_dataset(synthetic, n=3)
    ivf = FaissConnector(synthetic["ivf_flat"], vectors=synthetic["vectors"])
    off = execute_run(ds, ivf, RunOptions(evidence=False))
    assert all(q["evidence"] is None for q in off["queries"])
    no_exact = execute_run(ds, FaissConnector(synthetic["ivf_flat"]), RunOptions())
    assert all(q["evidence"] is None for q in no_exact["queries"])
    hnsw = execute_run(ds, FaissConnector(synthetic["hnsw_flat"]), RunOptions())
    assert all(q["evidence"] is None for q in hnsw["queries"])


def test_evidence_errors_do_not_fail_queries() -> None:
    class Flaky(Scripted):
        def evidence(self, request, found, exact):
            raise ConnectorError(ErrorCode.QUERY, "plan unavailable")

    run = execute_run(_two_dim(2), Flaky(lambda call, request: ["a"]), RunOptions(warmup=0))
    assert run["status"] == "complete"
    assert run["queries"][0]["evidence"]["kind"] == "unavailable"
    assert run["queries"][0]["evidence"]["error"]["message"] == "plan unavailable"
