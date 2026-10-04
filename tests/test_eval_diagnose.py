"""Diagnoses: findings with evidence levels."""

from __future__ import annotations

import copy
from typing import Any

from tests.test_eval_compare import fake_run
from tests.test_eval_runner import make_dataset
from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.report import render_html, render_markdown
from vectraudit.evaluation.runner import RunOptions, execute_run

ANN_POLICY = parse_policy({"metrics": {"ann_recall": {"max_drop": 0.01}}})


def ids(result: dict[str, Any]) -> list[str]:
    return [f["id"] for f in result["findings"]]


def finding(result: dict[str, Any], fid: str) -> dict[str, Any]:
    return next(f for f in result["findings"] if f["id"] == fid)


def test_ivf_unprobed_lists(uniform_ivf) -> None:
    index, vectors, ds = uniform_ivf
    conn = FaissConnector(index, vectors=vectors)
    base = execute_run(ds, conn, RunOptions(k=10, params={"nprobe": 16}))
    cand = execute_run(ds, conn, RunOptions(k=10, params={"nprobe": 1}))
    result = compare_runs(base, cand, ANN_POLICY)
    assert result["verdict"] == "fail"
    assert ids(result)[:3] == ["approximation-loss", "ivf-unprobed", "approximation-config"]
    unprobed = finding(result, "ivf-unprobed")
    assert unprobed["level"] == "verified"
    assert "Raise nprobe to" in unprobed["suggestion"]
    assert unprobed["data"]["recommended_nprobe"] > 1
    assert "list not probed" in unprobed["facts"][0]
    assert "nprobe: 16 → 1" in finding(result, "approximation-config")["facts"][0]
    assert finding(result, "approximation-config")["level"] == "supported"


def test_pq_quantization(synthetic) -> None:
    ds = make_dataset(synthetic, n=30)
    flat = FaissConnector(synthetic["ivf_flat"], vectors=synthetic["vectors"])
    pq = FaissConnector(synthetic["ivf_pq"], vectors=synthetic["vectors"])
    base = execute_run(ds, flat, RunOptions(k=10, params={"nprobe": 16}))
    cand = execute_run(ds, pq, RunOptions(k=10, params={"nprobe": 16}))
    result = compare_runs(base, cand, ANN_POLICY)
    q = finding(result, "ivf-quantization")
    assert q["level"] == "verified"
    assert "Raising nprobe will not recover" in q["suggestion"]
    assert "ivf-unprobed" not in ids(result)


def test_relevance_change_without_approximation_loss() -> None:
    base = fake_run([1.0] * 40, ann=[1.0] * 40)
    cand = fake_run([1.0] * 20 + [0.0] * 20, ann=[1.0] * 40)
    cand["provenance"]["labels"] = {"embedding_model": "e5-small"}
    result = compare_runs(base, cand)
    f = finding(result, "relevance-not-approximation")
    assert f["level"] == "supported"
    assert "labels.embedding_model" in f["facts"][-1]
    assert "approximation-loss" not in ids(result)


def test_relevance_drift_is_only_a_hypothesis() -> None:
    base = fake_run([1.0] * 40, ann=[1.0] * 40)
    cand = fake_run([1.0] * 20 + [0.0] * 20, ann=[1.0] * 40)
    f = finding(compare_runs(base, cand), "relevance-drift")
    assert f["level"] == "hypothesis"


def test_relevance_drop_without_exact_reference() -> None:
    result = compare_runs(fake_run([1.0] * 40), fake_run([1.0] * 20 + [0.0] * 20))
    f = finding(result, "relevance-drop")
    assert f["level"] == "verified"
    assert "exact reference" in f["suggestion"]


def _with_plans(run: dict[str, Any], uses_index: list[bool]) -> dict[str, Any]:
    for q, indexed in zip(run["queries"], uses_index, strict=True):
        q["evidence"] = {
            "kind": "pg_plan",
            "source": "planner estimate (EXPLAIN)",
            "nodes": ["Limit", "Index Scan"] if indexed else ["Limit", "Sort", "Seq Scan"],
            "indexes": ["ix"] if indexed else [],
        }
    return run


def test_plan_lost_index_with_latency() -> None:
    base = _with_plans(fake_run([1.0] * 40, latency=1.0), [True] * 40)
    cand = _with_plans(fake_run([1.0] * 40, latency=5.0), [False] * 30 + [True] * 10)
    result = compare_runs(base, cand)
    f = finding(result, "plan-lost-index")
    assert "30 of 40 queries" in f["facts"][0]
    assert "p95 latency rose 400%" in f["facts"][1]
    assert "latency-change" in ids(result)
    back = compare_runs(cand, base)
    assert finding(back, "plan-gained-index")["level"] == "verified"


def test_short_filtered_results() -> None:
    base = fake_run([1.0] * 40)
    cand = copy.deepcopy(base)
    for q in cand["queries"][:5]:
        q["ids"] = ["a"]
        q["filtered"] = True
    f = finding(compare_runs(base, cand), "short-results")
    assert "5 queries returned fewer than k = 2" in f["facts"][0]
    assert "All of them are filtered" in f["facts"][1]
    assert "iterative scans" in f["suggestion"]


def test_concentrations() -> None:
    tags = [["faq"]] * 8 + [["other"]] * 32
    base = fake_run([1.0] * 40, tags=tags)
    cand = fake_run([0.0] * 8 + [1.0] * 32, tags=tags)
    for q in cand["queries"][:8]:
        q["filtered"] = True
    for q in base["queries"][:8]:
        q["filtered"] = True
    result = compare_runs(base, cand)
    assert (
        "8 of 8 regressed queries are tagged 'faq'"
        in finding(result, "cohort-concentration")["facts"][0]
    )
    assert finding(result, "filtered-concentration")["level"] == "verified"


def test_latency_with_parameter_change_and_nondeterminism() -> None:
    base = fake_run([1.0] * 40, latency=1.0)
    cand = fake_run([1.0] * 40, latency=2.0, search={"k": 2, "params": {"nprobe": 64}})
    cand["queries"][0]["nondeterministic"] = True
    result = compare_runs(base, cand)
    f = finding(result, "latency-params")
    assert f["level"] == "supported"
    assert "nprobe: 8 → 64" in f["facts"][1]
    assert finding(result, "nondeterministic")["level"] == "verified"


def test_no_change_and_level_ordering() -> None:
    run = fake_run([1.0, 0.5] * 20)
    assert ids(compare_runs(run, copy.deepcopy(run))) == ["no-change"]
    base = fake_run([1.0] * 40, ann=[1.0] * 40)
    cand = fake_run([1.0] * 20 + [0.0] * 20, ann=[1.0] * 40, latency=3.0)
    levels = [f["level"] for f in compare_runs(base, cand)["findings"]]
    assert levels == sorted(levels, key=["verified", "supported", "hypothesis"].index)


def test_incomparable_checks_have_no_findings() -> None:
    run = fake_run([1.0] * 5)
    other = fake_run([1.0] * 5, search={"k": 3, "params": {}})
    assert compare_runs(run, other)["findings"] == []


def test_findings_in_reports() -> None:
    base = fake_run([1.0] * 40, ann=[1.0] * 40)
    cand = fake_run([1.0] * 20 + [0.0] * 20, ann=[1.0] * 40)
    cand["provenance"]["labels"] = {"embedding_model": "<b>e5</b>"}
    result = compare_runs(base, cand)
    page = render_html(result)
    assert ">Diagnosis</h2>" in page
    assert '<span class="level supported">supported</span>' in page
    assert "&lt;b&gt;e5&lt;/b&gt;" in page
    md = render_markdown(result)
    assert "**Diagnosis**" in md
    assert "_supported_: Relevance fell" in md


def test_shared_misses_are_not_blamed_on_the_change(uniform_ivf, tmp_path) -> None:
    """Both runs miss unprobed neighbours at nprobe 4; only PQ's new misses count."""
    import faiss
    import numpy as np

    index, vectors, ds = uniform_ivf
    x = np.load(vectors)
    flat = faiss.read_index(str(index))
    pq = faiss.IndexIVFPQ(faiss.IndexFlatL2(16), 16, 16, 4, 8)
    pq.train(x)
    pq.add(x)
    faiss.write_index(pq, str(tmp_path / "pq.index"))
    assert flat.nlist == pq.nlist
    base = execute_run(
        ds, FaissConnector(index, vectors=vectors), RunOptions(k=10, params={"nprobe": 4})
    )
    cand = execute_run(
        ds,
        FaissConnector(tmp_path / "pq.index", vectors=vectors),
        RunOptions(k=10, params={"nprobe": 4}),
    )
    result = compare_runs(base, cand, ANN_POLICY)
    q = finding(result, "ivf-quantization")
    assert "the baseline found were missed by the candidate" in q["facts"][0]
    assert "predate this change" in q["facts"][1]
    assert "ivf-unprobed" not in ids(result)
