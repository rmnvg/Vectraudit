"""Comparisons, policies and verdicts."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from tests.test_eval_runner import make_dataset
from vectraudit.connectors.faiss_index import FaissConnector
from vectraudit.evaluation.compare import Verdict, compare_runs, config_diff
from vectraudit.evaluation.policy import (
    DEFAULT_POLICY,
    PolicyError,
    load_policy,
    parse_policy,
)
from vectraudit.evaluation.runfile import check_run
from vectraudit.evaluation.runner import RunOptions, execute_run, summarize

ENV = {"system": "Linux", "machine": "x86_64", "cpu_count": 4, "python": "3.12.1"}


def fake_run(
    values: list[float],
    *,
    ann: list[float] | None = None,
    latency: float = 1.0,
    tags: list[list[str]] | None = None,
    labelled: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """A valid complete run: query i has nDCG/recall/MRR/precision = values[i]."""
    queries = []
    for i, v in enumerate(values):
        queries.append(
            {
                "query_id": f"q{i}",
                "tags": (tags or [[]] * len(values))[i],
                "ids": ["a", "b"] if v >= 0.5 else ["x", "y"],
                "exact_ids": ["a", "b"] if ann is not None else None,
                "latency_ms": latency,
                "metrics": (
                    {"recall": v, "precision": v, "mrr": v, "ndcg": v} if labelled else None
                ),
                "ann_recall": ann[i] if ann is not None else None,
                "nondeterministic": False,
                "error": None,
            }
        )
    run: dict[str, Any] = {
        "format": "vectraudit.run",
        "format_version": 1,
        "run_id": "r",
        "created_at": "2026-10-04T00:00:00+00:00",
        "label": None,
        "status": "complete",
        "error": None,
        "dataset": {"sha256": "d" * 64, "n_queries": len(values)},
        "connection": {"backend": "faiss"},
        "backend": {"backend": "faiss", "distance": "l2", "snapshot": "sha256:s", "config": {}},
        "capabilities": {},
        "search": {"k": 2, "params": {"nprobe": 8}},
        "timing": {},
        "metric_contract": {"version": 1, "k": 2},
        "provenance": {"runner": dict(ENV), "labels": {}},
        "queries": queries,
    }
    run.update(overrides)
    run["summary"] = summarize(run["queries"], len(values))
    return check_run(json.loads(json.dumps(run)))


def reasons(result: dict[str, Any]) -> str:
    return " | ".join(r["message"] for r in result["reasons"])


# --- verdicts ----------------------------------------------------------------------------


def test_identical_runs_pass() -> None:
    run = fake_run([1.0, 0.5, 0.0] * 20, ann=[1.0] * 60)
    result = compare_runs(run, copy.deepcopy(run))
    assert result["verdict"] == "pass"
    assert result["reasons"] == []
    assert result["n_queries"] == 60
    assert result["metrics"]["ndcg"]["delta"] == 0
    assert result["metrics"]["ann_recall"]["candidate"] == 1.0
    assert result["queries"]["primary_metric"] == "ndcg"
    assert result["queries"]["regressed_count"] == 0
    assert result["latency"]["compared"]
    assert result["config_diff"] == []
    json.dumps(result, allow_nan=False)


def test_quality_regression_fails_with_evidence() -> None:
    base = fake_run([1.0] * 40)
    cand = fake_run([1.0] * 30 + [0.0] * 10)
    result = compare_runs(base, cand)
    assert result["verdict"] == "fail"
    assert "ndcg dropped by 0.2500" in reasons(result)
    assert result["metrics"]["ndcg"]["ci_high"] < 0
    q = result["queries"]
    assert q["regressed_count"] == 10
    assert q["regressed"][0]["delta"] == -1.0
    assert q["regressed"][0]["lost_ids"] == ["a", "b"]
    assert q["regressed"][0]["new_ids"] == ["x", "y"]


def test_improvements_pass() -> None:
    result = compare_runs(fake_run([0.0] * 40), fake_run([1.0] * 40))
    assert result["verdict"] == "pass"
    assert result["queries"]["improved_count"] == 40


@pytest.mark.parametrize("status", ["partial", "failed", "cancelled"])
def test_incomplete_runs_are_errors(status) -> None:
    base = fake_run([1.0] * 5)
    cand = fake_run([1.0] * 5, status=status)
    if status == "partial":
        cand["queries"][0]["error"] = {"code": "timeout", "message": "slow", "hint": None}
    if status == "failed":
        cand["error"] = {"code": "auth", "message": "invalid api key", "hint": None}
    result = compare_runs(base, cand)
    assert result["verdict"] == "error"
    assert f"The candidate run is {status}" in reasons(result)
    if status == "failed":
        assert "invalid api key" in reasons(result)
    assert result["metrics"] == {}


def test_error_takes_precedence_over_not_comparable() -> None:
    base = fake_run([1.0] * 5)
    cand = fake_run([1.0] * 5, status="cancelled", search={"k": 3, "params": {}})
    assert compare_runs(base, cand)["verdict"] == "error"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dataset": {"sha256": "e" * 64, "n_queries": 5}}, "different dataset versions"),
        ({"search": {"k": 3, "params": {}}}, "different k"),
        ({"metric_contract": {"version": 99, "k": 2}}, "metric contract"),
    ],
)
def test_incompatible_runs_are_not_comparable(change, message) -> None:
    result = compare_runs(fake_run([1.0] * 5), fake_run([1.0] * 5, **change))
    assert result["verdict"] == "not_comparable"
    assert message in reasons(result)


def test_different_query_sets_are_not_comparable() -> None:
    cand = fake_run([1.0] * 5)
    cand["queries"][0]["query_id"] = "other"
    result = compare_runs(fake_run([1.0] * 5), cand)
    assert result["verdict"] == "not_comparable"
    assert "different query sets" in reasons(result)


def test_missing_measurements_never_pass() -> None:
    unlabelled = fake_run([1.0] * 5, labelled=False)
    result = compare_runs(unlabelled, copy.deepcopy(unlabelled))
    assert result["verdict"] == "not_comparable"
    assert "No quality metric" in reasons(result)

    strict = parse_policy({"metrics": {"ann_recall": {"min": 0.9}}})
    labelled = fake_run([1.0] * 5)
    result = compare_runs(labelled, copy.deepcopy(labelled), strict)
    assert result["verdict"] == "not_comparable"
    assert "requires ann_recall, which needs an exact reference" in reasons(result)


def test_default_policy_judges_available_metrics_only() -> None:
    ann_only = fake_run([1.0] * 40, ann=[1.0] * 40, labelled=False)
    worse = fake_run([1.0] * 40, ann=[0.5] * 40, labelled=False)
    assert compare_runs(ann_only, copy.deepcopy(ann_only))["verdict"] == "pass"
    result = compare_runs(ann_only, worse)
    assert result["verdict"] == "fail"
    assert "ann_recall dropped by 0.5000" in reasons(result)
    assert result["queries"]["primary_metric"] == "ann_recall"


def test_min_queries_and_small_sample_warning() -> None:
    run = fake_run([1.0] * 5)
    policy = parse_policy({"min_queries": 10, "metrics": {"ndcg": {"max_drop": 0}}})
    result = compare_runs(run, copy.deepcopy(run), policy)
    assert result["verdict"] == "not_comparable"
    assert "below the policy minimum of 10" in reasons(result)
    result = compare_runs(run, copy.deepcopy(run))
    assert any("Only 5 independent query families" in w for w in result["warnings"])


def test_minimum_and_regressed_query_rules() -> None:
    base = fake_run([0.8] * 40)
    cand = fake_run([0.8] * 37 + [0.0] * 3)
    policy = parse_policy({"metrics": {"ndcg": {"min": 0.9}}, "max_regressed_queries": 2})
    result = compare_runs(base, cand, policy)
    assert result["verdict"] == "fail"
    assert "below the policy minimum 0.9000" in reasons(result)
    assert "3 queries regressed on ndcg; the policy allows 2" in reasons(result)
    lenient = parse_policy({"metrics": {"ndcg": {"max_drop": 0.1}}, "max_regressed_queries": 3})
    assert compare_runs(base, cand, lenient)["verdict"] == "pass"


def test_require_significance_ignores_uncertain_drops() -> None:
    # One query out of 40 drops: the mean drop exceeds 0.02 but its interval reaches 0.
    base = fake_run([1.0] * 40)
    cand = fake_run([1.0] * 39 + [0.0])
    plain = parse_policy({"metrics": {"ndcg": {"max_drop": 0.02}}})
    significant = parse_policy(
        {"metrics": {"ndcg": {"max_drop": 0.02}}, "require_significance": True}
    )
    assert compare_runs(base, cand, plain)["verdict"] == "fail"
    assert compare_runs(base, cand, significant)["verdict"] == "pass"


def test_cohort_rules() -> None:
    tags = [["faq"]] * 10 + [["billing"]] * 30
    base = fake_run([1.0] * 40, tags=tags)
    cand = fake_run([0.5] * 10 + [1.0] * 30, tags=tags)
    policy = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0.2}},
            "cohorts": {"min_queries": 10, "metrics": {"ndcg": {"max_drop": 0.1}}},
        }
    )
    result = compare_runs(base, cand, policy)
    assert result["verdict"] == "fail"
    assert "cohort 'faq': ndcg dropped by 0.5000" in reasons(result)
    assert "billing" not in reasons(result)
    assert result["cohorts"]["faq"]["n"] == 10
    assert result["cohorts"]["faq"]["judged"]
    small = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0.2}},
            "cohorts": {"min_queries": 11, "metrics": {"ndcg": {"max_drop": 0.1}}},
        }
    )
    result = compare_runs(base, cand, small)
    assert result["verdict"] == "pass"
    assert not result["cohorts"]["faq"]["judged"]


def test_latency_rules() -> None:
    base = fake_run([1.0] * 40, latency=10.0)
    slow = fake_run([1.0] * 40, latency=16.0)
    result = compare_runs(base, slow)
    assert result["verdict"] == "fail"
    assert "p95 latency rose 60.0%" in reasons(result)
    assert result["latency"]["p95_change_pct"] == pytest.approx(60.0)

    other_env = copy.deepcopy(slow)
    other_env["provenance"]["runner"]["cpu_count"] = 64
    result = compare_runs(base, other_env)
    assert result["verdict"] == "pass"
    assert not result["latency"]["compared"]
    assert any("Latency was not compared" in w for w in result["warnings"])

    absolute = parse_policy({"metrics": {"ndcg": {"max_drop": 0}}, "latency": {"p95_max_ms": 12}})
    result = compare_runs(base, other_env, absolute)
    assert result["verdict"] == "fail"
    assert "above the policy limit 12.00 ms" in reasons(result)

    anywhere = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0}},
            "latency": {"p95_max_increase_pct": 10, "require_same_environment": False},
        }
    )
    result = compare_runs(base, other_env, anywhere)
    assert result["verdict"] == "fail"
    assert any("across different runner environments" in w for w in result["warnings"])


def test_context_warnings() -> None:
    base = fake_run([1.0] * 40)
    cand = copy.deepcopy(base)
    cand["backend"]["snapshot"] = None
    cand["backend"]["distance"] = "cosine"
    cand["queries"][0]["nondeterministic"] = True
    warnings = " | ".join(compare_runs(base, cand)["warnings"])
    assert "no immutable snapshot" in warnings
    assert "Distance differs" in warnings
    assert "1 candidate queries returned different results" in warnings
    cand["backend"]["snapshot"] = "sha256:other"
    assert "different corpus snapshots" in " | ".join(compare_runs(base, cand)["warnings"])


def test_config_diff() -> None:
    base = fake_run([1.0])
    cand = fake_run([1.0], search={"k": 2, "params": {"nprobe": 2}})
    cand["provenance"]["labels"] = {"embedding_model": "e5"}
    assert config_diff(base, cand) == [
        {"path": "search.params.nprobe", "baseline": 8, "candidate": 2},
        {"path": "labels.embedding_model", "baseline": None, "candidate": "e5"},
    ]


def test_bootstrap_is_seeded() -> None:
    base = fake_run([1.0, 0.0] * 20)
    cand = fake_run([0.0, 1.0, 1.0, 1.0] * 10)
    a = compare_runs(base, cand, seed=3)["metrics"]["ndcg"]
    b = compare_runs(base, cand, seed=3)["metrics"]["ndcg"]
    assert a == b
    assert a["ci_low"] <= a["delta"] <= a["ci_high"]


def test_real_faiss_regression(synthetic) -> None:
    ds = make_dataset(synthetic, n=40)
    exact = FaissConnector(synthetic["ivf_flat"], vectors=synthetic["vectors"])
    pq = FaissConnector(synthetic["ivf_pq"], vectors=synthetic["vectors"])
    base = execute_run(ds, exact, RunOptions(k=5, params={"nprobe": 16}))
    again = execute_run(ds, exact, RunOptions(k=5, params={"nprobe": 16}))
    cand = execute_run(ds, pq, RunOptions(k=5, params={"nprobe": 1}))
    policy = parse_policy({"metrics": {"recall": {"max_drop": 0.01}, "ann_recall": {"min": 0.95}}})
    assert compare_runs(base, again, policy)["verdict"] == "pass"
    result = compare_runs(base, cand, policy)
    assert result["verdict"] == Verdict.FAIL.value
    assert result["queries"]["regressed_count"] > 0
    assert any(d["path"] == "backend.config.kind" for d in result["config_diff"])


# --- policy files ------------------------------------------------------------------------


def test_policy_parsing(tmp_path) -> None:
    raw = {
        "version": 1,
        "metrics": {"ndcg": {"max_drop": 0.02}, "ann_recall": {"min": 0.9}},
        "latency": {"p95_max_increase_pct": 25, "p95_max_ms": 80},
        "min_queries": 30,
        "max_regressed_queries": 10,
        "primary_metric": "ndcg",
        "cohorts": {"min_queries": 10, "metrics": {"mrr": {"max_drop": 0.05}}},
        "require_significance": True,
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    policy = load_policy(path)
    assert policy.metrics["ndcg"].max_drop == 0.02
    assert policy.metrics["ann_recall"].min == 0.9
    assert policy.metrics["ndcg"].required
    assert policy.latency.p95_max_ms == 80
    assert policy.cohorts.metrics["mrr"].max_drop == 0.05
    assert policy.require_significance
    assert not policy.is_default
    assert DEFAULT_POLICY.is_default
    json.dumps(policy.as_dict())


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "JSON object"),
        ({"metrics": {"ndcg": {"max_drop": 0.1}}, "extra": 1}, "unknown field"),
        ({"version": 2, "metrics": {"ndcg": {"max_drop": 0.1}}}, "version"),
        ({"metrics": {"bleu": {"max_drop": 0.1}}}, "unknown metric"),
        ({"metrics": {"ndcg": {}}}, "sets no limit"),
        ({"metrics": {"ndcg": {"max_drop": -1}}}, "max_drop must be"),
        ({"metrics": {"ndcg": {"min": 2}}}, "must be <= 1"),
        ({"metrics": {"ndcg": {"max_drop": True}}}, "max_drop must be"),
        ({"metrics": {"ndcg": {"maxdrop": 0.1}}}, "unknown field"),
        ({"latency": {"p95_max_ms": 10}}, "at least one metric"),
        ({"metrics": {"ndcg": {"min": 0.5}}, "min_queries": 0}, "min_queries"),
        ({"metrics": {"ndcg": {"min": 0.5}}, "primary_metric": "bleu"}, "primary_metric"),
        ({"metrics": {"ndcg": {"min": 0.5}}, "latency": {"p95": 1}}, "unknown field"),
        ({"metrics": {"ndcg": {"min": 0.5}}, "require_significance": "yes"}, "true or false"),
    ],
)
def test_invalid_policies(raw, message) -> None:
    with pytest.raises(PolicyError, match=message):
        parse_policy(raw)


def test_policy_file_errors(tmp_path) -> None:
    with pytest.raises(PolicyError, match="not found"):
        load_policy(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(PolicyError, match="not valid JSON"):
        load_policy(tmp_path / "bad.json")


def test_tiny_latency_rises_are_noise() -> None:
    base = fake_run([1.0] * 40, latency=0.02)
    jitter = fake_run([1.0] * 40, latency=0.06)  # +200%, but only 0.04 ms
    assert compare_runs(base, jitter)["verdict"] == "pass"
    strict = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0}},
            "latency": {"p95_max_increase_pct": 50, "min_increase_ms": 0},
        }
    )
    assert compare_runs(base, jitter, strict)["verdict"] == "fail"
    assert strict.latency.min_increase_ms == 0
    assert DEFAULT_POLICY.latency.min_increase_ms == 2.0


def test_minimum_applies_to_measured_pairs_not_unlabelled_suite() -> None:
    base = fake_run([1.0] * 40, ann=[1.0] * 40)
    for q in base["queries"][2:]:
        q["metrics"] = None
    policy = parse_policy({"min_queries": 30, "metrics": {"ndcg": {"max_drop": 0.1}}})
    result = compare_runs(base, copy.deepcopy(base), policy)
    assert result["verdict"] == "not_comparable"
    assert "2 measured query pairs" in reasons(result)
    assert result["metrics"]["ndcg"]["n"] == 2
    assert any("2/40" in warning for warning in result["warnings"])
    ann_policy = parse_policy({"min_queries": 30, "metrics": {"ann_recall": {"min": 0.9}}})
    assert compare_runs(base, copy.deepcopy(base), ann_policy)["verdict"] == "pass"


def test_sparse_cohort_reports_without_judging() -> None:
    base = fake_run([1.0] * 40, tags=[["faq"]] * 40, ann=[1.0] * 40)
    cand = copy.deepcopy(base)
    for b, c in zip(base["queries"][2:], cand["queries"][2:], strict=True):
        b["metrics"] = c["metrics"] = None
    for q in cand["queries"][:2]:
        q["metrics"] = {m: 0.0 for m in ("recall", "precision", "mrr", "ndcg")}
    policy = parse_policy(
        {
            "min_queries": 30,
            "metrics": {"ann_recall": {"min": 0.9}},
            "cohorts": {"min_queries": 10, "metrics": {"ndcg": {"max_drop": 0.1}}},
        }
    )
    result = compare_runs(base, cand, policy)
    assert result["verdict"] == "pass"
    assert not result["cohorts"]["faq"]["judged"]
    assert result["cohorts"]["faq"]["metrics"]["ndcg"]["n"] == 2
    assert any("reported without judging" in warning for warning in result["warnings"])


def test_regressed_count_cannot_use_sparse_primary_metric() -> None:
    base = fake_run([1.0] * 40, ann=[1.0] * 40)
    for q in base["queries"][2:]:
        q["metrics"] = None
    policy = parse_policy(
        {
            "min_queries": 30,
            "metrics": {"ann_recall": {"min": 0.9}},
            "primary_metric": "ndcg",
            "max_regressed_queries": 10,
        }
    )
    assert compare_runs(base, copy.deepcopy(base), policy)["verdict"] == "not_comparable"


def test_batched_bootstrap_preserves_seeded_intervals(monkeypatch) -> None:
    import numpy as np

    import vectraudit.evaluation.compare as compare

    monkeypatch.setattr(compare, "BOOTSTRAP_MAX_CELLS", 100)
    values = np.linspace(0.0, 1.0, 40)
    base = fake_run([0.0] * 40)
    cand = fake_run(values.tolist())
    seed, samples = 17, 101
    idx = np.random.default_rng(seed).integers(0, 40, size=(samples, 40))
    expected = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    stats = compare_runs(base, cand, seed=seed, n_bootstrap=samples)["metrics"]["recall"]
    assert stats["ci_low"] == pytest.approx(expected[0])
    assert stats["ci_high"] == pytest.approx(expected[1])
