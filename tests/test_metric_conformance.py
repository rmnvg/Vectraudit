"""Vectraudit's label metrics against ranx 0.3.21 (fixture from tests/fixtures/ranx/generate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vectraudit.core.relevance import relevance_metrics
from vectraudit.evaluation.metrics import semantic_metrics

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ranx/metrics.json").read_text())
TOLERANCE = 1e-9


def test_fixture_covers_the_cases_the_contract_names():
    queries = FIXTURE["queries"]
    assert FIXTURE["ranx_version"] == "0.3.21"
    assert len(queries) >= 200
    grades = {g for q in queries for g in q["labels"].values()}
    assert grades == {0, 1, 2, 3}
    assert any(sum(g > 0 for g in q["labels"].values()) == 1 for q in queries)
    assert any(len(q["ranking"]) < 10 for q in queries)
    assert any(len(q["ranking"]) < 100 for q in queries)
    assert any(q["tied_scores"] for q in queries)
    assert all(len(set(q["ranking"])) == len(q["ranking"]) for q in queries)


@pytest.mark.parametrize("k", FIXTURE["ks"])
def test_semantic_metrics_match_ranx(k):
    for q in FIXTURE["queries"]:
        ours = semantic_metrics(q["ranking"], q["labels"], k)
        for name in FIXTURE["metrics"]:
            expected = q["ranx"][f"{name}@{k}"]
            assert ours[name] == pytest.approx(expected, abs=TOLERANCE), (q["id"], name, k)


@pytest.mark.parametrize("k", FIXTURE["ks"])
def test_core_relevance_matches_ranx(k):
    for q in FIXTURE["queries"]:
        ours = relevance_metrics(q["ranking"], q["labels"], k)
        for name in ("recall", "mrr", "ndcg"):
            expected = q["ranx"][f"{name}@{k}"]
            assert getattr(ours, name) == pytest.approx(expected, abs=TOLERANCE), (q["id"], name)


def test_duplicate_hits_earn_no_extra_credit():
    """ranx runs cannot hold duplicates, so the contract's rule is checked directly."""
    labels = {"a": 2, "b": 1, "c": 0}
    deduped = semantic_metrics(["a", "x", "b"], labels, 3)
    repeated = semantic_metrics(["a", "a", "b"], labels, 3)
    assert repeated["recall"] == deduped["recall"] == 1.0
    assert repeated["precision"] == deduped["precision"] == 2 / 3
    assert repeated["mrr"] == 1.0
    assert repeated["map"] == deduped["map"]
    assert repeated["ndcg"] == pytest.approx(deduped["ndcg"])
    assert semantic_metrics(["a", "a", "a"], labels, 3)["ndcg"] < deduped["ndcg"]
