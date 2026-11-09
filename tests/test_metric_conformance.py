"""Vectraudit's label metrics against ranx 0.3.21 and trec_eval (fixtures under tests/fixtures)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vectraudit.core.relevance import relevance_metrics
from vectraudit.evaluation.metrics import semantic_metrics

FIXTURE = json.loads((Path(__file__).parent / "fixtures/ranx/metrics.json").read_text())
TREC = json.loads((Path(__file__).parent / "fixtures/trec_eval/metrics.json").read_text())
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


# --- trec_eval (through pytrec_eval 0.5) ---------------------------------------------------


def test_the_trec_eval_fixture_scored_the_same_queries_as_the_ranx_fixture():
    queries = [[q["id"], q["labels"], q["ranking"]] for q in FIXTURE["queries"]]
    blob = json.dumps(queries, sort_keys=True).encode()
    assert TREC["source_alignment_sha256"] == hashlib.sha256(blob).hexdigest()
    assert set(TREC["queries"]) == {q["id"] for q in FIXTURE["queries"]}
    assert TREC["ks"] == FIXTURE["ks"]
    assert TREC["pytrec_eval_version"] == "0.5"
    assert TREC["relevance_level"] == 1


@pytest.mark.parametrize("k", FIXTURE["ks"])
def test_semantic_metrics_match_trec_eval(k):
    for q in FIXTURE["queries"]:
        ours = semantic_metrics(q["ranking"], q["labels"], k)
        for name in FIXTURE["metrics"]:
            expected = TREC["queries"][q["id"]][f"{name}@{k}"]
            assert ours[name] == pytest.approx(expected, abs=TOLERANCE), (q["id"], name, k)


def test_the_two_references_agree_with_each_other():
    """If they ever disagree, the difference is in the references, not in Vectraudit."""
    for q in FIXTURE["queries"]:
        for key, value in q["ranx"].items():
            assert TREC["queries"][q["id"]][key] == pytest.approx(value, abs=TOLERANCE), (
                q["id"],
                key,
            )


def test_precision_divides_by_k_even_when_fewer_documents_come_back():
    """trec_eval's P_k and Vectraudit's precision@k both divide by k, not by the list length."""
    short = [q for q in FIXTURE["queries"] if len(q["ranking"]) < 5][:1]
    assert short, "the fixture must contain a ranking shorter than 5"
    q = short[0]
    relevant = sum(1 for d in q["ranking"][:5] if q["labels"].get(d, 0) > 0)
    assert semantic_metrics(q["ranking"], q["labels"], 5)["precision"] == pytest.approx(
        relevant / 5
    )
    assert TREC["queries"][q["id"]]["precision@5"] == pytest.approx(relevant / 5)
