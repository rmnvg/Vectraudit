"""The decision-focused judgment queue and its stop rule."""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any

import numpy as np
import pytest

from tests.test_eval_compare import ENV
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import Dataset, parse_dataset
from vectraudit.evaluation.decision_queue import (
    BOUNDED,
    DecisionQueueError,
    _bounds,
    _questions,
    decision_queue,
)
from vectraudit.evaluation.metrics import metric_contract, repeat_quality, semantic_metrics
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.runfile import check_run
from vectraudit.evaluation.runner import summarize

K = 4


def make_dataset(labels: dict[str, dict[str, float]], **extra: dict[str, Any]) -> Dataset:
    rows = [
        {"query_id": qid, "vector": [1.0, 0.0], "relevant": rel, **extra.get(qid, {})}
        for qid, rel in labels.items()
    ]
    return parse_dataset("\n".join(json.dumps(r) for r in rows))


def make_run(dataset: Dataset, rankings: dict[str, list[str]], run_id: str) -> dict[str, Any]:
    queries = []
    for q in dataset.queries:
        ids = rankings[q.query_id]
        quality = repeat_quality([ids], q.relevant, K)
        queries.append(
            {
                "query_id": q.query_id,
                "tags": list(q.cohort_tags),
                "ids": ids,
                "repeat_ids": [ids],
                "exact_ids": None,
                "latency_ms": 1.0,
                **quality,
                "nondeterministic": False,
                "error": None,
            }
        )
    run: dict[str, Any] = {
        "format": "vectraudit.run",
        "format_version": 1,
        "run_id": run_id,
        "created_at": "2026-11-21T00:00:00+00:00",
        "label": None,
        "status": "complete",
        "error": None,
        "dataset": {"sha256": dataset.sha256, "n_queries": len(dataset)},
        "connection": {"backend": "faiss"},
        "backend": {"backend": "faiss", "distance": "l2", "snapshot": "sha256:s", "config": {}},
        "capabilities": {},
        "search": {"k": K, "params": {}},
        "timing": {"repeats": 1},
        "metric_contract": metric_contract(K),
        "provenance": {"runner": dict(ENV), "labels": {}},
        "queries": queries,
    }
    run["summary"] = summarize(run["queries"], len(queries))
    return check_run(json.loads(json.dumps(run)))


POLICY = parse_policy(
    {
        "metrics": {
            "ndcg": {"max_drop": 0.01},
            "recall": {"max_drop": 0.01},
            "mrr": {"max_drop": 0.02},
        },
        "min_queries": 1,
    }
)


def rescored(run: dict[str, Any], labels: dict[str, dict[str, float]]) -> dict[str, Any]:
    out = copy.deepcopy(run)
    for q in out["queries"]:
        q.update(repeat_quality(q["repeat_ids"], labels[q["query_id"]], K))
    return out


def random_case(
    rng: np.random.Generator, n_questions: int = 2, n_docs: int = 7, shared_tail: bool = False
):
    """Small runs with two unjudged pooled documents per question.

    With ``shared_tail`` both runs return the same two unjudged documents at ranks 3 and 4
    and differ only above them, so a clear difference can be settled.
    """
    labels: dict[str, dict[str, float]] = {}
    base: dict[str, list[str]] = {}
    cand: dict[str, list[str]] = {}
    docs = [f"d{i}" for i in range(n_docs)]
    for i in range(n_questions):
        qid = f"q{i}"
        if shared_tail:
            picked = list(rng.choice(docs, 5, replace=False))
            head, tail = picked[:3], picked[3:]
            base[qid] = [head[0], head[1], *tail]
            cand[qid] = [head[2], head[0], *tail]
            grades = {d: float(rng.integers(0, 4)) for d in head}
            if not any(g > 0 for g in grades.values()):
                grades[head[0]] = 3.0
        else:
            base[qid] = list(rng.choice(docs, K, replace=False))
            cand[qid] = list(rng.choice(docs, K, replace=False))
            pool = list(dict.fromkeys(base[qid] + cand[qid]))
            rng.shuffle(pool)
            judged = pool[2:]
            grades = {d: float(rng.integers(0, 4)) for d in judged}
            if not any(g > 0 for g in grades.values()):
                grades[judged[0]] = float(rng.integers(1, 4))
        labels[qid] = grades
    return labels, base, cand


def assignments(dataset: Dataset, base: dict[str, Any], cand: dict[str, Any]):
    questions = _questions(base, cand, dataset, {})
    pool = [(qid, doc) for qid, q in questions.items() for doc in q.unjudged]
    for grades in itertools.product((0.0, 1.0, 2.0, 3.0), repeat=len(pool)):
        labels = {qid: dict(q.labels) for qid, q in questions.items()}
        for (qid, doc), g in zip(pool, grades, strict=True):
            labels[qid][doc] = g
        yield labels


# --- acceptance: the stop rule never fires while a judgment could flip the verdict --------------


POLICIES = {
    "default shape": POLICY,
    "significance": {
        "metrics": {"ndcg": {"max_drop": 0.05}, "recall": {"max_drop": 0.05}},
        "require_significance": True,
    },
    "tolerance": {"metrics": {"ndcg": {"max_drop": 0.05}}, "non_inferiority": True},
    "floors": {
        "metrics": {
            "recall": {"min": 0.5},
            "mrr": {"min": 0.6},
            "precision": {"max_drop": 0.0},
            "hit_rate": {"max_drop": 0.0},
        }
    },
    "regressed queries": {"metrics": {}, "max_regressed_queries": 0, "primary_metric": "ndcg"},
}


@pytest.mark.parametrize("name", list(POLICIES))
def test_the_stop_rule_never_fires_while_a_judgment_could_change_the_verdict(name: str) -> None:
    """Every way the remaining documents could be judged (grades 0 to 3), checked."""
    raw = POLICIES[name]
    policy = raw if not isinstance(raw, dict) else parse_policy({**raw, "min_queries": 1})
    rng = np.random.default_rng(46)
    settled = open_ = flippable = 0
    for i in range(16):
        labels, base_ids, cand_ids = random_case(rng, shared_tail=i % 2 == 0)
        dataset = make_dataset(labels)
        base = make_run(dataset, base_ids, "base")
        cand = make_run(dataset, cand_ids, "cand")
        report = decision_queue(base, cand, dataset, policy, n_bootstrap=40)
        verdicts = {
            compare_runs(rescored(base, a), rescored(cand, a), policy, n_bootstrap=40)["verdict"]
            for a in assignments(dataset, base, cand)
        }
        if report["state"] == "settled":
            settled += 1
            assert verdicts == {report["verdict"]}, report["message"]
        else:
            assert report["state"] == "open"
            open_ += 1
            flippable += len(verdicts) > 1
    # Not vacuous: both states occur, and open verdicts were really changeable.
    assert settled
    assert open_
    assert flippable


def test_bounds_hold_every_judgment_and_are_exact_where_claimed() -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        labels, base_ids, cand_ids = random_case(rng, n_questions=1, n_docs=8)
        dataset = make_dataset(labels)
        base = make_run(dataset, base_ids, "base")
        cand = make_run(dataset, cand_ids, "cand")
        (q,) = _questions(base, cand, dataset, {}).values()
        bounds = _bounds(q, K, 3.0)
        seen: dict[str, list[tuple[float, float]]] = {m: [] for m in BOUNDED}
        for a in assignments(dataset, base, cand):
            mb = semantic_metrics(q.base[0], a[q.query.query_id], K)
            mc = semantic_metrics(q.cand[0], a[q.query.query_id], K)
            for m in BOUNDED:
                seen[m].append((mc[m] - mb[m], mc[m]))
        for m in BOUNDED:
            d_lo, d_hi, c_lo, c_hi = bounds[m]
            changes = [d for d, _ in seen[m]]
            values = [c for _, c in seen[m]]
            assert d_lo - 1e-12 <= min(changes), m
            assert max(changes) <= d_hi + 1e-12, m
            assert c_lo - 1e-12 <= min(values), m
            assert max(values) <= c_hi + 1e-12, m
            if m != "ndcg":  # exact for the binary metrics; nDCG's bound is only safe
                assert (min(changes), max(changes)) == pytest.approx((d_lo, d_hi)), m
                assert (min(values), max(values)) == pytest.approx((c_lo, c_hi)), m


# --- the report -------------------------------------------------------------------------------


def fixed_case() -> tuple[Dataset, dict[str, Any], dict[str, Any]]:
    labels = {
        "q0": {"a": 3, "b": 0},  # candidate's top result "x" is unjudged
        "q1": {"c": 2},
        "q2": {"e": 1, "f": 0},
    }
    dataset = make_dataset(labels)
    base = make_run(dataset, {"q0": ["a", "b", "p", "q"], "q1": ["c", "s", "t", "u"],
                              "q2": ["e", "f", "g", "h"]}, "base")  # fmt: skip
    cand = make_run(dataset, {"q0": ["x", "a", "b", "p"], "q1": ["c", "s", "t", "u"],
                              "q2": ["e", "f", "g", "h"]}, "cand")  # fmt: skip
    return dataset, base, cand


def test_documents_only_one_run_returns_near_the_top_come_first() -> None:
    dataset, base, cand = fixed_case()
    report = decision_queue(base, cand, dataset, POLICY)
    assert report["format"] == "vectraudit.decision_queue"
    assert report["state"] == "open"
    first = report["items"][0]
    assert (first["query_id"], first["doc_id"]) == ("q0", "x")
    assert (first["candidate_rank"], first["baseline_rank"]) == (1, None)
    # q1 and q2 return the same documents at the same ranks: they move no difference.
    same_rank = [i for i in report["items"] if i["query_id"] in ("q1", "q2")]
    assert same_rank
    assert all(i["swing"] < first["swing"] for i in same_rank)
    assert report["message"].startswith("About ")
    assert "about a third fewer, not half" in report["saving"]


def test_a_verdict_no_judgment_can_change_is_settled() -> None:
    labels = {f"q{i}": {"a": 3} for i in range(3)}
    dataset = make_dataset(labels)
    rankings = {f"q{i}": ["a", "n1", "n2", "n3"] for i in range(3)}
    base = make_run(dataset, rankings, "base")
    cand = make_run(dataset, rankings, "cand")
    # Identical rankings: every judgment moves both runs alike, except through recall's
    # denominator and nDCG's ideal, which only shrink a zero change.
    report = decision_queue(base, cand, dataset, POLICY)
    assert report["verdict"] == "pass"
    assert report["state"] == "settled", report["rules"]
    assert report["n_could_change"] == 0
    assert report["message"].startswith("No remaining judgment")
    assert not any(i["could_change"] for i in report["items"])


def test_held_out_and_already_judged_documents_are_never_queued() -> None:
    labels = {"q0": {"a": 3}, "q1": {"c": 2}}
    dataset = make_dataset(labels, q1={"split": "heldout"})
    base = make_run(dataset, {"q0": ["a", "m", "n", "o"], "q1": ["c", "y", "z", "w"]}, "base")
    cand = make_run(dataset, {"q0": ["m", "a", "n", "o"], "q1": ["y", "c", "z", "w"]}, "cand")
    report = decision_queue(base, cand, dataset, POLICY, irrelevant={"q0": ["m"]})
    assert {i["query_id"] for i in report["items"]} == {"q0"}
    assert {i["doc_id"] for i in report["items"]} == {"n", "o"}


def test_an_unlabelled_exploratory_question_keeps_the_verdict_open() -> None:
    rows = [
        {"query_id": "q0", "vector": [1.0, 0.0], "relevant": {"a": 3}},
        {"query_id": "q1", "vector": [1.0, 0.0]},
    ]
    dataset = parse_dataset("\n".join(json.dumps(r) for r in rows))
    rankings = {"q0": ["a", "b", "c", "d"], "q1": ["e", "f", "g", "h"]}
    base = make_run(dataset, rankings, "base")
    cand = make_run(dataset, rankings, "cand")
    report = decision_queue(base, cand, dataset, POLICY)
    assert report["state"] == "open"
    assert report["unlabelled_questions"] == ["q1"]
    assert "could join the check" in report["message"]
    assert any(i["joins_check"] and i["could_change"] for i in report["items"])


def test_rules_without_a_bound_are_not_certified() -> None:
    dataset, base, cand = fixed_case()
    policy = parse_policy({"metrics": {"map": {"max_drop": 0.01}}, "min_queries": 1})
    report = decision_queue(base, cand, dataset, policy)
    assert report["state"] == "not_certifiable"
    assert "map" in report["message"]
    assert report["items"]  # still ordered for a reviewer


def test_runs_of_another_dataset_version_are_refused() -> None:
    dataset, base, cand = fixed_case()
    other = make_dataset({"q0": {"a": 1}, "q1": {"c": 2}, "q2": {"e": 1}})
    with pytest.raises(DecisionQueueError, match="exact dataset version"):
        decision_queue(base, cand, other, POLICY)
    with pytest.raises(DecisionQueueError, match="max_grade"):
        decision_queue(base, cand, dataset, POLICY, max_grade=2)


def test_the_command_writes_the_queue_and_prints_the_stop_rule(tmp_path) -> None:
    from typer.testing import CliRunner

    from vectraudit.cli import app

    dataset, base, cand = fixed_case()
    data = tmp_path / "ds.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"query_id": q.query_id, "vector": [1.0, 0.0], "relevant": q.relevant})
            for q in dataset.queries
        )
    )
    (tmp_path / "b.json").write_text(json.dumps(base))
    (tmp_path / "c.json").write_text(json.dumps(cand))
    out = tmp_path / "queue.json"
    args = ["workbench", "decision-queue", str(tmp_path / "b.json"), str(tmp_path / "c.json")]
    result = CliRunner().invoke(app, [*args, "--dataset", str(data), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "more judgments could still change the verdict" in result.output
    assert "q0 / x (baseline rank -, candidate rank 1)" in result.output
    assert "about a third fewer, not half" in result.output
    assert json.loads(out.read_text())["format"] == "vectraudit.decision_queue"
