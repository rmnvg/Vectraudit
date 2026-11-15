"""MAP@k and Hit Rate@k: definitions, recording, validation and comparison."""

from __future__ import annotations

import copy
import json

import pytest

from tests.test_eval_compare import fake_run
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.metrics import (
    average_precision,
    metric_contract,
    repeat_quality,
    semantic_metrics,
)
from vectraudit.evaluation.policy import parse_policy
from vectraudit.evaluation.runfile import RunFileError, check_run
from vectraudit.evaluation.runner import summarize
from vectraudit.integrations.sharing import project_run


def test_average_precision_matches_trec_eval_map_cut():
    relevant = {"a", "b", "c"}
    # hits at ranks 1 and 3: (1/1 + 2/3) / 3 relevant documents
    assert average_precision(["a", "x", "b", "y", "z"], relevant, 5) == pytest.approx(
        (1 + 2 / 3) / 3
    )
    assert average_precision(["a", "b", "c"], relevant, 3) == 1.0
    assert average_precision(["x", "y", "a"], relevant, 2) == 0.0  # the hit is past k
    assert average_precision(["a", "a", "b"], relevant, 3) == pytest.approx((1 + 2 / 3) / 3)
    assert average_precision(["a"], set(), 3) == 0.0


def test_losing_a_relevant_document_lowers_map_unlike_evidentlys_denominator():
    relevant = {"a", "b"}
    found_both = average_precision(["a", "b"], relevant, 5)
    found_one = average_precision(["a", "x"], relevant, 5)
    assert found_both == 1.0
    assert found_one == 0.5  # Evidently's (divide by relevant retrieved) would give 1.0


def test_hit_rate_and_graded_labels():
    labels = {"a": 2, "b": 0, "c": 1}  # grade 0 is not relevant
    m = semantic_metrics(["b", "x", "c"], labels, 3)
    assert m["hit_rate"] == 1.0
    assert m["map"] == pytest.approx((1 / 3) / 2)
    assert semantic_metrics(["b", "x"], labels, 2)["hit_rate"] == 0.0
    assert set(m) == {"recall", "precision", "mrr", "ndcg", "map", "hit_rate"}


def test_repeats_average_map_and_hit_rate_like_the_other_metrics():
    out = repeat_quality([["a", "x"], ["x", "y"]], {"a": 1}, 2)
    assert out["metrics"]["hit_rate"] == 0.5
    assert out["metrics"]["map"] == 0.5
    assert all("map" in s for s in out["repeat_metrics"])


def test_the_contract_defines_both_and_keeps_its_version():
    contract = metric_contract(10)
    assert contract["version"] == 2  # additive: existing recordings stay comparable
    assert "map_cut_k" in contract["map"]
    assert "optional" in contract["hit_rate"]


def with_rank(run: dict, scale: float = 1.0) -> dict:
    """The fake run with MAP and Hit Rate recorded, as a run made today would have."""
    out = copy.deepcopy(run)
    for q in out["queries"]:
        if q["metrics"] is not None:
            v = q["metrics"]["ndcg"]
            q["metrics"]["map"] = v * scale
            q["metrics"]["hit_rate"] = 1.0 if v > 0 else 0.0
    out["summary"] = summarize(out["queries"], len(out["queries"]))
    return check_run(json.loads(json.dumps(out)))


STEADY = [1.0, 0.5, 0.0] * 20


def test_new_runs_validate_and_summarise_both_and_older_runs_still_validate():
    old = fake_run(STEADY)
    new = with_rank(old)
    assert "map" not in old["summary"]["metrics"]
    assert new["summary"]["metrics"]["map"] == pytest.approx(0.5)
    assert new["summary"]["metrics"]["hit_rate"] == pytest.approx(2 / 3)


def test_old_and_new_recordings_compare_as_before_under_existing_policies():
    old, new = fake_run(STEADY), with_rank(fake_run(STEADY))
    assert compare_runs(old, new)["verdict"] == "pass"
    assert "map" not in compare_runs(old, new)["metrics"]  # measured on one side only


def test_a_policy_that_needs_map_is_not_comparable_against_an_older_run():
    policy = parse_policy({"metrics": {"map": {"max_drop": 0.02}}})
    result = compare_runs(fake_run(STEADY), with_rank(fake_run(STEADY)), policy)
    assert result["verdict"] == "not_comparable"
    assert "map" in " ".join(r["message"] for r in result["reasons"])


def test_a_map_regression_fails_and_map_can_be_the_primary_metric():
    policy = parse_policy({"metrics": {"map": {"max_drop": 0.02}}, "primary_metric": "map"})
    base = with_rank(fake_run(STEADY))
    worse = with_rank(fake_run(STEADY), scale=0.5)  # same nDCG, MAP halved
    result = compare_runs(base, worse, policy)
    assert result["verdict"] == "fail"
    assert "map dropped by" in " ".join(r["message"] for r in result["reasons"])
    assert result["queries"]["primary_metric"] == "map"
    assert compare_runs(base, copy.deepcopy(base), policy)["verdict"] == "pass"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda q: q["metrics"].update(map=1.5), "metrics.map must be in"),
        (lambda q: q["metrics"].update(hit_rate=float("nan")), "metrics.hit_rate must be in"),
    ],
)
def test_invalid_values_are_refused(change, message):
    run = with_rank(fake_run(STEADY))
    change(run["queries"][0])
    with pytest.raises(RunFileError, match=message):
        check_run(run)


def test_optional_metrics_must_cover_every_repeat():
    labels = {"a": 1}
    quality = repeat_quality([["a"], ["x"]], labels, 1)
    run = with_rank(fake_run(STEADY))
    q = run["queries"][0]
    q.update(
        metrics=quality["metrics"],
        repeat_metrics=quality["repeat_metrics"],
        ids=["a"],
        repeat_ids=[["a"], ["x"]],
        repeat_ann_recall=[None, None],
        latency_samples_ms=[1.0, 1.0],
    )
    run["timing"] = {"repeats": 2}
    run["metric_contract"] = metric_contract(run["search"]["k"])  # version 2: repeats checked
    run["queries"] = [q]
    run["dataset"]["n_queries"] = 1
    run["summary"] = summarize(run["queries"], 1)
    check_run(json.loads(json.dumps(run)))  # consistent: accepted
    del q["repeat_metrics"][1]["map"]
    with pytest.raises(RunFileError, match="every repeat"):
        check_run(json.loads(json.dumps(run)))


def test_uploads_carry_both_and_older_projections_are_unchanged():
    old = fake_run(STEADY)
    new = with_rank(old)
    projected = project_run(new)
    assert projected["queries"][0]["metrics"]["map"] == new["queries"][0]["metrics"]["map"]
    assert "hit_rate" in projected["queries"][0]["metrics"]
    assert "map" not in project_run(old)["queries"][0]["metrics"]
