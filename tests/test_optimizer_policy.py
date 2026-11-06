"""The optimizer under the release policy: exploratory results never look approved."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from vectraudit.cli import app
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.policy import parse_policy
from vectraudit.pipelines import Hit, PipelineResult
from vectraudit.pipelines.optimize import optimize

EXPLORE = ["e1", "e2", "e3"]
HELDOUT = ["h1", "h2", "h3"]


def dataset():
    rows = [{"query_id": q, "text": f"question {q}", "relevant": {f"doc-{q}": 1}} for q in EXPLORE]
    rows += [
        {"query_id": q, "text": f"question {q}", "relevant": {f"doc-{q}": 1}, "split": "heldout"}
        for q in HELDOUT
    ]
    return parse_dataset("\n".join(map(json.dumps, rows)))


def pipeline(hits_for, calls=None, name=""):
    """A pipeline that returns ``hits_for(query_id)`` and records which questions it saw."""

    def run(request):
        if calls is not None:
            calls.append((name, request.query_id))
        return PipelineResult(hits=[Hit(id=h) for h in hits_for(request.query_id)])

    return run


def right(q):
    return [f"doc-{q}", "filler"]


def second(q):
    return ["filler", f"doc-{q}"]


def wrong(q):
    return ["filler", "other"]


POLICY = parse_policy({"metrics": {"ndcg": {"max_drop": 0.01}}, "min_queries": 3})


def test_without_a_policy_the_result_is_only_exploratory():
    report = optimize(dataset(), [("a", pipeline(right), {})], repeats=1)
    assert report["status"] == "exploratory_only"
    assert report["approved"] is False
    assert "not checked under a release policy" in report["status_label"]
    assert report["policy"] is None
    assert report["heldout"]["check"] is None


def test_a_better_candidate_is_approved_only_by_the_heldout_policy_check():
    calls = []
    report = optimize(
        dataset(),
        [
            ("current", pipeline(second, calls, "current"), {}),
            ("better", pipeline(right, calls, "better"), {}),
            ("worse", pipeline(wrong, calls, "worse"), {}),
        ],
        repeats=1,
        policy=POLICY,
        baseline="current",
    )
    assert report["selected"] == "better"
    assert report["status"] == "approved_under_policy"
    assert report["approved"] is True
    assert report["heldout"]["check"]["verdict"] == "pass"
    heldout_callers = {name for name, q in calls if q in HELDOUT}
    assert heldout_callers == {"better", "current"}  # never the unchosen candidate
    screens = {c["name"]: c["policy_screen"] for c in report["candidates"]}
    assert screens["current"] is None
    assert screens["worse"]["verdict"] == "fail"
    assert "metric:ndcg.max_drop" in screens["worse"]["failing_rules"]
    worse = next(c for c in report["candidates"] if c["name"] == "worse")
    assert worse["feasible"] is False
    assert any("fails the release policy" in r for r in worse["reasons"])


def test_a_critical_miss_on_exploratory_questions_excludes_even_a_higher_score():
    def mostly_right(q):
        return ["filler", "other"] if q == "e1" else right(q)

    policy = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0.5}},
            "min_queries": 3,
            "critical_queries": [
                {"query_id": "e1", "doc_id": "doc-e1", "max_rank": 2},
                {"query_id": "h1", "doc_id": "doc-h1", "max_rank": 2},
            ],
        }
    )
    report = optimize(
        dataset(),
        [
            ("current", pipeline(second), {}),
            ("risky", pipeline(mostly_right), {}),
        ],
        repeats=1,
        policy=policy,
        baseline="current",
    )
    risky = next(c for c in report["candidates"] if c["name"] == "risky")
    assert "critical:e1:doc-e1" in risky["policy_screen"]["failing_rules"]
    assert risky["feasible"] is False
    assert report["selected"] == "current"
    assert report["status"] == "baseline_remains_best"
    assert report["approved"] is False
    assert report["heldout"]["check"] is None  # the baseline is not compared with itself
    assert report["policy"]["critical_queries"] == {
        "screened_on_exploratory": ["e1"],
        "checked_on_heldout": ["h1"],
    }


def test_a_heldout_regression_is_not_approved():
    def great_then_bad(q):
        return right(q) if q in EXPLORE else wrong(q)

    report = optimize(
        dataset(),
        [("current", pipeline(second), {}), ("overfit", pipeline(great_then_bad), {})],
        repeats=1,
        policy=POLICY,
        baseline="current",
    )
    assert report["selected"] == "overfit"
    assert report["status"] == "not_approved"
    assert report["heldout"]["check"]["verdict"] == "fail"
    assert "best exploratory candidate" in report["status_label"].casefold()


def test_too_little_heldout_evidence_is_never_approved():
    strict = parse_policy({"metrics": {"ndcg": {"max_drop": 0.01}}, "min_queries": 30})
    report = optimize(
        dataset(),
        [("current", pipeline(second), {}), ("better", pipeline(right), {})],
        repeats=1,
        policy=strict,
        baseline="current",
    )
    assert report["selected"] == "better"  # the screen was inconclusive, not a failure
    better = next(c for c in report["candidates"] if c["name"] == "better")
    assert better["policy_screen"]["verdict"] == "not_comparable"
    assert report["status"] == "not_approved"
    assert report["heldout"]["check"]["verdict"] == "not_comparable"


def test_policy_and_baseline_go_together():
    candidates = [("a", pipeline(right), {})]
    with pytest.raises(ValueError, match="baseline"):
        optimize(dataset(), candidates, repeats=1, policy=POLICY)
    with pytest.raises(ValueError, match="baseline"):
        optimize(dataset(), candidates, repeats=1, baseline="a")
    with pytest.raises(ValueError, match="not one of the candidates"):
        optimize(dataset(), candidates, repeats=1, policy=POLICY, baseline="missing")


ADAPTERS = """
from vectraudit.pipelines import Hit, PipelineResult


def _answer(request, correct):
    doc = "doc-" + request.query_id
    hits = [doc, "filler"] if correct else ["filler", doc]
    return PipelineResult(hits=[Hit(id=h) for h in hits])


def current(request):
    return _answer(request, correct=False)


def better(request):
    return _answer(request, correct=True)


def overfit(request):
    # Best on exploratory questions, misses the document entirely on held-out ones.
    if request.query_id.startswith("e"):
        return _answer(request, correct=True)
    return PipelineResult(hits=[Hit(id="filler"), Hit(id="other")])
"""


@pytest.mark.parametrize(
    ("candidate", "status", "code"),
    [
        ("better", "approved_under_policy", 0),
        ("overfit", "not_approved", 3),
    ],
)
def test_cli_exit_codes_follow_the_policy_decision(tmp_path, candidate, status, code):
    rows = [{"query_id": q, "text": q, "relevant": {f"doc-{q}": 1}} for q in EXPLORE]
    rows += [
        {"query_id": q, "text": q, "relevant": {f"doc-{q}": 1}, "split": "heldout"} for q in HELDOUT
    ]
    (tmp_path / "queries.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "policy_adapters.py").write_text(ADAPTERS)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "queries.jsonl",
                "k": 2,
                "repeats": 1,
                "candidates": [
                    {
                        "name": name,
                        "adapter": {"kind": "python", "target": f"policy_adapters:{name}"},
                    }
                    for name in ("current", candidate)
                ],
            }
        )
    )
    (tmp_path / "policy.json").write_text(
        json.dumps({"metrics": {"ndcg": {"max_drop": 0.01}}, "min_queries": 3})
    )
    runner = CliRunner()
    out = tmp_path / "optimization.json"
    base = ["workbench", "optimize", str(tmp_path / "manifest.json"), "--out", str(out)]
    policy = ["--policy", str(tmp_path / "policy.json"), "--baseline", "current"]
    result = runner.invoke(app, [*base, *policy])
    assert result.exit_code == code, result.output
    report = json.loads(out.read_text())
    assert report["format_version"] == 2
    assert report["selected"] == candidate
    assert report["status"] == status
    assert report["status_label"] in result.output
    assert "Held-out release check:" in result.output
    lonely = runner.invoke(app, [*base, "--policy", str(tmp_path / "policy.json")])
    assert lonely.exit_code == 2
    assert "used together" in lonely.output


def test_a_quality_tie_keeps_the_baseline_whatever_the_latency():
    report = optimize(
        dataset(),
        [("current", pipeline(right), {}), ("same-quality", pipeline(right), {})],
        repeats=1,
        policy=POLICY,
        baseline="current",
    )
    assert report["selected"] == "current"
    assert report["status"] == "baseline_remains_best"


def test_three_candidates_show_which_the_best_cannot_be_told_apart_from():
    calls = []
    candidates = [
        ("current", pipeline(second, calls, "current"), {}),
        ("better", pipeline(right, calls, "better"), {}),
        ("worse", pipeline(wrong, calls, "worse"), {}),
    ]
    report = optimize(dataset(), candidates, repeats=1)
    significance = report["significance"]
    assert significance["method"] == "randomized_tukey_hsd"
    assert significance["leader"] == "better"
    assert "does not change which candidate is selected" in significance["note"]
    by = {c["name"]: c for c in report["candidates"]}
    assert by["better"]["p_vs_leader_adjusted"] is None
    assert all(0 < by[n]["p_vs_leader_adjusted"] <= 1 for n in ("current", "worse"))
    assert all(isinstance(c["hsd_group"], str) and c["hsd_group"] for c in by.values())
    # Information only: selection and the single held-out run are what they were without it.
    assert report["selected"] == "better"
    assert {name for name, q in calls if q in HELDOUT} == {"better"}
    assert [q for name, q in calls if q in HELDOUT] == HELDOUT
    json.dumps(report, allow_nan=False)


def test_two_candidates_have_no_significance_entry():
    report = optimize(
        dataset(), [("a", pipeline(right), {}), ("b", pipeline(second), {})], repeats=1
    )
    assert "significance" not in report
    assert all("hsd_group" not in c for c in report["candidates"])
