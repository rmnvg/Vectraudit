"""Absolute retrieval promises must not be hidden by improving averages."""

import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from tests.test_eval_compare import fake_run
from tests.test_eval_runner import Scripted, _two_dim
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import PolicyError, parse_policy, policy_sha256
from vectraudit.evaluation.report import render_html, render_markdown
from vectraudit.evaluation.runfile import RunFileError, check_run
from vectraudit.evaluation.runner import RunOptions, execute_run


def repeated(run, *later):
    """Record per-repeat rankings: the stored ranking first, then ``later`` for every query."""
    for q in run["queries"]:
        q["repeat_ids"] = [list(q["ids"]), *(list(r) for r in later)]
    run["timing"] = {"repeats": 1 + len(later)}
    return run


def policy(**changes):
    return parse_policy(
        {"critical_queries": [{"query_id": "q0", "doc_id": "b", "max_rank": 1, **changes}]}
    )


def test_critical_failure_overrides_improving_average_and_significance():
    rules = parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 1}},
            "require_significance": True,
            "critical_queries": [{"query_id": "q0", "doc_id": "b", "max_rank": 1}],
        }
    )
    result = compare_runs(fake_run([0.0] * 40), repeated(fake_run([1.0] * 40)), rules)
    assert result["metrics"]["ndcg"]["delta"] == 1
    assert result["verdict"] == "fail"
    assert result["critical_queries"] == [
        {
            "query_id": "q0",
            "doc_id": "b",
            "max_rank": 1,
            "rank": 2,
            "verdict": "fail",
            "executions": 1,
            "failed_executions": 1,
            "failures": [{"attempt": None, "repeat": 1, "ids": ["a", "b"], "rank": 2}],
        }
    ]
    assert "observed rank 2" in result["reasons"][0]["message"]


def test_assertion_only_policy_needs_no_relevance_labels():
    run = repeated(fake_run([1.0], labelled=False))
    assert compare_runs(run, run, policy(max_rank=2))["verdict"] == "pass"
    assert compare_runs(run, run, policy(doc_id="absent"))["verdict"] == "fail"
    # The baseline may also fail: this is an absolute candidate requirement.
    assert compare_runs(run, run, policy())["verdict"] == "fail"


@pytest.mark.parametrize("changes", [{"query_id": "not-in-this-split"}, {"max_rank": 3}])
def test_missing_query_or_insufficient_depth_never_passes(changes):
    run = repeated(fake_run([1.0]))
    assert compare_runs(run, run, policy(**changes))["verdict"] == "not_comparable"


def test_incomplete_or_incompatible_run_still_takes_precedence():
    run = repeated(fake_run([1.0]))
    candidate = copy.deepcopy(run)
    candidate["status"] = "cancelled"
    assert compare_runs(run, candidate, policy())["verdict"] == "error"
    candidate["status"] = "complete"
    candidate["dataset"]["sha256"] = "different"
    assert compare_runs(run, candidate, policy())["verdict"] == "not_comparable"


@pytest.mark.parametrize(
    "changes",
    [
        {"query_id": " "},
        {"doc_id": 1},
        {"max_rank": True},
        {"max_rank": 0},
        {"max_rank": 1.5},
        {"max_rank": 10001},
        {"unexpected": "ignored?"},
    ],
)
def test_rejects_invalid_assertions(changes):
    with pytest.raises(PolicyError):
        policy(**changes)


def test_rejects_malformed_and_duplicate_rules():
    for rules in [
        None,
        {},
        [None],
        [{"query_id": "q0", "doc_id": "b"}],
        [{"query_id": "q0", "doc_id": "b", "max_rank": 1}] * 2,
    ]:
        with pytest.raises(PolicyError):
            parse_policy({"critical_queries": rules})


def test_existing_policy_identity_is_unchanged_and_new_rules_are_hashed():
    old = parse_policy({"metrics": {"ndcg": {"max_drop": 0.1}}})
    legacy = asdict(old)
    legacy.pop("critical_queries")
    legacy.pop("exceptions")  # newer optional field, also omitted when absent
    legacy["cohorts"].pop("control")  # likewise omitted while it is "none"
    legacy.pop("non_inferiority")  # and while it is false
    legacy.pop("critical_context")  # and while it lists no questions
    legacy.pop("forbidden_stages")  # and while it names no stage kinds
    legacy.pop("prompt_changes")  # and while prompt changes are only reported
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode())
    assert policy_sha256(old) == digest.hexdigest()
    assert policy_sha256(policy()) != policy_sha256(policy(max_rank=2))


def test_reports_show_passed_rules_and_escape_document_ids():
    run = fake_run([1.0])
    doc = "<script>alert(1)</script>"
    run["queries"][0]["ids"][0] = doc
    result = compare_runs(run, repeated(run, ["b", doc]), policy(doc_id=doc))
    for report in [render_html(result), render_markdown(result)]:
        assert "Critical questions" in report
        assert "Failing executions" in report
        assert "&lt;script&gt;" in report
        assert "<script>alert(1)</script>" not in report


def test_document_that_disappears_on_a_later_repeat_fails():
    """Acceptance: rank 2 on the first execution, absent on the second."""
    run = repeated(fake_run([1.0]), ["a", "x"])
    result = compare_runs(run, run, policy(max_rank=2))
    assert result["verdict"] == "fail"
    [rule] = result["critical_queries"]
    assert (rule["rank"], rule["executions"], rule["failed_executions"]) == (None, 2, 1)
    assert rule["failures"] == [{"attempt": None, "repeat": 2, "ids": ["a", "x"], "rank": None}]
    assert (
        "failed on 1 of 2 executions; first at repeat 2: document was not retrieved"
        in (result["reasons"][0]["message"])
    )
    markdown = render_markdown(result)
    assert "| q0 | b | repeat 2 | Not observed | a, x |" in markdown
    assert "1 of 2 failed" in markdown


def test_every_repeat_must_meet_the_rank_and_worst_rank_is_reported():
    run = repeated(fake_run([1.0]), ["b", "a"], ["a", "b"])
    result = compare_runs(run, run, policy(max_rank=2))
    assert result["verdict"] == "pass"
    assert result["critical_queries"][0]["rank"] == 2
    assert result["critical_queries"][0]["executions"] == 3
    assert result["critical_queries"][0]["failures"] == []
    result = compare_runs(run, run, policy(max_rank=1))
    assert [f["repeat"] for f in result["critical_queries"][0]["failures"]] == [1, 3]


def test_runs_without_repeat_evidence_are_insufficient():
    legacy = fake_run([1.0])
    result = compare_runs(legacy, legacy, policy(max_rank=2))
    assert result["verdict"] == "not_comparable"
    assert "no per-repeat rankings" in result["reasons"][0]["message"]
    assert result["critical_queries"][0]["executions"] == 0
    assert "No repeat evidence" in render_markdown(result)
    # Fewer rankings than the run's declared repeats is incomplete evidence, too.
    short = repeated(fake_run([1.0]))
    short["timing"]["repeats"] = 3
    result = compare_runs(short, short, policy(max_rank=2))
    assert result["verdict"] == "not_comparable"
    assert "only 1 of 3 repeat rankings" in result["reasons"][0]["message"]
    # Rankings that contradict the stored ranking cannot be trusted either.
    forged = repeated(fake_run([1.0]))
    forged["queries"][0]["repeat_ids"][0] = ["b"]
    assert compare_runs(forged, forged, policy(max_rank=2))["verdict"] == "not_comparable"


def test_retried_attempts_keep_their_failing_repeats():
    run = repeated(fake_run([1.0]))
    query = run["queries"][0]
    query["attempt_count"] = 2
    query["attempts"] = [
        {"attempt": 1, "status": "failed", "result": {"repeat_ids": [["x"]], "error": {}}},
        {"attempt": 2, "status": "complete", "result": {"repeat_ids": [["a", "b"]]}},
    ]
    result = compare_runs(run, run, policy(max_rank=2))
    assert result["verdict"] == "fail"
    rule = result["critical_queries"][0]
    assert rule["executions"] == 2
    assert rule["failures"] == [{"attempt": 1, "repeat": 1, "ids": ["x"], "rank": None}]
    assert "first at attempt 1, repeat 1" in result["reasons"][0]["message"]


@pytest.mark.parametrize(
    "repeat_ids",
    [{"a": 1}, [["a", "b", "c"]], [[1]], [], [["x", "y"]]],
)
def test_run_files_reject_malformed_repeat_rankings(repeat_ids):
    run = fake_run([1.0])
    run["queries"][0]["repeat_ids"] = repeat_ids
    with pytest.raises(RunFileError):
        check_run(run)
    run["queries"][0]["repeat_ids"] = None
    run["queries"][0]["attempts"] = [{"result": {"repeat_ids": [[1]]}}]
    with pytest.raises(RunFileError):
        check_run(run)


def test_connector_runner_records_every_repeat_ranking():
    def flaky(call, request):
        return ["a", "b"] if call % 2 else ["a"]

    run = execute_run(_two_dim(1), Scripted(flaky), RunOptions(k=2, warmup=0, repeats=2))
    query = check_run(run)["queries"][0]
    assert query["repeat_ids"] == [["a", "b"], ["a"]]
    rules = parse_policy({"critical_queries": [{"query_id": "q0", "doc_id": "b", "max_rank": 2}]})
    assert compare_runs(run, run, rules)["verdict"] == "fail"
