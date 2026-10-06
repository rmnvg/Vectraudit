"""Absolute retrieval promises must not be hidden by improving averages."""

import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from tests.test_eval_compare import fake_run
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.policy import PolicyError, parse_policy, policy_sha256
from vectraudit.evaluation.report import render_html, render_markdown


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
    result = compare_runs(fake_run([0.0] * 40), fake_run([1.0] * 40), rules)
    assert result["metrics"]["ndcg"]["delta"] == 1
    assert result["verdict"] == "fail"
    assert result["critical_queries"] == [
        {"query_id": "q0", "doc_id": "b", "max_rank": 1, "rank": 2, "verdict": "fail"}
    ]
    assert "observed rank 2" in result["reasons"][0]["message"]


def test_assertion_only_policy_needs_no_relevance_labels():
    run = fake_run([1.0], labelled=False)
    assert compare_runs(run, run, policy(max_rank=2))["verdict"] == "pass"
    assert compare_runs(run, run, policy(doc_id="absent"))["verdict"] == "fail"
    # The baseline may also fail: this is an absolute candidate requirement.
    assert compare_runs(run, run, policy())["verdict"] == "fail"


@pytest.mark.parametrize("changes", [{"query_id": "not-in-this-split"}, {"max_rank": 3}])
def test_missing_query_or_insufficient_depth_never_passes(changes):
    run = fake_run([1.0])
    assert compare_runs(run, run, policy(**changes))["verdict"] == "not_comparable"


def test_incomplete_or_incompatible_run_still_takes_precedence():
    run = fake_run([1.0])
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
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode())
    assert policy_sha256(old) == digest.hexdigest()
    assert policy_sha256(policy()) != policy_sha256(policy(max_rank=2))


def test_reports_show_passed_rules_and_escape_document_ids():
    run = fake_run([1.0])
    doc = "<script>alert(1)</script>"
    run["queries"][0]["ids"][0] = doc
    result = compare_runs(run, run, policy(doc_id=doc))
    for report in [render_html(result), render_markdown(result)]:
        assert "Critical questions" in report
        assert "&lt;script&gt;" in report
        assert "<script>alert(1)</script>" not in report
