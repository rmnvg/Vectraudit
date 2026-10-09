"""Audited exceptions: stable rule keys, policy gates, and "fail, accepted" never "pass"."""

from __future__ import annotations

import pytest

from tests.test_eval_compare import fake_run
from vectraudit.evaluation.compare import Verdict, compare_runs
from vectraudit.evaluation.exceptions import (
    apply_exceptions,
    failing_rules,
    required_role,
    rule_kind,
)
from vectraudit.evaluation.policy import PolicyError, parse_policy, policy_sha256
from vectraudit.integrations.github import _CONCLUSIONS, _TITLES
from vectraudit.release_cli import EXIT_CODES

OPEN = {
    "allow": ["metric", "cohort", "regressed_queries", "latency", "critical"],
    "approver": "engineer",
}


def policy(**extra):
    return parse_policy(
        {
            "metrics": {"ndcg": {"max_drop": 0.01}},
            "min_queries": 20,
            "max_regressed_queries": 0,
            "latency": {"p95_max_ms": 5, "p95_max_increase_pct": 10},
            "cohorts": {"min_queries": 10, "metrics": {"ndcg": {"max_drop": 0.01}}},
            "critical_queries": [{"query_id": "q0", "doc_id": "a", "max_rank": 1}],
            **extra,
        }
    )


def failing_pair():
    tags = [["billing"]] * 20
    base = fake_run([1.0] * 20, tags=tags)
    cand = fake_run([0.0] + [1.0] * 19, tags=tags, latency=9.0)
    for run in (base, cand):
        for q in run["queries"]:
            q["repeat_ids"] = [q["ids"]]
        run["timing"] = {"repeats": 1}
    cand["queries"][1].update(forbidden_ids=["secret"], permission_checked=True)
    base["queries"][1].update(forbidden_ids=[], permission_checked=True)
    cand["queries"][2]["no_answer_correct"] = False
    base["queries"][2]["no_answer_correct"] = True
    return base, cand


def test_every_failure_names_a_stable_rule():
    base, cand = failing_pair()
    result = compare_runs(base, cand, policy())
    assert result["verdict"] == "fail"
    rules = failing_rules(result)
    assert set(rules) == {
        "safety.forbidden:q1",
        "safety.no_answer:q2",
        "critical:q0:a",
        "metric:ndcg.max_drop",
        "regressed_queries",
        "cohort:billing:ndcg.max_drop",
        "latency.p95_max_ms",
        "latency.p95_increase",
    }
    assert compare_runs(base, cand, policy())["reasons"] == result["reasons"]  # stable
    kinds = {rule_kind(r) for r in rules}
    assert kinds == {"safety", "critical", "metric", "regressed_queries", "cohort", "latency"}
    with pytest.raises(ValueError, match="Unknown rule"):
        rule_kind("something.else")


def test_policies_open_exceptions_explicitly_and_keep_old_hashes():
    plain = {"metrics": {"ndcg": {"max_drop": 0.01}}}
    assert "exceptions" not in parse_policy(plain).as_dict()
    assert policy_sha256(parse_policy(plain)) == policy_sha256(parse_policy(dict(plain)))
    opened = parse_policy({**plain, "exceptions": OPEN})
    assert policy_sha256(opened) != policy_sha256(parse_policy(plain))
    assert opened.as_dict()["exceptions"]["allow"] == OPEN["allow"]
    assert required_role(parse_policy(plain), "metric:ndcg.max_drop") is None
    assert required_role(opened, "metric:ndcg.max_drop") == "engineer"
    assert required_role(opened, "safety.forbidden:q1") is None
    owner_safety = parse_policy({**plain, "exceptions": {**OPEN, "safety": "owner"}})
    assert required_role(owner_safety, "safety.no_answer:q2") == "owner"
    only_latency = parse_policy({**plain, "exceptions": {"allow": ["latency"]}})
    assert required_role(only_latency, "latency.p95_max_ms") == "owner"  # default approver
    assert required_role(only_latency, "metric:ndcg.max_drop") is None
    for bad, match in (
        ({"allow": ["safety"]}, "allow"),
        ({"approver": "viewer"}, "approver"),
        ({"safety": "engineer"}, "safety"),
        ({"max_days": 400}, "max_days"),
        ({"unknown": 1}, "unknown"),
    ):
        with pytest.raises(PolicyError, match=match):
            parse_policy({**plain, "exceptions": bad})


def exception(rule, i=0):
    return {
        "id": f"e{i}",
        "rule": rule,
        "accepted_by": "lead@example.test",
        "reason": "Known trade-off",
        "expires_at": "2026-11-01T23:59:59+00:00",
    }


def test_partial_coverage_stays_fail_and_marks_only_named_rules():
    base, cand = failing_pair()
    rules = policy(exceptions={**OPEN, "safety": "owner"})
    result = compare_runs(base, cand, rules)
    applied = apply_exceptions(result, [exception("metric:ndcg.max_drop")], rules)
    assert applied["verdict"] == "fail"
    assert "verdict_before_exceptions" not in applied
    accepted = [r["rule"] for r in applied["reasons"] if "accepted" in r]
    assert accepted == ["metric:ndcg.max_drop"]
    assert result["verdict"] == "fail"  # the input is not modified
    assert not any("accepted" in r for r in result["reasons"])


def test_full_coverage_reads_accepted_never_pass():
    base, cand = failing_pair()
    rules = policy(exceptions={**OPEN, "safety": "owner"})
    result = compare_runs(base, cand, rules)
    every = [exception(rule, i) for i, rule in enumerate(failing_rules(result))]
    applied = apply_exceptions(result, every, rules)
    assert applied["verdict"] == Verdict.ACCEPTED.value
    assert applied["verdict_before_exceptions"] == "fail"
    assert len(applied["exceptions_applied"]) == len(every)
    assert EXIT_CODES[Verdict.ACCEPTED] == 0
    assert _CONCLUSIONS["accepted"] == "neutral"
    assert _TITLES["accepted"] == "Fail, accepted by exception"


def test_policy_closed_kinds_errors_and_incomparable_runs_are_never_accepted():
    base, cand = failing_pair()
    closed = policy(exceptions=OPEN)  # safety stays "never"
    result = compare_runs(base, cand, closed)
    every = [exception(rule, i) for i, rule in enumerate(failing_rules(result))]
    applied = apply_exceptions(result, every, closed)
    assert applied["verdict"] == "fail"
    still = [
        r["rule"] for r in applied["reasons"] if r["verdict"] == "fail" and "accepted" not in r
    ]
    assert set(still) == {"safety.forbidden:q1", "safety.no_answer:q2"}
    # Without an exceptions block, nothing applies at all.
    plain = policy()
    assert apply_exceptions(compare_runs(base, cand, plain), every, plain)["verdict"] == "fail"
    errored = dict(cand, status="partial")
    error = compare_runs(base, errored, closed)
    assert apply_exceptions(error, every, closed)["verdict"] == "error"
    other = dict(cand, dataset={"sha256": "e" * 64, "n_queries": 20})
    incomparable = compare_runs(base, other, closed)
    assert apply_exceptions(incomparable, every, closed)["verdict"] == "not_comparable"


def test_existing_policy_hashes_match_the_previous_release():
    """Pinned against main before exceptions existed; exported checks record these hashes."""
    shapes = {
        "8b62dae18abfcce8": {"metrics": {"ndcg": {"max_drop": 0.1}}},
        "6f1178693b014c58": {
            "max_regressed_queries": 3,
            "critical_queries": [{"query_id": "q", "doc_id": "d", "max_rank": 2}],
        },
    }
    for prefix, raw in shapes.items():
        assert policy_sha256(parse_policy(raw)).startswith(prefix)
