"""The prompt_changes policy rule: gate releases on changed prompt versions."""

from __future__ import annotations

import pytest

from tests.test_prompt_versions import import_versions
from vectraudit.evaluation.calibration import calibrate
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.exceptions import apply_exceptions, required_role, rule_kind
from vectraudit.evaluation.policy import PolicyError, parse_policy, policy_sha256
from vectraudit.integrations.sharing import project_run

RULES = {"metrics": {"ndcg": {"max_drop": 0.5}}}
GATED = {**RULES, "prompt_changes": "fail"}


def reasons(result):
    return [(r["verdict"], r.get("rule"), r["message"]) for r in result["reasons"]]


def runs(tmp_path, before, after):
    base, _ = import_versions(tmp_path, before, "a.jsonl")
    cand, _ = import_versions(tmp_path, after, "b.jsonl")
    return base, cand


def test_by_default_a_prompt_change_is_reported_not_gated(tmp_path):
    base, cand = runs(tmp_path, {"tf1": 12, "tf2": 12}, {"tf1": 13, "tf2": 13})
    result = compare_runs(base, cand, parse_policy(RULES))
    assert result["verdict"] == "pass"
    assert "backend.config.prompts.support-answer" in [d["path"] for d in result["config_diff"]]


def test_under_the_rule_a_changed_prompt_fails_with_its_own_rule_key(tmp_path):
    base, cand = runs(tmp_path, {"tf1": 12, "tf2": 12}, {"tf1": 13, "tf2": 13})
    result = compare_runs(base, cand, parse_policy(GATED))
    assert result["verdict"] == "fail"
    ((verdict, rule, message),) = reasons(result)
    assert (verdict, rule) == ("fail", "prompt:support-answer")
    assert "Prompt support-answer changed (12 → 13)" in message
    same = compare_runs(base, base, parse_policy(GATED))
    assert same["verdict"] == "pass"


def test_a_partly_rolled_out_prompt_is_a_change(tmp_path):
    base, cand = runs(tmp_path, {"tf1": 12, "tf2": 12}, {"tf1": 12, "tf2": 13})
    result = compare_runs(base, cand, parse_policy(GATED))
    assert result["verdict"] == "fail"
    assert "(12 → 12 + 13)" in reasons(result)[0][2]


def test_an_added_or_removed_prompt_is_a_change(tmp_path):
    base, cand = runs(tmp_path, {"tf1": 12, "tf2": 12}, {"tf1": 12, "tf2": 12})
    cand["backend"]["config"]["prompts"]["query-rewrite"] = "3"  # a new prompt in the pipeline
    added = compare_runs(base, cand, parse_policy(GATED))
    assert reasons(added)[0][:2] == ("fail", "prompt:query-rewrite")
    assert "(absent → 3)" in reasons(added)[0][2]
    removed = compare_runs(cand, base, parse_policy(GATED))
    assert "(3 → absent)" in reasons(removed)[0][2]


def test_without_prompt_versions_on_both_sides_the_rule_cannot_pass(tmp_path):
    recorded, unrecorded = runs(tmp_path, {"tf1": 12, "tf2": 12}, {"tf1": None, "tf2": None})
    for base, cand, side in ((recorded, unrecorded, "candidate"), (unrecorded, recorded,
                             "baseline"), (unrecorded, unrecorded, "baseline")):  # fmt: skip
        result = compare_runs(base, cand, parse_policy(GATED))
        assert result["verdict"] == "not_comparable"
        assert f"the {side} run records no prompt versions" in reasons(result)[-1][2]


def test_an_audited_exception_can_accept_a_reviewed_prompt_change(tmp_path):
    base, cand = runs(tmp_path, {"tf1": 12, "tf2": 12}, {"tf1": 13, "tf2": 13})
    open_policy = parse_policy({**GATED, "exceptions": {"allow": ["prompt"]}})
    result = compare_runs(base, cand, open_policy)
    assert rule_kind("prompt:support-answer") == "prompt"
    assert required_role(open_policy, "prompt:support-answer") == "owner"
    assert required_role(parse_policy(GATED), "prompt:support-answer") is None
    exception = {
        "id": "x1", "rule": "prompt:support-answer", "accepted_by": "lead@example.com",
        "reason": "v13 reviewed", "expires_at": "2026-12-01T00:00:00Z",
    }  # fmt: skip
    accepted = apply_exceptions(result, [exception], open_policy)
    assert accepted["verdict"] == "accepted"
    closed = apply_exceptions(result, [exception], parse_policy(GATED))
    assert closed["verdict"] == "fail"


def test_policy_hashes_are_unchanged_unless_the_rule_is_on_and_values_are_checked():
    assert policy_sha256(parse_policy({**RULES, "prompt_changes": "report"})) == policy_sha256(
        parse_policy(RULES)
    )
    gated = parse_policy(GATED)
    assert gated.as_dict()["prompt_changes"] == "fail"
    assert policy_sha256(gated) != policy_sha256(parse_policy(RULES))
    with pytest.raises(PolicyError, match="prompt_changes must be one of"):
        parse_policy({**RULES, "prompt_changes": "block"})


def test_calibration_never_treats_a_prompt_change_as_noise(tmp_path):
    base, cand = runs(tmp_path, {"tf1": 12, "tf2": 12}, {"tf1": 13, "tf2": 13})
    report = calibrate([base, cand], GATED)
    assert "prompt:support-answer" in report["not_noise"]


def test_hosted_projections_gate_on_fingerprinted_prompt_names(tmp_path):
    base, cand = runs(tmp_path, {"tf1": 12, "tf2": 12}, {"tf1": 13, "tf2": 13})
    result = compare_runs(project_run(base), project_run(cand), parse_policy(GATED))
    assert result["verdict"] == "fail"
    rule = reasons(result)[0][1]
    assert rule.startswith("prompt:sha256:")
    assert "support-answer" not in rule
