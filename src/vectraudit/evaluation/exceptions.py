"""Audited exceptions: accept named, failing policy rules without ever reporting a pass.

A check result's failing reasons carry a stable ``rule`` key (see :mod:`compare`). An
exception names one rule key; while active, it marks that reason accepted. Only when every
failing reason is accepted does the verdict become ``accepted`` ("fail, accepted by …"),
never ``pass``. Errors and incomparable runs cannot be accepted. The policy decides which
rule kinds are open to exceptions and who may grant them; exceptions themselves (who, why,
until when) are stored and audited by the caller, for example the hosted service.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.evaluation.policy import Policy

ACCEPTED = "accepted"


def rule_kind(rule: str) -> str:
    """``metric``, ``cohort``, ``regressed_queries``, ``latency``, ``critical`` or ``safety``."""
    for kind in ("metric", "cohort", "critical"):
        if rule.startswith(kind + ":"):
            return kind
    if rule.startswith("safety."):
        return "safety"
    if rule.startswith("latency."):
        return "latency"
    if rule == "regressed_queries":
        return "regressed_queries"
    raise ValueError(f"Unknown rule key {rule!r}")


def required_role(policy: Policy, rule: str) -> str | None:
    """The least role that may accept ``rule`` under ``policy``; ``None`` if it cannot be."""
    rules = policy.exceptions
    if rules is None:
        return None
    kind = rule_kind(rule)
    if kind == "safety":
        return "owner" if rules.safety == "owner" else None
    return rules.approver if kind in rules.allow else None


def failing_rules(result: Mapping[str, Any]) -> list[str]:
    """Rule keys of the failing reasons in a check result, in report order."""
    return [
        r["rule"] for r in result.get("reasons", []) if r.get("verdict") == "fail" and "rule" in r
    ]


def apply_exceptions(
    result: Mapping[str, Any], exceptions: Sequence[Mapping[str, Any]], policy: Policy
) -> dict[str, Any]:
    """Return a copy of ``result`` with active ``exceptions`` applied.

    Each exception is a mapping with ``id``, ``rule``, ``accepted_by``, ``reason`` and
    ``expires_at``; the caller passes only active ones. An exception whose rule the current
    policy no longer opens to exceptions is ignored. Reasons that are not failures, and
    failures without a matching exception, are left exactly as they were.
    """
    out = copy.deepcopy(dict(result))
    out["exceptions_applied"] = []
    if out.get("verdict") != "fail":
        return out
    active = {e["rule"]: e for e in exceptions if _rule_open(policy, e.get("rule", ""))}
    for reason in out.get("reasons", []):
        exception = active.get(reason.get("rule", "")) if reason.get("verdict") == "fail" else None
        if exception is None:
            continue
        reason["accepted"] = {
            "exception_id": exception["id"],
            "accepted_by": exception["accepted_by"],
            "reason": exception["reason"],
            "expires_at": exception["expires_at"],
        }
        out["exceptions_applied"].append({"exception_id": exception["id"], "rule": reason["rule"]})
    failing = [r for r in out.get("reasons", []) if r.get("verdict") != "pass"]
    if failing and all(r.get("verdict") == "fail" and "accepted" in r for r in failing):
        out["verdict_before_exceptions"] = "fail"
        out["verdict"] = ACCEPTED
    return out


def _rule_open(policy: Policy, rule: str) -> bool:
    try:
        return required_role(policy, rule) is not None
    except ValueError:
        return False
