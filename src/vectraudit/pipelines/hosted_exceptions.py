"""Show a hosted project's audited exceptions beside a local workbench comparison.

Exceptions are granted, audited and applied only in the hosted project, under its own
policy. The workbench reads them to answer one question: if this comparison were recorded
there, which failures would an active exception accept, and which would still block? The
local verdict never changes, and nothing is sent except the read requests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from vectraudit.evaluation.exceptions import apply_exceptions, failing_rules
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy, parse_policy, policy_sha256

MAX_LISTED = 50
_FIELDS = (
    "id",
    "rule",
    "kind",
    "reason",
    "owner",
    "accepted_by",
    "created_at",
    "expires_at",
    "max_checks",
    "uses",
    "status",
    "origin_check_id",
)


def hosted_policy(raw: Mapping[str, Any] | None) -> Policy:
    """The policy a hosted project judges under: its raw policy, or the built-in default."""
    return DEFAULT_POLICY if raw is None else parse_policy(dict(raw))


def check_policy_sha256(check: Mapping[str, Any]) -> str:
    """The policy identity of a check result, from the effective rules it records."""
    blob = json.dumps(check["policy"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def hosted_view(
    check: Mapping[str, Any],
    exceptions: Sequence[Mapping[str, Any]],
    remote: Policy,
    *,
    hosted_default: bool,
) -> dict[str, Any]:
    """Join one local check with the hosted project's exceptions (newest first) and policy.

    ``coverage`` has one entry per failing rule: ``covered`` with the active exception;
    the newest inactive one (``expired``, ``used``, ``revoked``) so a lapsed acceptance is
    visible; an active one that cannot apply (``not_open`` under the hosted policy, or
    ``not_applicable`` because the result is an error or not comparable); or ``none``.
    ``hosted_verdict`` is what the hosted project would record, given only when it judges
    under the same policy as this comparison; otherwise its rules differ and the answer
    would be a guess.
    """
    same_policy = check_policy_sha256(check) == policy_sha256(remote)
    active = [e for e in exceptions if e.get("status") == "active"]
    applied = apply_exceptions(check, active, remote)
    accepted = {
        r["rule"]: r["accepted"]["exception_id"]
        for r in applied.get("reasons", [])
        if "accepted" in r and "rule" in r
    }
    by_id = {e["id"]: e for e in exceptions}
    coverage = []
    for rule in failing_rules(check):
        exception: Mapping[str, Any] | None
        if rule in accepted:
            exception = by_id[accepted[rule]]
            state = "covered"
        else:
            exception = next((e for e in exceptions if e.get("rule") == rule), None)
            state = exception["status"] if exception else "none"
            if state == "active":
                # Active but not applied: errors and incomparable results are never
                # accepted, otherwise the hosted policy no longer opens this rule.
                state = "not_applicable" if check.get("verdict") != "fail" else "not_open"
        coverage.append(
            {"rule": rule, "status": state, "exception": _public(exception) if exception else None}
        )
    hosted_verdict = None
    if same_policy and check.get("verdict") == "fail":
        hosted_verdict = applied["verdict"]
    return {
        "same_policy": same_policy,
        "hosted_policy_sha256": policy_sha256(remote),
        "hosted_policy_default": hosted_default,
        "hosted_policy_allows": remote.exceptions is not None,
        "hosted_verdict": hosted_verdict,
        "coverage": coverage,
        "exceptions": [_public(e) for e in exceptions[:MAX_LISTED]],
        "truncated": len(exceptions) > MAX_LISTED,
    }


def _public(exception: Mapping[str, Any]) -> dict[str, Any]:
    return {key: exception.get(key) for key in _FIELDS}
