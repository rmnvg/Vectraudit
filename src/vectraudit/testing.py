"""Release checks inside a test suite.

Teams that already gate merges on ``pytest`` can make a retrieval release check one more
test, with the same verdict as ``vectraudit check``::

    from vectraudit.testing import assert_release

    def test_retrieval_release():
        assert_release("runs/baseline.json", "runs/candidate.json", policy="policy.json")

:func:`assert_release` raises :class:`ReleaseCheckFailed` (an ``AssertionError``) unless the
check passes, with the verdict, the failing rules and the changed settings in the message.
:func:`check_release` returns the result without asserting. Both compare the two runs
exactly as ``vectraudit check`` does with local files: the same validation, policy, seed
and bootstrap, so a test and the CLI cannot disagree.

With the ``vectraudit`` pytest plugin (registered automatically when Vectraudit is
installed), every check made during a session is listed in the terminal summary, and
``--vectraudit-report PATH`` writes all of them as one Markdown report for the CI job.

This module imports no test framework; it works under any runner.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vectraudit.evaluation.compare import Verdict, compare_runs
from vectraudit.evaluation.policy import DEFAULT_POLICY, Policy, load_policy, parse_policy
from vectraudit.evaluation.runfile import check_run, load_run

# The verdicts on which ``vectraudit check`` exits 0 (release_cli.EXIT_CODES). Accepted only
# arises from hosted, audited exceptions, never from two local files.
PASSING = frozenset({Verdict.PASS.value, Verdict.ACCEPTED.value})
RunInput = str | os.PathLike[str] | Mapping[str, Any]
PolicyInput = str | os.PathLike[str] | Mapping[str, Any] | Policy | None

_lock = threading.Lock()
_recorded: list[ReleaseCheck] = []


class ReleaseCheckFailed(AssertionError):
    """The release check did not pass; ``check`` holds the full result."""

    def __init__(self, check: ReleaseCheck) -> None:
        super().__init__(check.explain())
        self.check = check


@dataclass(frozen=True)
class ReleaseCheck:
    """One release check's result, as ``vectraudit check --json`` writes it."""

    result: dict[str, Any]
    test: str | None = None
    baseline: str | None = None
    candidate: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """``pass``, ``fail``, ``not_comparable``, ``error`` or ``accepted``."""
        return str(self.result["verdict"])

    @property
    def passed(self) -> bool:
        """Whether ``vectraudit check`` would exit 0."""
        return self.verdict in PASSING

    @property
    def reasons(self) -> list[str]:
        """The failing or blocking reasons, in report order."""
        return [str(r.get("message")) for r in self.result.get("reasons") or []]

    @property
    def changed_settings(self) -> list[str]:
        """Setting paths that differ between the runs (including prompt versions)."""
        return [str(d["path"]) for d in self.result.get("config_diff") or []]

    def explain(self, limit: int = 10) -> str:
        """A plain-text account for an assertion message or a log."""
        lines = [
            f"Release check {self.verdict.upper().replace('_', ' ')}"
            + (f": {self.baseline} → {self.candidate}" if self.baseline else "")
        ]
        reasons = self.reasons
        lines += [f"  - {r}" for r in reasons[:limit]]
        if len(reasons) > limit:
            lines.append(f"  … and {len(reasons) - limit} more")
        if self.changed_settings:
            shown = ", ".join(self.changed_settings[:limit])
            more = len(self.changed_settings) - limit
            lines.append("Changed settings: " + shown + (f" (+{more} more)" if more > 0 else ""))
        warnings = self.result.get("warnings") or []
        if warnings:
            lines.append(f"{len(warnings)} warning(s); see the report for details.")
        return "\n".join(lines)

    def markdown(self) -> str:
        """The same Markdown summary ``vectraudit check --markdown`` writes."""
        from vectraudit.evaluation.report import render_markdown

        return render_markdown(self.result)


def _run(value: RunInput, which: str) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        return check_run(json.loads(json.dumps(value))), f"<{which} run>"
    return load_run(Path(value)), str(value)


def _policy(value: PolicyInput) -> Policy:
    if value is None:
        return DEFAULT_POLICY
    if isinstance(value, Policy):
        return value
    if isinstance(value, Mapping):
        return parse_policy(dict(value))
    return load_policy(Path(value))


def check_release(
    baseline: RunInput,
    candidate: RunInput,
    policy: PolicyInput = None,
    *,
    seed: int = 0,
    n_bootstrap: int = 2000,
) -> ReleaseCheck:
    """Compare a candidate run with a baseline under a policy, as ``vectraudit check`` does.

    Runs are file paths or run records; the policy is a path, a parsed policy, a policy
    record, or ``None`` for the built-in policy. Invalid files raise the same errors the
    CLI reports. The check is recorded for the pytest plugin's summary.
    """
    base, base_name = _run(baseline, "baseline")
    cand, cand_name = _run(candidate, "candidate")
    result = compare_runs(base, cand, _policy(policy), seed=seed, n_bootstrap=n_bootstrap)
    test = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0] or None
    check = ReleaseCheck(result=result, test=test, baseline=base_name, candidate=cand_name)
    with _lock:
        _recorded.append(check)
    return check


def assert_release(
    baseline: RunInput,
    candidate: RunInput,
    policy: PolicyInput = None,
    *,
    seed: int = 0,
    n_bootstrap: int = 2000,
) -> ReleaseCheck:
    """:func:`check_release`, raising :class:`ReleaseCheckFailed` unless it passes."""
    check = check_release(baseline, candidate, policy, seed=seed, n_bootstrap=n_bootstrap)
    if not check.passed:
        raise ReleaseCheckFailed(check)
    return check


def recorded() -> list[ReleaseCheck]:
    """Checks made in this process since the last :func:`reset`, oldest first."""
    with _lock:
        return list(_recorded)


def reset() -> None:
    """Forget recorded checks (the pytest plugin does this at session start)."""
    with _lock:
        _recorded.clear()


def session_report(checks: list[ReleaseCheck]) -> str:
    """All checks of a session as one Markdown document."""
    if not checks:
        return "# Retrieval release checks\n\nNo release checks ran.\n"
    passed = sum(c.passed for c in checks)
    parts = [
        "# Retrieval release checks",
        "",
        f"{passed} of {len(checks)} passed.",
        "",
    ]
    for check in checks:
        parts += [f"## {check.test or 'release check'}", "", check.markdown().strip(), ""]
    return "\n".join(parts)
