"""The ``vectraudit`` pytest plugin: summarise release checks made during a test session.

Registered through the ``pytest11`` entry point, so it loads whenever Vectraudit is
installed alongside pytest. It does nothing unless a test calls
:func:`vectraudit.testing.check_release` or :func:`~vectraudit.testing.assert_release`:

- the terminal summary lists each check with its test and verdict;
- ``--vectraudit-report PATH`` writes every check as one Markdown report.

Checks are recorded per process; under ``pytest-xdist`` each worker keeps its own.
Disable the plugin with ``-p no:vectraudit``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def pytest_addoption(parser: Any) -> None:
    """``--vectraudit-report PATH``."""
    group = parser.getgroup("vectraudit", "Vectraudit retrieval release checks")
    group.addoption(
        "--vectraudit-report",
        metavar="PATH",
        default=None,
        help="Write every release check made in this session as one Markdown report.",
    )


def pytest_sessionstart(session: Any) -> None:
    """Start each session with no recorded checks."""
    from vectraudit import testing

    testing.reset()


def pytest_terminal_summary(terminalreporter: Any, exitstatus: Any, config: Any) -> None:
    """List the session's release checks, if any ran."""
    from vectraudit import testing

    checks = testing.recorded()
    if not checks:
        return
    terminalreporter.section("Vectraudit release checks")
    for check in checks:
        label = check.verdict.upper().replace("_", " ")
        terminalreporter.write_line(
            f"{label:<15} {check.test or '(outside a test)'}"
            + (f"  [{len(check.reasons)} reason(s)]" if check.reasons else "")
        )
    path = config.getoption("vectraudit_report")
    if path:
        terminalreporter.write_line(f"Report: {path}")


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    """Write ``--vectraudit-report`` when asked, even if no check ran."""
    from vectraudit import testing

    path = session.config.getoption("vectraudit_report")
    if path:
        Path(path).write_text(testing.session_report(testing.recorded()), encoding="utf-8")
