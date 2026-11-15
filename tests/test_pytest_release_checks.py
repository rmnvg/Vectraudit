"""Release checks inside pytest, with the same verdict as ``vectraudit check``."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import textwrap

import pytest
from typer.testing import CliRunner

from tests.test_eval_compare import fake_run
from vectraudit import testing
from vectraudit.cli import app
from vectraudit.evaluation.compare import Verdict
from vectraudit.evaluation.policy import PolicyError, parse_policy
from vectraudit.evaluation.runfile import RunFileError
from vectraudit.release_cli import EXIT_CODES
from vectraudit.testing import ReleaseCheckFailed, assert_release, check_release

STEADY = [1.0, 0.5, 0.0] * 20
REGRESSED = [0.5, 0.0, 0.0] * 20  # mean nDCG 0.17 against the baseline's 0.5
POLICY = {"metrics": {"ndcg": {"max_drop": 0.05}}}


def runs(tmp_path, candidate_values=STEADY, **candidate_overrides):
    base = tmp_path / "baseline.json"
    cand = tmp_path / "candidate.json"
    base.write_text(json.dumps(fake_run(STEADY)))
    cand.write_text(json.dumps(fake_run(candidate_values, **candidate_overrides)))
    return base, cand


def test_a_steady_candidate_passes_and_is_recorded(tmp_path):
    testing.reset()
    base, cand = runs(tmp_path)
    check = assert_release(base, cand, POLICY)
    assert (check.verdict, check.passed, check.reasons) == ("pass", True, [])
    assert testing.recorded() == [check]
    assert check.test is not None
    assert check.test.endswith("::test_a_steady_candidate_passes_and_is_recorded")  # no phase


def test_a_regression_fails_with_its_reasons_and_changed_settings(tmp_path):
    base, cand = runs(
        tmp_path,
        REGRESSED,
        search={"k": 2, "params": {"nprobe": 2}},
    )
    with pytest.raises(ReleaseCheckFailed) as info:
        assert_release(base, cand, POLICY)
    assert isinstance(info.value, AssertionError)  # pytest reports a failure, not an error
    message = str(info.value)
    assert message.startswith(f"Release check FAIL: {base} → {cand}")
    assert "ndcg dropped by" in message
    assert "Changed settings: search.params.nprobe" in message
    assert info.value.check.verdict == "fail"
    assert "## " not in message  # plain text, not the Markdown report


@pytest.mark.parametrize(
    ("values", "overrides"),
    [
        (STEADY, {}),  # pass
        (REGRESSED, {}),  # fail
        (STEADY, {"dataset": {"sha256": "e" * 64, "n_queries": 60}}),  # not comparable
        (STEADY, {"status": "partial"}),  # error
    ],
)
def test_the_verdict_matches_the_cli_exactly(tmp_path, values, overrides):
    base, cand = runs(tmp_path, values, **overrides)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY))
    out = tmp_path / "result.json"
    cli = CliRunner().invoke(
        app,
        ["check", str(base), str(cand), "--policy", str(policy), "--json", str(out),
         "--no-github-summary"],
    )  # fmt: skip
    check = check_release(base, cand, policy)
    expected = json.loads(out.read_text())
    assert check.verdict == expected["verdict"]
    assert check.reasons == [r["message"] for r in expected["reasons"]]
    assert (cli.exit_code == 0) == check.passed
    assert cli.exit_code == EXIT_CODES[Verdict(check.verdict)]


def test_passing_verdicts_are_exactly_the_cli_exit_zero_verdicts():
    assert {v.value for v, code in EXIT_CODES.items() if code == 0} == testing.PASSING


def test_runs_and_policies_can_be_given_as_paths_records_or_parsed(tmp_path):
    base, cand = runs(tmp_path, REGRESSED)
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps(POLICY))
    records = json.loads(base.read_text()), json.loads(cand.read_text())
    results = [
        check_release(base, cand, policy_file).result,
        check_release(str(base), str(cand), POLICY).result,
        check_release(*records, parse_policy(POLICY)).result,
    ]
    assert all(r == results[0] for r in results)
    record_check = check_release(*records, POLICY)
    assert (record_check.baseline, record_check.candidate) == ("<baseline run>", "<candidate run>")
    assert check_release(base, cand).result["policy"] != results[0]["policy"]  # built-in policy


def test_invalid_inputs_raise_what_the_cli_reports(tmp_path):
    base, cand = runs(tmp_path)
    with pytest.raises(PolicyError):
        check_release(base, cand, {"metrics": {"ndcg": {"max_drop": "lots"}}})
    broken = tmp_path / "broken.json"
    broken.write_text("{}")
    with pytest.raises(RunFileError):
        check_release(broken, cand)
    record = json.loads(base.read_text())
    del record["queries"]
    with pytest.raises((RunFileError, ValueError)):
        check_release(record, cand)


def test_records_are_copied_not_changed(tmp_path):
    base, cand = runs(tmp_path)
    record = json.loads(base.read_text())
    before = copy.deepcopy(record)
    check_release(record, cand)
    assert record == before


SUITE = """
from vectraudit.testing import assert_release, check_release

def test_release_passes():
    assert_release({base!r}, {base!r}, {policy!r})

def test_release_regresses():
    assert_release({base!r}, {regressed!r}, {policy!r})

def test_no_release_check():
    assert True
"""


def _pytest(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path / "suite"), "-q", "-p", "no:cacheprovider",
         "--rootdir", str(tmp_path / "suite"), *args],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )  # fmt: skip


def test_the_installed_plugin_summarises_checks_and_writes_a_report(tmp_path):
    base, regressed = runs(tmp_path, REGRESSED)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY))
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_retrieval.py").write_text(
        textwrap.dedent(SUITE).format(base=str(base), regressed=str(regressed), policy=str(policy))
    )
    report = tmp_path / "release.md"
    done = _pytest(tmp_path, "--vectraudit-report", str(report))
    out = done.stdout
    assert done.returncode == 1, out  # one failing test
    assert "1 failed, 2 passed" in out
    assert "Release check FAIL" in out  # the assertion message
    assert "Vectraudit release checks" in out
    assert "PASS            test_retrieval.py::test_release_passes" in out
    assert "FAIL            test_retrieval.py::test_release_regresses" in out
    assert "test_no_release_check" not in out.split("Vectraudit release checks", 1)[1]
    text = report.read_text()
    assert "1 of 2 passed." in text
    assert "## test_retrieval.py::test_release_regresses" in text


def test_the_plugin_is_silent_without_checks_and_can_be_disabled(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_plain.py").write_text("def test_plain():\n    assert True\n")
    report = tmp_path / "release.md"
    done = _pytest(tmp_path, "--vectraudit-report", str(report))
    assert done.returncode == 0, done.stdout
    assert "Vectraudit release checks" not in done.stdout
    assert report.read_text() == "# Retrieval release checks\n\nNo release checks ran.\n"
    disabled = _pytest(tmp_path, "-p", "no:vectraudit")
    assert disabled.returncode == 0, disabled.stdout
    unknown = _pytest(tmp_path, "-p", "no:vectraudit", "--vectraudit-report", "x.md")
    assert unknown.returncode != 0  # the option belongs to the plugin
