"""The reusable retrieval-check action: inputs, secret scoping and real runs on a git checkout."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vectraudit.evaluation.dataset_tools import ColumnMap, import_rows, read_records, render_dataset
from vectraudit.scaffold import CHECK_ACTION, ci_files, init_project
from vectraudit.templates import retrieval_check as action

BASE = {"VA_BASE_REF": "main", "VA_PR_NUMBER": "42"}


class FakeRunner(action.Runner):
    """Records each command and its environment; exit codes come from ``codes``."""

    def __init__(self, environ, codes=None):
        super().__init__(environ)
        self.calls = []
        self.codes = codes or {}

    def run(self, title, command, env):
        self.calls.append((title, list(command), dict(env)))
        if command[0] == "git":
            Path(command[4]).mkdir(parents=True)
        return self.codes.get(title.split(" (")[0], 0)

    def find(self, title):
        return next(c for c in self.calls if c[0].split(" (")[0] == title)


# --- inputs ------------------------------------------------------------------------------


def test_env_block_parses_one_assignment_per_line():
    text = "QDRANT_API_KEY=abc=def\n\n  SEARCH_URL=https://x?a=1\r\nEMPTY=\n"
    assert action.parse_env_block(text, "baseline") == {
        "QDRANT_API_KEY": "abc=def",
        "SEARCH_URL": "https://x?a=1",
        "EMPTY": "",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no-equals-s3cret", "line 1: use one NAME=value"),
        ("OK=1\n9BAD=s3cret", "line 2: use one NAME=value"),
        ("PATH=/tmp/s3cret", "PATH is reserved"),
        ("LD_PRELOAD=s3cret", "reserved"),
        ("GITHUB_TOKEN=s3cret", "reserved"),
        ("PYTHONPATH=s3cret", "reserved"),
        ("VA_MODE=s3cret", "reserved"),
        ("A=s3cret\nA=other", "A is set twice"),
    ],
)
def test_env_block_errors_never_echo_values(text, message):
    with pytest.raises(action.InputError, match=message) as error:
        action.parse_env_block(text, "baseline")
    assert "s3cret" not in str(error.value)


def test_inputs_have_safe_defaults():
    inputs = action.read_inputs(BASE)
    assert inputs.mode == "connection"
    assert inputs.candidate_label == "pr-42"
    assert (inputs.dataset, inputs.connection, inputs.policy, inputs.out) == (
        "retrieval/dataset.jsonl",
        "retrieval/connection.json",
        "retrieval/policy.json",
        "runs",
    )
    assert inputs.repeats == 3
    assert inputs.publish
    label = action.read_inputs({"VA_BASE_REF": "main", "GITHUB_SHA": "a" * 40}).candidate_label
    assert label == "a" * 12


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"VA_BASE_REF": ""}, "No target branch"),
        ({"VA_BASE_REF": "--upload-pack=x"}, "not a valid branch"),
        ({"VA_MODE": "shell"}, "mode must be"),
        ({"VA_REPEATS": "0"}, "repeats"),
        ({"VA_REPEATS": "3; rm"}, "repeats"),
        ({"VA_PUBLISH": "yes"}, "publish"),
        ({"VA_CONNECTION": "/etc/connection.json"}, "connection must be a relative path"),
        ({"VA_DATASET": "../other/dataset.jsonl"}, "dataset must be a relative path"),
        ({"VA_OUT": "C:\\runs"}, "out must be"),
        ({"VA_MODE": "release"}, "release must be a relative path"),
    ],
)
def test_bad_inputs_are_refused(change, message):
    with pytest.raises(action.InputError, match=message):
        action.read_inputs({**BASE, **change})


def test_input_errors_fail_the_step_with_outputs(tmp_path, capsys):
    output = tmp_path / "output"
    environ = {"VA_BASE_REF": "", "GITHUB_OUTPUT": str(output)}
    assert action.main(environ, FakeRunner(environ)) == 1
    assert "::error::No target branch" in capsys.readouterr().out
    assert output.read_text() == "verdict=\nexit-code=1\n"


# --- secret scoping ----------------------------------------------------------------------


def test_each_command_sees_only_its_own_credentials(tmp_path, capsys):
    environ = {
        **BASE,
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "VA_BASELINE_ENV": "API_KEY=base-secret",
        "VA_CANDIDATE_ENV": "API_KEY=candidate-secret\nEXTRA=1",
        "VA_GITHUB_TOKEN": "gh-token",
        "GH_TOKEN": "inherited",
        "VECTRAUDIT_DOWNLOAD_TOKEN": "download",
        "KEEP": "runner-value",
    }
    runner = FakeRunner(environ, {"Check": 3})
    assert action.main(environ, runner) == 3
    assert (tmp_path / "output").read_text() == "verdict=\nexit-code=3\n"
    printed = capsys.readouterr().out
    assert "::add-mask::base-secret" in printed
    assert "::add-mask::candidate-secret" in printed
    titles = [c[0].split(" (")[0] for c in runner.calls]
    assert titles == [
        "Validate the dataset",
        "Check out the target branch",
        "Record the baseline",
        "Record the candidate",
        "Check",
    ]
    for _, _, env in runner.calls:
        assert env["KEEP"] == "runner-value"
        assert not {"GH_TOKEN", "VECTRAUDIT_DOWNLOAD_TOKEN"} & set(env)
        assert not any(key.startswith("VA_") for key in env)
    _, command, env = runner.find("Record the baseline")
    assert env["API_KEY"] == "base-secret"
    assert "GITHUB_TOKEN" not in env
    base = Path(runner.find("Check out the target branch")[1][4])
    assert command[command.index("--connection") + 1] == str(base / "retrieval/connection.json")
    assert command[command.index("--label") + 1] == "main"
    _, command, env = runner.find("Record the candidate")
    assert (env["API_KEY"], env["EXTRA"]) == ("candidate-secret", "1")
    assert command[command.index("--connection") + 1] == "retrieval/connection.json"
    assert command[command.index("--label") + 1] == "pr-42"
    _, command, env = runner.find("Check")
    assert env["GITHUB_TOKEN"] == "gh-token"
    assert "API_KEY" not in env
    assert command[-2:] == ["--github-check", "--github-comment"]
    # The pinned interpreter runs the CLI, never another vectraudit on PATH.
    assert command[:3] == [sys.executable, "-m", "vectraudit"]


def test_unpublished_checks_get_no_github_token(tmp_path):
    environ = {**BASE, "RUNNER_TEMP": str(tmp_path), "VA_PUBLISH": "false", "VA_GITHUB_TOKEN": "t"}
    runner = FakeRunner(environ)
    assert action.main(environ, runner) == 0
    _, command, env = runner.find("Check")
    assert "GITHUB_TOKEN" not in env
    assert "--github-check" not in command


def test_invalid_dataset_or_missing_target_branch_records_nothing(tmp_path, capsys):
    environ = {**BASE, "RUNNER_TEMP": str(tmp_path)}
    runner = FakeRunner(environ, {"Validate the dataset": 1})
    assert action.main(environ, runner) == 1
    assert len(runner.calls) == 1
    runner = FakeRunner(environ, {"Check out the target branch": 128})
    assert action.main(environ, runner) == 1
    assert "fetch-depth: 0" in capsys.readouterr().out
    assert not any(c[0].startswith("Record") for c in runner.calls)


def test_release_mode_without_an_approved_configuration(tmp_path):
    summary = tmp_path / "summary.md"
    environ = {
        **BASE,
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(summary),
        "VA_MODE": "release",
        "VA_RELEASE": "retrieval/release/release.json",
    }
    runner = FakeRunner(environ)
    assert action.main(environ, runner) == 0
    assert "no approved release configuration yet" in summary.read_text()
    assert [c[0].split(" (")[0] for c in runner.calls][1:] == [
        "Validate this pull request's configuration",
        "Record the candidate",
    ]
    assert "--approved" not in runner.find("Record the candidate")[1]
    # Without anything to compare with, the candidate must at least run.
    runner = FakeRunner(environ, {"Record the candidate": 1})
    assert action.main(environ, runner) == 1


# --- the action definition ---------------------------------------------------------------


def test_action_passes_every_input_through_the_environment():
    yaml = pytest.importorskip("yaml")
    files = ci_files()
    definition = yaml.safe_load(files[f"{CHECK_ACTION}/action.yml"])
    steps = definition["runs"]["steps"]
    run = next(s for s in steps if s.get("id") == "check")
    assert "${{" not in run["run"]
    used = set(re.findall(r'"(VA_[A-Z_]+)"', files[f"{CHECK_ACTION}/check.py"]))
    assert used <= set(run["env"])
    names = {key.replace("-", "_").upper() for key in definition["inputs"]}
    for key, value in run["env"].items():
        if "inputs." in value:
            assert key.removeprefix("VA_") in names
    assert definition["outputs"]["verdict"]["value"] == "${{ steps.check.outputs.verdict }}"
    upload = steps[-1]
    assert upload["if"].startswith("always()")
    # Secrets are inputs, so the install step never sees them.
    assert "env" not in steps[0]


# --- real runs on a git checkout ---------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def git_repo(repo: Path) -> None:
    """Commit everything on main and make it the remote target branch, as checkout does."""
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "switch", "-q", "-c", "pr")


def run_action(repo: Path, tmp_path: Path, **inputs: str) -> tuple[int, dict[str, str], str]:
    """Run the generated check.py as the action does. Returns (code, outputs, job summary)."""
    script = repo / CHECK_ACTION / "check.py"
    output, summary = tmp_path / "github-output", tmp_path / "summary.md"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("GITHUB_", "VA_"))},
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
        "VA_BASE_REF": "main",
        "VA_PR_NUMBER": "7",
        "VA_PUBLISH": "false",
        **{f"VA_{k.upper()}": v for k, v in inputs.items()},
    }
    output.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["exit-code"] == str(proc.returncode), proc.stdout[-3000:]
    text = summary.read_text() if summary.exists() else ""
    return proc.returncode, outputs, text


def test_init_project_check_passes_and_catches_a_regression(synthetic, tmp_path):
    """The generated action, run on a real checkout with the real CLI."""
    import faiss

    repo = tmp_path / "repo"
    init_project(repo, "faiss")
    retrieval = repo / "retrieval"
    data = synthetic["vectors"].parent
    connection = json.loads((retrieval / "connection.json").read_text())
    connection.update(index=str(data / "ivf_flat.index"), vectors=str(data / "vectors.npy"))
    (retrieval / "connection.json").write_text(json.dumps(connection))
    vectors = np.load(synthetic["vectors"])
    queries = np.load(synthetic["queries"])[:40]
    _, nn = faiss.knn(queries, vectors, 3)
    with (tmp_path / "queries.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "vector", "relevant"])
        for i, (q, ids) in enumerate(zip(queries, nn, strict=True)):
            relevant = "; ".join(f"{d}:{3 - r}" for r, d in enumerate(ids))
            writer.writerow([f"q{i}", json.dumps(q.tolist()), relevant])
    rows = import_rows(read_records(tmp_path / "queries.csv"), ColumnMap(vector="vector"))
    (retrieval / "dataset.jsonl").write_text(render_dataset(rows))
    git_repo(repo)

    code, outputs, summary = run_action(repo, tmp_path, repeats="1")
    assert (code, outputs["verdict"]) == (0, "pass")
    assert (repo / "runs" / "report.html").is_file()
    assert summary
    result = json.loads((repo / "runs" / "result.json").read_text())
    assert result["verdict"] == "pass"

    # The pull request moves to a product-quantized index; the baseline still reads the
    # target branch's connection file.
    connection["index"] = str(data / "ivf_pq.index")
    (retrieval / "connection.json").write_text(json.dumps(connection))
    code, outputs, _ = run_action(repo, tmp_path, repeats="1")
    assert (code, outputs["verdict"]) == (3, "fail")
    baseline = json.loads((repo / "runs" / "baseline.json").read_text())
    candidate = json.loads((repo / "runs" / "candidate.json").read_text())
    assert (baseline["label"], candidate["label"]) == ("main", "pr-7")
