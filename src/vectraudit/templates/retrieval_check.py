"""Record the target branch and this pull request on one runner, then check them.

Runs inside the reusable ``retrieval-check`` GitHub action after the pinned install. Standard
library only. Inputs arrive as ``VA_*`` environment variables (never interpolated into shell
code). Each side's secrets reach only that side's record command, and the GitHub token
reaches only the check command.

Exit codes follow ``vectraudit check``: 0 pass, 1 error, 3 fail, 4 not comparable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MODES = ("connection", "release")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Variables that would change how the runner, git, Python or this script behave.
_RESERVED = {"PATH", "HOME", "SHELL", "ENV", "BASH_ENV", "IFS", "GITHUB_TOKEN"}
_RESERVED_PREFIXES = ("LD_", "DYLD_", "PYTHON", "GITHUB_", "RUNNER_", "ACTIONS_", "GH_", "VA_")
_PRIVATE = {"GITHUB_TOKEN", "GH_TOKEN", "GH_ENTERPRISE_TOKEN", "VECTRAUDIT_DOWNLOAD_TOKEN"}


class InputError(ValueError):
    """A configuration problem the workflow author must fix; the message never holds secrets."""


@dataclass(frozen=True)
class Inputs:
    """The action inputs after validation."""

    mode: str
    base_ref: str
    candidate_label: str
    out: str
    publish: bool
    repeats: int
    dataset: str
    connection: str
    policy: str
    release: str
    baseline_env: dict[str, str]
    candidate_env: dict[str, str]
    github_token: str


def parse_env_block(text: str, side: str) -> dict[str, str]:
    """``NAME=value`` per line (blank lines ignored). Errors name the line, never the value."""
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not _NAME.fullmatch(name):
            raise InputError(
                f"{side}-env line {number}: use one NAME=value per line "
                "(values must fit on one line)."
            )
        if name in _RESERVED or name.startswith(_RESERVED_PREFIXES):
            raise InputError(f"{side}-env line {number}: {name} is reserved for the runner.")
        if name in values:
            raise InputError(f"{side}-env line {number}: {name} is set twice.")
        values[name] = value
    return values


def _relative(value: str, field: str) -> str:
    """A path inside the repository, so the target branch's checkout resolves it the same way."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise InputError(f"{field} must be a relative path inside the repository.")
    return value


def read_inputs(environ: Mapping[str, str]) -> Inputs:
    """Validate the ``VA_*`` variables the action sets."""
    mode = environ.get("VA_MODE", "connection").strip() or "connection"
    if mode not in MODES:
        raise InputError(f"mode must be one of {', '.join(MODES)}.")
    base_ref = environ.get("VA_BASE_REF", "").strip()
    if not base_ref:
        raise InputError(
            "No target branch: run on pull_request, or set base-ref to the branch to compare with."
        )
    if base_ref.startswith("-") or not re.fullmatch(r"[A-Za-z0-9._/+-]+", base_ref):
        raise InputError("base-ref is not a valid branch name.")
    label = environ.get("VA_CANDIDATE_LABEL", "").strip()
    if not label:
        number = environ.get("VA_PR_NUMBER", "").strip()
        label = f"pr-{number}" if number else (environ.get("GITHUB_SHA", "")[:12] or "candidate")
    repeats_text = environ.get("VA_REPEATS", "3").strip() or "3"
    if not repeats_text.isdigit() or not 1 <= int(repeats_text) <= 20:
        raise InputError("repeats must be a whole number from 1 to 20.")
    publish = environ.get("VA_PUBLISH", "true").strip().lower()
    if publish not in {"true", "false"}:
        raise InputError("publish must be true or false.")
    release = environ.get("VA_RELEASE", "").strip()
    if mode == "release":
        release = _relative(release, "release")
    return Inputs(
        mode=mode,
        base_ref=base_ref,
        candidate_label=label,
        out=_relative(environ.get("VA_OUT", "runs").strip() or "runs", "out"),
        publish=publish == "true",
        repeats=int(repeats_text),
        dataset=_relative(environ.get("VA_DATASET", "retrieval/dataset.jsonl").strip(), "dataset"),
        connection=_relative(
            environ.get("VA_CONNECTION", "retrieval/connection.json").strip(), "connection"
        ),
        policy=_relative(environ.get("VA_POLICY", "retrieval/policy.json").strip(), "policy"),
        release=release,
        baseline_env=parse_env_block(environ.get("VA_BASELINE_ENV", ""), "baseline"),
        candidate_env=parse_env_block(environ.get("VA_CANDIDATE_ENV", ""), "candidate"),
        github_token=environ.get("VA_GITHUB_TOKEN", ""),
    )


def child_env(environ: Mapping[str, str], extra: Mapping[str, str]) -> dict[str, str]:
    """The runner environment without action inputs or credentials, plus ``extra``."""
    clean = {k: v for k, v in environ.items() if not k.startswith("VA_") and k not in _PRIVATE}
    return {**clean, **extra}


def _cli(*args: str) -> list[str]:
    # The interpreter the pinned wheel was installed into, never another vectraudit on PATH.
    return [sys.executable, "-m", "vectraudit", *args]


class Runner:
    """Runs commands with logs grouped per step; ``run`` is replaced in tests."""

    def __init__(self, environ: Mapping[str, str]) -> None:
        self.environ = environ

    def run(self, title: str, command: Sequence[str], env: Mapping[str, str]) -> int:
        print(f"::group::{title}", flush=True)
        try:
            return subprocess.run(list(command), env=dict(env), check=False).returncode
        finally:
            print("::endgroup::", flush=True)


def _append(environ: Mapping[str, str], key: str, text: str) -> None:
    target = environ.get(key)
    if target:
        with open(target, "a", encoding="utf-8") as stream:
            stream.write(text)


def _verdict(path: Path) -> str:
    try:
        verdict = json.loads(path.read_text(encoding="utf-8")).get("verdict", "")
    except (OSError, ValueError, AttributeError):
        return ""
    return verdict if isinstance(verdict, str) and re.fullmatch(r"[a-z_]+", verdict) else ""


def _base_checkout(inputs: Inputs, runner: Runner, env: Mapping[str, str]) -> Path:
    temp = runner.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    base = Path(tempfile.mkdtemp(prefix="vectraudit-", dir=temp)) / "base"
    code = runner.run(
        f"Check out the target branch ({inputs.base_ref})",
        ["git", "worktree", "add", "--detach", str(base), f"origin/{inputs.base_ref}"],
        env,
    )
    if code:
        raise InputError(
            f"Could not check out origin/{inputs.base_ref}; "
            "use actions/checkout with fetch-depth: 0."
        )
    return base


def check(inputs: Inputs, runner: Runner) -> tuple[int, str]:
    """Record both sides and check them. Returns (exit code, verdict or "")."""
    plain = child_env(runner.environ, {})
    out = Path(inputs.out)
    if inputs.mode == "connection":
        return _check_connection(inputs, runner, plain, out)
    return _check_release(inputs, runner, plain, out)


def _publish_flags(inputs: Inputs) -> list[str]:
    return ["--github-check", "--github-comment"] if inputs.publish else []


def _check_env(inputs: Inputs, plain: Mapping[str, str]) -> dict[str, str]:
    token = {"GITHUB_TOKEN": inputs.github_token} if inputs.publish and inputs.github_token else {}
    return {**plain, **token}


def _check_connection(
    inputs: Inputs, runner: Runner, plain: Mapping[str, str], out: Path
) -> tuple[int, str]:
    code = runner.run("Validate the dataset", _cli("dataset", "validate", inputs.dataset), plain)
    if code:
        return 1, ""
    base = _base_checkout(inputs, runner, plain)
    baseline, candidate = out / "baseline.json", out / "candidate.json"
    common = ["--dataset", inputs.dataset, "--repeats", str(inputs.repeats)]
    # A failed recording still writes its file; `check` reports it as an error.
    runner.run(
        "Record the baseline (target branch configuration)",
        _cli(
            "record",
            "--connection",
            str(base / inputs.connection),
            *common,
            "--out",
            str(baseline),
            "--label",
            inputs.base_ref,
        ),
        child_env(runner.environ, inputs.baseline_env),
    )
    runner.run(
        "Record the candidate (this pull request)",
        _cli(
            "record",
            "--connection",
            inputs.connection,
            *common,
            "--out",
            str(candidate),
            "--label",
            inputs.candidate_label,
        ),
        child_env(runner.environ, inputs.candidate_env),
    )
    result = out / "result.json"
    code = runner.run(
        "Check",
        _cli(
            "check",
            str(baseline),
            str(candidate),
            "--policy",
            inputs.policy,
            "--html",
            str(out / "report.html"),
            "--json",
            str(result),
            *_publish_flags(inputs),
        ),
        _check_env(inputs, plain),
    )
    return code, _verdict(result)


def _check_release(
    inputs: Inputs, runner: Runner, plain: Mapping[str, str], out: Path
) -> tuple[int, str]:
    # The target branch's configuration is the approved one; this pull request's is a proposal.
    base = _base_checkout(inputs, runner, plain)
    approved = base / inputs.release
    if not approved.is_file():
        notice = (
            f"{inputs.base_ref} has no approved release configuration yet. This pull request "
            "introduces it; the comparison starts after merge."
        )
        print(f"::notice::{notice}", flush=True)
        _append(runner.environ, "GITHUB_STEP_SUMMARY", notice + "\n")
    if runner.run(
        "Validate this pull request's configuration",
        _cli("release", "validate", inputs.release),
        plain,
    ):
        return 1, ""
    baseline, candidate = out / "baseline.run.json", out / "candidate.run.json"
    if approved.is_file():
        runner.run(
            "Record the baseline (target branch pipeline)",
            _cli(
                "release",
                "run",
                str(approved),
                "--out",
                str(baseline),
                "--label",
                inputs.base_ref,
            ),
            child_env(runner.environ, inputs.baseline_env),
        )
    candidate_code = runner.run(
        "Record the candidate (this pull request)",
        _cli(
            "release",
            "run",
            inputs.release,
            *(["--approved", str(approved)] if approved.is_file() else []),
            "--out",
            str(candidate),
            "--label",
            inputs.candidate_label,
        ),
        child_env(runner.environ, inputs.candidate_env),
    )
    if not approved.is_file():
        # Nothing to compare with: the candidate must at least run.
        return (1 if candidate_code else 0), ""
    result = out / "result.json"
    code = runner.run(
        "Check",
        _cli(
            "release",
            "check",
            str(approved),
            str(baseline),
            str(candidate),
            "--proposed",
            inputs.release,
            "--html",
            str(out / "report.html"),
            "--json",
            str(result),
            *_publish_flags(inputs),
        ),
        _check_env(inputs, plain),
    )
    return code, _verdict(result)


def main(environ: Mapping[str, str] | None = None, runner: Runner | None = None) -> int:
    """Entry point for the action; writes ``verdict`` and ``exit-code`` step outputs."""
    environ = os.environ if environ is None else environ
    runner = runner or Runner(environ)
    try:
        inputs = read_inputs(environ)
        for value in (*inputs.baseline_env.values(), *inputs.candidate_env.values()):
            if value:
                print(f"::add-mask::{value}", flush=True)
        code, verdict = check(inputs, runner)
    except InputError as exc:
        print(f"::error::{exc}", flush=True)
        code, verdict = 1, ""
    _append(environ, "GITHUB_OUTPUT", f"verdict={verdict}\nexit-code={code}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
