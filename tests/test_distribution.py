"""CI installs are pinned to an exact release wheel and verified before pip runs."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from vectraudit.pipelines.release import ReleasePipeline, render_release_workflow
from vectraudit.scaffold import BACKENDS, ci_files, render_workflow
from vectraudit.templates import install_wheel as installer

ROOT = Path(__file__).parents[1]


def wheel_file(tmp_path, *, name="vectraudit", version="0.1.0"):
    wheel = tmp_path / "input.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n"
        )
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_offline_install_verifies_bytes_and_removes_credentials(tmp_path, monkeypatch):
    wheel, digest = wheel_file(tmp_path)
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN"):
        monkeypatch.setenv(key, "runner-secret")
    calls = []

    def pip_run(command, **kwargs):
        calls.append(command)
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        assert "--force-reinstall" in command  # Same version may already contain different bytes.
        assert command[-1].endswith("vectraudit-0.1.0-py3-none-any.whl[faiss-cpu]")
        assert Path(command[-1].removesuffix("[faiss-cpu]")).read_bytes() == wheel.read_bytes()
        assert "runner-secret" not in kwargs["env"].values()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(installer.subprocess, "run", pip_run)
    installer.install(
        repository="", version="0.1.0", sha256=digest, wheel=wheel, extras="faiss-cpu"
    )
    assert len(calls) == 1
    assert not Path(calls[0][-1].removesuffix("[faiss-cpu]")).exists()


@pytest.mark.parametrize(
    ("version", "digest", "extras"),
    [
        ("latest", "a" * 64, ""),
        ("v0.1.0", "a" * 64, ""),
        ("0.1.0", "", ""),
        ("0.1.0", "REPLACE_WITH_64_HEX_SHA256", ""),
        ("0.1.0", "a" * 64, "faiss-cpu] --index-url https://bad"),
    ],
)
def test_missing_or_invalid_pins_never_download_or_install(version, digest, extras, monkeypatch):
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *a, **k: pytest.fail("must not execute")
    )
    with pytest.raises(ValueError, match=r"exact version|SHA-256|extras"):
        installer.install(repository="", version=version, sha256=digest, extras=extras)


@pytest.mark.parametrize("repository", ["--repo=x", "owner", "a/b/c", "x/y?z"])
def test_invalid_repository_never_downloads(repository, monkeypatch):
    monkeypatch.setattr(installer, "download", lambda *a: pytest.fail("must not download"))
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *a, **k: pytest.fail("must not execute")
    )
    with pytest.raises(ValueError, match="repository"):
        installer.install(repository=repository, version="0.1.0", sha256="a" * 64)


@pytest.mark.parametrize("case", ["changed-bytes", "wrong-version", "wrong-package", "not-a-wheel"])
def test_invalid_wheel_never_reaches_pip(case, tmp_path, monkeypatch):
    wheel, digest = wheel_file(
        tmp_path,
        name="other" if case == "wrong-package" else "vectraudit",
        version="0.2.0" if case == "wrong-version" else "0.1.0",
    )
    if case == "changed-bytes":
        wheel.write_bytes(wheel.read_bytes() + b"tampered")
    elif case == "not-a-wheel":
        wheel.write_bytes(b"not a zip")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *a, **k: pytest.fail("must not install")
    )
    with pytest.raises(ValueError, match=r"checksum mismatch|package/version|Invalid wheel"):
        installer.install(repository="", version="0.1.0", sha256=digest, wheel=wheel)


@pytest.mark.parametrize(
    ("repository", "expected"),
    [("", installer.DEFAULT_REPOSITORY), ("someone/fork.git", "someone/fork")],
)
def test_download_uses_the_exact_public_release_asset(tmp_path, monkeypatch, repository, expected):
    wheel, digest = wheel_file(tmp_path)
    urls, calls = [], []

    def download(url, target):
        urls.append(url)
        target.write_bytes(wheel.read_bytes())

    monkeypatch.setattr(installer, "download", download)
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda command, **k: calls.append(command) or subprocess.CompletedProcess(command, 0),
    )
    installer.install(repository=repository, version="0.1.0", sha256=digest)
    assert urls == [
        f"https://github.com/{expected}/releases/download/v0.1.0/vectraudit-0.1.0-py3-none-any.whl"
    ]
    assert len(calls) == 1
    assert calls[0][3] == "install"


def test_download_failure_and_oversized_files_never_install(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *a, **k: pytest.fail("must not install")
    )

    def unreachable(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(installer.urllib.request, "urlopen", unreachable)
    monkeypatch.setattr(sys, "argv", ["install.py", "--version", "0.1.0", "--sha256", "a" * 64])
    assert installer.main() == 1
    assert "download failed" in capsys.readouterr().err

    class Huge:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            return b"x" * size

    monkeypatch.setattr(installer, "MAX_WHEEL_BYTES", 3 * 1024 * 1024)
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *a, **k: Huge())
    with pytest.raises(ValueError, match="too large"):
        installer.download("https://example.invalid/x.whl", tmp_path / "x.whl")


def test_generated_installer_runs_without_vectraudit_installed(tmp_path):
    files = ci_files()
    script = tmp_path / "install.py"
    script.write_text(files[".github/actions/install-vectraudit/install.py"])
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--version", "0.1.0", "--sha256", "missing"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert result.returncode == 1
    assert "64-character SHA-256" in result.stderr


def test_all_ci_surfaces_share_the_pinned_install_action():
    yaml = pytest.importorskip("yaml")
    workflows = [render_workflow(backend) for backend in BACKENDS]
    workflows.append(
        render_release_workflow(
            ReleasePipeline(kind="hybrid", name="Search", corpus="corpus.jsonl")
        )
    )
    for workflow in workflows:
        assert "gh release download" not in workflow
        assert "token: ${{ secrets." not in workflow
        job = next(iter(yaml.safe_load(workflow)["jobs"].values()))
        check = next(s for s in job["steps"] if "retrieval-check" in s.get("uses", ""))
        assert check["uses"] == "./.github/actions/retrieval-check"
        assert check["with"]["sha256"] == "${{ env.VECTRAUDIT_SHA256 }}"
        assert "REPLACE" in job["env"]["VECTRAUDIT_SHA256"]
    action = yaml.safe_load(ci_files()[".github/actions/install-vectraudit/action.yml"])
    assert action["runs"]["using"] == "composite"
    assert "${{ inputs." not in action["runs"]["steps"][0]["run"]
    check = yaml.safe_load(ci_files()[".github/actions/retrieval-check/action.yml"])
    install = check["runs"]["steps"][0]
    assert install["uses"] == "./.github/actions/install-vectraudit"
    assert set(install["with"]) == {"repository", "version", "sha256", "extras"}


def test_release_staging_keeps_only_the_wheel_and_refuses_stale_assets(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "stage_release", ROOT / "scripts/stage_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "dist"
    source.mkdir()
    wheel, digest = wheel_file(source)
    filename = "vectraudit-0.1.0-py3-none-any.whl"
    wheel.rename(source / filename)
    (source / "vectraudit-0.1.0.tar.gz").write_bytes(b"source archive")
    (source / "notes.txt").write_text("notes")
    destination = tmp_path / "release-dist"
    module.prepare(source, destination)
    assert {p.name for p in destination.iterdir()} == {filename, "SHA256SUMS"}
    assert (destination / "SHA256SUMS").read_text() == f"{digest}  {filename}\n"
    with pytest.raises(ValueError, match="new directory"):
        module.prepare(source, destination)
    (source / "vectraudit-0.2.0-py3-none-any.whl").write_bytes(b"stale release")
    with pytest.raises(ValueError, match="exactly one"):
        module.prepare(source, tmp_path / "another")


RELEASE_VERSION_SURFACES = {
    "README.md": r"vectraudit-(\d+\.\d+\.\d+)-py3-none-any\.whl|download/v(\d+\.\d+\.\d+)/",
    "docs/first-check.md": r"vectraudit-(\d+\.\d+\.\d+)-py3-none-any\.whl",
    "examples/github/retrieval-check.yml": r'VECTRAUDIT_VERSION: "(\d+\.\d+\.\d+)"',
    ".github/ISSUE_TEMPLATE/bug_report.yml": r"vectraudit (\d+\.\d+\.\d+),",
}


def test_documented_install_commands_name_the_package_version():
    """A version bump must update every install command and example pin that names a release."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE).group(1)
    for path, pattern in RELEASE_VERSION_SURFACES.items():
        named = {
            v
            for match in re.finditer(pattern, (ROOT / path).read_text())
            for v in match.groups()
            if v
        }
        assert named <= {version}, f"{path} names {sorted(named)}; the package is {version}"
    changelog = (ROOT / "CHANGELOG.md").read_text()
    heading = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$"
    assert re.search(heading, changelog, re.MULTILINE), f"CHANGELOG.md has no dated {version}"
