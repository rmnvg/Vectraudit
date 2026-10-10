"""Customer installs are pinned, source-independent and verified before pip runs."""

from __future__ import annotations

import hashlib
import importlib.util
import os
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
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "VECTRAUDIT_DOWNLOAD_TOKEN"):
        monkeypatch.setenv(key, "private-download-secret")
    calls = []

    def pip_run(command, **kwargs):
        calls.append(command)
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        assert "--force-reinstall" in command  # Same version may already contain different bytes.
        assert command[-1].endswith("vectraudit-0.1.0-py3-none-any.whl[faiss-cpu]")
        assert Path(command[-1].removesuffix("[faiss-cpu]")).read_bytes() == wheel.read_bytes()
        assert "private-download-secret" not in kwargs["env"].values()
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
        installer.install(
            repository="vendor/distribution", version=version, sha256=digest, extras=extras
        )


@pytest.mark.parametrize("repository", ["", "rmnvg/Vectraudit", "RMNVG/VECTRAUDIT.git", "--repo=x"])
def test_source_or_invalid_repository_never_downloads(repository, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "secret")
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


def test_download_is_exact_and_failure_redacts_credentials(tmp_path, monkeypatch, capsys):
    wheel, digest = wheel_file(tmp_path)
    monkeypatch.setenv("GH_TOKEN", "private-secret")
    monkeypatch.setenv("GH_HOST", "untrusted.example")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "gh":
            assert command == [
                "gh",
                "release",
                "download",
                "v0.1.0",
                "--repo",
                "vendor/distribution",
                "--pattern",
                "vectraudit-0.1.0-py3-none-any.whl",
                "--dir",
                command[-1],
            ]
            assert kwargs["env"]["GH_HOST"] == "github.com"
            (Path(command[-1]) / command[-3]).write_bytes(wheel.read_bytes())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(installer.subprocess, "run", run)
    installer.install(repository="vendor/distribution", version="0.1.0", sha256=digest)
    assert len(calls) == 2

    def denied(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gh", stderr=b"private-secret denied")

    monkeypatch.setattr(installer.subprocess, "run", denied)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install.py",
            "--repository",
            "vendor/distribution",
            "--version",
            "0.1.0",
            "--sha256",
            digest,
        ],
    )
    assert installer.main() == 1
    output = capsys.readouterr().err
    assert "download failed" in output
    assert "private-secret" not in output


def test_missing_credential_never_runs_gh(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *a, **k: pytest.fail("must not execute")
    )
    with pytest.raises(ValueError, match="customer-specific"):
        installer.install(repository="vendor/distribution", version="0.1.0", sha256="a" * 64)


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
        assert "rmnvg/Vectraudit" not in workflow
        assert "gh release download" not in workflow
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
    assert set(install["with"]) == {"repository", "version", "sha256", "token", "extras"}


def test_distribution_staging_excludes_source_archives_and_refuses_stale_assets(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_distribution", ROOT / "scripts/prepare_distribution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "dist"
    source.mkdir()
    wheel, digest = wheel_file(source)
    filename = "vectraudit-0.1.0-py3-none-any.whl"
    wheel.rename(source / filename)
    (source / "vectraudit-0.1.0.tar.gz").write_bytes(b"private source archive")
    (source / "private.txt").write_text("private notes")
    destination = tmp_path / "customer-dist"
    module.prepare(source, destination)
    assert {p.name for p in destination.iterdir()} == {filename, "SHA256SUMS"}
    assert (destination / "SHA256SUMS").read_text() == f"{digest}  {filename}\n"
    with pytest.raises(ValueError, match="new directory"):
        module.prepare(source, destination)
    (source / "vectraudit-0.2.0-py3-none-any.whl").write_bytes(b"stale release")
    with pytest.raises(ValueError, match="exactly one"):
        module.prepare(source, tmp_path / "another")
