"""Checkpoint symlink protection covers managed paths, not folders the user chose.

On macOS ``/tmp`` and ``/var`` are symlinks, so a user-chosen output folder often sits
below a link. Those ancestors are trusted; links inside managed folders are refused.
"""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_pipeline_checkpoints import Adapter, dataset, execute_pipeline
from vectraudit.cli import app
from vectraudit.pipelines.checkpoint import validate_path


@pytest.fixture
def linked(tmp_path):
    """A user folder reached through a symlink, like macOS /tmp -> /private/tmp."""
    real = tmp_path / "private" / "tmp"
    real.mkdir(parents=True)
    link = tmp_path / "tmp"
    link.symlink_to(real, target_is_directory=True)
    return link


def test_runs_and_resumes_below_a_symlinked_user_folder(linked):
    pipe = Adapter()
    path = linked / "runs" / "run.sqlite"
    path.parent.mkdir()
    first = execute_pipeline(dataset(), pipe, checkpoint=path)
    again = execute_pipeline(dataset(), pipe, checkpoint=path, resume=True)
    assert first["status"] == again["status"] == "complete"
    assert pipe.calls == ["0", "1", "2"]


@pytest.mark.parametrize("suffix", ["", ".lock", "-journal", "-wal", "-shm"])
def test_links_at_the_journal_or_its_companions_are_refused(tmp_path, suffix):
    target = tmp_path / "elsewhere"
    target.write_text("")
    path = tmp_path / "run.sqlite"
    Path(str(path) + suffix).symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        validate_path(path)


def test_links_inside_a_managed_folder_are_refused(tmp_path, linked):
    outside = tmp_path / "outside"
    outside.mkdir()
    out = linked / "runs"
    out.mkdir()
    (out / ".checkpoints").symlink_to(outside, target_is_directory=True)
    path = out / ".checkpoints" / "batch" / "batch.json"
    with pytest.raises(ValueError, match="symlinks"):
        validate_path(path, out)
    # The chosen folder itself, and anything above it, are trusted as given.
    validate_path(out / "batch.json", out)
    validate_path(linked / "batch.json", linked)


def test_relative_output_folders_are_checked_below_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outside").mkdir()
    os.symlink("outside", ".checkpoints")
    with pytest.raises(ValueError, match="symlinks"):
        validate_path(Path(".checkpoints") / "batch" / "batch.json", Path("."))


def _manifest(folder):
    (folder / "queries.jsonl").write_text('{"query_id":"q","text":"hello","no_answer":true}\n')
    (folder / "corpus.jsonl").write_text('{"doc_id":"a","text":"unrelated"}\n')
    manifest = folder / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "queries.jsonl",
                "repeats": 1,
                "candidates": [
                    {"name": "local", "adapter": {"kind": "hybrid", "corpus": "corpus.jsonl"}}
                ],
            }
        )
    )
    return manifest


def test_cli_run_and_resume_below_a_symlinked_output_folder(tmp_path, linked):
    manifest = _manifest(tmp_path)
    out = linked / "workbench-runs"
    args = ["workbench", "run", str(manifest), "--out", str(out)]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    batch = next((out / ".checkpoints").iterdir())
    resumed = CliRunner().invoke(app, [*args, "--resume", str(batch)])
    assert resumed.exit_code == 0, resumed.output


def test_cli_refuses_a_linked_checkpoint_folder_inside_the_output(tmp_path):
    manifest = _manifest(tmp_path)
    out = tmp_path / "workbench-runs"
    out.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (out / ".checkpoints").symlink_to(elsewhere, target_is_directory=True)
    result = CliRunner().invoke(app, ["workbench", "run", str(manifest), "--out", str(out)])
    assert result.exit_code == 2
    assert "symlinks" in result.output
    assert not any(elsewhere.iterdir())


def test_guided_workspace_still_refuses_linked_evaluation_folders(tmp_path):
    from vectraudit.pipelines.guided import GuidedWorkspace

    root = tmp_path / "workspace"
    (root / ".workbench").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / ".workbench" / "evaluations").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        GuidedWorkspace(root).evaluation_path("0" * 32)
