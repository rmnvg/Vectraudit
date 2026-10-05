"""Durability and concurrent edits must preserve human decisions."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.experiments.corpus import Document, make_corpus
from vectraudit.labelling.operations import judge
from vectraudit.labelling.pooling import pool
from vectraudit.labelling.storage import TaskBusy, TaskConflict, task_lock, update_task
from vectraudit.labelling.task import Judgment, load_task, save_task


@pytest.fixture
def task_path(tmp_path):
    task = pool(
        parse_dataset('{"query_id":"q","text":"answer"}'),
        make_corpus([Document("doc", "answer text")]),
        k=1,
    )
    path = tmp_path / "task.json"
    save_task(task, path)
    return path


def test_stale_snapshot_cannot_overwrite_new_decisions(task_path):
    older, newer = load_task(task_path), load_task(task_path)
    newer.name = "newer"
    save_task(newer, task_path)
    older.name = "stale"
    with pytest.raises(TaskConflict, match="changed"):
        save_task(older, task_path)
    assert load_task(task_path).name == "newer"


def test_failed_transaction_leaves_original_intact(task_path):
    before = task_path.read_bytes()

    def fail(task):
        task.name = "not committed"
        raise ValueError("test failure")

    with pytest.raises(ValueError, match="test failure"):
        update_task(task_path, fail)
    assert task_path.read_bytes() == before
    with (
        task_lock(task_path),
        pytest.raises(TaskBusy, match="busy"),
        task_lock(task_path, timeout=0),
    ):
        pass
    update_task(task_path, lambda task: setattr(task, "name", "recovered"))
    assert load_task(task_path).name == "recovered"


def test_concurrent_writers_keep_every_annotation(task_path):
    def annotate(i):
        row = Judgment(query_id="q", doc_id="doc", annotator=f"person-{i}", grade=2)
        update_task(task_path, lambda task: judge(task, row))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(annotate, range(24)))
    assert len(load_task(task_path).judgments) == 24


def test_lock_serializes_separate_processes(task_path):
    import subprocess
    import sys

    code = """
import sys
from pathlib import Path
from vectraudit.labelling.storage import update_task
from vectraudit.labelling.operations import judge
from vectraudit.labelling.task import Judgment
row = Judgment(query_id="q",doc_id="doc",annotator=sys.argv[2],grade=2)
update_task(Path(sys.argv[1]),lambda task: judge(task,row))
"""
    children = [
        subprocess.Popen([sys.executable, "-c", code, str(task_path), str(i)]) for i in range(4)
    ]
    assert all(child.wait(timeout=20) == 0 for child in children)
    assert len(load_task(task_path).judgments) == 4


def test_initial_save_does_not_replace_existing_task(task_path):
    copy = load_task(task_path)
    copy.sha256 = ""
    with pytest.raises(TaskConflict, match="changed"):
        save_task(copy, task_path)
