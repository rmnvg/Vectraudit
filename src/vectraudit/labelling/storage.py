"""Cross-process task locking and optimistic concurrency, using the standard library."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from vectraudit.labelling.task import Task, atomic_write, load_task

T = TypeVar("T")


class TaskConflict(ValueError):
    """Saved content changed after it was read; a stale writer must reload."""


class TaskBusy(ValueError):
    """Another process held the task lock longer than the short interactive timeout."""


@contextmanager
def task_lock(path: Path, *, timeout: float = 5.0) -> Iterator[Path]:
    """Lock a stable companion file across atomic replacements, on Unix and Windows.

    Lock files stay on disk; the OS releases locks after a process exits or crashes.
    Removing a lock file while writers run would create independent lock domains.
    """
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.with_name(path.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        if os.name == "nt":
            from importlib import import_module

            windows: Any = import_module("msvcrt")

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
        else:
            import fcntl
        deadline = time.monotonic() + timeout
        while not acquired:
            try:
                if os.name == "nt":
                    os.lseek(fd, 0, os.SEEK_SET)
                    windows.locking(fd, windows.LK_NBLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise TaskBusy("Task is busy. Retry after the other writer finishes.") from None
                time.sleep(0.02)
        yield path
    finally:
        if acquired:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                windows.locking(fd, windows.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write(task: Task, path: Path) -> None:
    Task.model_validate(task.model_dump())
    digest = task.digest()
    value = task.model_copy(update={"sha256": digest})
    atomic_write(path, value.model_dump_json(indent=2) + "\n")
    task.sha256 = digest


def commit_task(task: Task, path: Path) -> None:
    """Save only if the file still matches the version this caller loaded."""
    with task_lock(path) as canonical:
        if canonical.exists():
            if not task.sha256 or load_task(canonical).sha256 != task.sha256:
                raise TaskConflict("Task changed since it was loaded. Reload before saving.")
        elif task.sha256:
            raise TaskConflict("Task was removed since it was loaded. Restore it before saving.")
        _write(task, canonical)


def update_task(path: Path, change: Callable[[Task], T]) -> T:
    """Serialize a read/modify/save transaction; a failed callback leaves the file intact."""
    with task_lock(path) as canonical:
        task = load_task(canonical)
        result = change(task)
        _write(task, canonical)
        return result
