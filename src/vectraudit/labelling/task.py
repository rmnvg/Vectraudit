"""Content-hashed tasks and atomic local persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Question(Model):
    query_id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)
    documents: list[str]


class Task(Model):
    format: str = "vectraudit.labelling.v1"
    name: str = Field(min_length=1)
    corpus_sha256: str
    sources: list[str]
    documents: dict[str, str]
    queries: list[Question]
    sha256: str = ""

    @model_validator(mode="after")
    def valid(self) -> Task:
        if self.format != "vectraudit.labelling.v1" or not self.queries:
            raise ValueError("Unsupported or empty task")
        if len({q.query_id for q in self.queries}) != len(self.queries):
            raise ValueError("Duplicate query ids")
        for q in self.queries:
            if len(set(q.documents)) != len(q.documents):
                raise ValueError("Duplicate pooled documents")
            if any(d not in self.documents for d in q.documents):
                raise ValueError("Pool contains an unknown source document")
        return self

    def digest(self) -> str:
        """Hash all task content except the digest itself."""
        blob = json.dumps(self.model_dump(exclude={"sha256"}), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    """Replace a file only after a complete, flushed write, with private permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_task(task: Task, path: Path) -> None:
    """Validate and atomically save a task with its current content digest."""
    Task.model_validate(task.model_dump())
    task.sha256 = task.digest()
    atomic_write(path, task.model_dump_json(indent=2) + "\n")


def load_task(path: Path) -> Task:
    """Load a task, rejecting corruption before presenting any labels."""
    task = Task.model_validate_json(path.read_text(encoding="utf-8"))
    if task.sha256 != task.digest():
        raise ValueError("Task content hash mismatch")
    return task
