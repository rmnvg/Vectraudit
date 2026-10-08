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
    family_id: str | None = Field(default=None, min_length=1, max_length=512)


class Judgment(Model):
    query_id: str
    doc_id: str
    annotator: str = Field(min_length=1, max_length=100)
    grade: int | None = Field(default=None, ge=0, le=3)
    suggestion_grade: int | None = Field(default=None, ge=0, le=3)
    suggestion_judge: str | None = None
    accepted_suggestion: bool = False
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid(self) -> Judgment:
        if self.accepted_suggestion and (self.grade is None or self.grade != self.suggestion_grade):
            raise ValueError("Accepted suggestion must match the human grade")
        if not self.annotator.strip():
            raise ValueError("Annotator name is required")
        if (self.start is None) != (self.end is None):
            raise ValueError("Both span offsets are required")
        if (
            self.start is not None
            and self.end is not None
            and (self.start >= self.end or not self.grade)
        ):
            raise ValueError("Spans require start < end and a positive grade")
        return self


class Suggestion(Model):
    query_id: str
    doc_id: str
    grade: int = Field(ge=0, le=3)
    judge: str = Field(min_length=1)


class Task(Model):
    format: str = "vectraudit.labelling.v1"
    name: str = Field(min_length=1)
    corpus_sha256: str
    sources: list[str]
    documents: dict[str, str]
    queries: list[Question]
    judgments: list[Judgment] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    resolutions: list[Judgment] = Field(default_factory=list)
    assignments: dict[str, list[str]] = Field(default_factory=dict)
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
        for rows in (self.judgments, self.resolutions):
            seen: set[tuple[str, str, str]] = set()
            for j in rows:
                key = (j.query_id, j.doc_id, j.annotator)
                if key in seen:
                    raise ValueError("Duplicate judgment")
                seen.add(key)
                query = next((q for q in self.queries if q.query_id == j.query_id), None)
                if query is None or j.doc_id not in query.documents:
                    raise ValueError("Judgment outside pool")
                if j.end is not None and j.end > len(self.documents[j.doc_id]):
                    raise ValueError("Span exceeds source document")
        resolved = [(j.query_id, j.doc_id) for j in self.resolutions]
        if len(set(resolved)) != len(resolved):
            raise ValueError("Duplicate reviewer resolution")
        suggested: set[tuple[str, str]] = set()
        for suggestion in self.suggestions:
            pair = (suggestion.query_id, suggestion.doc_id)
            if pair in suggested:
                raise ValueError("Duplicate suggestion")
            suggested.add(pair)
            query = next((q for q in self.queries if q.query_id == suggestion.query_id), None)
            if query is None or suggestion.doc_id not in query.documents:
                raise ValueError("Suggestion outside pool")
        for person, ids in self.assignments.items():
            if not person.strip() or not ids or not set(ids) <= {q.query_id for q in self.queries}:
                raise ValueError("Invalid second-pass assignment")
        return self

    def digest(self) -> str:
        """Hash all task content except the digest itself."""
        content = self.model_dump(exclude={"sha256"})
        # Preserve digests of tasks created before optional family provenance.
        for query in content["queries"]:
            if query["family_id"] is None:
                del query["family_id"]
        blob = json.dumps(content, sort_keys=True, ensure_ascii=False)
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
    from vectraudit.labelling.storage import commit_task

    commit_task(task, path)


def load_task(path: Path) -> Task:
    """Load a task, rejecting corruption before presenting any labels."""
    task = Task.model_validate_json(path.read_text(encoding="utf-8"))
    if task.sha256 != task.digest():
        raise ValueError("Task content hash mismatch")
    return task
