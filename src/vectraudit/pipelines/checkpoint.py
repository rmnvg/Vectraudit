"""Transactional, local query journals. Only an unfinished query may be replayed."""

from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from vectraudit.evaluation.dataset import Dataset
from vectraudit.evaluation.runner import RunOptions, runner_environment
from vectraudit.pipelines.runtime import Pipeline


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def identity(
    dataset: Dataset,
    pipeline: Pipeline,
    options: RunOptions,
    config: dict[str, Any] | None,
    corpus_version: str | None,
) -> str:
    """Fingerprint inputs and local evaluator/adapter sources; store only their digest."""
    target = pipeline if inspect.isfunction(pipeline) else type(pipeline)
    try:
        source = inspect.getsourcefile(target)
    except TypeError:
        source = None
    code = getattr(target, "__code__", None)
    if code is None:
        code = getattr(target.__call__, "__code__", None)
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    if source and Path(source).is_file():
        sources["adapter_source"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    value = {
        "dataset": dataset.sha256,
        "callable": {
            "module": getattr(target, "__module__", None),
            "name": getattr(target, "__qualname__", None),
            "code": hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None,
        },
        "environment": runner_environment(),
        "packages": sorted(((d.metadata["Name"] or ""), d.version) for d in distributions()),
        "options": asdict(options),
        "config": config,
        "corpus_version": corpus_version,
        "sources": sources,
        "evaluation": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).parent.parent / "evaluation").glob("*.py")
        },
    }
    return hashlib.sha256(_json(value).encode()).hexdigest()


def validate_path(path: Path, root: Path | None = None) -> None:
    """Do not follow symlinks for journals, locks, SQLite companion files, or directories
    Vectraudit manages below ``root``.

    ``root`` (default: the file's own directory) is the folder the caller chose, so it and its
    ancestors are trusted as given. That keeps system links such as macOS ``/tmp`` and
    ``/var`` usable while still refusing a link planted inside a managed directory.
    """
    base = path.parent if root is None else root
    managed = [p for p in path.parents if p != base and p.is_relative_to(base)]
    for item in (
        path,
        *managed,
        *(Path(str(path) + suffix) for suffix in (".lock", "-journal", "-wal", "-shm")),
    ):
        if item.is_symlink():
            raise ValueError("Checkpoint paths must not contain symlinks")


class Checkpoint:
    """One SQLite journal, protected by the caller's process-wide execution lock."""

    def __init__(self, path: Path, fingerprint: str, *, resume: bool) -> None:
        validate_path(path)
        if path.exists() != resume:
            raise ValueError(
                "Checkpoint exists; use resume" if path.exists() else "Checkpoint does not exist"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        try:
            self.db.execute("PRAGMA synchronous=FULL")
            if not resume:
                os.chmod(path, 0o600)
                with self.db:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.execute(
                        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                    )
                    self.db.execute(
                        "CREATE TABLE attempts (query_id TEXT, number INTEGER, "
                        "started TEXT NOT NULL, result TEXT, error INTEGER, "
                        "PRIMARY KEY(query_id, number))"
                    )
                    self.put("identity", fingerprint)
            elif self.get("identity") != fingerprint:
                raise ValueError(
                    "Cannot resume: dataset, pipeline, corpus or execution settings changed"
                )
            self.record = self.get("record")
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        """Release the database connection."""
        self.db.close()

    def get(self, key: str) -> Any:
        """Load journal metadata."""
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        """Commit metadata and its compact listing summary together."""
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, _json(value)))
            if key == "record":
                state = {
                    "run_id": value["run_id"],
                    "status": value["status"],
                    "total": value["dataset"]["n_queries"],
                }
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES ('state', ?)", (_json(state),)
                )

    def latest(self, query_id: str) -> dict[str, Any] | None:
        """Return the latest result, or None for an unfinished query."""
        row = self.db.execute(
            "SELECT result FROM attempts WHERE query_id=? ORDER BY number DESC LIMIT 1", (query_id,)
        ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def begin(self, query_id: str) -> int:
        """Commit an attempt marker before invoking the pipeline."""
        row = self.db.execute(
            "SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE query_id=?", (query_id,)
        ).fetchone()
        number = int(row[0])
        with self.db:
            self.db.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, NULL, NULL)",
                (query_id, number, datetime.now(timezone.utc).isoformat()),
            )
        return number

    def finish(self, entry: dict[str, Any], number: int) -> None:
        """Commit a complete query result, including any sanitized failure."""
        with self.db:
            self.db.execute(
                "UPDATE attempts SET result=?, error=? WHERE query_id=? AND number=?",
                (_json(entry), bool(entry["error"]), entry["query_id"], number),
            )

    def history(self, query_id: str) -> list[dict[str, Any]]:
        """Keep prior failures and interrupted attempts in exported evidence."""
        return [
            {
                "attempt": number,
                "started_at": started,
                "status": "interrupted"
                if result is None
                else ("failed" if json.loads(result)["error"] else "complete"),
                "result": json.loads(result) if result else None,
            }
            for number, started, result in self.db.execute(
                "SELECT number, started, result FROM attempts WHERE query_id=? ORDER BY number",
                (query_id,),
            )
        ]


def checkpoint_state(path: Path) -> dict[str, Any]:
    """Read a compact committed snapshot without opening or creating a writable journal."""
    validate_path(path)
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("BEGIN")
        row = db.execute("SELECT value FROM metadata WHERE key='state'").fetchone()
        state = json.loads(row[0]) if row else {"status": "partial", "total": 0, "run_id": None}
        done, failed = db.execute(
            "SELECT COUNT(error), COALESCE(SUM(error),0) FROM attempts a WHERE number = "
            "(SELECT MAX(number) FROM attempts b WHERE b.query_id=a.query_id)"
        ).fetchone()
        return {**state, "finished": done, "failed": failed}
    finally:
        db.close()
