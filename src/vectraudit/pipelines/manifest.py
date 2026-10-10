"""Local manifest loading; Python imports are an explicit trusted-code operation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from vectraudit.pipelines.document_lookup import LookupConfig
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.models import Contract
from vectraudit.pipelines.runtime import HTTPPipeline, Pipeline


class Adapter(Contract):
    kind: Literal["python", "http", "hybrid"]
    target: str | None = None
    corpus: str | None = None
    token_env: str | None = None
    timeout: float = Field(default=30, gt=0, le=300)

    @model_validator(mode="after")
    def configured(self) -> Adapter:
        if self.kind == "hybrid" and not self.corpus:
            raise ValueError("hybrid adapter requires corpus")
        if self.kind != "hybrid" and not self.target:
            raise ValueError("python/http adapter requires target")
        return self


class Candidate(Contract):
    name: str = Field(min_length=1, max_length=128)
    adapter: Adapter
    pipeline_version: str | None = Field(default=None, min_length=1, max_length=256)
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    concurrency: int = Field(default=1, ge=1, le=32)
    """Queries in flight at once; above 1 the adapter must be thread-safe."""
    max_rps: float | None = Field(default=None, gt=0, le=1000)
    """Upper bound on calls per second to this pipeline, retries included."""
    cost_per_call_usd: float | None = Field(default=None, ge=0, le=1000)
    """Declared price of one pipeline call, used by ``workbench preview`` when no earlier run
    of this candidate reported ``cost_usd``."""


class Manifest(Contract):
    dataset: str
    candidates: list[Candidate] = Field(min_length=1, max_length=100)
    k: int = Field(default=10, ge=1, le=10000)
    repeats: int = Field(default=3, ge=1, le=100)
    corpus_version: str | None = None
    max_candidates: int = Field(default=20, ge=1, le=100)
    max_p95_ms: float = Field(default=150, gt=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_memory_bytes: int | None = Field(default=None, ge=0)
    document_lookup: LookupConfig | None = None


def load_manifest(path: Path) -> Manifest:
    """Validate manifest syntax before opening any adapters."""
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def open_adapter(adapter: Adapter, root: Path) -> Pipeline:
    """Open an explicitly authorized local adapter; no import path is accepted via HTTP."""
    if adapter.kind == "hybrid":
        corpus = root / str(adapter.corpus)
        docs = [
            json.loads(line)
            for line in corpus.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return hybrid_pipeline(docs)
    if adapter.kind == "http":
        token = None
        if adapter.token_env:
            token = os.environ.get(adapter.token_env)
            if not token:
                raise ValueError(f"required environment variable {adapter.token_env} is not set")
        return HTTPPipeline(str(adapter.target), token=token, timeout=adapter.timeout)
    module, sep, name = str(adapter.target).partition(":")
    if not sep or not name.isidentifier():
        raise ValueError("Python adapter target must be module:function")
    # A module beside the manifest is importable without installing it; only for this
    # import, so the runner's sys.path is otherwise unchanged.
    folder = str(root.resolve())
    added = folder not in sys.path
    if added:
        sys.path.insert(0, folder)
    try:
        imported = importlib.import_module(module)
    finally:
        if added:
            sys.path.remove(folder)
    fn: Pipeline = getattr(imported, name)
    if not callable(fn):
        raise ValueError("Python adapter target is not callable")
    return fn


def candidate_config(candidate: Candidate, root: Path) -> dict[str, Any]:
    """Safe identity for drift tracking; exclude endpoint credentials and local paths."""
    # Execution settings and declared prices change how fast or costly queries run, not what
    # the pipeline returns, so they stay out of the configuration identity (and existing
    # identities stay unchanged).
    excluded = {"concurrency", "max_rps", "cost_per_call_usd"}
    if candidate.pipeline_version is None:
        excluded.add("pipeline_version")
    raw = candidate.model_dump_json(exclude=excluded)
    out: dict[str, Any] = {
        "name": candidate.name,
        "adapter": candidate.adapter.kind,
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    if candidate.pipeline_version:
        out["pipeline_version"] = candidate.pipeline_version
    if candidate.adapter.kind == "hybrid":
        out["corpus_sha256"] = hashlib.sha256(
            (root / str(candidate.adapter.corpus)).read_bytes()
        ).hexdigest()
    return out
