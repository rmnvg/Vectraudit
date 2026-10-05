"""Local manifest loading; Python imports are an explicit trusted-code operation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

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
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)


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
    fn: Pipeline = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise ValueError("Python adapter target is not callable")
    return fn


def candidate_config(candidate: Candidate, root: Path) -> dict[str, Any]:
    """Safe identity for drift tracking; exclude endpoint credentials and local paths."""
    raw = candidate.model_dump_json()
    out: dict[str, Any] = {
        "name": candidate.name,
        "adapter": candidate.adapter.kind,
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    if candidate.adapter.kind == "hybrid":
        out["corpus_sha256"] = hashlib.sha256(
            (root / str(candidate.adapter.corpus)).read_bytes()
        ).hexdigest()
    return out
