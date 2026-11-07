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
    corpus_overlay: str | None = None
    """Hybrid only: a poison suite's ``poison-corpus.jsonl``, loaded in memory beside the corpus."""

    @model_validator(mode="after")
    def configured(self) -> Adapter:
        if self.corpus_overlay is not None and self.kind != "hybrid":
            raise ValueError(
                "corpus_overlay is for the hybrid baseline; declare a staging environment and "
                "overlay on the candidate for other pipelines"
            )
        if self.kind == "hybrid" and not self.corpus:
            raise ValueError("hybrid adapter requires corpus")
        if self.kind != "hybrid" and not self.target:
            raise ValueError("python/http adapter requires target")
        return self


FIXABLE = (
    "not_observed",
    "lost_downstream",
    "lost_in_context",
    "answered_wrong_despite_evidence",
)
"""Failure classes a declared fix may address; permission and no-answer failures are not
among them, because loosening access is never a repair."""


class Fix(Contract):
    """A bounded change the pipeline supports, as parameter overrides, and what it targets.

    ``addresses`` names failure classes (``lost_downstream``), optionally narrowed to a stage
    kind (``lost_downstream:rerank``). Only declared fixes are ever suggested.
    """

    name: str = Field(min_length=1, max_length=128)
    addresses: list[str] = Field(min_length=1, max_length=16)
    params: dict[str, str | int | float | bool] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def known(self) -> Fix:
        for item in self.addresses:
            kind, _, stage = item.partition(":")
            if kind not in FIXABLE or (stage and kind != "lost_downstream"):
                raise ValueError(
                    f"fix {self.name!r} addresses {item!r}; use one of {', '.join(FIXABLE)}, "
                    "or lost_downstream:<stage kind>"
                )
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
    fixes: list[Fix] = Field(default_factory=list, max_length=20)
    """Changes this pipeline supports, offered by ``workbench investigate`` and ``try-fix``."""
    environment: Literal["staging"] | None = None
    """Declares that this candidate runs against a staging index, not production."""
    overlay: str | None = None
    """A poison suite's ``poison-suite.json``, whose documents *you* loaded into the staging
    index. Vectraudit inserts nothing; it checks the suite's hashes and reads the run."""

    @model_validator(mode="after")
    def unique_fixes(self) -> Candidate:
        if len({fix.name for fix in self.fixes}) != len(self.fixes):
            raise ValueError("fix names must be unique within a candidate")
        if self.overlay is not None and self.environment != "staging":
            raise ValueError(
                'an overlay needs "environment": "staging": poisoned documents belong in a '
                "staging index, never production"
            )
        return self


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


def _read_corpus(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def open_adapter(adapter: Adapter, root: Path) -> Pipeline:
    """Open an explicitly authorized local adapter; no import path is accepted via HTTP."""
    if adapter.kind == "hybrid":
        docs = _read_corpus(root / str(adapter.corpus))
        if adapter.corpus_overlay is not None:
            extra = _read_corpus(root / adapter.corpus_overlay)
            clash = sorted({d["doc_id"] for d in docs} & {d["doc_id"] for d in extra})
            if clash:
                raise ValueError(
                    f"corpus_overlay repeats {len(clash)} document IDs of the corpus "
                    f"(first: {clash[0]!r}); overlay IDs must be new"
                )
            docs = [*docs, *extra]
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
    excluded = {"concurrency", "max_rps", "cost_per_call_usd", "fixes"}
    if candidate.pipeline_version is None:
        excluded.add("pipeline_version")
    # Fields added later stay out of the identity while unset, so existing identities hold.
    excluded |= {f for f in ("environment", "overlay") if getattr(candidate, f) is None}
    exclude: dict[str, Any] = dict.fromkeys(excluded, True)
    if candidate.adapter.corpus_overlay is None:
        exclude["adapter"] = {"corpus_overlay"}
    raw = candidate.model_dump_json(exclude=exclude)
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
        if candidate.adapter.corpus_overlay is not None:
            out["corpus_overlay_sha256"] = hashlib.sha256(
                (root / candidate.adapter.corpus_overlay).read_bytes()
            ).hexdigest()
    if candidate.overlay is not None:
        out["overlay_sha256"] = hashlib.sha256((root / candidate.overlay).read_bytes()).hexdigest()
    return out
