"""Local guided evaluation: versioned drafts and bounded, pre-authorized jobs."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from vectraudit.core.jobs import JobCapacityError, JobRunner, ProgressFn
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.runfile import load_run
from vectraudit.evaluation.triage import triage
from vectraudit.labelling.storage import TaskBusy, task_lock
from vectraudit.labelling.task import atomic_write
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.manifest import candidate_config, load_manifest, open_adapter
from vectraudit.pipelines.models import Contract
from vectraudit.pipelines.release import corpus_version as documents_version
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline
from vectraudit.pipelines.runtime import HTTPPipeline, Pipeline

MAX_BODY = 5 * 1024 * 1024
ID = re.compile(r"^[a-f0-9]{32}$")


class DraftIn(Contract):
    id: str | None = None
    revision: str | None = None
    name: str = Field(min_length=1, max_length=128)
    dataset: str = Field(max_length=MAX_BODY)
    corpus: str = Field(default="", max_length=MAX_BODY)
    irrelevant: dict[str, list[str]] = Field(default_factory=dict)


class Document(Contract):
    doc_id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=100000)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    vector: list[float] | None = Field(default=None, max_length=4096)


class EvaluationIn(Contract):
    draft_id: str
    revision: str
    pipeline: str = "builtin"
    label: str = Field(default="Candidate", min_length=1, max_length=128)
    mode: Literal["explore", "heldout"] = "explore"
    confirm_heldout: bool = False
    confirm_remote: bool = False
    k: int = Field(default=5, ge=1, le=100)
    repeats: int = Field(default=1, ge=1, le=5)
    depth: int = Field(default=50, ge=1, le=10000)
    vector_weight: float = Field(default=0, ge=0, le=1)
    reranker: Literal["none", "phrase"] = "none"
    corpus: str = Field(default="", max_length=MAX_BODY)


class ComparisonIn(Contract):
    baseline: str
    candidate: str
    policy: str | None = None
    """Saved or startup policy id; ``None`` compares under the exploratory defaults."""


def _identifier(value: str) -> str:
    if not ID.fullmatch(value):
        raise HTTPException(404, "Unknown local resource")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


class GuidedWorkspace:
    """One local artifact directory, durable drafts, and a single evaluation worker."""

    def __init__(
        self, root: Path, manifest: Path | None = None, policy: Path | None = None
    ) -> None:
        from vectraudit.pipelines.release_workflow import PolicyStore

        self.root = root.resolve()
        self.drafts = self.root / ".workbench" / "drafts"
        self.token = secrets.token_urlsafe(32)
        self.jobs = JobRunner(max_workers=1, max_pending=0, max_completed=32)
        self.manifest = load_manifest(manifest) if manifest else None
        self.manifest_root = manifest.resolve().parent if manifest else self.root
        # Registration is trusted process configuration, never accepted from a browser request.
        self.registry = {
            f"configured-{i}": candidate
            for i, candidate in enumerate(self.manifest.candidates if self.manifest else [])
        }

        for candidate in self.registry.values():
            if candidate.adapter.kind == "http":
                HTTPPipeline(str(candidate.adapter.target), timeout=candidate.adapter.timeout)
        self.policies = PolicyStore(self, policy)

    def _draft_path(self, identifier: str) -> Path:
        path = self.drafts / f"{_identifier(identifier)}.json"
        for parent in (self.root / ".workbench", self.drafts, path, path.with_suffix(".json.lock")):
            if parent.is_symlink():
                raise HTTPException(400, "Workspace paths must not be symbolic links")
        return path

    def load_draft(self, identifier: str) -> dict[str, Any]:
        path = self._draft_path(identifier)
        if not path.is_file():
            raise HTTPException(404, "Benchmark draft not found")
        if path.stat().st_size > MAX_BODY * 2:
            raise HTTPException(413, "Draft exceeds the local size limit")
        value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return value

    def save_draft(self, body: DraftIn) -> dict[str, Any]:
        """Validate annotation semantics before an optimistic, atomic draft update."""
        from vectraudit.pipelines.workflows import documents

        ds = parse_dataset(body.dataset)
        documents(body.corpus)
        queries = {q.query_id: q for q in ds.queries}
        for query_id, ids in body.irrelevant.items():
            query = queries.get(query_id)
            if query is None or len(ids) > 5000 or any(not d or len(d) > 512 for d in ids):
                raise ValueError("Invalid irrelevant-document judgments")
            positive = {d for d, grade in query.relevant.items() if grade > 0}
            positive |= {span.doc_id for span in query.relevant_spans}
            if positive & set(ids):
                raise ValueError("A document cannot be both relevant and irrelevant")
        if len(ds) > 500:
            raise HTTPException(422, "Guided benchmarks support at most 500 queries")
        identifier = body.id or uuid.uuid4().hex
        path = self._draft_path(identifier)
        with task_lock(path):
            previous = self.load_draft(identifier) if path.exists() else None
            if (previous and previous["revision"] != body.revision) or (
                not previous and (body.revision or body.id)
            ):
                raise HTTPException(409, "Benchmark changed. Reload it before saving.")
            dataset = render_dataset(dataset_rows(ds), ds.meta)
            revision = hashlib.sha256(
                json.dumps(
                    [body.name, dataset, body.corpus, body.irrelevant], sort_keys=True
                ).encode()
            ).hexdigest()
            value = {
                "id": identifier,
                "revision": revision,
                "name": body.name,
                "dataset": dataset,
                "corpus": body.corpus,
                "irrelevant": body.irrelevant,
                "audit": audit_benchmark(ds),
            }
            _write(path, value)
        return value

    def describe(self) -> dict[str, Any]:
        pipelines = [
            {"id": "builtin", "name": "Local BM25 / hybrid", "kind": "hybrid", "destination": None}
        ]
        for key, candidate in self.registry.items():
            pipelines.append(
                {
                    "id": key,
                    "name": candidate.name,
                    "kind": candidate.adapter.kind,
                    "destination": candidate.adapter.target
                    if candidate.adapter.kind == "http"
                    else None,
                }
            )
        drafts = []
        for path in sorted(self.drafts.glob("*.json"))[:100]:
            if path.is_symlink() or not ID.fullmatch(path.stem):
                continue
            draft = self.load_draft(path.stem)
            drafts.append({k: draft[k] for k in ("id", "revision", "name")})
        return {"token": self.token, "pipelines": pipelines, "drafts": drafts}

    def run_path(self, identifier: str) -> Path:
        path = self.root / f"{_identifier(identifier)}.run.json"
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, "Run not found in this workspace")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise HTTPException(413, "Run exceeds the local size limit")
        return path

    def start(self, body: EvaluationIn) -> str:
        """Snapshot the draft and authorized configuration before queuing execution."""
        draft = self.load_draft(body.draft_id)
        if draft["revision"] != body.revision:
            raise HTTPException(409, "Benchmark changed. Reload before running.")
        source = parse_dataset(draft["dataset"])
        if body.mode == "heldout" and not body.confirm_heldout:
            raise HTTPException(422, "Confirm final held-out evaluation explicitly")
        rows = [r for r in dataset_rows(source) if r.get("split", "explore") == body.mode]
        if not rows:
            raise HTTPException(422, f"This benchmark has no {body.mode} queries")
        ds = parse_dataset(render_dataset(rows, source.meta))
        if body.depth < body.k:
            raise HTTPException(422, "Retrieval depth must be at least k")
        corpus_version: str | None
        if body.pipeline == "builtin":
            documents = [
                Document.model_validate_json(line).model_dump(exclude_none=True)
                for line in body.corpus.splitlines()
                if line.strip()
            ]
            if not 1 <= len(documents) <= 5000:
                raise HTTPException(422, "Provide a JSONL corpus containing 1-5000 documents")
            # Prevalidate the corpus; build the index in the background worker.
            if len({d["doc_id"] for d in documents}) != len(documents):
                raise HTTPException(422, "Corpus document ids must be unique")
            corpus_version = documents_version(documents)
            params: dict[str, Any] = {
                "depth": body.depth,
                "vector_weight": body.vector_weight,
                "reranker": body.reranker,
            }
            config = {"adapter": "hybrid", "params": params}

            def factory() -> Pipeline:
                return hybrid_pipeline(documents)
        else:
            candidate = self.registry.get(body.pipeline)
            if candidate is None:
                raise HTTPException(422, "Select a pipeline registered at server startup")
            if candidate.adapter.kind == "http" and not body.confirm_remote:
                raise HTTPException(422, "Confirm query transfer to the configured endpoint")
            params = dict(candidate.params)
            config = candidate_config(candidate, self.manifest_root)
            corpus_version = self.manifest.corpus_version if self.manifest else None
            corpus_version = corpus_version or (
                str(config["corpus_sha256"]) if config.get("corpus_sha256") else None
            )
            adapter = candidate.adapter

            def factory() -> Pipeline:
                return open_adapter(adapter, self.manifest_root)

        options = PipelineOptions(
            k=body.k, repeats=body.repeats, warmup=0, label=body.label, params=params
        )
        identifier = uuid.uuid4().hex

        def work(progress: ProgressFn) -> dict[str, Any]:
            progress(0, "Preparing pipeline")
            pipeline = factory()
            run = execute_pipeline(
                ds,
                pipeline,
                options,
                config=config,
                corpus_version=corpus_version,
                progress=lambda done, total: progress(done / total, f"{done}/{total} queries"),
            )
            progress(1, "Saving results")
            run["benchmark_source"] = {
                "draft_id": draft["id"],
                "revision": draft["revision"],
                "sha256": source.sha256,
                "split": body.mode,
            }
            _write(self.root / f"{run['run_id']}.run.json", run)
            return run

        self.jobs.get_or_start(identifier, work)
        return identifier


def guided_router(workspace: GuidedWorkspace) -> APIRouter:
    """Local-only API; mutations require a per-process browser token and same origin."""
    router = APIRouter(prefix="/api/guided")

    def authorize(request: Request) -> None:
        if not secrets.compare_digest(
            request.headers.get("x-workbench-token", ""), workspace.token
        ):
            raise HTTPException(403, "Reload the workbench to establish a local session")
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
            raise HTTPException(403, "Cross-origin workbench requests are not allowed")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "Cross-site workbench requests are not allowed")

    auth = [Depends(authorize)]

    @router.get("/session")
    def session() -> dict[str, Any]:
        return workspace.describe()

    @router.post("/drafts", dependencies=auth)
    def save(body: DraftIn) -> dict[str, Any]:
        try:
            return workspace.save_draft(body)
        except TaskBusy as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/drafts/{identifier}", dependencies=auth)
    def draft(identifier: str) -> dict[str, Any]:
        return workspace.load_draft(identifier)

    @router.post("/jobs", dependencies=auth, status_code=202)
    def start(body: EvaluationIn) -> dict[str, str]:
        try:
            return {"id": workspace.start(body)}
        except JobCapacityError as exc:
            raise HTTPException(
                429, "An evaluation is already running. Wait or cancel it."
            ) from exc

    @router.get("/jobs/{identifier}", dependencies=auth)
    def job(identifier: str) -> dict[str, Any]:
        found = workspace.jobs.get(_identifier(identifier))
        if found is None:
            raise HTTPException(404, "Evaluation expired or the server restarted")
        state = found.as_dict()
        if state["error"]:
            state["error"] = (
                "Evaluation failed. Use Test pipeline to diagnose the registered adapter."
            )
        if found.is_done:
            state["run"] = found.result
        return state

    @router.post("/jobs/{identifier}/cancel", dependencies=auth)
    def cancel(identifier: str) -> dict[str, bool]:
        return {"cancel_requested": workspace.jobs.cancel(_identifier(identifier))}

    @router.post("/compare", dependencies=auth)
    def compare(body: ComparisonIn) -> dict[str, Any]:
        base_path = workspace.run_path(body.baseline)
        cand_path = workspace.run_path(body.candidate)
        baseline, candidate = load_run(base_path), load_run(cand_path)
        rules, policy = workspace.policies.resolve(body.policy)
        check = compare_runs(baseline, candidate, rules)
        identifier = uuid.uuid4().hex
        _write(workspace.root / f"{identifier}.check.json", check)
        # Name the files on disk so every finding points at the evidence it came from.
        files = {
            "baseline": base_path.name,
            "candidate": cand_path.name,
            "check": f"{identifier}.check.json",
        }
        return {
            "check": check,
            "policy": policy,
            "triage": triage(baseline, candidate),
            "files": files,
            "baseline": baseline,
            "candidate": candidate,
        }

    from vectraudit.pipelines.diagnostics import diagnostic_router
    from vectraudit.pipelines.release_workflow import release_router
    from vectraudit.pipelines.review import review_router
    from vectraudit.pipelines.workflows import workflow_router

    router.include_router(workflow_router(workspace), dependencies=auth)
    router.include_router(release_router(workspace), dependencies=auth)
    router.include_router(review_router(workspace), dependencies=auth)
    router.include_router(diagnostic_router(workspace), dependencies=auth)
    return router
