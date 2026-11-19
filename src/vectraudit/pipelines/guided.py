"""Local guided evaluation: versioned drafts and bounded, pre-authorized jobs."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from vectraudit.core.jobs import JobCapacityError, JobRunner, ProgressFn
from vectraudit.evaluation.compare import compare_runs
from vectraudit.evaluation.dataset import parse_dataset
from vectraudit.evaluation.dataset_tools import dataset_rows, render_dataset
from vectraudit.evaluation.exceptions import acceptance
from vectraudit.evaluation.runfile import load_run, read_run_header
from vectraudit.evaluation.triage import triage
from vectraudit.labelling.storage import TaskBusy, task_lock
from vectraudit.labelling.task import atomic_write
from vectraudit.pipelines.benchmark import audit_benchmark
from vectraudit.pipelines.checkpoint import checkpoint_state, validate_path
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.local_display import run_doc_ids, run_document_titles, run_questions
from vectraudit.pipelines.manifest import candidate_config, load_manifest, open_adapter
from vectraudit.pipelines.models import Contract
from vectraudit.pipelines.release import corpus_version as documents_version
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline
from vectraudit.pipelines.runtime import HTTPPipeline, Pipeline

MAX_BODY = 5 * 1024 * 1024
MAX_QUERIES = 1000
"""Guided benchmark size: a 1,000-question suite."""
MAX_RUN_BYTES = 64 * 1024 * 1024
"""A 1,000-question run with stage traces at retrieval depth 100 is about 22 MiB."""
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
    title: str | None = Field(default=None, min_length=1, max_length=512)
    """Shown beside the ID in findings and cases; ``metadata.title`` is used when absent."""
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
    vector_weight: float = Field(default=0.0, ge=0, le=1)
    reranker: Literal["none", "phrase"] = "none"
    corpus: str = Field(default="", max_length=MAX_BODY)


class ResumeIn(Contract):
    retry_errors: bool = False
    confirm_remote: bool = False
    confirm_heldout: bool = False


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


def write_run(path: Path, run: dict[str, Any]) -> None:
    """Pipeline runs are written compactly: indentation nearly triples a 1,000-question run
    with stage traces (39 MiB becomes 104 MiB). Readers parse JSON, so nothing else changes."""
    text = json.dumps(run, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    atomic_write(path, text + "\n")


MAX_EVALUATIONS = 2
"""Evaluations one workspace runs at the same time (the PRD's two concurrent runs)."""


class PipelineBusy(Exception):
    """The registered pipeline already has an evaluation running in this workspace."""


class _Overlap:
    """Which evaluations are running, so each run records whether others ran beside it.

    A registered pipeline runs one evaluation at a time: its declared concurrency and rate
    limit describe one run, and two at once would silently double the load on its endpoint.
    Built-in hybrid runs build their own index and may overlap freely.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}

    def enter(self, identifier: str, pipeline: str) -> None:
        with self._lock:
            if pipeline != "builtin" and any(
                a["pipeline"] == pipeline for a in self._active.values()
            ):
                raise PipelineBusy(pipeline)
            others = set(self._active)
            for other in others:
                self._active[other]["seen"].add(identifier)
            self._active[identifier] = {"pipeline": pipeline, "seen": others}

    def leave(self, identifier: str) -> int:
        """Forget ``identifier``; returns how many other evaluations overlapped it."""
        with self._lock:
            entry = self._active.pop(identifier, None)
            return len(entry["seen"]) if entry else 0


class GuidedWorkspace:
    """One local artifact directory, durable drafts, and up to two concurrent evaluations."""

    def __init__(
        self,
        root: Path,
        manifest: Path | None = None,
        policy: Path | None = None,
        lineage: Path | None = None,
        lineage_ids: str = "chunk",
        changes: Path | None = None,
    ) -> None:
        from vectraudit.pipelines.release_workflow import PolicyStore

        self.root = root.resolve()
        self.drafts = self.root / ".workbench" / "drafts"
        self.token = secrets.token_urlsafe(32)
        self.jobs = JobRunner(max_workers=MAX_EVALUATIONS, max_pending=0, max_completed=32)
        self.running = _Overlap()
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
        # A lineage export is trusted startup configuration, like the manifest: validated
        # now, so a broken export fails at start rather than inside an investigation.
        from vectraudit.evaluation.lineage import LineageError, read_manifest

        if lineage_ids not in ("chunk", "source"):
            raise LineageError("lineage IDs must be chunk or source")
        self.lineage = read_manifest(lineage) if lineage else None
        self.lineage_ids = lineage_ids
        if changes is not None and self.lineage is None:
            raise LineageError("a change log needs the lineage export it is checked against")
        from vectraudit.evaluation.freshness import read_changes

        self.changes = read_changes(changes) if changes else None
        self._run_summaries: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}

    def run_summary(self, identifier: str) -> dict[str, Any]:
        """A validated run without its per-query records, cached by file size and mtime.

        Read one question at a time (:func:`read_run_header`), so a run of any size is listed;
        ``too_large`` says it is above the limit for opening it whole.
        """
        path = self.root / f"{_identifier(identifier)}.run.json"
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, "Run not found in this workspace")
        stat = path.stat()
        key = (stat.st_size, stat.st_mtime_ns)
        cached = self._run_summaries.get(identifier)
        if cached is None or cached[0] != key:
            summary = read_run_header(path)
            summary["too_large"] = stat.st_size > MAX_RUN_BYTES
            cached = (key, summary)
            self._run_summaries[identifier] = cached
        return cached[1]

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
        if len(ds) > MAX_QUERIES:
            raise HTTPException(422, f"Guided benchmarks support at most {MAX_QUERIES} queries")
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
        if path.stat().st_size > MAX_RUN_BYTES:
            raise HTTPException(413, "Run exceeds the local size limit")
        return path

    def evaluation_path(self, identifier: str) -> Path:
        """Resolve only an opaque local evaluation ID."""
        path = self.root / ".workbench" / "evaluations" / (_identifier(identifier) + ".json")
        validate_path(path, self.root)
        return path

    def evaluations(self) -> list[dict[str, Any]]:
        """Discover saved evaluations after a browser or server restart."""
        directory = self.evaluation_path("0" * 32).parent
        entries = []
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                saved = json.loads(self.evaluation_path(path.stem).read_text())
                journal = path.with_suffix(".sqlite")
                blocker = saved.get("resume_blocker")
                try:
                    if saved.get("started") and not journal.exists():
                        raise ValueError("Missing checkpoint")
                    state = (
                        checkpoint_state(journal)
                        if journal.exists()
                        else {
                            "status": "partial",
                            "finished": 0,
                            "failed": 0,
                            "total": saved["total"],
                        }
                    )
                except (sqlite3.Error, ValueError, OSError):
                    state = {
                        "status": "unavailable",
                        "finished": 0,
                        "failed": 0,
                        "total": saved["total"],
                    }
                    blocker = (
                        "Saved progress is missing or unreadable. "
                        "Restore the checkpoint or start a new evaluation."
                    )
                if (
                    state["status"] == "complete"
                    and not (self.root / f"{state['run_id']}.run.json").is_file()
                ):
                    state["status"] = (
                        "partial"  # Recover an export interrupted after journal commit.
                    )
                job = self.jobs.get(saved["job_id"])
                entries.append(
                    {
                        "id": path.stem,
                        "label": saved["body"]["label"],
                        "job_id": saved["job_id"],
                        "remote": saved["remote"],
                        "heldout": saved["body"]["mode"] == "heldout",
                        "resume_blocker": blocker,
                        **state,
                        "running": job is not None and job.as_dict()["status"] == "running",
                    }
                )
            except (OSError, ValueError, sqlite3.Error, KeyError):
                continue
        return entries

    def resume(self, identifier: str, body: ResumeIn) -> str:
        """Revalidate the original request and require fresh transfer/held-out consent."""
        path = self.evaluation_path(identifier)
        if not path.is_file():
            raise HTTPException(404, "Unknown evaluation")
        saved = json.loads(path.read_text())
        if saved.get("started") and not path.with_suffix(".sqlite").exists():
            raise HTTPException(
                409, "Saved progress is missing. Restore it or start a new evaluation."
            )
        original = EvaluationIn.model_validate(saved["body"])
        original.confirm_remote = body.confirm_remote
        original.confirm_heldout = body.confirm_heldout
        return self.start(original, evaluation_id=identifier, retry_errors=body.retry_errors)

    def start(
        self, body: EvaluationIn, *, evaluation_id: str | None = None, retry_errors: bool = False
    ) -> str:
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

        registered = self.registry.get(body.pipeline)
        options = PipelineOptions(
            k=body.k,
            repeats=body.repeats,
            warmup=0,
            label=body.label,
            params=params,
            concurrency=registered.concurrency if registered else 1,
            max_rps=registered.max_rps if registered else None,
        )
        identifier = uuid.uuid4().hex
        evaluation_path = self.evaluation_path(evaluation_id or identifier)
        journal = evaluation_path.with_suffix(".sqlite")
        resuming = evaluation_id is not None and journal.exists()
        saved = {
            "body": body.model_dump(exclude={"confirm_remote", "confirm_heldout"}),
            "job_id": identifier,
            "started": journal.exists(),
            "total": len(ds),
            "resume_blocker": (
                "This evaluation needs explicit pipeline_version and corpus_version to resume. "
                "Set them in the manifest before starting a new evaluation."
                if config.get("adapter") != "hybrid"
                and (not corpus_version or not config.get("pipeline_version"))
                else None
            ),
            "remote": body.pipeline != "builtin"
            and self.registry[body.pipeline].adapter.kind == "http",
        }

        def work(progress: ProgressFn) -> dict[str, Any]:
            try:
                run = evaluate(progress)
            finally:
                overlapped = self.running.leave(identifier)
            # Latency measured while another evaluation ran is latency under extra load.
            run["timing"]["overlapped_runs"] = overlapped
            if run["status"] != "cancelled":
                write_run(self.root / f"{run['run_id']}.run.json", run)
            return run

        def evaluate(progress: ProgressFn) -> dict[str, Any]:
            progress(0, "Preparing pipeline")
            pipeline = factory()
            with task_lock(evaluation_path):
                metadata = json.loads(evaluation_path.read_text())
                if metadata["job_id"] != identifier:
                    raise TaskBusy("Another process resumed this evaluation")
                metadata["started"] = True
                _write(evaluation_path, metadata)
            run = execute_pipeline(
                ds,
                pipeline,
                options,
                config=config,
                corpus_version=corpus_version,
                progress=lambda done, total: progress(done / total, f"{done}/{total} queries"),
                checkpoint=journal,
                resume=resuming,
                retry_errors=retry_errors and resuming,
            )
            run["benchmark_source"] = {
                "draft_id": draft["id"],
                "revision": draft["revision"],
                "sha256": source.sha256,
                "split": body.mode,
            }
            # Cancelled runs remain in the journal; they are not release artifacts.
            return run

        with task_lock(evaluation_path, timeout=0):
            previous = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
            if previous:
                active = self.jobs.get(previous["job_id"])
                if active and active.as_dict()["status"] == "running":
                    raise JobCapacityError("This evaluation is already running")
            self.running.enter(identifier, body.pipeline)
            # Persist before scheduling: a process may stop before the worker starts.
            _write(evaluation_path, saved)
            try:
                self.jobs.get_or_start(identifier, work)
            except JobCapacityError:
                self.running.leave(identifier)
                if previous:
                    _write(evaluation_path, previous)
                else:
                    evaluation_path.unlink(missing_ok=True)
                raise
        return identifier


def _document_display(
    workspace: GuidedWorkspace, baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """One title map for the documents either run returned, from each run's own benchmark."""
    ids = run_doc_ids(baseline, candidate)
    sides = [run_document_titles(workspace, run, ids) for run in (baseline, candidate)]
    titles = {**sides[1]["titles"], **sides[0]["titles"]}
    failed = None if titles else next((s for s in sides if s["reason"]), None)
    return {
        "titles": titles,
        "reason": failed["reason"] if failed else None,
        "code": failed["code"] if failed else None,
        "verified": all(s["verified"] for s in sides),
    }


CAPACITY = (
    f"{MAX_EVALUATIONS} evaluations are already running. Wait for one to finish or cancel it."
)
PIPELINE_BUSY = (
    "This pipeline already has an evaluation running. Its concurrency and rate limit apply to "
    "one evaluation at a time; wait for it to finish or cancel it."
)


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
            raise HTTPException(429, CAPACITY) from exc
        except PipelineBusy as exc:
            raise HTTPException(409, PIPELINE_BUSY) from exc

    @router.get("/evaluations", dependencies=auth)
    def evaluations() -> list[dict[str, Any]]:
        return workspace.evaluations()

    @router.post("/evaluations/{identifier}/resume", dependencies=auth, status_code=202)
    def resume(identifier: str, body: ResumeIn) -> dict[str, str]:
        try:
            return {"id": workspace.resume(identifier, body)}
        except JobCapacityError as exc:
            raise HTTPException(429, CAPACITY) from exc
        except PipelineBusy as exc:
            raise HTTPException(409, PIPELINE_BUSY) from exc

    @router.get("/jobs/{identifier}", dependencies=auth)
    def job(identifier: str) -> dict[str, Any]:
        found = workspace.jobs.get(_identifier(identifier))
        if found is None:
            raise HTTPException(404, "Evaluation expired or the server restarted")
        state = found.as_dict()
        if state["error"]:
            state["error"] = (
                "Evaluation failed. Check that saved inputs, pipeline/corpus versions and runner "
                "environment are unchanged, or use Test pipeline to diagnose the adapter."
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
            "question_display": {
                "baseline": run_questions(workspace, baseline),
                "candidate": run_questions(workspace, candidate),
            },
            "document_display": _document_display(workspace, baseline, candidate),
            # Who may accept each failure under this policy; exceptions themselves are
            # granted and audited only in the hosted project.
            "acceptance": acceptance(check, rules),
        }

    from vectraudit.pipelines.diagnostics import diagnostic_router
    from vectraudit.pipelines.investigation_web import investigation_router
    from vectraudit.pipelines.release_workflow import release_router
    from vectraudit.pipelines.review import review_router
    from vectraudit.pipelines.workflows import workflow_router

    router.include_router(workflow_router(workspace), dependencies=auth)
    router.include_router(release_router(workspace), dependencies=auth)
    router.include_router(review_router(workspace), dependencies=auth)
    router.include_router(diagnostic_router(workspace), dependencies=auth)
    router.include_router(investigation_router(workspace), dependencies=auth)
    return router
