"""Workbench API for the investigation screen: explain a question, try a fix, validate it.

A fix can be tried only on the configuration that produced the run: the run's saved
benchmark must be unchanged, the built-in pipeline's corpus must match the run's corpus
version, and a registered pipeline must still be registered with the same configuration.
Otherwise the screen explains why instead of running something else.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from pydantic import Field

from vectraudit.core.jobs import JobCapacityError, ProgressFn
from vectraudit.evaluation.dataset import Dataset, parse_dataset
from vectraudit.evaluation.runfile import load_run
from vectraudit.labelling.storage import task_lock
from vectraudit.pipelines.guided import (
    CAPACITY,
    PIPELINE_BUSY,
    Document,
    GuidedWorkspace,
    PipelineBusy,
    _identifier,
    _write,
    write_run,
)
from vectraudit.pipelines.investigate import (
    available_fixes,
    investigate,
    try_fix,
    try_plan,
    validate_fix,
)
from vectraudit.pipelines.investigation_records import (
    close,
    link_result,
    measures,
    new_record,
    protection_policy,
    record_protection,
    request_evidence,
    set_diagnosis,
    summarize,
)
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.local_display import run_questions
from vectraudit.pipelines.manifest import Adapter, Candidate, candidate_config, open_adapter
from vectraudit.pipelines.models import Contract
from vectraudit.pipelines.release import corpus_version as documents_version
from vectraudit.pipelines.runtime import Pipeline
from vectraudit.pipelines.workflows import LocalRecords


class InvestigateIn(Contract):
    run: str
    query_id: str = Field(min_length=1, max_length=512)
    baseline: str | None = None


class TrialIn(Contract):
    run: str
    fix: str = Field(min_length=1, max_length=128)
    query_ids: list[str] = Field(min_length=1, max_length=50)
    sample: int = Field(default=20, ge=0, le=200)
    seed: int = Field(default=0, ge=0, le=2**31)
    confirm_remote: bool = False


Name = Annotated[str, Field(min_length=1, max_length=80)]


class OpenIn(Contract):
    run: str
    query_id: str = Field(min_length=1, max_length=512)
    baseline: str | None = None
    by: Name


class DiagnosisIn(Contract):
    status: str
    by: Name
    reason: str | None = Field(default=None, max_length=2000)


class RequestIn(Contract):
    what: str = Field(min_length=1, max_length=2000)
    by: Name


class LinkIn(Contract):
    result: str
    by: Name


class CloseIn(Contract):
    outcome: str
    by: Name
    note: str | None = Field(default=None, max_length=2000)


class ProtectIn(Contract):
    policy: str
    by: Name
    context: bool = False
    doc_id: str | None = Field(default=None, min_length=1, max_length=512)
    max_rank: int | None = Field(default=None, ge=1, le=10000)


class ValidateIn(Contract):
    run: str
    fix: str = Field(min_length=1, max_length=128)
    policy: str | None = None
    confirm_heldout: bool = False
    confirm_remote: bool = False


@dataclass
class _Source:
    """The configuration that produced a run, if it can be run again here."""

    candidate: Candidate
    pipeline_key: str
    dataset: Dataset | None
    factory: Any
    config: dict[str, Any]
    corpus_version: str | None
    remote: str | None
    has_vectors: bool
    blocker: str | None


def _source(workspace: GuidedWorkspace, run: dict[str, Any]) -> _Source:
    backend = run.get("backend") or {}
    config = dict(backend.get("config") or {})
    label = str(run.get("label") or "run")[:128]
    source = run.get("benchmark_source") or {}
    dataset = None
    blocker = None
    draft: dict[str, Any] | None = None
    try:
        draft = workspace.load_draft(str(source.get("draft_id")))
    except HTTPException:
        blocker = "This run's saved benchmark is not in this workspace."
    if draft is not None:
        if draft["revision"] != source.get("revision"):
            blocker = "The benchmark changed after this run; run it again before trying a fix."
        dataset = parse_dataset(draft["dataset"])
    if config.get("adapter") == "hybrid":
        documents = [
            Document.model_validate_json(line).model_dump(exclude_none=True)
            for line in ((draft or {}).get("corpus") or "").splitlines()
            if line.strip()
        ]
        if blocker is None and documents_version(documents) != backend.get("corpus_version"):
            blocker = "The benchmark's corpus differs from the one this run used."
        params = {
            k: v
            for k, v in (config.get("params") or {}).items()
            if type(v) in (str, int, float, bool)
        }
        candidate = Candidate(
            name=label, adapter=Adapter(kind="hybrid", corpus="draft corpus"), params=params
        )

        def built() -> Pipeline:
            return hybrid_pipeline(documents)

        return _Source(
            candidate,
            "builtin",
            dataset,
            built,
            config,
            backend.get("corpus_version"),
            None,
            bool(documents) and all("vector" in d for d in documents),
            blocker,
        )
    for key, registered in workspace.registry.items():
        if candidate_config(registered, workspace.manifest_root).get("sha256") == config.get(
            "sha256"
        ):
            adapter = registered.adapter

            def opened(adapter: Adapter = adapter) -> Pipeline:
                return open_adapter(adapter, workspace.manifest_root)

            version = workspace.manifest.corpus_version if workspace.manifest else None
            return _Source(
                registered,
                key,
                dataset,
                opened,
                config,
                version or backend.get("corpus_version"),
                str(adapter.target) if adapter.kind == "http" else None,
                False,
                blocker,
            )
    candidate = Candidate(
        name=label, adapter=Adapter(kind="python", target="unregistered:pipeline")
    )
    return _Source(
        candidate,
        "",
        dataset,
        None,
        config,
        None,
        None,
        False,
        blocker
        or "The pipeline that produced this run is not registered with the same configuration "
        "in this workbench session.",
    )


def _fix(source: _Source, name: str) -> Any:
    for fix in available_fixes(source.candidate, has_vectors=source.has_vectors):
        if fix.name == name:
            return fix
    raise HTTPException(422, f"{source.candidate.name} declares no fix named {name!r}")


def investigation_router(workspace: GuidedWorkspace) -> APIRouter:
    """Routes under ``/api/guided``; the caller adds the session authorization."""
    router = APIRouter()
    trials = workspace.root / ".workbench" / "fixes"

    def load(identifier: str) -> dict[str, Any]:
        return load_run(workspace.run_path(identifier))

    @router.post("/investigate")
    def explain(body: InvestigateIn) -> dict[str, Any]:
        run = load(body.run)
        source = _source(workspace, run)
        fixes = available_fixes(source.candidate, has_vectors=source.has_vectors)
        baseline = load(body.baseline) if body.baseline else None
        result = investigate(run, body.query_id, fixes, baseline=baseline)
        held = sum(q.split == "heldout" for q in source.dataset.queries) if source.dataset else 0
        explore = (
            {q.query_id for q in source.dataset.queries if q.split == "explore"}
            if (source.dataset)
            else set()
        )
        return {
            **result,
            "question": run_questions(workspace, run)["texts"].get(body.query_id),
            "pipeline": {
                "name": source.candidate.name,
                "remote": source.remote,
                "params": dict(source.candidate.params),
            },
            "trial": {
                "available": source.blocker is None and body.query_id in explore,
                "reason": source.blocker
                or (
                    None
                    if body.query_id in explore
                    else "Held-out questions are reserved for validation; fixes are tried on "
                    "exploratory questions."
                ),
                "explore_questions": len(explore),
                "heldout_questions": held,
                "repeats": (run.get("timing") or {}).get("repeats") or 1,
                "k": run["search"]["k"],
            },
        }

    def start(
        run_id: str, confirm_remote: bool, kind: str, work: Any, source: _Source
    ) -> dict[str, str]:
        if source.blocker is not None or source.factory is None or source.dataset is None:
            raise HTTPException(409, source.blocker or "This run cannot be run again here.")
        if source.remote and not confirm_remote:
            raise HTTPException(422, f"Confirm query transfer to {source.remote}")
        identifier = uuid.uuid4().hex
        trials.mkdir(parents=True, exist_ok=True)

        def job(progress: ProgressFn) -> dict[str, Any]:
            try:
                progress(0, "Preparing pipeline")
                result: dict[str, Any] = work(source.factory(), progress)
            finally:
                workspace.running.leave(identifier)
            folder = trials / identifier
            folder.mkdir(exist_ok=True)
            for side, recorded in result.pop("runs").items():
                write_run(folder / f"{side}.run.json", recorded)
            result.update(kind=kind, run=run_id, id=identifier)
            _write(trials / f"{identifier}.json", result)
            return result

        workspace.running.enter(identifier, source.pipeline_key)
        try:
            workspace.jobs.get_or_start(identifier, job)
        except JobCapacityError as exc:
            workspace.running.leave(identifier)
            raise HTTPException(429, CAPACITY) from exc
        return {"id": identifier}

    @router.post("/fixes/try", status_code=202)
    def trial(body: TrialIn) -> dict[str, str]:
        run = load(body.run)
        source = _source(workspace, run)
        fix = _fix(source, body.fix)
        if source.dataset is not None:
            try_plan(source.dataset, body.query_ids, body.sample, body.seed)  # validate first
        k, repeats = run["search"]["k"], (run.get("timing") or {}).get("repeats") or 1

        def work(pipeline: Pipeline, progress: ProgressFn) -> dict[str, Any]:
            progress(0.1, "Running current and fixed configurations")
            assert source.dataset is not None
            return try_fix(
                source.dataset,
                pipeline,
                source.candidate,
                fix,
                body.query_ids,
                k=k,
                repeats=repeats,
                sample=body.sample,
                seed=body.seed,
                config=source.config,
                corpus_version=source.corpus_version,
            )

        try:
            return start(body.run, body.confirm_remote, "trial", work, source)
        except PipelineBusy as exc:
            raise HTTPException(409, PIPELINE_BUSY) from exc

    @router.post("/fixes/validate", status_code=202)
    def validate(body: ValidateIn) -> dict[str, str]:
        run = load(body.run)
        source = _source(workspace, run)
        fix = _fix(source, body.fix)
        if not body.confirm_heldout:
            raise HTTPException(422, "Confirm validation on held-out questions explicitly")
        if source.dataset is not None and not any(
            q.split == "heldout" for q in source.dataset.queries
        ):
            raise HTTPException(
                422, "The benchmark has no held-out questions to validate the fix on"
            )
        rules, _ = workspace.policies.resolve(body.policy)
        k, repeats = run["search"]["k"], (run.get("timing") or {}).get("repeats") or 1

        def work(pipeline: Pipeline, progress: ProgressFn) -> dict[str, Any]:
            progress(0.1, "Running both configurations on held-out questions")
            assert source.dataset is not None
            return validate_fix(
                source.dataset,
                pipeline,
                source.candidate,
                fix,
                rules if body.policy is not None else None,
                k=k,
                repeats=repeats,
                config=source.config,
                corpus_version=source.corpus_version,
            )

        try:
            return start(body.run, body.confirm_remote, "validation", work, source)
        except PipelineBusy as exc:
            raise HTTPException(409, PIPELINE_BUSY) from exc

    def saved_result(identifier: str) -> dict[str, Any]:
        path = trials / f"{_identifier(identifier)}.json"
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, "Fix trial not found")
        value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return value

    @router.get("/fixes/{identifier}")
    def saved(identifier: str) -> dict[str, Any]:
        return saved_result(identifier)

    records = LocalRecords(workspace, "investigations")

    def change(identifier: str, edit: Any) -> dict[str, Any]:
        path = records.path(identifier)
        with task_lock(path):
            record = records.read(identifier)
            if record["status"] != "open":
                raise HTTPException(409, "This investigation is closed")
            try:
                edit(record)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            return records.write(record)

    @router.post("/investigations")
    def open_record(body: OpenIn) -> dict[str, Any]:
        run = load(body.run)
        losses = [loss["class"] for loss in investigate(run, body.query_id)["losses"]]
        for existing in records.list():
            if (
                existing["status"] == "open"
                and existing["run"] == body.run
                and existing["query_id"] == body.query_id
            ):
                return existing
        identifier = uuid.uuid4().hex
        return records.write(
            new_record(identifier, body.run, body.query_id, body.baseline, body.by, losses)
        )

    @router.get("/investigations")
    def summary() -> dict[str, Any]:
        return summarize(records.list())

    @router.get("/investigations/{identifier}")
    def read_record(identifier: str) -> dict[str, Any]:
        return {**records.read(identifier), "measures": measures(records.read(identifier))}

    @router.post("/investigations/{identifier}/diagnosis")
    def diagnosis(identifier: str, body: DiagnosisIn) -> dict[str, Any]:
        return change(identifier, lambda r: set_diagnosis(r, body.status, body.by, body.reason))

    @router.post("/investigations/{identifier}/requests")
    def evidence(identifier: str, body: RequestIn) -> dict[str, Any]:
        return change(identifier, lambda r: request_evidence(r, body.what, body.by))

    @router.post("/investigations/{identifier}/results")
    def link(identifier: str, body: LinkIn) -> dict[str, Any]:
        result = saved_result(body.result)  # read from disk: the browser cannot invent results

        def attach(record: dict[str, Any]) -> None:
            if result.get("run") != record["run"]:
                raise ValueError("this result tried a fix on a different run")
            link_result(record, result, body.by)

        return change(identifier, attach)

    @router.post("/investigations/{identifier}/protect")
    def protect(identifier: str, body: ProtectIn) -> dict[str, Any]:
        from vectraudit.pipelines.release_workflow import STARTUP, PolicyIn

        if body.policy == STARTUP:
            raise HTTPException(
                422, "The startup policy is read-only; save it as a workspace policy first"
            )
        if (body.doc_id is None) != (body.max_rank is None):
            raise HTTPException(422, "A rank rule needs both a document and a maximum rank")
        path = records.path(identifier)
        with task_lock(path):
            record = records.read(identifier)
            if record.get("outcome") != "fixed":
                raise HTTPException(
                    409, "Protect a question after its fix is verified and the record closed"
                )
            run = load(record["run"])
            query = next(q for q in run["queries"] if q["query_id"] == record["query_id"])
            states = [r.get("context") for r in query.get("repeat_sufficiency") or []]
            if body.context and (not states or "unrecorded" in states):
                # A rule the pipeline cannot be judged on would block every later release.
                raise HTTPException(
                    422,
                    "This question cannot be protected in context: it needs required_evidence "
                    "in the benchmark and a pipeline that records what reaches the model",
                )
            if body.max_rank is not None and body.max_rank > run["search"]["k"]:
                raise HTTPException(422, "The maximum rank cannot exceed the run's k")
            current = workspace.policies.record(body.policy)
            rank = (body.doc_id, body.max_rank) if body.doc_id and body.max_rank else None
            try:
                raw, added = protection_policy(
                    current["policy"], record["query_id"], context=body.context, rank=rank
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            saved = workspace.policies.save(
                PolicyIn(
                    name=current["name"],
                    policy=raw,
                    note=f"Protects {record['query_id']} after investigation "
                    f"{identifier[:8]} (fixed); added by {body.by}.",
                )
            )
            record_protection(record, saved, added, body.by)
            records.write(record)
        return {"record": record, "policy": saved, "added": added}

    @router.post("/investigations/{identifier}/close")
    def finish(identifier: str, body: CloseIn) -> dict[str, Any]:
        return change(identifier, lambda r: close(r, body.outcome, body.by, body.note))

    return router
