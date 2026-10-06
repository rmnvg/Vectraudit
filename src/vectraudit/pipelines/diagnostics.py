"""Live connection check: one sample call to a registered pipeline in a fresh process.

The probe runs out of process so a hung adapter can be killed at its time limit, and so
adapter output (which may contain credentials) never reaches the workbench. Only the
probe's allowlisted :class:`DiagnosticReport` crosses back.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import Field, ValidationError

from vectraudit.pipelines.diagnostic_report import DiagnosticReport, failure_report
from vectraudit.pipelines.models import Contract, PipelineRequest

if TYPE_CHECKING:
    from vectraudit.pipelines.guided import GuidedWorkspace

MAX_PAYLOAD = 6 * 1024 * 1024
MAX_REPORT = 1024 * 1024
Scalar = str | int | float | bool


class DiagnosticIn(Contract):
    pipeline: str = "builtin"
    confirm_remote: bool = False
    text: str = Field(min_length=1, max_length=10000)
    vector: list[float] | None = Field(default=None, max_length=4096)
    k: int = Field(default=5, ge=1, le=100)
    filters: dict[str, Scalar] = Field(default_factory=dict, max_length=32)
    context: dict[str, Scalar] = Field(default_factory=dict, max_length=32)
    forbidden: list[str] = Field(default_factory=list, max_length=1000)
    no_answer: bool = False
    corpus: str = Field(default="", max_length=5 * 1024 * 1024)
    """JSONL passages used for document lookup; required for the built-in pipeline."""
    timeout_s: float = Field(default=60, ge=1, le=300)


class DiagnosticRunner:
    """At most one probe at a time; a cancel kills the probe's whole process group."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._cancelled = False

    def run(self, payload: dict[str, Any], timeout: float) -> DiagnosticReport:
        """Run the probe with ``payload``; failures become allowlisted report categories."""
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if len(data) > MAX_PAYLOAD:
            raise HTTPException(413, "Diagnostic input exceeds the 6 MiB limit")
        if not self._lock.acquire(blocking=False):
            raise HTTPException(429, "A pipeline test is already running. Wait or cancel it.")
        try:
            self._cancelled = False
            process = subprocess.Popen(
                [sys.executable, "-m", "vectraudit.pipelines.diagnostic_probe"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
            self._process = process
            try:
                out, _ = process.communicate(data, timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill(process)
                process.communicate()
                return failure_report("timeout")
            finally:
                self._process = None
            if self._cancelled:
                return failure_report("cancelled")
            if len(out) > MAX_REPORT:
                return failure_report("worker_failed")
            try:
                return DiagnosticReport.model_validate_json(out)
            except ValidationError:
                return failure_report("worker_failed")
        finally:
            self._lock.release()

    def cancel(self) -> bool:
        """Kill the running probe, if any; returns whether one was running."""
        process = self._process
        if process is None or process.poll() is not None:
            return False
        self._cancelled = True
        _kill(process)
        return True


def _kill(process: subprocess.Popen[bytes]) -> None:
    # Adapters may start helper processes; on POSIX the probe leads its own group.
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass


def diagnostic_payload(workspace: GuidedWorkspace, body: DiagnosticIn) -> dict[str, Any]:
    """Resolve the trusted registration; the browser never supplies an adapter or path."""
    from vectraudit.pipelines.guided import Document

    documents = [
        Document.model_validate_json(line).model_dump(exclude_none=True)
        for line in body.corpus.splitlines()
        if line.strip()
    ]
    if len(documents) > 5000:
        raise HTTPException(422, "Provide at most 5000 documents for lookup")
    if len({d["doc_id"] for d in documents}) != len(documents):
        raise HTTPException(422, "Corpus document ids must be unique")
    request = PipelineRequest(
        query_id="connection-check",
        text=body.text,
        vector=body.vector,
        k=body.k,
        filters=body.filters,
        context=body.context,
    )
    payload: dict[str, Any] = {
        "documents": documents,
        "forbidden": body.forbidden,
        "no_answer": body.no_answer,
        "root": str(workspace.manifest_root),
    }
    if body.pipeline == "builtin":
        if not documents:
            raise HTTPException(422, "Provide a JSONL corpus to test the built-in pipeline")
        payload.update(builtin=True, adapter=None, corpus_version_known=True)
    else:
        candidate = workspace.registry.get(body.pipeline)
        if candidate is None:
            raise HTTPException(422, "Select a pipeline registered at server startup")
        if candidate.adapter.kind == "http" and not body.confirm_remote:
            raise HTTPException(422, "Confirm query transfer to the configured endpoint")
        request.params = dict(candidate.params)
        known = bool(
            (workspace.manifest and workspace.manifest.corpus_version)
            or candidate.adapter.kind == "hybrid"
        )
        payload.update(
            builtin=False,
            adapter=candidate.adapter.model_dump(),
            corpus_version_known=known,
        )
    payload["request"] = request.model_dump()
    return payload


def diagnostic_router(workspace: GuidedWorkspace) -> APIRouter:
    """Inherit the guided router's token/origin checks; writes no run artifacts."""
    router = APIRouter()
    runner = DiagnosticRunner()

    @router.post("/diagnostics")
    def diagnose(body: DiagnosticIn) -> dict[str, Any]:
        payload = diagnostic_payload(workspace, body)
        report = runner.run(payload, body.timeout_s)
        return {"pipeline": body.pipeline, "report": report.model_dump()}

    @router.post("/diagnostics/cancel")
    def cancel() -> dict[str, bool]:
        return {"cancel_requested": runner.cancel()}

    return router
