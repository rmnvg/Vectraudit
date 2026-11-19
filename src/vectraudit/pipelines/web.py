"""Loopback workbench with protected guided evaluation and artifact browsing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from vectraudit.labelling.storage import TaskBusy
from vectraudit.pipelines.guided import MAX_BODY, MAX_RUN_BYTES, GuidedWorkspace, guided_router
from vectraudit.server.app import STATIC_DIR, _mount_frontend

MAX_ATLAS_BYTES = 64 * 1024 * 1024


def create_workbench(
    directory: Path,
    manifest: Path | None = None,
    policy: Path | None = None,
    lineage: Path | None = None,
    lineage_ids: str = "chunk",
    changes: Path | None = None,
) -> FastAPI:
    """Serve a bounded artifact directory and the shared product UI."""
    root = directory.resolve()
    workspace = GuidedWorkspace(root, manifest, policy, lineage, lineage_ids, changes)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            workspace.jobs.close()

    app = FastAPI(title="Vectraudit workbench", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.guided = workspace
    app.include_router(guided_router(workspace))

    @app.middleware("http")
    async def bound_body(request: Request, call_next: Any) -> Response:
        if request.url.path.startswith("/api/guided"):
            if request.method not in {"GET", "HEAD"}:
                chunks, size = [], 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BODY:
                        return JSONResponse({"detail": "Request exceeds 5 MiB"}, status_code=413)
                    chunks.append(chunk)
                request._body = b"".join(chunks)
            response: Response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        other_response: Response = await call_next(request)
        return other_response

    @app.exception_handler(TaskBusy)
    async def busy(_: Request, exc: TaskBusy) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(_: Request, exc: ValueError) -> JSONResponse:
        # Validation errors name the field, not the full submitted user payload.
        from pydantic import ValidationError

        message = "Invalid workbench input" if isinstance(exc, ValidationError) else str(exc)
        return JSONResponse({"detail": message}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {"detail": "Check the guided workflow fields and limits"}, status_code=422
        )

    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )

    @app.get("/api/workbench")
    def artifacts() -> dict[str, Any]:
        entries = []
        for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[
            :200
        ]:
            size = path.stat().st_size
            large_run = size > MAX_RUN_BYTES and path.name.endswith(".run.json")
            if path.is_symlink() or (size > MAX_RUN_BYTES and not large_run):
                continue
            # A run too large to open is still listed: opening it shows its validated summary.
            entries.append({"name": path.name, "bytes": size, "too_large": large_run})
        return {"local": True, "artifacts": entries}

    @app.get("/api/workbench/{name}")
    def artifact(name: str) -> dict[str, Any]:
        path = root / name
        if (
            path.is_symlink()
            or path.resolve().parent != root
            or path.suffix != ".json"
            or not path.is_file()
        ):
            raise HTTPException(404, "Artifact not found")
        if path.stat().st_size > MAX_RUN_BYTES:
            if not path.name.endswith(".run.json"):
                raise HTTPException(413, "Artifact is larger than 64 MiB")
            from vectraudit.evaluation.runfile import RunFileError, run_header_report

            try:
                return run_header_report(path)
            except RunFileError as exc:
                raise HTTPException(422, "Invalid workbench artifact") from exc
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or not str(result.get("format", "")).startswith(
                "vectraudit."
            ):
                raise ValueError("not an artifact")
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Invalid workbench artifact") from exc
        return result

    @app.get("/api/workbench/atlas/{name}")
    def atlas_points(name: str) -> Response:
        """The binary points of ``atlas-<name>.bin`` (layout: ``vectraudit.atlas`` sidecar)."""
        from vectraudit.pipelines.atlas_build import NAME

        path = root / f"atlas-{name}.bin"
        if (
            not NAME.match(name)
            or path.is_symlink()
            or path.resolve().parent != root
            or not path.is_file()
        ):
            raise HTTPException(404, "Atlas not found")
        if path.stat().st_size > MAX_ATLAS_BYTES:
            raise HTTPException(413, "Atlas is larger than 64 MiB")
        return Response(
            path.read_bytes(),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    # Let the common shell load without an index; no index APIs are enabled.
    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {"workbench": True, "supported": False, "inputs": {"embedder_status": None}}

    _mount_frontend(app, STATIC_DIR)
    return app
