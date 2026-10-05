"""Read-only loopback workbench; execution stays in trusted CLI/Python processes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from vectraudit.server.app import STATIC_DIR, _mount_frontend


def create_workbench(directory: Path) -> FastAPI:
    """Serve a bounded artifact directory and the shared product UI."""
    root = directory.resolve()
    app = FastAPI(title="Vectraudit workbench", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )

    @app.get("/api/workbench")
    def artifacts() -> dict[str, Any]:
        entries = []
        for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[
            :200
        ]:
            if path.is_symlink() or path.stat().st_size > 20 * 1024 * 1024:
                continue
            entries.append({"name": path.name, "bytes": path.stat().st_size})
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
        if path.stat().st_size > 20 * 1024 * 1024:
            raise HTTPException(413, "Artifact is larger than 20 MiB")
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or not str(result.get("format", "")).startswith(
                "vectraudit."
            ):
                raise ValueError("not an artifact")
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Invalid workbench artifact") from exc
        return result

    # Let the common shell load without an index; no index APIs are enabled.
    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {"workbench": True, "supported": False, "inputs": {"embedder_status": None}}

    _mount_frontend(app, STATIC_DIR)
    return app
