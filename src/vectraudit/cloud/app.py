"""FastAPI app factory for the hosted control plane."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from vectraudit import __version__
from vectraudit.cloud.db import make_engine, session_factory
from vectraudit.cloud.errors import CloudError
from vectraudit.cloud.migrate import current_revision, head_revision
from vectraudit.cloud.settings import Settings

log = logging.getLogger("vectraudit.cloud")


def _error(status: int, code: str, message: str, hint: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error_code": code, "message": message, "hint": hint}
    )


def create_app(settings: Settings, engine: Engine | None = None) -> FastAPI:
    """Build the API. ``engine`` defaults to one for ``settings.database_url``."""
    engine = engine or make_engine(settings.database_url)
    factory = session_factory(engine)
    app = FastAPI(
        title="Vectraudit Cloud",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.sessions = factory

    @app.middleware("http")
    async def limit_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > settings.max_upload_bytes:
            return _error(
                413,
                "payload_too_large",
                f"Request body exceeds {settings.max_upload_bytes:,} bytes.",
                "Run files above the limit usually mean too many queries for one run.",
            )
        return await call_next(request)

    @app.exception_handler(CloudError)
    async def cloud_error(_: Request, e: CloudError) -> JSONResponse:
        return _error(e.status, e.code, e.message, e.hint)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, e: RequestValidationError) -> JSONResponse:
        first = e.errors()[0] if e.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()))
        return _error(422, "invalid_request", f"{where}: {first.get('msg', 'invalid value')}")

    @app.exception_handler(Exception)
    async def unexpected(_: Request, e: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _error(500, "internal_error", "Something went wrong on our side.")

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> JSONResponse:
        """Readiness: the database answers and its schema is at the latest migration."""
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            current = current_revision(engine)
        except Exception as e:  # any database problem means "not ready"
            return _error(503, "database_unavailable", f"Database check failed: {e}")
        head = head_revision()
        if current != head:
            return _error(
                503,
                "migrations_pending",
                f"Database schema is at {current}, the code expects {head}.",
                "Run `vectraudit-cloud migrate`.",
            )
        return JSONResponse({"status": "ready", "schema": current})

    from vectraudit.cloud.routes import register

    register(app)
    return app


def db_session(request: Request) -> Iterator[Session]:
    """Request-scoped session: commit on success, roll back on error."""
    factory: sessionmaker[Session] = request.app.state.sessions
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
