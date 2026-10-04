"""API routers, mounted under ``/api/v1``."""

from __future__ import annotations

from fastapi import FastAPI


def register(app: FastAPI) -> None:
    """Mount every router on the app."""
    from vectraudit.cloud.routes import accounts, checks, history, runs

    app.include_router(accounts.router)
    app.include_router(runs.router)
    app.include_router(checks.router)
    app.include_router(history.router)

    from fastapi import Request
    from fastapi.responses import RedirectResponse, Response

    from vectraudit.cloud import web

    app.include_router(web.router)

    @app.exception_handler(web._Redirect)
    async def redirect(_: Request, exc: web._Redirect) -> Response:
        return RedirectResponse(exc.location, status_code=303)
