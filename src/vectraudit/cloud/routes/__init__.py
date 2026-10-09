"""API routers, mounted under ``/api/v1``."""

from __future__ import annotations

from fastapi import FastAPI


def register(app: FastAPI) -> None:
    """Mount every router on the app."""
    from vectraudit.cloud.routes import accounts, checks, history, records, runs

    app.include_router(accounts.router)
    app.include_router(runs.router)
    app.include_router(checks.router)
    app.include_router(history.router)
    app.include_router(records.router)

    from fastapi import Request
    from fastapi.responses import RedirectResponse, Response

    from vectraudit.cloud import web

    app.include_router(web.router)
    from vectraudit.cloud.web_records import router as records_web

    app.include_router(records_web)
    from vectraudit.cloud.web_lead import router as lead_web

    app.include_router(lead_web)

    @app.exception_handler(web._Redirect)
    async def redirect(_: Request, exc: web._Redirect) -> Response:
        return RedirectResponse(exc.location, status_code=303)
