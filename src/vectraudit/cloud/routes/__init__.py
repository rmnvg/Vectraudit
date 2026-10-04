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
