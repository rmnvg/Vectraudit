"""API routers, mounted under ``/api/v1``."""

from __future__ import annotations

from fastapi import FastAPI


def register(app: FastAPI) -> None:
    """Mount every router on the app."""
