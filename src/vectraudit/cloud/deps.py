"""FastAPI dependencies: one database session per request, and the calling principal."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session, sessionmaker

from vectraudit.cloud.auth import Principal, authenticate
from vectraudit.cloud.settings import Settings


def db_session(request: Request) -> Iterator[Session]:
    """Request-scoped session: commit on success, roll back on error.

    FastAPI caches it per request, so the principal lookup and the route share it.
    """
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


SessionDep = Annotated[Session, Depends(db_session)]


def get_principal(request: Request, session: SessionDep) -> Principal:
    return authenticate(session, request.headers.get("authorization"))


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


PrincipalDep = Annotated[Principal, Depends(get_principal)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
