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
    """Bearer token; or, for reads only, a dashboard session cookie.

    Cookies never authorize API writes: those need an explicit token, which rules out
    cross-site request forgery against the JSON API.
    """
    header = request.headers.get("authorization")
    if header is None and request.method in ("GET", "HEAD"):
        from vectraudit.cloud.web_auth import SESSION_COOKIE, resolve_session, session_principal

        found = resolve_session(session, request.cookies.get(SESSION_COOKIE))
        if found is not None:
            return session_principal(found[1])
    return authenticate(session, header)


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


PrincipalDep = Annotated[Principal, Depends(get_principal)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
