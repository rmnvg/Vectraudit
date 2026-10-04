"""FastAPI dependencies: one database session per request, and the calling principal."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session, sessionmaker

from vectraudit.cloud.auth import Principal, authenticate
from vectraudit.cloud.errors import CloudError, rate_limited
from vectraudit.cloud.security import looks_like_token
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
log = logging.getLogger("vectraudit.cloud.security")


def client_key(request: Request) -> str:
    """The client address (behind a trusted proxy, uvicorn fills it from X-Forwarded-For)."""
    return request.client.host if request.client else "unknown"


def guard_failures(request: Request) -> None:
    """Refuse callers that have failed authentication too often."""
    retry = request.app.state.auth_failures.retry_after(client_key(request))
    if retry is not None:
        raise rate_limited(retry)


def record_failure(request: Request, what: str, token_prefix: str | None = None) -> None:
    request.app.state.auth_failures.failure(client_key(request))
    log.warning(
        "auth_failed",
        extra={
            "fields": {
                "what": what,
                "client": client_key(request),
                "token_prefix": token_prefix,
            }
        },
    )


def get_principal(request: Request, session: SessionDep) -> Principal:
    """Bearer token; or, for reads only, a dashboard session cookie.

    Cookies never authorize API writes: those need an explicit token, which rules out
    cross-site request forgery against the JSON API. Repeated failures from one address
    are rate limited.
    """
    guard_failures(request)
    header = request.headers.get("authorization")
    if header is None and request.method in ("GET", "HEAD"):
        from vectraudit.cloud.web_auth import SESSION_COOKIE, resolve_session, session_principal

        found = resolve_session(session, request.cookies.get(SESSION_COOKIE))
        if found is not None:
            return session_principal(found[1])
    try:
        return authenticate(session, header)
    except CloudError:
        if header is not None:  # a missing header is not a guess
            token = header.removeprefix("Bearer ").strip()
            prefix = token[:15] if looks_like_token(token) else None
            record_failure(request, "bearer", prefix)
        raise


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


PrincipalDep = Annotated[Principal, Depends(get_principal)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
