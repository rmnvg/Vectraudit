"""Browser sessions for the dashboard, and CSRF tokens bound to them.

The cookie carries a random session id (HttpOnly, SameSite=Lax, Secure over HTTPS); only
its SHA-256 is stored, so a database leak can't be replayed as sessions. Every
state-changing dashboard form carries a CSRF token derived from the session.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from vectraudit.cloud.auth import Principal
from vectraudit.cloud.models import Invite, User, WebSession, utcnow

SESSION_COOKIE = "va_session"
SESSION_DAYS = 14


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_session(session: Session, user: User, method: str) -> str:
    """Start a session; returns the raw id for the cookie (never stored)."""
    raw = secrets.token_urlsafe(32)
    session.add(
        WebSession(
            session_hash=_hash(raw),
            user_id=user.id,
            method=method,
            expires_at=utcnow() + timedelta(days=SESSION_DAYS),
        )
    )
    return raw


def resolve_session(session: Session, raw: str | None) -> tuple[WebSession, User] | None:
    if not raw:
        return None
    row = session.scalars(select(WebSession).where(WebSession.session_hash == _hash(raw))).first()
    if row is None or row.revoked_at is not None or row.expires_at <= utcnow():
        return None
    user = session.get(User, row.user_id)
    if user is None or user.disabled:
        return None
    return row, user


def revoke_session(session: Session, raw: str | None) -> None:
    found = resolve_session(session, raw)
    if found is not None:
        found[0].revoked_at = utcnow()


def session_principal(user: User) -> Principal:
    return Principal(
        kind="user", token_id=None, token_prefix=None, user_id=user.id, is_admin=user.is_admin
    )


def csrf_token(secret: str, raw_session: str) -> str:
    return hmac.new(secret.encode(), f"csrf:{raw_session}".encode(), hashlib.sha256).hexdigest()


def csrf_ok(secret: str, raw_session: str | None, submitted: str | None) -> bool:
    if not raw_session or not submitted:
        return False
    return hmac.compare_digest(csrf_token(secret, raw_session), submitted)


INVITE_HOURS = 72


def create_invite(session: Session, workspace_id: str, user: User, actor: str) -> str:
    """A single-use sign-in link token for ``user``; returns the raw token (shown once)."""
    raw = secrets.token_urlsafe(32)
    session.add(
        Invite(
            token_hash=_hash(raw),
            workspace_id=workspace_id,
            user_id=user.id,
            created_by=actor,
            expires_at=utcnow() + timedelta(hours=INVITE_HOURS),
        )
    )
    return raw


def redeem_invite(session: Session, raw: str) -> tuple[User, str] | None:
    """Use an invite once: returns (user, workspace id), or ``None`` if invalid."""
    row = session.scalars(select(Invite).where(Invite.token_hash == _hash(raw))).first()
    if row is None or row.used_at is not None or row.expires_at <= utcnow():
        return None
    user = session.get(User, row.user_id)
    if user is None or user.disabled:
        return None
    row.used_at = utcnow()
    return user, row.workspace_id
