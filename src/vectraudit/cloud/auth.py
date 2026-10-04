"""Who is calling: resolve a bearer token to a principal."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from vectraudit.cloud.errors import unauthorized
from vectraudit.cloud.models import ApiToken, User, utcnow
from vectraudit.cloud.security import hash_token, looks_like_token

RUNNER_SCOPES = ("runs:write", "runs:read", "baselines:read", "checks:write")
_TOUCH_EVERY = timedelta(minutes=1)


@dataclass(frozen=True)
class Principal:
    """An authenticated caller: a user (personal token / session) or a project runner."""

    kind: str
    """``user`` or ``runner``."""
    token_id: str | None
    token_prefix: str | None
    user_id: str | None = None
    is_admin: bool = False
    workspace_id: str | None = None
    """Runner tokens are bound to one workspace and project."""
    project_id: str | None = None
    scopes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def actor(self) -> str:
        """For audit logs: never a secret."""
        if self.kind == "user":
            return f"user:{self.user_id}"
        return f"token:{self.token_prefix}"


def authenticate(session: Session, authorization: str | None) -> Principal:
    """Resolve an ``Authorization: Bearer <token>`` header, or raise 401."""
    if not authorization or not authorization.startswith("Bearer "):
        raise unauthorized()
    token = authorization.removeprefix("Bearer ").strip()
    if not looks_like_token(token):
        raise unauthorized()
    row = session.scalars(select(ApiToken).where(ApiToken.token_hash == hash_token(token))).first()
    now = utcnow()
    if row is None or row.revoked_at is not None:
        raise unauthorized("This token is invalid or has been revoked.")
    if row.expires_at is not None and row.expires_at <= now:
        raise unauthorized("This token has expired.")
    if row.last_used_at is None or now - row.last_used_at > _TOUCH_EVERY:
        row.last_used_at = now
    if row.kind == "personal":
        user = session.get(User, row.user_id) if row.user_id else None
        if user is None or user.disabled:
            raise unauthorized("This account is disabled.")
        return Principal(
            kind="user",
            token_id=row.id,
            token_prefix=row.prefix,
            user_id=user.id,
            is_admin=user.is_admin,
        )
    return Principal(
        kind="runner",
        token_id=row.id,
        token_prefix=row.prefix,
        workspace_id=row.workspace_id,
        project_id=row.project_id,
        scopes=tuple(row.scopes or ()),
    )
