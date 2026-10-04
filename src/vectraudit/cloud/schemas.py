"""Request and response bodies (pydantic)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UserOut(BaseModel):
    id: str
    email: str
    name: str


class WorkspaceOut(BaseModel):
    slug: str
    name: str
    plan: str
    role: str | None = None
    """The caller's role (``runner`` for runner tokens)."""


class MeOut(BaseModel):
    user: UserOut
    workspaces: list[WorkspaceOut]


class WorkspaceIn(BaseModel):
    slug: str = Field(max_length=63)
    name: str = Field(min_length=1, max_length=200)
    owner_email: str = Field(max_length=320)


class MemberOut(BaseModel):
    user: UserOut
    role: str
    since: datetime


class MemberIn(BaseModel):
    email: str = Field(max_length=320)
    role: str


class RoleIn(BaseModel):
    role: str


class ProjectIn(BaseModel):
    slug: str = Field(max_length=63)
    name: str = Field(min_length=1, max_length=200)


class ProjectOut(BaseModel):
    slug: str
    name: str
    created_at: datetime


class TokenIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class RunnerTokenIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    scopes: list[str] | None = None


class TokenOut(BaseModel):
    id: str
    kind: str
    prefix: str
    name: str
    scopes: list[str]
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None


class NewTokenOut(TokenOut):
    token: str
    """The secret. Shown once; store it in your CI secrets."""


class AuditOut(BaseModel):
    actor: str
    action: str
    target_type: str
    target_id: str
    details: dict[str, Any]
    created_at: datetime
