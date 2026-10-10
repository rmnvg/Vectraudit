"""Service configuration, read from ``VECTRAUDIT_CLOUD_*`` environment variables."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

MIN_SECRET_LENGTH = 32


class SettingsError(ValueError):
    """The environment does not describe a valid configuration."""


@dataclass(frozen=True)
class Settings:
    database_url: str
    secret_key: str
    """Signs share links and sessions. At least 32 characters; rotate to invalidate them."""
    public_url: str = "http://localhost:8800"
    max_upload_bytes: int = 20 * 1024 * 1024
    max_queries_per_run: int = 10_000
    share_links: bool = False
    """Signed public report links are off unless explicitly enabled."""
    environment: str = "production"
    sso: dict[str, str] | None = None
    github_client_id: str | None = None
    github_client_secret: str | None = None
    """GitHub OAuth app for dashboard sign-in; without it, members sign in with a token."""
    webhook_allow_private: bool = False
    """Development only: allow http webhooks and private/loopback addresses."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            return env.get(f"VECTRAUDIT_CLOUD_{name}")

        url = get("DATABASE_URL")
        if not url:
            raise SettingsError("VECTRAUDIT_CLOUD_DATABASE_URL is not set.")
        secret = get("SECRET_KEY") or ""
        if len(secret) < MIN_SECRET_LENGTH:
            raise SettingsError(
                f"VECTRAUDIT_CLOUD_SECRET_KEY must be at least {MIN_SECRET_LENGTH} characters. "
                'Generate one: python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        try:
            max_upload = int(get("MAX_UPLOAD_BYTES") or cls.max_upload_bytes)
        except ValueError as e:
            raise SettingsError("VECTRAUDIT_CLOUD_MAX_UPLOAD_BYTES must be an integer.") from e
        try:
            max_queries = int(get("MAX_QUERIES_PER_RUN") or cls.max_queries_per_run)
        except ValueError as e:
            raise SettingsError("VECTRAUDIT_CLOUD_MAX_QUERIES_PER_RUN must be an integer.") from e
        sso = None
        if get("SSO_JSON"):
            try:
                sso = json.loads(get("SSO_JSON") or "")
                if not isinstance(sso, dict):
                    raise ValueError("expected object")
            except ValueError as e:
                raise SettingsError("VECTRAUDIT_CLOUD_SSO_JSON must be a JSON object.") from e
        return cls(
            sso=sso,
            database_url=normalize_database_url(url),
            max_queries_per_run=max_queries,
            secret_key=secret,
            public_url=(get("PUBLIC_URL") or cls.public_url).rstrip("/"),
            max_upload_bytes=max_upload,
            share_links=(get("SHARE_LINKS") or "false").lower() in ("1", "true", "yes"),
            environment=get("ENVIRONMENT") or cls.environment,
            github_client_id=get("GITHUB_CLIENT_ID") or None,
            github_client_secret=get("GITHUB_CLIENT_SECRET") or None,
            webhook_allow_private=(get("WEBHOOK_ALLOW_PRIVATE") or "false").lower()
            in ("1", "true", "yes"),
        )


def normalize_database_url(url: str) -> str:
    """Use psycopg 3 for plain ``postgres://`` / ``postgresql://`` URLs."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url
