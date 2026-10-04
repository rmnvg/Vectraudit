"""API tokens: random secrets with a visible prefix; only their SHA-256 is stored."""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass

KIND_PREFIX = {"personal": "va_pat_", "runner": "va_run_"}
TOKEN_RE = re.compile(r"^va_(pat|run)_[A-Za-z0-9]{8}_[A-Za-z0-9_-]{32,}$")
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"


@dataclass(frozen=True)
class NewToken:
    token: str
    """The secret: shown to the user once, never stored."""
    prefix: str
    """Safe to display and log (e.g. ``va_run_Ab3dEf9h``); cannot authenticate."""
    token_hash: str


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def generate_token(kind: str) -> NewToken:
    """A new token of ``kind`` (``personal`` or ``runner``)."""
    prefix = KIND_PREFIX[kind] + "".join(secrets.choice(_ALPHABET) for _ in range(8))
    token = f"{prefix}_{secrets.token_urlsafe(32)}"
    return NewToken(token, prefix, hash_token(token))


def looks_like_token(value: str) -> bool:
    return bool(TOKEN_RE.fullmatch(value))
