"""Enterprise OAuth code flow with PKCE and verified userinfo for invited accounts.

The configured provider is a deployment trust boundary. Tokens stay server-side;
existing workspace membership and disabled-user checks still determine access.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import urllib.request
from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select

from vectraudit.cloud.deps import SessionDep, SettingsDep, guard_failures, record_failure
from vectraudit.cloud.errors import CloudError
from vectraudit.cloud.models import User
from vectraudit.cloud.settings import Settings
from vectraudit.cloud.web import _safe_next, _secure, _signed_in
from vectraudit.cloud.web_auth import create_session

router = APIRouter(include_in_schema=False)
COOKIE = "va_sso_state"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _configuration(settings: Settings) -> dict[str, str]:
    config = settings.sso or {}
    required = {"client_id", "client_secret", "authorize_url", "token_url", "userinfo_url"}
    if set(config) != required or any(not isinstance(v, str) or not v for v in config.values()):
        raise CloudError(404, "sso_unavailable", "Single sign-on is not configured.")
    for key in ("authorize_url", "token_url", "userinfo_url"):
        url = urlsplit(config[key])
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise CloudError(503, "sso_configuration", "SSO endpoints must be fixed HTTPS URLs.")
    if not settings.public_url.startswith("https://"):
        raise CloudError(503, "sso_configuration", "SSO requires an HTTPS public URL.")
    return config


def _seal(payload: dict[str, Any], secret: str) -> str:
    raw = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signature = hmac.new(secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return raw + "." + signature


def _unseal(cookie: str, secret: str) -> dict[str, Any]:
    raw, signature = cookie.rsplit(".", 1)
    expected = hmac.new(secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("invalid state")
    payload: dict[str, Any] = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
    if not 0 <= time.time() - payload["issued"] <= 600:
        raise ValueError("expired state")
    return payload


@router.get("/auth/sso")
def start(settings: SettingsDep, next: str = "/app") -> Response:
    config = _configuration(settings)
    state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
    query = urlencode(
        {
            "response_type": "code",
            "client_id": config["client_id"],
            "redirect_uri": settings.public_url + "/auth/sso/callback",
            "scope": "openid email profile",
            "state": state,
            "code_challenge": _b64(hashlib.sha256(verifier.encode()).digest()),
            "code_challenge_method": "S256",
        }
    )
    response = RedirectResponse(config["authorize_url"] + "?" + query, status_code=303)
    response.set_cookie(
        COOKIE,
        _seal(
            {"state": state, "verifier": verifier, "next": _safe_next(next), "issued": time.time()},
            settings.secret_key,
        ),
        max_age=600,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/auth/sso",
    )
    return response


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _request(url: str, headers: dict[str, str], data: bytes | None = None) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(
        urllib.request.Request(url, data=data, headers=headers), timeout=15
    ) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("oversized provider response")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("invalid provider response")
    return result


@router.get("/auth/sso/callback")
def callback(
    request: Request, session: SessionDep, settings: SettingsDep, code: str = "", state: str = ""
) -> Response:
    guard_failures(request)
    config = _configuration(settings)
    try:
        payload = _unseal(request.cookies.get(COOKIE, ""), settings.secret_key)
        if not code or not state or not secrets.compare_digest(state, payload["state"]):
            raise ValueError("state mismatch")
        token = _request(
            config["token_url"],
            {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            urlencode(
                {
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": config["client_id"],
                    "client_secret": config["client_secret"],
                    "code_verifier": payload["verifier"],
                    "redirect_uri": settings.public_url + "/auth/sso/callback",
                }
            ).encode(),
        )
        access_token = token.get("access_token")
        if (
            not isinstance(access_token, str)
            or not access_token
            or not isinstance(token.get("token_type"), str)
            or token["token_type"].lower() != "bearer"
        ):
            raise ValueError("missing bearer token")
        # Authenticate using userinfo fetched directly from the fixed trusted provider.
        # ID tokens are deliberately not parsed or used as identity evidence.
        userinfo = _request(config["userinfo_url"], {"Authorization": "Bearer " + access_token})
        if userinfo.get("email_verified") is not True or not isinstance(userinfo.get("sub"), str):
            raise ValueError("unverified identity")
        email = userinfo.get("email")
        if not isinstance(email, str) or not email.strip():
            raise ValueError("missing email")
        user = session.scalars(select(User).where(User.email == email.strip().lower())).first()
        if user is None or user.disabled:
            raise ValueError("not an invited active user")
    except (ValueError, KeyError, TypeError, OSError):
        record_failure(request, "sso")
        response: Response = RedirectResponse(
            "/login?error=Enterprise%20sign-in%20failed", status_code=303
        )
    else:
        raw_session = create_session(session, user, "sso")
        response = _signed_in(
            RedirectResponse(_safe_next(payload["next"]), status_code=303), raw_session, settings
        )
    response.delete_cookie(COOKIE, path="/auth/sso", secure=_secure(settings), httponly=True)
    return response
