"""Client for the hosted control plane, used by the CLI and runners (stdlib only).

Configured by ``VECTRAUDIT_CLOUD_URL``, ``VECTRAUDIT_CLOUD_TOKEN`` (a runner token in CI)
and a project as ``workspace/project`` (``--project`` or ``VECTRAUDIT_CLOUD_PROJECT``).
Tokens are only ever sent over HTTPS, except to a local development server.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

from vectraudit import __version__
from vectraudit.integrations.sharing import project_run

TIMEOUT_S = 30
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})
Send = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, bytes]]


class CloudClientError(RuntimeError):
    """A failed call, with the server's structured error when it sent one."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.hint = hint


@dataclass(frozen=True)
class CloudTarget:
    url: str
    token: str
    workspace: str
    project: str

    @classmethod
    def from_env(
        cls, project: str | None = None, env: Mapping[str, str] | None = None
    ) -> CloudTarget:
        env = os.environ if env is None else env
        url = (env.get("VECTRAUDIT_CLOUD_URL") or "").rstrip("/")
        token = env.get("VECTRAUDIT_CLOUD_TOKEN") or ""
        spec = project or env.get("VECTRAUDIT_CLOUD_PROJECT") or ""
        if not url:
            raise CloudClientError("VECTRAUDIT_CLOUD_URL is not set.")
        if not token:
            raise CloudClientError(
                "VECTRAUDIT_CLOUD_TOKEN is not set.",
                hint="Create a runner token for the project and store it as a CI secret.",
            )
        workspace, sep, proj = spec.partition("/")
        if not sep or not workspace or not proj or "/" in proj:
            raise CloudClientError(
                "Name the cloud project as workspace/project (--project or "
                "VECTRAUDIT_CLOUD_PROJECT)."
            )
        parts = urlsplit(url)
        if parts.scheme != "https" and parts.hostname not in LOCAL_HOSTS:
            raise CloudClientError(
                f"Refusing to send a token over {parts.scheme}:// to {parts.hostname}.",
                hint="Use the https:// address of the service.",
            )
        return cls(url, token, workspace, proj)


def urllib_send(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()
    except (urllib.error.URLError, TimeoutError) as e:
        raise CloudClientError(f"Cannot reach the Vectraudit service: {e}") from e


class CloudClient:
    def __init__(self, target: CloudTarget, send: Send | None = None) -> None:
        self.target = target
        self._send = send or (lambda *args: urllib_send(*args))

    @property
    def project_path(self) -> str:
        t = self.target
        return f"/api/v1/workspaces/{quote(t.workspace)}/projects/{quote(t.project)}"

    def _call(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        all_headers = {
            "Authorization": f"Bearer {self.target.token}",
            "Accept": "application/json",
            "User-Agent": f"vectraudit/{__version__}",
            **({"Content-Type": "application/json"} if body is not None else {}),
            **(headers or {}),
        }
        status, raw = self._send(method, f"{self.target.url}{path}", all_headers, body)
        try:
            data = json.loads(raw) if raw else None
        except ValueError:
            data = None
        if status >= 400:
            err = data if isinstance(data, dict) else {}
            raise CloudClientError(
                str(err.get("message") or f"HTTP {status} from the Vectraudit service."),
                status=status,
                code=err.get("error_code"),
                hint=err.get("hint"),
            )
        return data

    def upload_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Upload an explicitly projected workbench revision, with transfer integrity."""
        from vectraudit.integrations.shared_records import encoded, validate_record

        body = encoded(validate_record(record))
        result: dict[str, Any] = self._call(
            "POST",
            f"{self.project_path}/records",
            body,
            {"X-Vectraudit-Digest": "sha256=" + hashlib.sha256(body).hexdigest()},
        )
        return result

    def upload_run(
        self, record: Mapping[str, Any], *, share_query_text: bool = False
    ) -> dict[str, Any]:
        """Upload the allowlisted projection; raw local artifacts stay unchanged."""
        try:
            projected = project_run(record, share_query_text=share_query_text)
        except ValueError as exc:
            raise CloudClientError("Run cannot be projected for sharing.", hint=str(exc)) from exc
        body = json.dumps(
            projected, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        ).encode()
        digest = "sha256=" + hashlib.sha256(body).hexdigest()
        result: dict[str, Any] = self._call(
            "POST", f"{self.project_path}/runs", body, {"X-Vectraudit-Digest": digest}
        )
        return result

    def get_run_file(self, run_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._call("GET", f"{self.project_path}/runs/{quote(run_id)}/file")
        return result

    def get_policy(self) -> dict[str, Any]:
        """``{"policy": <raw or None>, "effective": {...}, "is_default": bool}``."""
        result: dict[str, Any] = self._call("GET", f"{self.project_path}/policy")
        return result

    def get_baseline_file(self, name: str) -> dict[str, Any]:
        result: dict[str, Any] = self._call(
            "GET", f"{self.project_path}/baselines/{quote(name)}/file"
        )
        return result

    def list_baselines(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._call("GET", f"{self.project_path}/baselines")
        return result

    def pin_baseline(self, name: str, run_id: str, note: str = "") -> dict[str, Any]:
        body = json.dumps({"run_id": run_id, "note": note}).encode()
        result: dict[str, Any] = self._call(
            "PUT", f"{self.project_path}/baselines/{quote(name)}", body
        )
        return result

    def create_check(
        self, candidate_run_id: str, baseline: str = "main", *, trigger: str = "release"
    ) -> dict[str, Any]:
        """Record a hosted check; ``trigger="monitor"`` marks a scheduled monitor run."""
        payload = {"candidate_run_id": candidate_run_id, "baseline": baseline}
        if trigger != "release":
            payload["trigger"] = trigger
        body = json.dumps(payload).encode()
        result: dict[str, Any] = self._call("POST", f"{self.project_path}/checks", body)
        return result

    def decision_summary(self, check_id: str) -> dict[str, Any]:
        """A release decision in plain language (``headline``, ``markdown``, …)."""
        result: dict[str, Any] = self._call(
            "GET", f"{self.project_path}/decisions/{quote(check_id)}/summary"
        )
        return result

    def list_exceptions(self, *, active: bool = False) -> list[dict[str, Any]]:
        """Audited exceptions, newest first; ``active`` ones apply to new checks."""
        suffix = "?active=true" if active else ""
        result: list[dict[str, Any]] = self._call("GET", f"{self.project_path}/exceptions{suffix}")
        return result

    def grant_exception(
        self,
        check_id: str,
        *,
        rule: str,
        reason: str,
        owner: str,
        expires_at: str,
        max_checks: int | None = None,
    ) -> dict[str, Any]:
        """Accept one failing rule of a hosted check (engineer/owner as the policy says)."""
        body = json.dumps(
            {
                "rule": rule,
                "reason": reason,
                "owner": owner,
                "expires_at": expires_at,
                "max_checks": max_checks,
            }
        ).encode()
        result: dict[str, Any] = self._call(
            "POST", f"{self.project_path}/checks/{quote(check_id, safe='')}/exceptions", body
        )
        return result

    def revoke_exception(self, exception_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._call(
            "POST", f"{self.project_path}/exceptions/{quote(exception_id, safe='')}/revoke", b""
        )
        return result

    def list_runs(self, limit: int = 20, **filters: str) -> dict[str, Any]:
        """``{"runs": [...], "next_cursor": ...}``, newest first."""
        from urllib.parse import urlencode

        query = urlencode({"limit": limit, **{k: v for k, v in filters.items() if v}})
        result: dict[str, Any] = self._call("GET", f"{self.project_path}/runs?{query}")
        return result
