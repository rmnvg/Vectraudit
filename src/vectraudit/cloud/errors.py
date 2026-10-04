"""Structured API errors: ``{error_code, message, hint}`` with an HTTP status."""

from __future__ import annotations


class CloudError(Exception):
    """An error the API reports to the client as structured JSON."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        hint: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.hint = hint
        self.headers = headers or {}


def not_found(what: str) -> CloudError:
    """404 for objects that don't exist *or* belong to another workspace (no oracle)."""
    return CloudError(404, "not_found", f"{what} not found.")


def forbidden(message: str, hint: str | None = None) -> CloudError:
    return CloudError(403, "forbidden", message, hint)


def unauthorized(message: str = "Missing or invalid credentials.") -> CloudError:
    return CloudError(
        401, "unauthorized", message, "Send `Authorization: Bearer <token>` with a valid token."
    )


def rate_limited(retry_after: int) -> CloudError:
    return CloudError(
        429,
        "rate_limited",
        "Too many failed attempts from this address.",
        f"Wait {retry_after} seconds before trying again.",
        headers={"Retry-After": str(retry_after)},
    )
