"""Bearer authentication middleware."""
from __future__ import annotations

import hmac
import os

from fastapi import Request
from fastapi.responses import JSONResponse


def _get_router_keys() -> list[str]:
    """Get valid router API keys from env."""
    raw = os.environ.get("ROUTER_API_KEY", "")
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _get_admin_key() -> str | None:
    return os.environ.get("ADMIN_API_KEY") or None


def _constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def extract_bearer_token(auth_header: str | None) -> str:
    """Extract token from Authorization header (RFC 6750 case-insensitive Bearer)."""
    if not auth_header:
        return ""
    stripped = auth_header.strip()
    if not stripped:
        return ""
    parts = stripped.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    if parts[0].lower() == "bearer":
        return ""
    return stripped


async def auth_middleware(request: Request, call_next):
    """Authenticate requests based on config."""
    return await call_next(request)


def check_router_auth(auth_header: str | None, config) -> bool:
    """Check if the request is authorized for /v1/* endpoints."""
    if not config.auth.enabled:
        return True

    keys = _get_router_keys()
    if not keys:
        return False

    token = extract_bearer_token(auth_header)
    if not token:
        return False

    return any(_constant_time_compare(token, key) for key in keys)


def check_admin_auth(auth_header: str | None, config) -> bool:
    """Check if the request is authorized for /admin/* endpoints."""
    admin_key = _get_admin_key()
    if not admin_key:
        return False

    token = extract_bearer_token(auth_header)
    if not token:
        return False

    return _constant_time_compare(token, admin_key)


def unauthorized_response(message: str = "Invalid API key") -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": message,
                "type": "invalid_api_key",
                "param": None,
                "code": "invalid_api_key",
            }
        },
    )
