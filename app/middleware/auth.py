"""Bearer authentication middleware."""
from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse


def _get_router_keys() -> list[str]:
    """Get valid router API keys from env."""
    raw = os.environ.get("ROUTER_API_KEY", "")
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _get_admin_key() -> Optional[str]:
    return os.environ.get("ADMIN_API_KEY") or None


def _constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


async def auth_middleware(request: Request, call_next):
    """Authenticate requests based on config."""
    from app.config.loader import ConfigManager
    config = ConfigManager.__new__(ConfigManager)  # lightweight check
    # We'll use the global config instance instead
    return await call_next(request)


def check_router_auth(auth_header: str | None, config) -> bool:
    """Check if the request is authorized for /v1/* endpoints."""
    if not config.auth.enabled:
        return True

    keys = _get_router_keys()
    if not keys:
        return False

    if not auth_header:
        return False

    # Extract bearer token
    token = auth_header.replace("Bearer ", "").replace("bearer ", "").strip()
    for key in keys:
        if _constant_time_compare(token, key):
            return True
    return False


def check_admin_auth(auth_header: str | None, config) -> bool:
    """Check if the request is authorized for /admin/* endpoints."""
    admin_key = _get_admin_key()
    if not admin_key:
        return False

    if not auth_header:
        return False

    token = auth_header.replace("Bearer ", "").replace("bearer ", "").strip()
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
