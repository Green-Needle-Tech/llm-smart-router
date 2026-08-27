"""Request ID generation and propagation."""
from __future__ import annotations

import uuid

from fastapi import Request


async def request_id_middleware(request: Request, call_next):
    """Generate or propagate X-Request-Id."""
    request_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response
