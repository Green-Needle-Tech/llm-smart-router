"""Error handling middleware: OpenAI-compatible error envelopes."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


async def error_middleware(request: Request, call_next):
    """Catch exceptions and return OpenAI-format errors."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        status = 500
        error_type = "internal_error"
        code = "router_internal_error"

        # Map common exceptions
        from fastapi import HTTPException
        if isinstance(e, HTTPException):
            status = e.status_code
            error_type = "invalid_request_error" if status == 400 else "internal_error"
            code = f"router_{status}"

        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": str(e),
                    "type": error_type,
                    "param": None,
                    "code": code,
                }
            },
        )
