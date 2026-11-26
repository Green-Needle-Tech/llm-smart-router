"""Error handling middleware: OpenAI-compatible error envelopes."""
from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.telemetry.logging import get_logger

logger = get_logger("errors")


async def error_middleware(request: Request, call_next):
    """Catch exceptions and return OpenAI-format errors."""
    try:
        return await call_next(request)
    except HTTPException as e:
        status = e.status_code
        error_type = "invalid_request_error" if status == 400 else "internal_error"
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": str(e.detail),
                    "type": error_type,
                    "param": None,
                    "code": f"router_{status}",
                }
            },
        )
    except Exception as e:
        logger.error("router.unhandled_exception", path=request.url.path, error=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal router error",
                    "type": "internal_error",
                    "param": None,
                    "code": "router_internal_error",
                }
            },
        )
