"""Request size limits and validation middleware."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Conservative defaults
MAX_MESSAGES = 100
MAX_MESSAGE_TEXT_BYTES = 256_000  # 256 KB per message
MAX_TOOLS = 50
MAX_TOOL_SCHEMA_BYTES = 32_000  # 32 KB per tool schema
MAX_N = 4
MAX_MAX_TOKENS = 32_000
MAX_TEMPERATURE = 2.0
MAX_PENALTY = 2.0


class RequestLimitsMiddleware(BaseHTTPMiddleware):
    """Enforce request body size and field bounds before JSON parsing."""

    def __init__(self, app, max_body_bytes: int = 10_485_760):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        # Skip non-POST and non-chat routes
        if request.method != "POST" or "/v1/chat/completions" not in request.url.path:
            return await call_next(request)

        # Check Content-Length early
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                cl = int(content_length)
                if cl > self.max_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"error": {
                            "message": f"Request body exceeds limit of {self.max_body_bytes} bytes",
                            "type": "request_too_large",
                            "code": "body_size_exceeded",
                        }},
                    )
            except ValueError:
                pass

        return await call_next(request)


def validate_request_bounds(body) -> JSONResponse | None:
    """Validate request field bounds after Pydantic parsing.

    Returns a JSONResponse error if bounds are violated, None if OK.
    """
    # Message count
    if len(body.messages) > MAX_MESSAGES:
        return JSONResponse(
            status_code=422,
            content={"error": {
                "message": f"Too many messages (max {MAX_MESSAGES})",
                "type": "invalid_request",
                "code": "too_many_messages",
            }},
        )

    # Message text size
    for msg in body.messages:
        content = msg.content
        if isinstance(content, str) and len(content.encode("utf-8")) > MAX_MESSAGE_TEXT_BYTES:
            return JSONResponse(
                status_code=422,
                content={"error": {
                    "message": f"Message content exceeds {MAX_MESSAGE_TEXT_BYTES} bytes",
                    "type": "invalid_request",
                    "code": "message_too_large",
                }},
            )
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    if len(block["text"].encode("utf-8")) > MAX_MESSAGE_TEXT_BYTES:
                        return JSONResponse(
                            status_code=422,
                            content={"error": {
                                "message": f"Message block exceeds {MAX_MESSAGE_TEXT_BYTES} bytes",
                                "type": "invalid_request",
                                "code": "message_too_large",
                            }},
                        )

    # Tool count and schema size
    if body.tools:
        if len(body.tools) > MAX_TOOLS:
            return JSONResponse(
                status_code=422,
                content={"error": {
                    "message": f"Too many tools (max {MAX_TOOLS})",
                    "type": "invalid_request",
                    "code": "too_many_tools",
                }},
            )
        import json
        for tool in body.tools:
            tool_str = json.dumps(tool)
            if len(tool_str) > MAX_TOOL_SCHEMA_BYTES:
                return JSONResponse(
                    status_code=422,
                    content={"error": {
                        "message": f"Tool schema exceeds {MAX_TOOL_SCHEMA_BYTES} bytes",
                        "type": "invalid_request",
                        "code": "tool_schema_too_large",
                    }},
                )

    # max_tokens bound
    if body.max_tokens is not None and body.max_tokens > MAX_MAX_TOKENS:
        return JSONResponse(
            status_code=422,
            content={"error": {
                "message": f"max_tokens exceeds limit of {MAX_MAX_TOKENS}",
                "type": "invalid_request",
                "code": "max_tokens_exceeded",
            }},
        )

    # n bound
    if body.n is not None and body.n > MAX_N:
        return JSONResponse(
            status_code=422,
            content={"error": {
                "message": f"n exceeds limit of {MAX_N}",
                "type": "invalid_request",
                "code": "n_exceeded",
            }},
        )

    # Temperature bound
    if body.temperature is not None and (body.temperature < 0 or body.temperature > MAX_TEMPERATURE):
        return JSONResponse(
            status_code=422,
            content={"error": {
                "message": f"temperature must be between 0 and {MAX_TEMPERATURE}",
                "type": "invalid_request",
                "code": "invalid_temperature",
            }},
        )

    # Penalty bounds
    for field_name in ("presence_penalty", "frequency_penalty"):
        val = getattr(body, field_name, None)
        if val is not None and (val < -MAX_PENALTY or val > MAX_PENALTY):
            return JSONResponse(
                status_code=422,
                content={"error": {
                    "message": f"{field_name} must be between -{MAX_PENALTY} and {MAX_PENALTY}",
                    "type": "invalid_request",
                    "code": f"invalid_{field_name}",
                }},
            )

    return None
