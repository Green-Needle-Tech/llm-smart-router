"""Request size limits and validation middleware."""
from __future__ import annotations

import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Conservative defaults
MAX_MESSAGES = 100
MAX_MESSAGE_TEXT_BYTES = 256_000  # 256 KB per message
MAX_TOOLS = 50
MAX_TOOL_SCHEMA_BYTES = 32_000  # 32 KB per tool schema
MAX_N = 4
MAX_MAX_TOKENS = 131_072
MAX_TEMPERATURE = 2.0
MAX_PENALTY = 2.0


def _err(status: int, msg: str, code: str) -> JSONResponse:
    """Build a standard error response."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": msg, "type": "invalid_request", "code": code}},
    )


def _check_message_count(body) -> JSONResponse | None:
    if len(body.messages) > MAX_MESSAGES:
        return _err(422, f"Too many messages (max {MAX_MESSAGES})", "too_many_messages")
    return None


def _check_message_sizes(body) -> JSONResponse | None:
    for msg in body.messages:
        content = msg.content
        if isinstance(content, str) and len(content.encode("utf-8")) > MAX_MESSAGE_TEXT_BYTES:
            return _err(422, f"Message content exceeds {MAX_MESSAGE_TEXT_BYTES} bytes", "message_too_large")
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                    and len(block["text"].encode("utf-8")) > MAX_MESSAGE_TEXT_BYTES
                ):
                    return _err(422, f"Message block exceeds {MAX_MESSAGE_TEXT_BYTES} bytes", "message_too_large")
    return None


def _check_tools(body) -> JSONResponse | None:
    if not body.tools:
        return None
    if len(body.tools) > MAX_TOOLS:
        return _err(422, f"Too many tools (max {MAX_TOOLS})", "too_many_tools")
    for tool in body.tools:
        if len(json.dumps(tool)) > MAX_TOOL_SCHEMA_BYTES:
            return _err(422, f"Tool schema exceeds {MAX_TOOL_SCHEMA_BYTES} bytes", "tool_schema_too_large")
    return None


def _check_numeric_bounds(body) -> JSONResponse | None:
    if body.max_tokens is not None and body.max_tokens > MAX_MAX_TOKENS:
        return _err(422, f"max_tokens exceeds limit of {MAX_MAX_TOKENS}", "max_tokens_exceeded")
    if body.n is not None and body.n > MAX_N:
        return _err(422, f"n exceeds limit of {MAX_N}", "n_exceeded")
    return None


def _check_temperature(body) -> JSONResponse | None:
    if body.temperature is not None and (body.temperature < 0 or body.temperature > MAX_TEMPERATURE):
        return _err(422, f"temperature must be between 0 and {MAX_TEMPERATURE}", "invalid_temperature")
    return None


def _check_penalties(body) -> JSONResponse | None:
    for field_name in ("presence_penalty", "frequency_penalty"):
        val = getattr(body, field_name, None)
        if val is not None and (val < -MAX_PENALTY or val > MAX_PENALTY):
            return _err(
                422,
                f"{field_name} must be between -{MAX_PENALTY} and {MAX_PENALTY}",
                f"invalid_{field_name}",
            )
    return None


class RequestLimitsMiddleware(BaseHTTPMiddleware):
    """Enforce request body size and field bounds before JSON parsing."""

    def __init__(self, app, max_body_bytes: int = 10_485_760):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or "/v1/chat/completions" not in request.url.path:
            return await call_next(request)

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                cl = int(content_length)
                if cl > self.max_body_bytes:
                    return _err(
                        413,
                        f"Request body exceeds limit of {self.max_body_bytes} bytes",
                        "body_size_exceeded",
                    )
            except ValueError:
                pass

        return await call_next(request)


def validate_request_bounds(body) -> JSONResponse | None:
    """Validate request field bounds after Pydantic parsing.

    Returns a JSONResponse error if bounds are violated, None if OK.
    """
    for check in (_check_message_count, _check_message_sizes, _check_tools,
                  _check_numeric_bounds, _check_temperature, _check_penalties):
        err = check(body)
        if err is not None:
            return err
    return None
