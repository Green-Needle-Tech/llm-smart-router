"""Request size limits and validation middleware."""
from __future__ import annotations

import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Generous defaults aligned with 1M-context long-running agentic sessions
MAX_MESSAGES = 2_000
MAX_MESSAGE_TEXT_BYTES = 2_097_152  # 2 MB per message
MAX_TOOLS = 256
MAX_TOOL_SCHEMA_BYTES = 131_072  # 128 KB per tool schema
MAX_N = 4
MAX_MAX_TOKENS = 524_288
MAX_TEMPERATURE = 2.0
MAX_PENALTY = 2.0


def _err(status: int, msg: str, code: str) -> JSONResponse:
    """Build a standard error response."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": msg, "type": "invalid_request", "code": code}},
    )


def _limit(config, attr: str, default: int) -> int:
    server_cfg = getattr(config, "server", None) if config is not None else None
    val = getattr(server_cfg, attr, None) if server_cfg is not None else None
    return val if isinstance(val, int) and val > 0 else default


def _check_message_count(body, config=None) -> JSONResponse | None:
    max_messages = _limit(config, "max_messages", MAX_MESSAGES)
    if len(body.messages) > max_messages:
        return _err(422, f"Too many messages (max {max_messages})", "too_many_messages")
    return None


def _check_message_sizes(body, config=None) -> JSONResponse | None:
    max_text_bytes = _limit(config, "max_message_text_bytes", MAX_MESSAGE_TEXT_BYTES)
    for msg in body.messages:
        content = msg.content
        if isinstance(content, str) and len(content.encode("utf-8")) > max_text_bytes:
            return _err(422, f"Message content exceeds {max_text_bytes} bytes", "message_too_large")
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                    and len(block["text"].encode("utf-8")) > max_text_bytes
                ):
                    return _err(422, f"Message block exceeds {max_text_bytes} bytes", "message_too_large")
    return None


def _check_tools(body, config=None) -> JSONResponse | None:
    if not body.tools:
        return None
    max_tools = _limit(config, "max_tools", MAX_TOOLS)
    max_schema_bytes = _limit(config, "max_tool_schema_bytes", MAX_TOOL_SCHEMA_BYTES)
    if len(body.tools) > max_tools:
        return _err(422, f"Too many tools (max {max_tools})", "too_many_tools")
    for tool in body.tools:
        if len(json.dumps(tool)) > max_schema_bytes:
            return _err(422, f"Tool schema exceeds {max_schema_bytes} bytes", "tool_schema_too_large")
    return None


def _check_numeric_bounds(body, config=None) -> JSONResponse | None:
    max_tokens_cap = _limit(config, "max_max_tokens", MAX_MAX_TOKENS)
    if body.max_tokens is not None and body.max_tokens > max_tokens_cap:
        return _err(422, f"max_tokens exceeds limit of {max_tokens_cap}", "max_tokens_exceeded")
    if body.n is not None and body.n > MAX_N:
        return _err(422, f"n exceeds limit of {MAX_N}", "n_exceeded")
    return None


def _check_temperature(body, config=None) -> JSONResponse | None:
    if body.temperature is not None and (body.temperature < 0 or body.temperature > MAX_TEMPERATURE):
        return _err(422, f"temperature must be between 0 and {MAX_TEMPERATURE}", "invalid_temperature")
    return None


def _check_penalties(body, config=None) -> JSONResponse | None:
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

    def __init__(self, app, max_body_bytes: int = 33_554_432):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or "/v1/chat/completions" not in request.url.path:
            return await call_next(request)

        max_bytes = self.max_body_bytes
        cm = getattr(getattr(request.app, "state", None), "config", None)
        if cm is not None and hasattr(cm, "get"):
            server_cfg = getattr(cm.get(), "server", None)
            if server_cfg is not None:
                max_bytes = getattr(server_cfg, "max_body_bytes", max_bytes)

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                cl = int(content_length)
                if cl > max_bytes:
                    return _err(
                        413,
                        f"Request body exceeds limit of {max_bytes} bytes",
                        "body_size_exceeded",
                    )
            except ValueError:
                pass

        return await call_next(request)


def validate_request_bounds(body, config=None) -> JSONResponse | None:
    """Validate request field bounds after Pydantic parsing.

    Returns a JSONResponse error if bounds are violated, None if OK.
    """
    for check in (_check_message_count, _check_message_sizes, _check_tools,
                  _check_numeric_bounds, _check_temperature, _check_penalties):
        err = check(body, config)
        if err is not None:
            return err
    return None
