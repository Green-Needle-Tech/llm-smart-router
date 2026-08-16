"""Session-id resolution from headers, body, user field, or fingerprint."""
from __future__ import annotations

from typing import Any, Optional, Tuple

from app.schemas.router import SessionSource
from app.schemas.openai import ChatCompletionRequest
from .fingerprint import derive_fingerprint, _extract_text


def _get_api_key_id(auth_header: Optional[str]) -> str:
    """Extract a short identifier from the Authorization header."""
    if not auth_header:
        return "anon"
    # Bearer <key> -> last 8 chars
    parts = auth_header.replace("Bearer ", "").strip()
    return parts[-8:] if len(parts) >= 8 else parts[:8]


def _get_tool_names(tools: list[dict] | None) -> list[str]:
    """Extract function names from tools array."""
    if not tools:
        return []
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            fn = tool.get("function", tool)
            name = fn.get("name", "")
            if name:
                names.append(name)
    return names


def resolve_session_id(
    request: ChatCompletionRequest,
    headers: dict[str, str],
    config,
    fingerprint_salt: str = "",
) -> Tuple[Optional[str], SessionSource]:
    """Resolve session_id in strict priority order.

    Returns (session_id, source). session_id is None if unresolvable.
    """
    # 1. X-Session-Id header
    header_name = config.session.id_header if config else "X-Session-Id"
    session_id = headers.get(header_name.lower()) or headers.get(header_name)
    if session_id:
        return session_id, SessionSource.HEADER

    # 2. router.session_id body field
    if request.router and isinstance(request.router, dict):
        sid = request.router.get("session_id")
        if sid:
            return str(sid), SessionSource.BODY

    # 3. user field (if enabled)
    if config and config.session.use_user_field and request.user:
        return request.user, SessionSource.USER_FIELD

    # 4. Fingerprint fallback
    if config and config.session.fingerprint_fallback:
        messages = request.messages
        if not messages:
            return None, SessionSource.NONE

        system_msg = ""
        first_user_msg = ""
        for msg in messages:
            if msg.role == "system" and not system_msg:
                system_msg = _extract_text(msg.content)
            if msg.role == "user" and not first_user_msg:
                first_user_msg = _extract_text(msg.content)
            if system_msg and first_user_msg:
                break

        if not first_user_msg:
            return None, SessionSource.NONE

        tool_names = _get_tool_names(request.tools)
        auth_header = headers.get("authorization", "")
        api_key_id = _get_api_key_id(auth_header)

        strip_patterns = config.session.fingerprint_strip_patterns if config else []

        fp = derive_fingerprint(
            system_message=system_msg,
            first_user_message=first_user_msg,
            tool_names=tool_names,
            api_key_id=api_key_id,
            salt=fingerprint_salt or (config.session.fingerprint_salt if config else ""),
            strip_patterns=strip_patterns,
        )
        return fp, SessionSource.FINGERPRINT

    return None, SessionSource.NONE
