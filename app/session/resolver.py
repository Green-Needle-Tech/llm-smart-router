"""Session-id resolution from headers, body, user field, or fingerprint."""
from __future__ import annotations

import hashlib
import hmac
import os
import re

from app.schemas.openai import ChatCompletionRequest
from app.schemas.router import SessionSource

from .fingerprint import _extract_text, derive_fingerprint

# Conservative validation for external session IDs.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-.:]{1,256}$")


def _get_api_key_identity(auth_header: str | None) -> str:
    """Extract a stable, non-reversible identity from the Authorization header.

    Uses SHA-256 of the full key (first 16 hex chars) so the raw key or its
    suffix is never exposed or used as a lookup key directly.
    """
    if not auth_header:
        return "anon"
    token = auth_header.replace("Bearer ", "").replace("bearer ", "").strip()
    if not token:
        return "anon"
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _get_api_key_id(auth_header: str | None) -> str:
    """Backward-compatible alias for fingerprint derivation.

    Returns the same stable identity as _get_api_key_identity.
    """
    return _get_api_key_identity(auth_header)


def _get_server_secret() -> str:
    """Get the server-side namespace secret from env."""
    return os.environ.get("SESSION_NAMESPACE_SECRET", "")


def derive_internal_session_id(
    api_key_identity: str,
    external_session_id: str,
    server_secret: str = "",  # nosec B107 — empty default is intentional; falls back to env
) -> str:
    """Compute an HMAC-SHA256 namespaced session key.

    This prevents cross-tenant session collision: two clients using the same
    external X-Session-Id get different internal keys because their API key
    identities differ.

    The server_secret adds a server-side component so that even if the API key
    identity is known, the internal key cannot be predicted without the secret.
    """
    secret = server_secret or _get_server_secret()
    message = f"{api_key_identity}:{external_session_id}"
    return hmac.new(
        secret.encode() if secret else b"lsr-default",
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


def validate_external_session_id(sid: str) -> bool:
    """Validate that an external session ID is safe to accept.

    Conservative: 1-256 chars, alphanumeric + dash/underscore/colon/dot only.
    """
    return bool(_SESSION_ID_RE.match(sid))


def resolve_session_id(
    request: ChatCompletionRequest,
    headers: dict[str, str],
    config,
    fingerprint_salt: str = "",
    *,
    api_key_identity: str | None = None,
    namespace: bool = False,
) -> tuple[str | None, SessionSource]:
    """Resolve session_id in strict priority order.

    Returns (session_id, source). session_id is None if unresolvable.

    When ``namespace=True`` and ``api_key_identity`` is provided, the returned
    session_id is the HMAC-namespaced internal key, not the raw external ID.
    This prevents cross-tenant session interference.
    """
    # 1. X-Session-Id header
    header_name = config.session.id_header if config else "X-Session-Id"
    session_id = headers.get(header_name.lower()) or headers.get(header_name)
    if session_id:
        # Validate external session ID
        if not validate_external_session_id(session_id):
            return None, SessionSource.NONE
        if namespace and api_key_identity:
            session_id = derive_internal_session_id(
                api_key_identity, session_id,
            )
        return session_id, SessionSource.HEADER

    # 2. router.session_id body field
    if request.router and isinstance(request.router, dict):
        sid = request.router.get("session_id")
        if sid:
            sid = str(sid)
            if not validate_external_session_id(sid):
                return None, SessionSource.NONE
            if namespace and api_key_identity:
                sid = derive_internal_session_id(api_key_identity, sid)
            return sid, SessionSource.BODY

    # 3. user field (if enabled)
    if config and config.session.use_user_field and request.user:
        uid = str(request.user)
        if namespace and api_key_identity:
            uid = derive_internal_session_id(api_key_identity, uid)
        return uid, SessionSource.USER_FIELD

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
        key_id = api_key_identity or _get_api_key_id(auth_header)

        strip_patterns = config.session.fingerprint_strip_patterns if config else []

        fp = derive_fingerprint(
            system_message=system_msg,
            first_user_message=first_user_msg,
            tool_names=tool_names,
            api_key_id=key_id,
            salt=fingerprint_salt or (config.session.fingerprint_salt if config else ""),
            strip_patterns=strip_patterns,
        )
        return fp, SessionSource.FINGERPRINT

    return None, SessionSource.NONE


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
