"""Conversation fingerprint derivation for session-id fallback."""
from __future__ import annotations

import contextlib
import hashlib
import re
from functools import lru_cache
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=64)
def _compile_strip_pattern(pattern: str) -> re.Pattern | None:
    with contextlib.suppress(re.error):
        return re.compile(pattern)
    return None


def _extract_text(content: Any) -> str:
    """Extract text from message content (string or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content) if content else ""


def _strip_patterns(text: str, patterns: list[str]) -> str:
    """Apply regex strip patterns to remove volatile content."""
    for pattern in patterns:
        compiled = _compile_strip_pattern(pattern)
        if compiled is not None:
            text = compiled.sub("", text)
    return text


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def derive_fingerprint(
    system_message: str,
    first_user_message: str,
    tool_names: list[str],
    api_key_id: str,
    salt: str,
    strip_patterns: list[str] | None = None,
) -> str:
    """Derive a stable fingerprint from conversation invariant parts.

    The fingerprint is stable across turns because OpenAI-style clients
    resend the whole history, so the head of the conversation doesn't change.
    """
    # Strip volatile patterns from system message
    sys_text = _strip_patterns(system_message, strip_patterns or [])
    sys_text = _collapse_whitespace(sys_text)

    # First user message, whitespace-collapsed
    user_text = _collapse_whitespace(first_user_message)

    # Sorted tool names
    tools_str = ",".join(sorted(tool_names))

    # Normalize and hash
    raw = f"{sys_text}|{user_text}|{tools_str}|{api_key_id}|{salt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
