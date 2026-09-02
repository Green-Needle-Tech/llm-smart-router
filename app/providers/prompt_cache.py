"""Upstream prompt-cache (KV cache) optimization.

Two features, both applied to the upstream payload just before forwarding:

1. session_id passthrough — OpenRouter uses `session_id` as the provider
   sticky-routing key, so consecutive turns of a conversation land on the
   same provider endpoint and its prefix/KV cache stays warm. Without it,
   stickiness only activates after a first cache hit and is derived from
   message hashing.

2. cache_control injection — Anthropic and Alibaba/Qwen models require
   explicit `cache_control: {type: "ephemeral"}` breakpoints on stable
   content (system prompt, tool definitions). OpenAI / Gemini / DeepSeek
   cache implicitly and need nothing.
"""
from __future__ import annotations

from typing import Any

# Model id prefixes that require explicit cache_control breakpoints.
# Everything else (OpenAI, Google, DeepSeek, Z.AI, ...) caches implicitly.
_EXPLICIT_CACHE_PREFIXES = ("anthropic/", "qwen/", "alibaba/")


class PromptCachingConfigView:
    """Lightweight accessor over config.provider.prompt_caching (may be absent)."""

    def __init__(self, cfg: Any):
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.forward_session_id = bool(getattr(cfg, "forward_session_id", True))
        self.inject_cache_control = bool(getattr(cfg, "inject_cache_control", True))
        self.anthropic_ttl = getattr(cfg, "anthropic_ttl", "5m") or "5m"
        self.min_tokens = int(getattr(cfg, "min_tokens", 1024))


def _view(config: Any) -> PromptCachingConfigView | None:
    provider = getattr(config, "provider", None)
    if provider is None:
        return None
    pc = getattr(provider, "prompt_caching", None)
    if pc is None or not getattr(pc, "enabled", False):
        return None
    return PromptCachingConfigView(pc)


def needs_explicit_cache_control(model: str) -> bool:
    """True for models whose providers require explicit cache_control."""
    return str(model).startswith(_EXPLICIT_CACHE_PREFIXES)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for floor checks."""
    return len(text) // 4


def _already_has_cache_control(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    return False


def _find_system_target(messages):
    """Find the first system/developer message. Returns (target, content) or None."""
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            return msg, msg.get("content")
    return None


def _extract_text_from_content(content) -> str | None:
    """Extract text from str or block-list content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return None


def _apply_cache_control(target, content, cc):
    """Apply cache_control to the target message. Returns True on success."""
    if isinstance(content, str):
        target["content"] = [{"type": "text", "text": content, "cache_control": cc}]
        return True
    if isinstance(content, list):
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cc
            return True
    return False


def _inject_cache_control(payload: dict[str, Any], pc: PromptCachingConfigView) -> bool:
    """Add an ephemeral cache_control breakpoint to the system/developer message."""
    messages = payload.get("messages") or []
    if not messages or _already_has_cache_control(messages):
        return False

    found = _find_system_target(messages)
    if found is None:
        return False
    target, content = found

    text = _extract_text_from_content(content)
    if text is None or _estimate_tokens(text) < pc.min_tokens:
        return False

    cc: dict[str, Any] = {"type": "ephemeral"}
    model = str(payload.get("model", ""))
    if model.startswith("anthropic/") and pc.anthropic_ttl in ("5m", "1h"):
        cc["ttl"] = pc.anthropic_ttl

    return _apply_cache_control(target, content, cc)


def extract_cache_usage(json_resp: dict[str, Any]) -> tuple[int, int]:
    """Extract (cached_tokens, cache_write_tokens) from an OpenRouter response.

    The ground truth lives in usage.prompt_tokens_details (cached_tokens,
    cache_write_tokens). Returns (0, 0) when absent.
    """
    usage = (json_resp or {}).get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    try:
        cached = int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        cached = 0
    try:
        written = int(details.get("cache_write_tokens") or 0)
    except (TypeError, ValueError):
        written = 0
    return cached, written


def apply_prompt_cache_features(
    payload: dict[str, Any],
    session_id: str | None,
    config: Any,
) -> dict[str, Any]:
    """Mutate the upstream payload for maximal provider-side cache hits."""
    pc = _view(config)
    if pc is None:
        return payload

    # 1. session_id → OpenRouter provider sticky routing (cache key).
    # Body field wins; do not clobber a client-provided value.
    if pc.forward_session_id and session_id and not payload.get("session_id"):
        payload["session_id"] = str(session_id)[:256]

    # 2. Explicit cache_control for Anthropic / Qwen routes.
    if pc.inject_cache_control and needs_explicit_cache_control(payload.get("model", "")):
        _inject_cache_control(payload, pc)

    return payload
