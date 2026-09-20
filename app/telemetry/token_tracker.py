"""Per-session cumulative token usage tracking.

Accumulates input (prompt) and output (completion) token counts per tier
across all turns of a session.  The accumulated totals are rendered into
a compact postfix appended to the assistant response so the user can see
cumulative token consumption and context-window usage at a glance:

[smart-router|L3|In:2102943/Out:8766|Ctx:84329/1M|v2.28.2]

The ``Ctx`` field shows the **last call's** prompt-token count (the actual
context-window consumption for the most recent turn) followed by the
configured context-window limit rendered as a human-readable value (e.g.
``1M`` for 1,000,000).  This lets the user monitor context-window growth
across a long conversation and know when they are approaching the limit.

The tracker stores data on the SessionPin (``pin.token_usage``) so it
survives across turns in the same way as ``cost_usd_total``.
"""
from __future__ import annotations

from typing import Any

from app.version import APPLICATION_VERSION


def _version_tag() -> str:
    """Compact version tag appended to every router postfix (e.g. ``v2.26.0``)."""
    return f"v{APPLICATION_VERSION}"


def extract_tokens(usage: dict[str, Any] | None) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an OpenRouter usage block."""
    if not usage:
        return 0, 0
    try:
        prompt = int(usage.get("prompt_tokens") or 0)
    except (TypeError, ValueError):
        prompt = 0
    try:
        completion = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        completion = 0
    return prompt, completion


def accumulate(
    token_usage: dict[str, dict[str, int]],
    level: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, dict[str, int]]:
    """Accumulate tokens for a tier into the usage dict (in-place).

    ``token_usage`` maps ``"L1"`` → ``{"prompt": N, "completion": N}``.
    Returns the same dict for convenience.
    """
    if level not in token_usage:
        token_usage[level] = {"prompt": 0, "completion": 0}
    token_usage[level]["prompt"] += prompt_tokens
    token_usage[level]["completion"] += completion_tokens
    return token_usage


def _format_context_limit(context_window: int) -> str:
    """Format a context-window size as a compact human-readable string.

    >>> _format_context_limit(1_000_000)
    '1M'
    >>> _format_context_limit(256_000)
    '256K'
    >>> _format_context_limit(131072)
    '128K'
    """
    if context_window <= 0:
        return ""
    if context_window >= 1_000_000:
        whole = context_window // 1_000_000
        frac = (context_window % 1_000_000) / 1_000_000
        if frac == 0:
            return f"{whole}M"
        # Round to 1 decimal for fractional millions (e.g. 1.5M)
        return f"{whole + frac:.1f}M".rstrip("0").rstrip(".")
    if context_window >= 1_000:
        return f"{context_window // 1_000}K"
    return str(context_window)


def render_postfix(
    token_usage: dict[str, dict[str, int]] | None,
    last_ctx_tokens: int = 0,
    context_window: int = 0,
    level: str | None = None,
) -> str:
    """Render the cumulative token usage into a compact postfix string.

    Only the current tier's usage appears (tiers are not aggregated in the
    new format).  Returns ``"L3|In:2102943/Out:8766|Ctx:84329/1M"`` style
    segments; the caller wraps them in ``[smart-router|...|vX]``.

    When ``last_ctx_tokens`` > 0, appends ``|Ctx:N/LIMIT`` showing the
    prompt-token count from the most recent LLM call (the actual
    context-window consumption for this turn).

    Example: ``"L3|In:2102943/Out:8766|Ctx:84329/1M"``
    Returns an empty string when there is no usage to report.
    """
    if not token_usage:
        # Still render Ctx if we have last-call data even with no cumulative
        if last_ctx_tokens > 0:
            parts = [f"Ctx:{last_ctx_tokens}"]
            limit = _format_context_limit(context_window)
            if limit:
                parts.append(limit)
            return "/".join(parts)
        return ""

    _ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    # Prefer the current tier; fall back to the lowest tier with usage.
    stats = None
    chosen = None
    if level and level in token_usage and (token_usage[level].get("prompt", 0) or token_usage[level].get("completion", 0)):
        stats = token_usage[level]
        chosen = level
    else:
        for lvl in sorted(token_usage.keys(), key=lambda v: _ORDER.get(v, 99)):
            s = token_usage[lvl]
            if s.get("prompt", 0) or s.get("completion", 0):
                stats = s
                chosen = lvl
                break
    if stats is None:
        # All tiers zero usage — fall back to ctx-only
        if last_ctx_tokens > 0:
            ctx_parts = [f"Ctx:{last_ctx_tokens}"]
            limit = _format_context_limit(context_window)
            if limit:
                ctx_parts.append(limit)
            return "/".join(ctx_parts)
        return ""
    p = stats.get("prompt", 0)
    c = stats.get("completion", 0)
    parts = [f"{chosen}|In:{p}/Out:{c}"]
    if last_ctx_tokens > 0:
        ctx_parts = [f"Ctx:{last_ctx_tokens}"]
        limit = _format_context_limit(context_window)
        if limit:
            ctx_parts.append(limit)
        parts.append("/".join(ctx_parts))
    return "|".join(parts)


def build_postfix(
    level: str,
    token_usage: dict[str, dict[str, int]] | None,
    show_in_postfix: bool = True,
    last_ctx_tokens: int = 0,
    context_window: int = 0,
) -> str:
    """Build the full postfix marker for a response.

    When ``show_in_postfix`` is True and there is token usage data, the
    format is::

        [smart-router|L3|In:2102943/Out:8766|Ctx:84329/1M|v2.28.2]

    When tracking is disabled or no usage exists, falls back to the
    classic format::

        [smart-router/L1|vX]

    Parameters
    ----------
    last_ctx_tokens
        Prompt-token count from the most recent LLM call.  This is the
        actual context-window consumption for the current turn — the
        metric that matters for staying under the limit.
    context_window
        The configured context-window limit (from ``provider.context_window``).
        Rendered as a compact suffix (e.g. ``1M``, ``256K``).
    """
    if show_in_postfix:
        token_part = render_postfix(token_usage, last_ctx_tokens, context_window, level=level)
        if token_part:
            return f"[smart-router|{token_part}|{_version_tag()}]"
    return f"[smart-router/{level}|{_version_tag()}]"
