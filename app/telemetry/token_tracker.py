"""Per-session cumulative token usage tracking.

Accumulates input (prompt) and output (completion) token counts per tier
across all turns of a session.  The accumulated totals are rendered into
a compact postfix appended to the assistant response so the user can see
cumulative token consumption and context-window usage at a glance:

    [smart-router/L1-In:3032|Out:1000, L2-In:10021|Out:6054/Ctx:6100/1M]

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
) -> str:
    """Render the cumulative token usage into a compact postfix string.

    Tiers are sorted L1 → L5.  Only tiers with non-zero usage appear.

    When ``last_ctx_tokens`` > 0, appends ``/Ctx:N`` showing the prompt-token
    count from the most recent LLM call (i.e. the actual context-window
    consumption for this turn).

    When ``context_window`` > 0, appends ``/LIMIT`` (e.g. ``/1M``) so the
    user can see the ceiling at a glance.

    Example: ``"L1-In:3032|Out:1000, L2-In:10021|Out:6054/Ctx:6100/1M"``
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
    parts: list[str] = []
    for level in sorted(token_usage.keys(), key=lambda v: _ORDER.get(v, 99)):
        stats = token_usage[level]
        p = stats.get("prompt", 0)
        c = stats.get("completion", 0)
        if p == 0 and c == 0:
            continue
        parts.append(f"{level}-In:{p}|Out:{c}")
    # Build ctx suffix
    ctx_parts: list[str] = []
    if last_ctx_tokens > 0:
        ctx_parts.append(f"Ctx:{last_ctx_tokens}")
    if context_window > 0:
        limit = _format_context_limit(context_window)
        if limit:
            ctx_parts.append(limit)
    ctx_suffix = "/".join(ctx_parts) if ctx_parts else ""
    if not parts:
        return ctx_suffix
    # Join tier parts with ", ", then append ctx suffix with "/"
    result = ", ".join(parts)
    if ctx_suffix:
        result = f"{result}/{ctx_suffix}"
    return result


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

        [smart-router/L1-In:3032|Out:1000, L2-In:10021|Out:6054/Ctx:6100/1M]

    When tracking is disabled or no usage exists, falls back to the
    classic format::

        [smart-router/L1]

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
        token_part = render_postfix(token_usage, last_ctx_tokens, context_window)
        if token_part:
            return f"[smart-router/{token_part}]"
    return f"[smart-router/{level}]"
