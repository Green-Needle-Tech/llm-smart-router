"""Per-session cumulative token usage tracking.

Accumulates input (prompt) and output (completion) token counts per tier
across all turns of a session.  The accumulated totals are rendered into
a compact postfix appended to the assistant response so the user can see
cumulative token consumption at a glance:

    [smart-router/L1-In:3032|Out:1000, L2-In:10021|Out:6054]

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


def render_postfix(token_usage: dict[str, dict[str, int]] | None) -> str:
    """Render the cumulative token usage into a compact postfix string.

    Tiers are sorted L1 → L5.  Only tiers with non-zero usage appear.

    Example: ``"L1-In:3032|Out:1000, L2-In:10021|Out:6054"``
    Returns an empty string when there is no usage to report.
    """
    if not token_usage:
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
    return ", ".join(parts)


def build_postfix(
    level: str,
    token_usage: dict[str, dict[str, int]] | None,
    show_in_postfix: bool = True,
) -> str:
    """Build the full postfix marker for a response.

    When ``show_in_postfix`` is True and there is token usage data, the
    format is::

        [smart-router/L1-In:3032|Out:1000, L2-In:10021|Out:6054]

    When tracking is disabled or no usage exists, falls back to the
    classic format::

        [smart-router/L1]
    """
    base = f"[smart-router/{level}"
    if show_in_postfix:
        token_part = render_postfix(token_usage)
        if token_part:
            return f"{base}/{token_part}]"
    return f"{base}]"
