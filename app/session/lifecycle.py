"""Session lifecycle: expiry checks, turn caps, config-change policy."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from app.schemas.router import SessionPin, Level, SessionStatus


def check_expiry(pin: SessionPin) -> Optional[str]:
    """Check if a pin has expired. Returns reason or None."""
    if pin.is_expired():
        # Determine reason
        now = datetime.now(timezone.utc)
        try:
            pinned = datetime.fromisoformat(pin.pinned_at)
            if pinned.tzinfo is None:
                pinned = pinned.replace(tzinfo=timezone.utc)
            max_ttl_delta = timedelta(seconds=86400)  # default
            if now > pinned + max_ttl_delta:
                return "absolute"
        except Exception:
            pass
        return "idle"
    return None


def check_turn_cap(pin: SessionPin, max_turns: Optional[int]) -> bool:
    """Check if the pin has exceeded the turn cap. Returns True if capped."""
    if max_turns is not None and pin.turn_count >= max_turns:
        return True
    return False


def should_reclassify_on_turn(
    pin: SessionPin,
    max_provisional_turns: int,
) -> bool:
    """For provisional sessions, check if we should retry classification."""
    if pin.status == SessionStatus.PROVISIONAL:
        return pin.provisional_turns < max_provisional_turns
    return False


def apply_config_change(
    pin: SessionPin,
    policy: str,
    new_model_for_level: Optional[str] = None,
) -> Optional[SessionPin]:
    """Apply config-change policy to an existing pin.

    Returns updated pin, or None if the pin should be flushed.
    """
    if policy == "flush":
        return None
    elif policy == "keep_model":
        # Keep the exact model slug; if it's no longer configured, invalidate
        # (caller checks if model still exists)
        return pin
    elif policy == "keep_level":
        # Re-resolve level -> model from new config
        if new_model_for_level:
            pin.model = new_model_for_level
        return pin
    return pin
