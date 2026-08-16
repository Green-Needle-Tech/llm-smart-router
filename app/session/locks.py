"""First-turn race protocol: prevent duplicate classification."""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from app.schemas.router import SessionPin, Level, SessionStatus
from .store import SessionStore


async def acquire_or_wait(
    store: SessionStore,
    session_id: str,
    ttl_seconds: int,
    wait_ms: int,
    poll_interval_ms: int = 25,
) -> tuple[bool, Optional[SessionPin]]:
    """Try to reserve a session for classification.

    Returns (won, existing_pin):
      - won=True: caller should classify and pin
      - won=False, pin=Some: another worker already pinned; use it
      - won=False, pin=None: timeout waiting; use default_level
    """
    won = await store.reserve(session_id, ttl_seconds)
    if won:
        return True, None

    # Wait for the winner to finish
    deadline = time.time() + (wait_ms / 1000.0)
    poll_interval = poll_interval_ms / 1000.0

    while time.time() < deadline:
        await asyncio.sleep(poll_interval)
        pin = await store.get(session_id)
        if pin is not None and pin.status != SessionStatus.CLASSIFYING:
            return False, pin

    # Timeout
    await store.release(session_id)
    return False, None
