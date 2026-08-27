"""In-memory TTL + LRU session store for single-worker mode."""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

from app.schemas.router import SessionPin

from .store import SessionStore


class MemorySessionStore(SessionStore):
    """TTL + LRU in-memory store. Thread-safe via asyncio.Lock."""

    def __init__(self, max_sessions: int = 50000):
        self._store: OrderedDict[str, SessionPin] = OrderedDict()
        self._reservations: dict[str, float] = {}  # session_id -> expiry timestamp
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._max_sessions = max_sessions

    async def get(self, session_id: str) -> SessionPin | None:
        async with self._global_lock:
            pin = self._store.get(session_id)
            if pin is None:
                return None
            if pin.is_expired():
                self._store.pop(session_id, None)
                return None
            # Move to end (LRU)
            self._store.move_to_end(session_id)
            return pin

    async def put(self, pin: SessionPin) -> None:
        async with self._global_lock:
            self._store[pin.session_id] = pin
            self._store.move_to_end(pin.session_id)
            # Evict oldest if over capacity
            while len(self._store) > self._max_sessions:
                self._store.popitem(last=False)

    async def delete(self, session_id: str) -> bool:
        async with self._global_lock:
            existed = self._store.pop(session_id, None) is not None
            self._reservations.pop(session_id, None)
            return existed

    async def delete_all(self) -> int:
        async with self._global_lock:
            count = len(self._store)
            self._store.clear()
            self._reservations.clear()
            return count

    async def list_sessions(self, level: str | None = None, offset: int = 0, limit: int = 50) -> list[SessionPin]:
        async with self._global_lock:
            pins = list(self._store.values())
            # Filter expired
            pins = [p for p in pins if not p.is_expired()]
            if level:
                pins = [p for p in pins if p.level.value == level]
            return pins[offset:offset + limit]

    async def count(self) -> int:
        async with self._global_lock:
            return len(self._store)

    async def reserve(self, session_id: str, ttl_seconds: int) -> bool:
        async with self._global_lock:
            now = time.time()
            # Check existing reservation
            exp = self._reservations.get(session_id)
            if exp is not None and exp > now:
                return False  # Already reserved
            # Check existing pin
            if session_id in self._store and not self._store[session_id].is_expired():
                return False  # Already pinned
            # Set reservation
            self._reservations[session_id] = now + ttl_seconds
            return True

    async def release(self, session_id: str) -> None:
        async with self._global_lock:
            self._reservations.pop(session_id, None)

    async def get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session asyncio.Lock."""
        async with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = asyncio.Lock()
            return self._locks[session_id]

    async def cleanup(self) -> None:
        """Remove expired entries and stale reservations."""
        async with self._global_lock:
            now = time.time()
            # Clean expired pins
            to_remove = [
                sid for sid, pin in self._store.items() if pin.is_expired()
            ]
            for sid in to_remove:
                self._store.pop(sid, None)
            # Clean stale reservations
            stale_res = [
                sid for sid, exp in self._reservations.items() if exp <= now
            ]
            for sid in stale_res:
                self._reservations.pop(sid, None)
            # Clean stale locks
            active = set(self._store.keys()) | set(self._reservations.keys())
            stale_locks = [sid for sid in self._locks if sid not in active]
            for sid in stale_locks:
                self._locks.pop(sid, None)
