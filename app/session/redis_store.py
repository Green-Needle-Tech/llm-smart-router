"""Redis-backed session store for multi-worker mode."""
from __future__ import annotations

import redis.asyncio as aioredis

from app.schemas.router import SessionPin

from .store import SessionStore


class RedisSessionStore(SessionStore):
    """Redis-backed store using SETNX for reservations."""

    def __init__(self, redis_url: str = "redis://redis:6379/0", max_sessions: int = 50000):
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._max_sessions = max_sessions

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def _res_key(session_id: str) -> str:
        return f"session:res:{session_id}"

    async def get(self, session_id: str) -> SessionPin | None:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        pin = SessionPin.model_validate_json(raw)
        if pin.is_expired():
            await self._redis.delete(self._key(session_id))
            return None
        return pin

    async def put(self, pin: SessionPin) -> None:
        # Compute TTL from expires_at instead of hardcoded 7200
        ttl = self._compute_ttl(pin)
        await self._redis.set(self._key(pin.session_id), pin.model_dump_json(), ex=ttl)

    @staticmethod
    def _compute_ttl(pin: SessionPin) -> int:
        """Compute remaining TTL from pin.expires_at."""
        if pin.expires_at is None:
            return 7200  # fallback
        from datetime import UTC, datetime
        try:
            exp = datetime.fromisoformat(pin.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=UTC)
            remaining = int((exp - datetime.now(UTC)).total_seconds())
            return max(remaining, 60)  # minimum 60s TTL
        except Exception:
            return 7200

    async def delete(self, session_id: str) -> bool:
        deleted = await self._redis.delete(self._key(session_id))
        await self._redis.delete(self._res_key(session_id))
        return deleted > 0

    async def delete_all(self) -> int:
        count = 0
        async for key in self._redis.scan_iter(match="session:*", count=100):
            if ":res:" not in key:
                await self._redis.delete(key)
                count += 1
        return count

    async def list_sessions(self, level: str | None = None, offset: int = 0, limit: int = 50) -> list[SessionPin]:
        pins = []
        async for key in self._redis.scan_iter(match="session:*", count=100):
            if ":res:" in key:
                continue
            raw = await self._redis.get(key)
            if raw:
                pin = SessionPin.model_validate_json(raw)
                if not pin.is_expired() and (level is None or pin.level.value == level):
                    pins.append(pin)
        return pins[offset:offset + limit]

    async def count(self) -> int:
        count = 0
        async for key in self._redis.scan_iter(match="session:*", count=100):
            if ":res:" not in key:
                count += 1
        return count

    async def reserve(self, session_id: str, ttl_seconds: int) -> bool:
        result = await self._redis.set(
            self._res_key(session_id), "1", nx=True, ex=ttl_seconds
        )
        return result is not None

    async def release(self, session_id: str) -> None:
        await self._redis.delete(self._res_key(session_id))

    async def close(self) -> None:
        """Close the Redis connection pool."""
        await self._redis.aclose()
