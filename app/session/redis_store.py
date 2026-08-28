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
        ttl = getattr(pin, "session_config_ttl", 7200)
        await self._redis.set(self._key(pin.session_id), pin.model_dump_json(), ex=ttl)

    async def delete(self, session_id: str) -> bool:
        deleted = await self._redis.delete(self._key(session_id))
        await self._redis.delete(self._res_key(session_id))
        return deleted > 0

    async def delete_all(self) -> int:
        keys = await self._redis.keys("session:*")
        res_keys = [k for k in keys if ":res:" not in k]
        if res_keys:
            await self._redis.delete(*res_keys)
        return len(res_keys) if res_keys else 0

    async def list_sessions(self, level: str | None = None, offset: int = 0, limit: int = 50) -> list[SessionPin]:
        keys = await self._redis.keys("session:*")
        keys = [k for k in keys if ":res:" not in k]
        pins = []
        for k in keys:
            raw = await self._redis.get(k)
            if raw:
                pin = SessionPin.model_validate_json(raw)
                if not pin.is_expired() and (level is None or pin.level.value == level):
                    pins.append(pin)
        return pins[offset:offset + limit]

    async def count(self) -> int:
        keys = await self._redis.keys("session:*")
        return len([k for k in keys if ":res:" not in k])

    async def reserve(self, session_id: str, ttl_seconds: int) -> bool:
        result = await self._redis.set(
            self._res_key(session_id), "1", nx=True, ex=ttl_seconds
        )
        return result is not None

    async def release(self, session_id: str) -> None:
        await self._redis.delete(self._res_key(session_id))
