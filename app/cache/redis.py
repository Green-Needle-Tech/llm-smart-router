"""Redis-backed classification cache."""
from __future__ import annotations

import json

import redis.asyncio as aioredis


class RedisClassificationCache:
    """Redis-backed TTL cache."""

    def __init__(self, redis_url: str = "redis://redis:6379/0", ttl_seconds: int = 3600):
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds

    @staticmethod
    def _key(k: str) -> str:
        return f"classify_cache:{k}"

    async def get(self, key: str) -> dict | None:
        raw = await self._redis.get(self._key(key))
        if raw is None:
            return None
        return json.loads(raw)

    async def put(self, key: str, value: dict) -> None:
        await self._redis.set(self._key(key), json.dumps(value), ex=self._ttl)

    async def delete(self, key: str) -> bool:
        return await self._redis.delete(self._key(key)) > 0

    async def clear(self) -> None:
        keys = await self._redis.keys("classify_cache:*")
        if keys:
            await self._redis.delete(*keys)

    async def count(self) -> int:
        keys = await self._redis.keys("classify_cache:*")
        return len(keys)
