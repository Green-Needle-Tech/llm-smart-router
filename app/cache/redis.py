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
        batch: list[str] = []
        async for key in self._redis.scan_iter(match="classify_cache:*", count=200):
            batch.append(key)
            if len(batch) >= 200:
                await self._redis.delete(*batch)
                batch.clear()
        if batch:
            await self._redis.delete(*batch)

    async def count(self) -> int:
        total = 0
        async for _ in self._redis.scan_iter(match="classify_cache:*", count=200):
            total += 1
        return total

    async def close(self) -> None:
        """Close the Redis connection pool."""
        await self._redis.aclose()
