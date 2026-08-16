"""In-memory classification cache (TTL + LRU)."""
from __future__ import annotations

from typing import Any, Optional

from cachetools import TTLCache


class MemoryClassificationCache:
    """TTL + LRU cache for classification results."""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 10000):
        self._cache: TTLCache[str, dict] = TTLCache(maxsize=max_entries, ttl=ttl_seconds)

    async def get(self, key: str) -> Optional[dict]:
        return self._cache.get(key)

    async def put(self, key: str, value: dict) -> None:
        self._cache[key] = value

    async def delete(self, key: str) -> bool:
        existed = key in self._cache
        self._cache.pop(key, None)
        return existed

    async def clear(self) -> None:
        self._cache.clear()

    async def count(self) -> int:
        return len(self._cache)
