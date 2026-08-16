"""OpenRouter provider adapter: request translation, streaming, retries."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

import httpx

from .base import ProviderAdapter
from app.routing.fallback import FallbackExecutor


class OpenRouterAdapter(ProviderAdapter):
    """OpenRouter API adapter with streaming pass-through and fallback."""

    def __init__(self, config, api_key: str):
        self.config = config
        self.api_key = api_key
        self.http = httpx.AsyncClient(
            http2=True,
            timeout=config.provider.timeout_seconds,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
        )
        self._fallback_executor = FallbackExecutor(config, self.http)
        self._pricing: dict[str, dict] = {}

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.config.provider.headers,
        }

    async def chat_completion(
        self,
        payload: dict[str, Any],
        fallback_models: list[str] | None = None,
        *,
        stream: bool = False,
    ) -> tuple[dict | None, httpx.Response | None, str, bool, str | None]:
        """Execute a chat completion with fallback chain."""
        primary_model = payload.get("model", "")
        fallbacks = fallback_models or []
        return await self._fallback_executor.execute_with_fallback(
            primary_model=primary_model,
            fallback_models=fallbacks,
            payload=payload,
            headers=self._headers(),
            stream=stream,
        )

    async def list_models(self) -> list[dict]:
        """Fetch model list from OpenRouter."""
        try:
            resp = await self.http.get(
                f"{self.config.provider.base_url}/models",
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            # Cache pricing
            for m in models:
                mid = m.get("id", "")
                pricing = m.get("pricing", {})
                if mid and pricing:
                    self._pricing[mid] = {
                        "prompt": float(pricing.get("prompt", 0) or 0),
                        "completion": float(pricing.get("completion", 0) or 0),
                    }
            return models
        except Exception:
            return []

    def get_pricing(self, model: str) -> dict | None:
        """Get cached pricing for a model."""
        return self._pricing.get(model)

    def estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """Estimate cost in USD for a request."""
        pricing = self.get_pricing(model)
        if pricing is None:
            return None
        cost = (prompt_tokens * pricing["prompt"]) + (completion_tokens * pricing["completion"])
        return round(cost, 6)

    async def close(self) -> None:
        await self.http.aclose()
