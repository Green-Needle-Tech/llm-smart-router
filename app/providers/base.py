"""Provider adapter abstract base."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx


class ProviderAdapter(ABC):
    """Abstract provider adapter for LLM API translation."""

    @abstractmethod
    async def chat_completion(
        self,
        payload: dict[str, Any],
        fallback_models: list[str] | None = None,
        *,
        stream: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> tuple[dict | None, httpx.Response | None, str, bool, str | None]:
        """Execute a chat completion request with optional fallback chain.

        Returns (json, stream_response, model, fallback_used, error).
        When base_url/api_key are provided, they override the adapter defaults.
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """List available models from the provider."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...
