"""Fallback chain execution: try primary, then fallbacks in order."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from app.schemas.router import Level


class FallbackExecutor:
    """Executes the fallback chain for a routing decision."""

    def __init__(self, config, http_client: httpx.AsyncClient):
        self.config = config
        self.http = http_client

    async def execute_with_fallback(
        self,
        primary_model: str,
        fallback_models: list[str],
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        stream: bool = False,
    ) -> tuple[dict | None, httpx.Response | None, str, bool, str | None]:
        """Try primary, then fallbacks. Returns (json_response, raw_response, model_used, fallback_used, error).

        For streaming, returns (None, response, model_used, fallback_used, error).
        """
        models_to_try = [primary_model] + list(fallback_models)
        last_error: str | None = None

        for i, model in enumerate(models_to_try):
            try:
                request_payload = {**payload, "model": model}

                if stream:
                    # httpx.stream() is an async context manager; we must not
                    # close the response before the caller finishes iterating.
                    # Use send() + aread() pattern to get the response without
                    # the context manager closing it.
                    req = self.http.build_request(
                        "POST",
                        f"{self.config.provider.base_url}/chat/completions",
                        json=request_payload,
                        headers=headers,
                        timeout=self.config.provider.timeout_seconds,
                    )
                    resp = await self.http.send(req, stream=True)
                    try:
                        if resp.status_code in self.config.provider.retry_on_status:
                            await resp.aread()
                            last_error = f"upstream {resp.status_code} for {model}"
                            await resp.aclose()
                            continue
                        resp.raise_for_status()
                        return None, resp, model, i > 0, None
                    except Exception:
                        await resp.aclose()
                        raise
                else:
                    resp = await self.http.post(
                        f"{self.config.provider.base_url}/chat/completions",
                        json=request_payload,
                        headers=headers,
                        timeout=self.config.provider.timeout_seconds,
                    )
                    if resp.status_code in self.config.provider.retry_on_status:
                        last_error = f"upstream {resp.status_code} for {model}"
                        continue
                    resp.raise_for_status()
                    return resp.json(), None, model, i > 0, None

            except (httpx.TimeoutException, asyncio.TimeoutError):
                last_error = f"timeout for {model}"
                continue
            except httpx.HTTPStatusError as e:
                last_error = f"upstream {e.response.status_code} for {model}"
                continue
            except Exception as e:
                last_error = f"error for {model}: {str(e)}"
                continue

        return None, None, primary_model, False, last_error or "all fallbacks exhausted"
