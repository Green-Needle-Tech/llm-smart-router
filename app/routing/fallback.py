"""Fallback chain execution: try primary, then fallbacks in order."""
from __future__ import annotations

from typing import Any

import httpx


class FallbackExecutor:
    """Executes the fallback chain for a routing decision."""

    def __init__(self, config, http_client: httpx.AsyncClient):
        self.config = config
        self.http = http_client

    async def _try_stream(self, url, request_payload, headers, model):
        """Try a streaming request. Returns (resp, error)."""
        req = self.http.build_request(
            "POST", f"{url}/chat/completions",
            json=request_payload, headers=headers,
            timeout=self.config.provider.timeout_seconds,
        )
        resp = await self.http.send(req, stream=True)
        try:
            if resp.status_code in self.config.provider.retry_on_status:
                await resp.aread()
                await resp.aclose()
                return None, f"upstream {resp.status_code} for {model}"
            resp.raise_for_status()
            return resp, None
        except Exception:
            await resp.aclose()
            raise

    async def _try_non_stream(self, url, request_payload, headers, model):
        """Try a non-streaming request. Returns (json, error)."""
        resp = await self.http.post(
            f"{url}/chat/completions",
            json=request_payload, headers=headers,
            timeout=self.config.provider.timeout_seconds,
        )
        if resp.status_code in self.config.provider.retry_on_status:
            return None, f"upstream {resp.status_code} for {model}"
        resp.raise_for_status()
        return resp.json(), None

    async def execute_with_fallback(
        self,
        primary_model: str,
        fallback_models: list[str],
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        stream: bool = False,
        base_url: str | None = None,
    ) -> tuple[dict | None, httpx.Response | None, str, bool, str | None]:
        """Try primary, then fallbacks."""
        models_to_try = [primary_model, *list(fallback_models)]
        last_error: str | None = None
        effective_base_url = (base_url or self.config.provider.base_url).rstrip("/")

        for i, model in enumerate(models_to_try):
            request_payload = {**payload, "model": model}
            try:
                if stream:
                    resp, err = await self._try_stream(
                        effective_base_url, request_payload, headers, model)
                    if err:
                        last_error = err
                        continue
                    return None, resp, model, i > 0, None
                else:
                    json_resp, err = await self._try_non_stream(
                        effective_base_url, request_payload, headers, model)
                    if err:
                        last_error = err
                        continue
                    return json_resp, None, model, i > 0, None
            except (TimeoutError, httpx.TimeoutException):
                last_error = f"timeout for {model}"
            except httpx.HTTPStatusError as e:
                last_error = f"upstream {e.response.status_code} for {model}"
            except Exception as e:
                last_error = f"error for {model}: {e!s}"

        return None, None, primary_model, False, last_error or "all fallbacks exhausted"
