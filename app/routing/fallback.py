"""Fallback chain execution: try primary, then fallbacks in order."""
from __future__ import annotations

import time
from typing import Any

import httpx

from app.telemetry import langfuse_tracing


class FallbackDeadlineExceeded(Exception):
    """Raised/reported when the wall-clock budget for a request is exhausted."""


class FallbackExecutor:
    """Executes the fallback chain for a routing decision."""

    def __init__(self, config, http_client: httpx.AsyncClient):
        self.config = config
        self.http = http_client

    def _remaining_budget(self, deadline: float | None) -> float | None:
        """Seconds left before the wall-clock deadline, or None when unbounded."""
        if deadline is None:
            return None
        return deadline - time.monotonic()

    def _attempt_timeout(self, deadline: float | None) -> float:
        """Timeout for a single upstream attempt, clamped to the remaining budget.

        Without this clamp a 3-model chain can hang for 3 x timeout_seconds,
        because each attempt gets a fresh full timeout.
        """
        base = float(self.config.provider.timeout_seconds)
        remaining = self._remaining_budget(deadline)
        if remaining is None:
            return base
        return max(1.0, min(base, remaining))

    async def _try_stream(self, url, request_payload, headers, model, timeout=None):
        """Try a streaming request. Returns (resp, error)."""
        req = self.http.build_request(
            "POST", f"{url}/chat/completions",
            json=request_payload, headers=headers,
            timeout=timeout if timeout is not None else self.config.provider.timeout_seconds,
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

    async def _try_non_stream(self, url, request_payload, headers, model, timeout=None):
        """Try a non-streaming request. Returns (json, error)."""
        resp = await self.http.post(
            f"{url}/chat/completions",
            json=request_payload, headers=headers,
            timeout=timeout if timeout is not None else self.config.provider.timeout_seconds,
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
        deadline: float | None = None,
    ) -> tuple[dict | None, httpx.Response | None, str, bool, str | None]:
        """Try primary, then fallbacks.

        ``deadline`` is an absolute ``time.monotonic()`` value bounding the
        whole chain.  Each attempt's timeout is clamped to the remaining
        budget, and the loop stops once the budget is gone, so a slow chain
        can no longer hang for the sum of every model's timeout.
        """
        models_to_try = [primary_model, *list(fallback_models)]
        last_error: str | None = None
        effective_base_url = (base_url or self.config.provider.base_url).rstrip("/")

        for i, model in enumerate(models_to_try):
            remaining = self._remaining_budget(deadline)
            if remaining is not None and remaining <= 0:
                last_error = (
                    f"request deadline exceeded before trying {model}"
                    if last_error is None
                    else f"{last_error}; deadline exceeded before trying {model}"
                )
                break

            attempt_timeout = self._attempt_timeout(deadline)
            request_payload = {**payload, "model": model}
            # Langfuse: one `generation` observation per upstream model
            # attempt (siblings under the request trace, so a fallback
            # chain is visible step by step). Must be a real `with` block
            # so the OTEL context nests it under the request trace.
            with langfuse_tracing.generation_cm(
                name=f"openrouter-{model}", model=model,
                input_data={"messages": request_payload.get("messages")},
            ) as gen:
                try:
                    if stream:
                        resp, err = await self._try_stream(
                            effective_base_url, request_payload, headers, model,
                            timeout=attempt_timeout)
                        if err:
                            last_error = err
                            langfuse_tracing._safe_update(
                                gen, level="ERROR", status_message=err[:500])
                            continue
                        langfuse_tracing._safe_update(
                            gen, output={"note": "streamed response", "model": model})
                        return None, resp, model, i > 0, None
                    else:
                        json_resp, err = await self._try_non_stream(
                            effective_base_url, request_payload, headers, model,
                            timeout=attempt_timeout)
                        if err:
                            last_error = err
                            langfuse_tracing._safe_update(
                                gen, level="ERROR", status_message=err[:500])
                            continue
                        jr = json_resp or {}
                        usage = jr.get("usage") or {}
                        gen_output = None
                        try:
                            choices = jr.get("choices") or []
                            first = choices[0] if choices else {}
                            message = first.get("message") or {}
                            gen_output = {
                                "content": langfuse_tracing._truncate(message.get("content")),
                                "finish_reason": first.get("finish_reason"),
                            }
                        except Exception:
                            gen_output = {"note": "non-streamed response"}
                        langfuse_tracing._safe_update(
                            gen,
                            output=gen_output,
                            usage_details={
                                "input": usage.get("prompt_tokens"),
                                "output": usage.get("completion_tokens"),
                                "total": usage.get("total_tokens"),
                            } if usage else None,
                        )
                        return json_resp, None, model, i > 0, None
                except (TimeoutError, httpx.TimeoutException):
                    last_error = f"timeout for {model}"
                    langfuse_tracing._safe_update(
                        gen, level="ERROR", status_message=last_error[:500])
                except httpx.HTTPStatusError as e:
                    last_error = f"upstream {e.response.status_code} for {model}"
                    langfuse_tracing._safe_update(
                        gen, level="ERROR", status_message=last_error[:500])
                except Exception as e:
                    last_error = f"error for {model}: {e!s}"
                    langfuse_tracing._safe_update(
                        gen, level="ERROR", status_message=last_error[:500])

        return None, None, primary_model, False, last_error or "all fallbacks exhausted"
