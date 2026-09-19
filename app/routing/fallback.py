"""Fallback chain execution: try primary, then fallbacks in order."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from app.telemetry import langfuse_tracing
from app.telemetry.logging import get_logger
from app.telemetry.metrics import router_provider_retries_total

logger = get_logger("fallback")


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
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                await resp.aclose()
                return None, f"upstream {resp.status_code} for {model}: {body}"
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
            body = resp.text[:500]
            return None, f"upstream {resp.status_code} for {model}: {body}"
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
        """Try primary, then fallbacks, with per-model retries.

        Each model in the chain is retried up to ``max_retries`` times (default
        2) on any provider error (timeout, retryable status, exception) before
        advancing to the next fallback model.  A backoff of
        ``retry_backoff_seconds`` is applied between retries on the same model.

        ``deadline`` is an absolute ``time.monotonic()`` value bounding the
        whole chain.  Each attempt's timeout is clamped to the remaining
        budget, and the loop stops once the budget is gone, so a slow chain
        can no longer hang for the sum of every model's timeout.
        """
        models_to_try = [primary_model, *list(fallback_models)]
        last_error: str | None = None
        effective_base_url = (base_url or self.config.provider.base_url).rstrip("/")
        chain_started = time.monotonic()
        attempts: list[dict[str, Any]] = []
        max_retries = self.config.provider.max_retries
        backoff = self.config.provider.retry_backoff_seconds
        remaining_budget = self._remaining_budget(deadline)
        logger.info(
            "fallback_chain_start",
            primary_model=primary_model,
            fallback_models=list(fallback_models),
            chain_depth=len(models_to_try),
            stream=stream,
            base_url=effective_base_url,
            deadline_seconds=round(remaining_budget, 3) if remaining_budget is not None else None,
            max_retries=max_retries,
        )

        for i, model in enumerate(models_to_try):
            for retry in range(max_retries + 1):
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
                attempt_started = time.monotonic()
                attempt_record: dict[str, Any] = {
                    "model": model,
                    "chain_position": i + 1,
                    "retry": retry,
                    "timeout_seconds": attempt_timeout,
                }
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
                                attempt_record["outcome"] = "retryable_status"
                                attempt_record["error"] = err
                                attempts.append(attempt_record)
                                logger.warning(
                                    "fallback_attempt_failed",
                                    **attempt_record,
                                    duration_ms=int((time.monotonic() - attempt_started) * 1000),
                                )
                                langfuse_tracing._safe_update(
                                    gen, level="ERROR", status_message=err[:500])
                                if retry < max_retries:
                                    router_provider_retries_total.labels(
                                        model=model, reason="retryable_status").inc()
                                    logger.info(
                                        "provider_retry_scheduled",
                                        model=model, retry=retry + 1,
                                        backoff_seconds=backoff,
                                    )
                                    await asyncio.sleep(backoff)
                                    continue
                                break  # retries exhausted, advance to next model
                            attempt_record["outcome"] = "success"
                            attempt_record["fallback_used"] = i > 0
                            attempts.append(attempt_record)
                            logger.info(
                                "fallback_chain_success",
                                model=model,
                                fallback_used=i > 0,
                                attempts=attempts,
                                total_ms=int((time.monotonic() - chain_started) * 1000),
                            )
                            langfuse_tracing._safe_update(
                                gen, output={"note": "streamed response", "model": model})
                            return None, resp, model, i > 0, None
                        else:
                            json_resp, err = await self._try_non_stream(
                                effective_base_url, request_payload, headers, model,
                                timeout=attempt_timeout)
                            if err:
                                last_error = err
                                attempt_record["outcome"] = "retryable_status"
                                attempt_record["error"] = err
                                attempts.append(attempt_record)
                                logger.warning(
                                    "fallback_attempt_failed",
                                    **attempt_record,
                                    duration_ms=int((time.monotonic() - attempt_started) * 1000),
                                )
                                langfuse_tracing._safe_update(
                                    gen, level="ERROR", status_message=err[:500])
                                if retry < max_retries:
                                    router_provider_retries_total.labels(
                                        model=model, reason="retryable_status").inc()
                                    logger.info(
                                        "provider_retry_scheduled",
                                        model=model, retry=retry + 1,
                                        backoff_seconds=backoff,
                                    )
                                    await asyncio.sleep(backoff)
                                    continue
                                break
                            attempt_record["outcome"] = "success"
                            attempt_record["fallback_used"] = i > 0
                            attempts.append(attempt_record)
                            logger.info(
                                "fallback_chain_success",
                                model=model,
                                fallback_used=i > 0,
                                attempts=attempts,
                                total_ms=int((time.monotonic() - chain_started) * 1000),
                            )
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
                        attempt_record["outcome"] = "timeout"
                        attempt_record["error"] = last_error
                        attempts.append(attempt_record)
                        logger.warning(
                            "fallback_attempt_failed",
                            **attempt_record,
                            duration_ms=int((time.monotonic() - attempt_started) * 1000),
                        )
                        langfuse_tracing._safe_update(
                            gen, level="ERROR", status_message=last_error[:500])
                        if retry < max_retries:
                            router_provider_retries_total.labels(
                                model=model, reason="timeout").inc()
                            logger.info(
                                "provider_retry_scheduled",
                                model=model, retry=retry + 1,
                                backoff_seconds=backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                        break
                    except httpx.HTTPStatusError as e:
                        last_error = f"upstream {e.response.status_code} for {model}"
                        attempt_record["outcome"] = "http_status_error"
                        attempt_record["error"] = last_error
                        attempts.append(attempt_record)
                        logger.warning(
                            "fallback_attempt_failed",
                            **attempt_record,
                            duration_ms=int((time.monotonic() - attempt_started) * 1000),
                        )
                        langfuse_tracing._safe_update(
                            gen, level="ERROR", status_message=last_error[:500])
                        if retry < max_retries:
                            router_provider_retries_total.labels(
                                model=model, reason="http_status_error").inc()
                            logger.info(
                                "provider_retry_scheduled",
                                model=model, retry=retry + 1,
                                backoff_seconds=backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                        break
                    except Exception as e:
                        last_error = f"error for {model}: {e!s}"
                        attempt_record["outcome"] = "exception"
                        attempt_record["error"] = last_error
                        attempts.append(attempt_record)
                        logger.warning(
                            "fallback_attempt_failed",
                            **attempt_record,
                            duration_ms=int((time.monotonic() - attempt_started) * 1000),
                        )
                        langfuse_tracing._safe_update(
                            gen, level="ERROR", status_message=last_error[:500])
                        if retry < max_retries:
                            router_provider_retries_total.labels(
                                model=model, reason="exception").inc()
                            logger.info(
                                "provider_retry_scheduled",
                                model=model, retry=retry + 1,
                                backoff_seconds=backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                        break
            # inner retry loop exhausted for this model — outer loop advances

        total_ms = int((time.monotonic() - chain_started) * 1000)
        logger.error(
            "fallback_chain_exhausted",
            primary_model=primary_model,
            attempts=attempts,
            last_error=last_error,
            total_ms=total_ms,
        )
        return None, None, primary_model, False, last_error or "all fallbacks exhausted"
