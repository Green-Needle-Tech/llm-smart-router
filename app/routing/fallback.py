"""Fallback chain execution: try primary, then fallbacks in order."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import httpx

from app.telemetry import langfuse_tracing
from app.telemetry.logging import get_logger
from app.telemetry.metrics import router_provider_retries_total

logger = get_logger("fallback")

# Deterministic 4xx client errors: retrying the identical request against the
# same model wastes latency/budget without recovering — immediately advance to
# the next fallback model instead (except 400 max_tokens/context overflow,
# which gets one retry with max_tokens stripped).
_NON_RETRYABLE_4XX = frozenset({400, 401, 403, 404, 413, 422})
_MAX_TOKENS_ERROR_HINTS = (
    "max_tokens",
    "max_completion_tokens",
    "context length",
    "maximum context",
    "too many tokens",
    "exceeds the context",
)


def _is_max_tokens_400(status_code: int, body: str) -> bool:
    """True when an upstream 400 error looks like a max_tokens / context overflow."""
    if status_code != 400 or not body:
        return False
    lower = body.lower()
    return any(hint in lower for hint in _MAX_TOKENS_ERROR_HINTS)


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

    def _build_httpx_timeout(self, read_timeout: float) -> httpx.Timeout:
        """Build an httpx.Timeout that preserves connect_timeout_seconds."""
        connect = float(getattr(self.config.provider, "connect_timeout_seconds", 10) or 10)
        return httpx.Timeout(read_timeout, connect=min(connect, read_timeout))

    def _stream_header_timeout(self, attempt_timeout: float) -> float:
        """Max seconds to wait for initial HTTP response headers on a stream."""
        first_tok = getattr(self.config.provider, "stream_first_token_timeout_seconds", 0) or 0
        try:
            first_tok = float(first_tok)
        except (TypeError, ValueError):
            first_tok = 0.0
        if first_tok > 0:
            return min(attempt_timeout, first_tok)
        return attempt_timeout

    async def _sleep_backoff(self, backoff: float, deadline: float | None) -> bool:
        """Sleep backoff if deadline budget permits. Returns False if budget expired."""
        remaining = self._remaining_budget(deadline)
        if remaining is not None and remaining <= max(backoff, 0.5):
            return False
        if backoff > 0:
            await asyncio.sleep(backoff)
        return True

    async def _try_stream(self, url, request_payload, headers, model, timeout=None):
        """Try a streaming request. Returns (resp, error)."""
        eff_timeout = float(timeout if timeout is not None else self.config.provider.timeout_seconds)
        httpx_timeout = self._build_httpx_timeout(eff_timeout)
        req = self.http.build_request(
            "POST", f"{url}/chat/completions",
            json=request_payload, headers=headers,
            timeout=httpx_timeout,
        )
        header_timeout = self._stream_header_timeout(eff_timeout)
        resp = await asyncio.wait_for(self.http.send(req, stream=True), timeout=header_timeout)
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
        eff_timeout = float(timeout if timeout is not None else self.config.provider.timeout_seconds)
        httpx_timeout = self._build_httpx_timeout(eff_timeout)
        resp = await self.http.post(
            f"{url}/chat/completions",
            json=request_payload, headers=headers,
            timeout=httpx_timeout,
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
        payload_adapter: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        skip_models: set[str] | None = None,
    ) -> tuple[dict | None, httpx.Response | None, str, bool, str | None]:
        """Try primary, then fallbacks, with per-model retries.

        Each model in the chain is retried up to ``max_retries`` times (default
        2) on transient provider errors (timeout, retryable status, transport
        exception) before advancing to the next fallback model. Deterministic
        4xx responses advance immediately (with one auto-retry if a 400
        indicates a max_tokens/context-length overflow). A backoff of
        ``retry_backoff_seconds`` is applied between retries on the same model.

        ``deadline`` is an absolute ``time.monotonic()`` value bounding the
        whole chain.  Each attempt's timeout is clamped to the remaining
        budget, and the loop stops once the budget is gone, so a slow chain
        can no longer hang for the sum of every model's timeout.
        """
        skip = skip_models or set()
        models_to_try: list[str] = []
        seen_in_chain: set[str] = set()
        for m in [primary_model, *list(fallback_models)]:
            if m and m not in seen_in_chain and m not in skip:
                seen_in_chain.add(m)
                models_to_try.append(m)

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
            is_fallback_model = model != primary_model
            stripped_max_tokens = False
            retry = 0
            while retry <= max_retries:
                remaining = self._remaining_budget(deadline)
                if remaining is not None and remaining <= 0:
                    last_error = (
                        f"request deadline exceeded before trying {model}"
                        if last_error is None
                        else f"{last_error}; deadline exceeded before trying {model}"
                    )
                    break

                attempt_timeout = self._attempt_timeout(deadline)
                if payload_adapter is not None:
                    request_payload = payload_adapter(model, payload)
                    request_payload["model"] = model
                else:
                    request_payload = {**payload, "model": model}
                if stripped_max_tokens:
                    request_payload.pop("max_tokens", None)
                    request_payload.pop("max_completion_tokens", None)

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
                                    if not await self._sleep_backoff(backoff, deadline):
                                        break
                                    retry += 1
                                    continue
                                break  # retries exhausted, advance to next model
                            attempt_record["outcome"] = "success"
                            attempt_record["fallback_used"] = is_fallback_model
                            attempts.append(attempt_record)
                            logger.info(
                                "fallback_chain_success",
                                model=model,
                                fallback_used=is_fallback_model,
                                attempts=attempts,
                                total_ms=int((time.monotonic() - chain_started) * 1000),
                            )
                            langfuse_tracing._safe_update(
                                gen, output={"note": "streamed response", "model": model})
                            return None, resp, model, is_fallback_model, None
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
                                    if not await self._sleep_backoff(backoff, deadline):
                                        break
                                    retry += 1
                                    continue
                                break
                            # Detect null/empty choices (reasoning models that
                            # exhausted max_tokens, or HTTP 200 JSON error bodies
                            # from OpenRouter). Treat as retryable so the fallback
                            # chain moves to the next model instead of returning a
                            # malformed/empty response.
                            _choices = (json_resp or {}).get("choices")
                            if not _choices:
                                _embedded_err = (json_resp or {}).get("error")
                                if _embedded_err:
                                    last_error = (
                                        f"upstream 200 error body for {model}: "
                                        f"{str(_embedded_err)[:300]}"
                                    )
                                else:
                                    last_error = (
                                        f"upstream returned null/empty choices for "
                                        f"{model} (likely reasoning model exhausted "
                                        f"max_tokens)"
                                    )
                                attempt_record["outcome"] = "retryable_status"
                                attempt_record["error"] = last_error
                                attempts.append(attempt_record)
                                logger.warning(
                                    "fallback_attempt_failed",
                                    **attempt_record,
                                    duration_ms=int((time.monotonic() - attempt_started) * 1000),
                                )
                                langfuse_tracing._safe_update(
                                    gen, level="ERROR",
                                    status_message=last_error[:500])
                                if retry < max_retries:
                                    router_provider_retries_total.labels(
                                        model=model,
                                        reason="null_choices").inc()
                                    logger.info(
                                        "provider_retry_scheduled",
                                        model=model, retry=retry + 1,
                                        backoff_seconds=backoff,
                                    )
                                    if not await self._sleep_backoff(backoff, deadline):
                                        break
                                    retry += 1
                                    continue
                                break
                            attempt_record["outcome"] = "success"
                            attempt_record["fallback_used"] = is_fallback_model
                            attempts.append(attempt_record)
                            logger.info(
                                "fallback_chain_success",
                                model=model,
                                fallback_used=is_fallback_model,
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
                            return json_resp, None, model, is_fallback_model, None
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
                            if not await self._sleep_backoff(backoff, deadline):
                                break
                            retry += 1
                            continue
                        break
                    except httpx.HTTPStatusError as e:
                        raw = ""
                        try:
                            raw = e.response.text or ""
                        except Exception:
                            raw = ""
                        if not raw:
                            try:
                                raw = (await e.response.aread()).decode("utf-8", "replace")
                                await e.response.aclose()
                            except Exception:
                                raw = ""
                        body = raw[:300].strip().replace("\n", " ")
                        status_code = e.response.status_code
                        last_error = (
                            f"upstream {status_code} for {model}: {body}"
                            if body
                            else f"upstream {status_code} for {model}"
                        )
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
                        # Auto-recover from 400 max_tokens/context overflow by
                        # stripping max_tokens once before giving up on this model.
                        if (
                            not stripped_max_tokens
                            and ("max_tokens" in request_payload or "max_completion_tokens" in request_payload)
                            and _is_max_tokens_400(status_code, body)
                        ):
                            stripped_max_tokens = True
                            router_provider_retries_total.labels(
                                model=model, reason="max_tokens_400").inc()
                            logger.info(
                                "provider_retry_strip_max_tokens",
                                model=model,
                            )
                            continue
                        # Do not waste retries/backoff on deterministic 4xx errors —
                        # advance immediately to the next fallback model.
                        if status_code in _NON_RETRYABLE_4XX:
                            break
                        if retry < max_retries:
                            router_provider_retries_total.labels(
                                model=model, reason="http_status_error").inc()
                            logger.info(
                                "provider_retry_scheduled",
                                model=model, retry=retry + 1,
                                backoff_seconds=backoff,
                            )
                            if not await self._sleep_backoff(backoff, deadline):
                                break
                            retry += 1
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
                            if not await self._sleep_backoff(backoff, deadline):
                                break
                            retry += 1
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
