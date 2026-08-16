"""POST /v1/chat/completions — the primary endpoint."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas.openai import ChatCompletionRequest
from app.schemas.router import (
    Level, ClassificationResult, ClassificationSource,
    RouteDecision, SessionPin, SessionStatus, SessionSource,
)
from app.session.resolver import resolve_session_id
from app.session.locks import acquire_or_wait
from app.session.lifecycle import check_expiry, check_turn_cap
from app.middleware.auth import check_router_auth, unauthorized_response
from app.telemetry.logging import get_logger
from app.telemetry.metrics import (
    router_requests_total, router_active_requests,
    router_sessions_active, router_sessions_created_total,
    router_session_lookups_total, router_session_turns,
    router_classifier_calls_total, router_classification_duration_seconds,
    router_upstream_duration_seconds, router_tokens_total,
    router_cost_usd_total, router_fallbacks_total,
    router_reclassifications_total, router_escalations_total,
    router_escalation_signals_total,
    router_cache_events_total,
)

router = APIRouter()
logger = get_logger("chat")


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Handle a chat completion request with session-pinned routing."""
    start = time.monotonic()
    config = request.app.state.config.get()

    # Auth
    if config.auth.enabled:
        auth_header = request.headers.get("Authorization", "")
        if not check_router_auth(auth_header, config):
            return unauthorized_response()

    # Parse routing directive from model field
    routing_engine = request.app.state.routing_engine
    directive = routing_engine.parse_model_directive(body.model)

    # Extract router overrides
    router_opts = body.router or {}
    task_text = router_opts.get("task_text")
    max_level = Level.from_str(router_opts["max_level"]) if router_opts.get("max_level") else None
    min_level = Level.from_str(router_opts["min_level"]) if router_opts.get("min_level") else None
    forced_level = Level.from_str(router_opts["level"]) if router_opts.get("level") else None
    forced_model = router_opts.get("model")
    reclassify = router_opts.get("reclassify", False) or request.headers.get("X-Router-Reclassify") == "true"
    repin = router_opts.get("repin", False) or request.headers.get("X-Router-Repin") == "true"
    stateless = router_opts.get("stateless", False)
    bypass_cache = router_opts.get("bypass_cache", False) or request.headers.get("X-Router-Bypass-Cache") == "true"
    include_metadata = router_opts.get("include_metadata", False) or config.telemetry.include_metadata_in_body

    # Check header overrides
    header_level = request.headers.get("X-Router-Level")
    if header_level:
        try:
            forced_level = Level.from_str(header_level)
        except ValueError:
            pass
    header_model = request.headers.get("X-Router-Model")
    if header_model:
        forced_model = header_model

    # Handle classify-only mode
    if directive["mode"] == "classify_only":
        return await _classify_only(request, body, task_text, router_opts)

    # Handle passthrough mode
    if directive["mode"] == "passthrough":
        if not routing_engine.is_passthrough_allowed(directive["model"]):
            raise HTTPException(status_code=400, detail=f"Passthrough not allowed for model: {directive['model']}")
        return await _passthrough(request, body, directive["model"])

    # Handle stateless mode
    if directive["mode"] == "stateless" or stateless or not config.session.enabled:
        return await _stateless_classify_and_forward(
            request, body, config, routing_engine, task_text, max_level, min_level, forced_model, bypass_cache
        )

    # Handle explicit level directive (smart-router/L3 etc.)
    if directive["mode"] == "level":
        forced_level = directive["level"]

    # Session-pinned routing (the main path)
    return await _session_pinned_route(
        request, body, config, routing_engine,
        directive, forced_level, forced_model, max_level, min_level,
        reclassify, repin, task_text, bypass_cache, include_metadata, start,
    )


async def _session_pinned_route(
    request, body, config, routing_engine,
    directive, forced_level, forced_model, max_level, min_level,
    reclassify, repin, task_text, bypass_cache, include_metadata, start,
):
    """The main session-pinned routing path."""
    store = request.app.state.session_store
    classifier = request.app.state.classifier

    # Resolve session id
    headers_dict = {k.lower(): v for k, v in request.headers.items()}
    session_id, session_source = resolve_session_id(body, headers_dict, config, config.session.fingerprint_salt)

    if session_id is None:
        # Unidentifiable — classify per request
        router_session_lookups_total.labels(result="disabled").inc()
        return await _stateless_classify_and_forward(
            request, body, config, routing_engine, task_text, max_level, min_level, forced_model, bypass_cache
        )

    # Session lookup
    pin = await store.get(session_id)

    if pin is not None and pin.status != SessionStatus.CLASSIFYING:
        # Check expiry
        expired_reason = check_expiry(pin)
        if expired_reason:
            await store.delete(session_id)
            pin = None
            router_session_lookups_total.labels(result="miss").inc()
        elif check_turn_cap(pin, config.session.max_turns):
            await store.delete(session_id)
            pin = None
            router_session_lookups_total.labels(result="miss").inc()
        else:
            router_session_lookups_total.labels(result="hit").inc()

    if pin is not None and not reclassify:
        # SESSION HIT — use pinned model
        pin.turn_count += 1
        pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)

        # Apply per-request overrides (don't mutate pin unless repin)
        effective_level = pin.level
        effective_model = pin.model

        if forced_level and repin:
            # Re-pin to the new level
            pin.level = forced_level
            pin.model = routing_engine.resolve_model_for_level(forced_level)
            effective_level = forced_level
            effective_model = pin.model
        elif forced_level:
            effective_level = forced_level
            effective_model = routing_engine.resolve_model_for_level(forced_level)
        if forced_model:
            effective_model = forced_model

        # Apply max/min level
        if max_level and effective_level > max_level:
            effective_level = max_level
            effective_model = routing_engine.resolve_model_for_level(effective_level)
        if min_level and effective_level < min_level:
            effective_level = min_level
            effective_model = routing_engine.resolve_model_for_level(effective_level)

        await store.put(pin)

        classification = ClassificationResult(
            level=effective_level,
            confidence=1.0,
            reason="session pin",
            source=ClassificationSource.SESSION,
            latency_ms=0,
        )

        # Check for escalation signals (free signals layer)
        if config.session.escalation.enabled and config.session.escalation.free_signals_enabled:
            escalation_result = _check_escalation_signals(body, pin, config)
            if escalation_result:
                effective_level, effective_model = escalation_result
                await store.put(pin)

        route = RouteDecision(
            level=effective_level,
            model=effective_model,
            params=pin.params,
            classification=classification,
        )

        return await _forward_to_provider(
            request, body, route, session_id, session_source, pin, include_metadata, start,
        )

    # SESSION MISS — first turn (or reclassify)
    router_session_lookups_total.labels(result="miss").inc()

    # Acquire classification lock
    won, existing_pin = await acquire_or_wait(
        store, session_id,
        ttl_seconds=config.session.lock_reservation_seconds,
        wait_ms=config.session.lock_wait_ms,
    )

    if not won and existing_pin is not None:
        # Another worker already classified — use their pin
        pin = existing_pin
        pin.turn_count += 1
        pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)
        await store.put(pin)

        classification = ClassificationResult(
            level=pin.level,
            confidence=1.0,
            reason="session pin (race resolved)",
            source=ClassificationSource.SESSION,
            latency_ms=0,
        )

        route = RouteDecision(
            level=pin.level,
            model=pin.model,
            params=pin.params,
            classification=classification,
        )
        return await _forward_to_provider(
            request, body, route, session_id, session_source, pin, include_metadata, start,
        )

    if not won and existing_pin is None:
        # Lock timeout — use default level for this turn only
        default_level = Level.from_str(config.classification.default_level)
        classification = ClassificationResult(
            level=default_level,
            confidence=0.0,
            reason="lock timeout",
            source=ClassificationSource.DEFAULT,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
        route = routing_engine.resolve(default_level, classification, max_level=max_level, min_level=min_level)
        # Don't pin — next turn will retry
        return await _forward_to_provider(
            request, body, route, session_id, session_source, None, include_metadata, start,
        )

    # We won the lock — classify
    if forced_level:
        # Skip classification, use forced level
        level = forced_level
        classification = ClassificationResult(
            level=level,
            confidence=1.0,
            reason="forced override",
            source=ClassificationSource.OVERRIDE,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
    else:
        # Check classification cache
        cache = request.app.state.classification_cache
        cache_key = None
        cached = None
        if not bypass_cache and config.classification.cache.enabled:
            # Build a quick digest for cache key
            _, digest_info = await classifier.classify(
                body.messages, body.tools, body.response_format,
                task_text=task_text, bypass_cache=True,
            )
            cache_key = classifier.digest_builder.digest_hash(
                digest_info["digest"],
                config.classification.model,
                config.classification.rubric_version,
            )
            cached = await cache.get(cache_key)
            if cached:
                router_cache_events_total.labels(result="hit").inc()
                level = Level.from_str(cached["level"])
                classification = ClassificationResult(
                    level=level,
                    confidence=cached.get("confidence", 1.0),
                    reason=cached.get("reason", ""),
                    source=ClassificationSource.CACHE,
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
            else:
                router_cache_events_total.labels(result="miss").inc()
        else:
            router_cache_events_total.labels(result="bypass").inc()

        if cached is None:
            # Run classifier
            classification, digest_info = await classifier.classify(
                body.messages, body.tools, body.response_format,
                task_text=task_text, bypass_cache=bypass_cache,
            )
            level = classification.level or Level.from_str(config.classification.default_level)

            # Write to cache
            if cache_key and config.classification.cache.enabled:
                await cache.put(cache_key, {
                    "level": level.value,
                    "confidence": classification.confidence,
                    "reason": classification.reason,
                    "source": classification.source.value,
                })

    # Apply max/min level
    effective_level = level
    if max_level and effective_level > max_level:
        effective_level = max_level
    if min_level and effective_level < min_level:
        effective_level = min_level

    # Create pin
    model = routing_engine.resolve_model_for_level(effective_level)
    params = config.routing.get_params(effective_level.value)

    # Handle UNKNOWN / provisional
    is_unknown = classification.reason and "UNKNOWN" in classification.reason.upper()
    status = SessionStatus.PROVISIONAL if is_unknown else SessionStatus.PINNED

    new_pin = SessionPin(
        session_id=session_id,
        level=effective_level,
        model=model,
        params=params,
        status=status,
        classification=classification,
        turn_count=1,
    )
    if not is_unknown:
        new_pin.escalation.original_level = effective_level
    new_pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)

    # Release lock and write pin
    await store.release(session_id)
    await store.put(new_pin)

    router_sessions_created_total.labels(level=effective_level.value, id_source=session_source.value).inc()
    router_sessions_active.labels(level=effective_level.value).inc()

    route = RouteDecision(
        level=effective_level,
        model=model,
        params=params,
        classification=classification,
    )

    return await _forward_to_provider(
        request, body, route, session_id, session_source, new_pin, include_metadata, start,
    )


def _check_escalation_signals(body, pin, config) -> Optional[tuple[Level, str]]:
    """Check free-signal escalation. Returns (new_level, new_model) if escalated."""
    esc_cfg = config.session.escalation
    if not esc_cfg.enabled:
        return None

    import re
    signals_fired = []

    # repair_language
    last_user_text = ""
    for msg in reversed(body.messages):
        if msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            last_user_text = content
            break

    if re.search(r"\b(no,|that's wrong|still (failing|broken)|doesn't work|try again|not what I|incorrect|you missed)\b", last_user_text, re.IGNORECASE):
        signals_fired.append(("repair_language", esc_cfg.signal_weights.get("repair_language", 3)))

    # deep_keywords
    if re.search(r"\b(architect|design a system|prove|derive|refactor the|threat model|optimize the algorithm)\b", last_user_text, re.IGNORECASE):
        signals_fired.append(("deep_keywords", esc_cfg.signal_weights.get("deep_keywords", 2)))

    # turn_depth
    if pin.turn_count >= esc_cfg.escalate_after_turns:
        signals_fired.append(("turn_depth", esc_cfg.signal_weights.get("turn_depth", 1)))

    if not signals_fired:
        # Decay
        pin.escalation.score = max(0, pin.escalation.score - esc_cfg.decay_per_turn)
        return None

    # Add to score
    for signal, weight in signals_fired:
        pin.escalation.score += weight
        router_escalation_signals_total.labels(signal=signal).inc()

    # Check threshold and cooldown
    if (pin.escalation.score >= esc_cfg.threshold and
        pin.escalation.count < esc_cfg.max_escalations_per_session and
        pin.turn_count >= pin.escalation.cooldown_until_turn):

        new_level = Level.from_numeric(pin.level.numeric + 1)
        if new_level <= Level.from_str(config.routing.global_max_level):
            old_level = pin.level
            pin.level = new_level
            pin.model = config.routing.get_model(new_level.value)
            pin.params = config.routing.get_params(new_level.value)
            pin.escalation.count += 1
            pin.escalation.last_escalated_turn = pin.turn_count
            pin.escalation.cooldown_until_turn = pin.turn_count + esc_cfg.cooldown_turns
            if pin.escalation.original_level is None:
                pin.escalation.original_level = old_level

            router_escalations_total.labels(
                from_level=old_level.value, to_level=new_level.value,
                trigger="free_signal", layer="free_signal",
            ).inc()

            return (new_level, pin.model)

    return None


async def _forward_to_provider(
    request, body, route, session_id, session_source, pin, include_metadata, start,
):
    """Forward the request to OpenRouter and return the response."""
    config = request.app.state.config.get()
    provider = request.app.state.provider

    # Build upstream payload
    payload = body.model_dump(exclude={"router"}, exclude_none=True)
    payload["model"] = route.model

    # Apply tier params (fill in what client omitted)
    for key, val in route.params.items():
        if key not in payload or payload.get(key) is None:
            payload[key] = val

    # Resolve max_tokens: "auto" → detect from OpenRouter; int → use as-is
    tier_max_tokens = config.routing.get_max_tokens(route.level.value)
    client_sent_max = (
        payload.get("max_tokens") is not None
        or payload.get("max_completion_tokens") is not None
    )
    if not client_sent_max:
        if isinstance(tier_max_tokens, str) and tier_max_tokens == "auto":
            # Auto-detect from OpenRouter cache
            detected = provider.get_max_completion_tokens(route.model)
            if detected is not None and detected > 0:
                payload["max_tokens"] = detected
        elif isinstance(tier_max_tokens, int) and tier_max_tokens > 0:
            payload["max_tokens"] = tier_max_tokens
        # else: leave unset, let OpenRouter use model default

    # Get fallbacks
    fallbacks = config.routing.get_fallbacks(route.level.value)

    router_active_requests.inc()

    try:
        if body.stream:
            return await _handle_stream(
                request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
            )
        else:
            return await _handle_non_stream(
                request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
            )
    finally:
        router_active_requests.dec()


async def _handle_non_stream(
    request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
):
    """Handle non-streaming request."""
    provider = request.app.state.provider
    config = request.app.state.config.get()

    upstream_start = time.monotonic()
    json_resp, _, model_used, fallback_used, error = await provider.chat_completion(
        payload, fallback_models=fallbacks, stream=False,
    )
    upstream_ms = int((time.monotonic() - upstream_start) * 1000)
    total_ms = int((time.monotonic() - start) * 1000)

    if error and json_resp is None:
        # All fallbacks exhausted
        status = 502 if "exhausted" in error else 504
        error_type = "upstream_error" if status == 502 else "upstream_timeout"
        return JSONResponse(
            status_code=status,
            content={"error": {
                "message": error, "type": error_type, "param": None, "code": f"router_upstream_{status}",
            }},
        )

    # Update response model to actual used
    json_resp["model"] = model_used

    # Update pin cost
    if pin:
        usage = json_resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = provider.estimate_cost(model_used, prompt_tokens, completion_tokens)
        if cost is not None:
            pin.cost_usd_total += cost
            router_cost_usd_total.labels(level=route.level.value, model=model_used).inc(cost)
        router_tokens_total.labels(level=route.level.value, model=model_used, kind="prompt").inc(prompt_tokens)
        router_tokens_total.labels(level=route.level.value, model=model_used, kind="completion").inc(completion_tokens)
        await request.app.state.session_store.put(pin)

    # Metrics
    router_requests_total.labels(
        level=route.level.value, model=model_used,
        source=route.classification.source.value, status=200,
    ).inc()
    router_upstream_duration_seconds.labels(level=route.level.value, model=model_used).observe(upstream_ms / 1000)
    if fallback_used:
        router_fallbacks_total.labels(
            level=route.level.value, from_model=route.model, to_model=model_used, reason="fallback",
        ).inc()

    # Add router metadata if requested
    if include_metadata:
        json_resp["router"] = {
            "level": route.level.value,
            "model": model_used,
            "session_id": session_id,
            "turn": pin.turn_count if pin else 0,
            "classification_source": route.classification.source.value,
            "total_ms": total_ms,
            "fallback_used": fallback_used,
        }

    # Build response with headers
    response = JSONResponse(content=json_resp)
    _add_router_headers(response, route, session_id, session_source, pin, total_ms, fallback_used)
    return response


async def _handle_stream(
    request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
):
    """Handle streaming request — pass through SSE chunks."""
    provider = request.app.state.provider
    config = request.app.state.config.get()

    upstream_start = time.monotonic()
    _, stream_resp, model_used, fallback_used, error = await provider.chat_completion(
        payload, fallback_models=fallbacks, stream=True,
    )

    if error and stream_resp is None:
        status = 502 if "exhausted" in error else 504
        return JSONResponse(
            status_code=status,
            content={"error": {
                "message": error, "type": "upstream_error", "param": None, "code": f"router_upstream_{status}",
            }},
        )

    total_ms = int((time.monotonic() - start) * 1000)

    async def stream_generator():
        """Pass through SSE chunks from upstream, adding router headers as first event."""
        # Send router metadata as first SSE comment
        metadata = {
            "level": route.level.value,
            "model": model_used,
            "session_id": session_id,
            "turn": pin.turn_count if pin else 0,
            "classification_source": route.classification.source.value,
            "fallback_used": fallback_used,
        }
        if include_metadata:
            yield f"data: {json.dumps({'router': metadata})}\n\n"

        try:
            async for line in stream_resp.aiter_lines():
                yield f"{line}\n"
                if line.strip() == "data: [DONE]":
                    break
        except Exception as e:
            # Stream broke — emit error event
            error_data = {"error": {"message": f"stream interrupted: {str(e)}", "type": "upstream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            await stream_resp.aclose()

            # Update metrics
            upstream_ms = int((time.monotonic() - upstream_start) * 1000)
            router_requests_total.labels(
                level=route.level.value, model=model_used,
                source=route.classification.source.value, status=200,
            ).inc()
            router_upstream_duration_seconds.labels(level=route.level.value, model=model_used).observe(upstream_ms / 1000)
            if fallback_used:
                router_fallbacks_total.labels(
                    level=route.level.value, from_model=route.model, to_model=model_used, reason="fallback",
                ).inc()

    response = StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
    )
    _add_router_headers(response, route, session_id, session_source, pin, total_ms, fallback_used)
    return response


def _add_router_headers(response, route, session_id, session_source, pin, total_ms, fallback_used):
    """Add X-Router-* headers to the response."""
    response.headers["X-Router-Level"] = route.level.value
    response.headers["X-Router-Model"] = route.model
    response.headers["X-Router-Session-Id"] = session_id or ""
    response.headers["X-Router-Session-Source"] = session_source.value if session_source else "none"
    response.headers["X-Router-Session-Turn"] = str(pin.turn_count if pin else 0)
    response.headers["X-Router-Classification-Source"] = route.classification.source.value
    response.headers["X-Router-Classification-Ms"] = str(route.classification.latency_ms)
    response.headers["X-Router-Total-Ms"] = str(total_ms)
    response.headers["X-Router-Fallback-Used"] = str(fallback_used).lower()
    if pin and pin.pinned_at:
        response.headers["X-Router-Session-Pinned-At"] = pin.pinned_at
    if route.escalated:
        response.headers["X-Router-Escalated"] = "true"
        if route.escalated_from:
            response.headers["X-Router-Escalated-From"] = route.escalated_from.value


async def _classify_only(request, body, task_text, router_opts):
    """Return classification result only, no downstream call."""
    classifier = request.app.state.classifier
    result, digest_info = await classifier.classify(
        body.messages, body.tools, body.response_format,
        task_text=task_text,
    )
    return JSONResponse(content={
        "level": result.level.value if result.level else "UNKNOWN",
        "confidence": result.confidence,
        "reason": result.reason,
        "source": result.source.value,
        "latency_ms": result.latency_ms,
    })


async def _passthrough(request, body, model):
    """Forward as-is to OpenRouter."""
    provider = request.app.state.provider
    payload = body.model_dump(exclude={"router"}, exclude_none=True)
    payload["model"] = model

    if body.stream:
        _, stream_resp, model_used, _, error = await provider.chat_completion(payload, stream=True)
        if error:
            return JSONResponse(status_code=502, content={"error": {"message": error, "type": "upstream_error"}})

        async def gen():
            async for line in stream_resp.aiter_lines():
                yield f"{line}\n"
                if line.strip() == "data: [DONE]":
                    break
            await stream_resp.aclose()

        return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        json_resp, _, model_used, _, error = await provider.chat_completion(payload)
        if error and json_resp is None:
            return JSONResponse(status_code=502, content={"error": {"message": error, "type": "upstream_error"}})
        return JSONResponse(content=json_resp)


async def _stateless_classify_and_forward(
    request, body, config, routing_engine, task_text, max_level, min_level, forced_model, bypass_cache,
):
    """Classify in isolation, no session pinning."""
    classifier = request.app.state.classifier

    classification, digest_info = await classifier.classify(
        body.messages, body.tools, body.response_format,
        task_text=task_text, bypass_cache=bypass_cache,
    )

    level = classification.level or Level.from_str(config.classification.default_level)
    route = routing_engine.resolve(level, classification, max_level=max_level, min_level=min_level, forced_model=forced_model)

    return await _forward_to_provider(
        request, body, route, None, None, None, False, time.monotonic(),
    )
