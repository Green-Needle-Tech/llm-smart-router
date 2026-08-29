"""POST /v1/chat/completions — the primary endpoint."""
from __future__ import annotations

import contextlib
import json
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.guardrails.rules import strip_invisible_text
from app.guardrails.scanner import GuardrailConfig, GuardrailEngine
from app.middleware.auth import check_router_auth, unauthorized_response
from app.privacy.ip_redaction import IPRedactionEngine
from app.providers.prompt_cache import apply_prompt_cache_features, extract_cache_usage
from app.schemas.openai import ChatCompletionRequest
from app.schemas.router import (
    ClassificationResult,
    ClassificationSource,
    Level,
    RouteDecision,
    SessionPin,
    SessionStatus,
)
from app.session.lifecycle import check_expiry, check_turn_cap
from app.session.locks import acquire_or_wait
from app.session.resolver import resolve_session_id
from app.telemetry.logging import get_logger
from app.telemetry.metrics import (
    router_active_requests,
    router_cache_events_total,
    router_cost_usd_total,
    router_escalation_signals_total,
    router_escalation_turn,
    router_escalations_total,
    router_fallbacks_total,
    router_guardrail_blocks_total,
    router_guardrail_findings_total,
    router_guardrail_secret_masks_total,
    router_privacy_redactions_total,
    router_prompt_cache_hit_ratio,
    router_prompt_cache_writes_total,
    router_prompt_cached_tokens_total,
    router_requests_total,
    router_session_lookups_total,
    router_sessions_active,
    router_sessions_created_total,
    router_stream_errors_total,
    router_tokens_total,
    router_upstream_duration_seconds,
)
from app.temporal_awareness.engine import TemporalAwarenessEngine

_SEV_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

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

    # Guardrails: input injection/jailbreak scan (runs BEFORE privacy
    # redaction so detection sees the original text).
    guardrail_block = _guardrail_scan_input(request, body)
    if guardrail_block is not None:
        return guardrail_block

    # Privacy middleware: redact raw IPs before classification/forwarding.
    # Re-hydration key uses the same session resolution as routing; when no
    # session is identifiable, a per-request UUID scopes the mapping.
    redaction_key = await _redact_incoming(request, body)

    # Temporal awareness: normalize temporal expressions (today, yesterday,
    # tomorrow, etc.) to concrete dates before classification/forwarding.
    _process_temporal_awareness(request, body)

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
        with contextlib.suppress(ValueError):
            forced_level = Level.from_str(header_level)
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
        return await _passthrough(request, body, directive["model"], redaction_key=redaction_key)

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
        redaction_key=redaction_key,
    )


def _guardrail_scan_input(request, body):
    """Scan incoming messages with the input guardrail.

    Returns a JSONResponse (400) when configured to block and the request
    trips the severity threshold; otherwise None. Findings are always
    counted in metrics and logged.

    Also detects invisible/zero-width characters and banned substrings,
    and strips invisible characters from message content before forwarding.
    """
    engine: GuardrailEngine | None = getattr(request.app.state, "guardrails", None)
    if engine is None:
        return None
    config = request.app.state.config.get()
    cfg = config.telemetry.guardrails
    engine.config = GuardrailConfig(
        input_enabled=cfg.input_enabled,
        input_action=cfg.input_action,
        block_on_severity=cfg.block_on_severity,
        output_enabled=cfg.output_enabled,
        output_action=cfg.output_action,
        invisible_text_detection=cfg.invisible_text_detection,
        pii_masking_enabled=cfg.pii_masking_enabled,
        banned_substrings=cfg.banned_substrings,
        refusal_detection=cfg.refusal_detection,
        malicious_url_detection=cfg.malicious_url_detection,
        system_prompt_leak_detection=getattr(cfg, "system_prompt_leak_detection", False),
        system_prompt_fragments=getattr(cfg, "system_prompt_fragments", []),
        system_prompt_leak_threshold=getattr(cfg, "system_prompt_leak_threshold", 0.85),
        homoglyph_normalization=getattr(cfg, "homoglyph_normalization", True),
        obfuscation_detection=getattr(cfg, "obfuscation_detection", True),
        entropy_threshold=getattr(cfg, "entropy_threshold", 4.5),
    )
    messages = [
        (m.model_dump() if hasattr(m, "model_dump") else m) for m in body.messages
    ]

    # Invisible text detection + stripping
    invisible_findings: list = []
    if cfg.invisible_text_detection:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str) and content:
                ifs = engine.scan_invisible_text(content)
                if ifs:
                    invisible_findings.extend(ifs)
                    # Strip invisible chars from the message in place
                    msg["content"] = strip_invisible_text(content)
                    # Also update the original body.messages list
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        ifs = engine.scan_invisible_text(block["text"])
                        if ifs:
                            invisible_findings.extend(ifs)
                            block["text"] = strip_invisible_text(block["text"])
        for f in invisible_findings:
            router_guardrail_findings_total.labels(
                rule_id=f.rule_id, severity=f.severity, direction="input",
            ).inc()
            logger.warning(
                "router.guardrail.invisible_text",
                rule=f.rule_id, severity=f.severity,
            )

    # Banned substrings detection
    banned_findings: list = []
    if cfg.banned_substrings:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                banned_findings.extend(engine.scan_banned_substrings(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        banned_findings.extend(engine.scan_banned_substrings(block["text"]))
        for f in banned_findings:
            router_guardrail_findings_total.labels(
                rule_id=f.rule_id, severity=f.severity, direction="input",
            ).inc()
            logger.warning(
                "router.guardrail.banned_substring",
                rule=f.rule_id, severity=f.severity, action=cfg.input_action,
            )

    # Obfuscation and high-entropy detection (Base64, Hex, URL-encoded)
    obfuscation_findings: list = []
    if getattr(cfg, "obfuscation_detection", True):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                obfuscation_findings.extend(engine.scan_obfuscation(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        obfuscation_findings.extend(engine.scan_obfuscation(block["text"]))
        for f in obfuscation_findings:
            router_guardrail_findings_total.labels(
                rule_id=f.rule_id, severity=f.severity, direction="input",
            ).inc()
            logger.warning(
                "router.guardrail.obfuscation",
                rule=f.rule_id, severity=f.severity, action=cfg.input_action,
            )

    # Standard injection scan
    result = engine.scan_messages(messages)
    for f in result.findings:
        router_guardrail_findings_total.labels(
            rule_id=f.rule_id, severity=f.severity, direction="input",
        ).inc()
        logger.warning(
            "router.guardrail.input_finding",
            rule=f.rule_id, severity=f.severity, action=cfg.input_action,
        )

    # Combine all findings for block decision
    all_findings = result.findings + banned_findings + obfuscation_findings
    blocked = False
    if cfg.input_action == "block" and all_findings:
        threshold = _SEV_ORDER.get(cfg.block_on_severity, 2)
        blocked = any(_SEV_ORDER.get(f.severity, 0) >= threshold for f in all_findings)

    if blocked:
        top = max(all_findings, key=lambda f: _SEV_ORDER.get(f.severity, 0))
        router_guardrail_blocks_total.labels(rule_id=top.rule_id, severity=top.severity).inc()
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": (
                    "Request blocked by router guardrail: potential prompt "
                    f"injection detected (rule: {top.rule_id}, severity: {top.severity})."
                ),
                "type": "guardrail_violation",
                "param": None,
                "code": "router_guardrail_blocked",
            }},
        )
    return None


def _guardrail_process_output(request, json_resp) -> None:
    """Mask (or log) secrets, PII, and malicious URLs in a non-streaming response, in place.

    Also logs refusal patterns (log-only monitoring). Findings are counted
    in metrics: secret/PII/URL masks use router_guardrail_secret_masks_total;
    refusals and log-mode findings use router_guardrail_findings_total.
    """
    engine: GuardrailEngine | None = getattr(request.app.state, "guardrails", None)
    if engine is None:
        return
    for choice in json_resp.get("choices", []):
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        findings = engine.process_response_content(message)
        for f in findings:
            if f.rule_id.startswith("refusal-"):
                # Refusals are always log-only, regardless of output_action
                router_guardrail_findings_total.labels(
                    rule_id=f.rule_id, severity=f.severity, direction="output",
                ).inc()
                logger.info(
                    "router.guardrail.refusal_detected", rule=f.rule_id,
                )
            elif f.rule_id.startswith("pii-") or f.rule_id == "malicious-url" or f.rule_id == "output-system-prompt-leak":
                if engine.config.output_action == "mask":
                    router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                else:
                    router_guardrail_findings_total.labels(
                        rule_id=f.rule_id, severity=f.severity, direction="output",
                    ).inc()
                logger.warning(
                    "router.guardrail.output_finding", rule=f.rule_id,
                    action=engine.config.output_action,
                )
            else:
                # Secret findings (existing behavior)
                if engine.config.output_action == "mask":
                    router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                else:
                    router_guardrail_findings_total.labels(
                        rule_id=f.rule_id, severity=f.severity, direction="output",
                    ).inc()
                logger.warning(
                    "router.guardrail.output_finding", rule=f.rule_id,
                    action=engine.config.output_action,
                )


async def _redact_incoming(request, body) -> str | None:
    """Run IP redaction over the incoming messages if enabled.

    Returns the re-hydration key (session id), or None when the privacy
    module is disabled. The key mirrors the routing session resolution so
    re-hydration on the response side looks up the same mapping bucket.
    """
    engine: IPRedactionEngine | None = getattr(
        request.app.state, "ip_redaction", None
    )
    if engine is None:
        return None
    headers_dict = {k.lower(): v for k, v in request.headers.items()}
    config = request.app.state.config.get()
    session_id, _ = resolve_session_id(body, headers_dict, config, config.session.fingerprint_salt)
    key = session_id or f"req-{uuid.uuid4()}"
    messages = [m.model_dump() if hasattr(m, "model_dump") else m for m in body.messages]
    await engine.redact_messages(messages, key)
    # Write redacted content back onto the pydantic messages.
    for orig, dumped in zip(body.messages, messages, strict=False):
        if orig.content != dumped.get("content"):
            orig.content = dumped.get("content")
    router_privacy_redactions_total.inc()
    return key


def _process_temporal_awareness(request, body) -> None:
    """Normalize temporal expressions (today, yesterday, tomorrow) to concrete dates."""
    engine: TemporalAwarenessEngine | None = getattr(
        request.app.state, "temporal_awareness_engine", None
    )
    if engine is None:
        return
    messages = [m.model_dump() if hasattr(m, "model_dump") else m for m in body.messages]
    processed = engine.process_messages(messages)
    for orig, proc in zip(body.messages, processed, strict=False):
        content = proc.get("content")
        if isinstance(content, str) and orig.content != content:
            orig.content = content


def _rehydrate_response_content(engine, json_resp, key: str | None) -> None:
    """Re-hydrate placeholders in a non-streaming response in place (sync path)."""
    if engine is None or key is None:
        return
    for choice in json_resp.get("choices", []):
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = engine.rehydrate_text_sync(content, key)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        block["text"] = engine.rehydrate_text_sync(block["text"], key)


async def _session_pinned_route(
    request, body, config, routing_engine,
    directive, forced_level, forced_model, max_level, min_level,
    reclassify, repin, task_text, bypass_cache, include_metadata, start,
    redaction_key=None,
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
        if expired_reason or check_turn_cap(pin, config.session.max_turns):
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
        effective_model = _resolve_effective_model(
            pin, routing_engine, config,
        )

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
            redaction_key=redaction_key,
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
            redaction_key=redaction_key,
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
            redaction_key=redaction_key,
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
            # Build digest for cache key (digest build only, no LLM call)
            digest_info = classifier.digest_builder.build(
                messages=body.messages,
                tools=body.tools,
                response_format=body.response_format,
                task_text=task_text,
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
        redaction_key=redaction_key,
    )


def _resolve_effective_model(pin, routing_engine, config) -> str:
    """Resolve the model for a session hit, honoring session.on_config_change.

    "keep_level" (default): keep the pinned LEVEL but re-resolve the MODEL
    from the live config on every turn — tier model changes (hot-reload)
    apply to existing sessions too. The pin is updated so /admin/sessions
    and later turns stay consistent.
    "keep_pin": keep the frozen model from classification time (old behavior).
    """
    if config.session.on_config_change == "keep_pin":
        return pin.model
    resolved = routing_engine.resolve_model_for_level(pin.level)
    if not resolved:
        return pin.model
    pin.model = resolved
    return resolved


def _extract_raw_user_text(body) -> str:
    """Extract the user's actual typed text, stripping injected context blocks.

    Hermes injects <memory-context>, <skill-context>, and similar blocks into
    the user message content. These are agent scaffolding, not user intent.
    Signal detection must only scan what the user actually typed.
    """
    last_user_text = ""
    for msg in reversed(body.messages):
        if msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            last_user_text = content
            break

    # Strip injected context blocks: <memory-context>...</memory-context>,
    # <skill-context>...</skill-context>, <system-note>...</system-note>, etc.
    import re
    last_user_text = re.sub(
        r"<(?:memory|skill|system|context|soul|persona|user_profile)[-_]?context>.*?</(?:memory|skill|system|context|soul|persona|user_profile)[-_]?context>",
        "",
        last_user_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return last_user_text.strip()


def _check_escalation_signals(body, pin, config) -> tuple[Level, str] | None:
    """Check free-signal escalation. Returns (new_level, new_model) if escalated."""
    esc_cfg = config.session.escalation
    if not esc_cfg.enabled:
        return None

    import re
    signals_fired = []

    # Extract only the user's raw typed text (not injected context)
    raw_user_text = _extract_raw_user_text(body)

    # repair_language
    if re.search(r"\b(no,|that's wrong|still (failing|broken)|doesn't work|try again|not what I|incorrect|you missed)\b", raw_user_text, re.IGNORECASE):
        signals_fired.append(("repair_language", esc_cfg.signal_weights.get("repair_language", 3)))

    # deep_keywords
    if re.search(r"\b(architect|design a system|prove|derive|refactor the|threat model|optimize the algorithm)\b", raw_user_text, re.IGNORECASE):
        signals_fired.append(("deep_keywords", esc_cfg.signal_weights.get("deep_keywords", 2)))

    # turn_depth
    if pin.turn_count >= esc_cfg.escalate_after_turns:
        signals_fired.append(("turn_depth", esc_cfg.signal_weights.get("turn_depth", 1)))

    # Drop zero-weight signals so a disabled signal cannot suppress decay
    signals_fired = [(s, w) for s, w in signals_fired if w > 0]

    if not signals_fired:
        # Decay
        pin.escalation.score = max(0, pin.escalation.score - esc_cfg.decay_per_turn)
        return None

    # Add to score
    signal_names = []
    for signal, weight in signals_fired:
        pin.escalation.score += weight
        signal_names.append(signal)
        router_escalation_signals_total.labels(signal=signal).inc()

    # Check threshold and cooldown
    if (pin.escalation.score >= esc_cfg.threshold and
        pin.escalation.count < esc_cfg.max_escalations_per_session and
        pin.turn_count >= pin.escalation.cooldown_until_turn):

        new_level = Level.from_numeric(pin.level.numeric + 1)
        if new_level <= Level.from_str(config.routing.global_max_level):
            # L5 guard: deep_keywords alone must not escalate to L5.
            # Require at least one "strong" signal (repair_language or
            # tool_error_loop) for the final tier jump.
            strong_signals = {"repair_language", "tool_error_loop"}
            has_strong = any(s in strong_signals for s, _ in signals_fired)
            if new_level == Level.L5 and not has_strong:
                # Cap at L4 — keywords alone are insufficient evidence for
                # the most expensive tier.
                return None

            old_level = pin.level
            pin.level = new_level
            pin.model = config.routing.get_model(new_level.value)
            pin.params = config.routing.get_params(new_level.value)
            pin.escalation.count += 1
            pin.escalation.last_escalated_turn = pin.turn_count
            pin.escalation.cooldown_until_turn = pin.turn_count + esc_cfg.cooldown_turns
            # Reset accumulated score: evidence has been consumed by this
            # escalation. Without this the score stays above threshold and the
            # session chains straight into the next tier once cooldown expires.
            pin.escalation.score = 0
            # Record which signals triggered this escalation for diagnostics
            pin.escalation.last_trigger = signal_names
            if pin.escalation.original_level is None:
                pin.escalation.original_level = old_level

            # Observe the turn number in the histogram for diagnostics
            router_escalation_turn.observe(float(pin.turn_count))

            router_escalations_total.labels(
                from_level=old_level.value, to_level=new_level.value,
                trigger="free_signal", layer="free_signal",
            ).inc()

            return (new_level, pin.model)

    return None


def _resolve_tier_provider(config, level: str) -> tuple[str | None, str | None]:
    """Resolve per-tier provider overrides for a given level.

    Returns (base_url, api_key). Both are None when the tier uses the
    global provider configuration.
    """
    import os
    # Defensive: tests may mock config as SimpleNamespace without get_tier
    routing = getattr(config, "routing", None)
    if routing is None or not hasattr(routing, "get_tier"):
        return None, None
    tier = routing.get_tier(level)
    if tier.base_url or tier.api_key_env:
        base_url = tier.base_url
        api_key = os.environ.get(tier.api_key_env, "") if tier.api_key_env else None
        return base_url, api_key
    return None, None


async def _forward_to_provider(
    request, body, route, session_id, session_source, pin, include_metadata, start,
    redaction_key=None,
):
    """Forward the request to the provider and return the response."""
    config = request.app.state.config.get()
    provider = request.app.state.provider

    # Build upstream payload
    payload = body.model_dump(exclude={"router"}, exclude_none=True)
    _strip_model_postfix_from_messages(payload.get("messages", []))
    payload["model"] = route.model

    # Prompt-cache optimization: session_id → provider sticky routing,
    # cache_control injection for Anthropic/Qwen routes.
    apply_prompt_cache_features(payload, session_id, config)

    # Apply tier params (fill in what client omitted)
    for key, val in route.params.items():
        if key not in payload or payload.get(key) is None:
            payload[key] = val

    # Resolve max_tokens: "auto" → always use OpenRouter-detected value
    # (overrides client-sent max_tokens to avoid OpenRouter 402 credit-reservation
    # errors when the client sends a large default like 65536); int → use as-is
    tier_max_tokens = config.routing.get_max_tokens(route.level.value)
    if isinstance(tier_max_tokens, str) and tier_max_tokens == "auto":
        # Always auto-detect from OpenRouter cache, ignoring client-sent value
        detected = provider.get_max_completion_tokens(route.model)
        if detected is not None and detected > 0:
            payload["max_tokens"] = detected
        # else: leave whatever client sent (or unset)
    elif isinstance(tier_max_tokens, int) and tier_max_tokens > 0:
        payload["max_tokens"] = tier_max_tokens
    # else: leave unset, let OpenRouter use model default

    # Get fallbacks
    fallbacks = config.routing.get_fallbacks(route.level.value)

    # Resolve per-tier provider overrides (base_url, api_key)
    tier_base_url, tier_api_key = _resolve_tier_provider(config, route.level.value)

    router_active_requests.inc()

    try:
        if body.stream:
            return await _handle_stream(
                request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
                redaction_key=redaction_key, tier_base_url=tier_base_url, tier_api_key=tier_api_key,
            )
        else:
            return await _handle_non_stream(
                request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
                redaction_key=redaction_key, tier_base_url=tier_base_url, tier_api_key=tier_api_key,
            )
    finally:
        router_active_requests.dec()


async def _handle_non_stream(
    request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
    redaction_key=None, tier_base_url=None, tier_api_key=None,
):
    """Handle non-streaming request."""
    provider = request.app.state.provider

    upstream_start = time.monotonic()
    json_resp, _, model_used, fallback_used, error = await provider.chat_completion(
        payload, fallback_models=fallbacks, stream=False,
        base_url=tier_base_url, api_key=tier_api_key,
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

    # Update response model to router tier label (hide actual upstream model)
    json_resp["model"] = f"smart-router/{route.level.value}"
    _add_model_postfix(json_resp, model_used, route)

    # Privacy middleware: re-hydrate IP placeholders in the LLM output.
    _rehydrate_response_content(
        getattr(request.app.state, "ip_redaction", None), json_resp, redaction_key,
    )

    # Guardrails: mask secrets in the LLM output before serving.
    _guardrail_process_output(request, json_resp)

    # Record upstream prompt-cache usage (cached_tokens / cache_write_tokens)
    cached_tokens, cache_written = extract_cache_usage(json_resp)
    if cached_tokens or cache_written:
        router_prompt_cached_tokens_total.labels(level=route.level.value, model=model_used).inc(cached_tokens)
        router_prompt_cache_writes_total.labels(level=route.level.value, model=model_used).inc(cache_written)
        prompt_tokens = (json_resp.get("usage") or {}).get("prompt_tokens") or 0
        if prompt_tokens:
            ratio = cached_tokens / prompt_tokens
            router_prompt_cache_hit_ratio.labels(level=route.level.value, model=model_used).set(ratio)

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
            "model": f"smart-router/{route.level.value}",
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
    redaction_key=None, tier_base_url=None, tier_api_key=None,
):
    """Handle streaming request — pass through SSE chunks."""
    provider = request.app.state.provider

    upstream_start = time.monotonic()
    _, stream_resp, model_used, fallback_used, error = await provider.chat_completion(
        payload, fallback_models=fallbacks, stream=True,
        base_url=tier_base_url, api_key=tier_api_key,
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
            "model": f"smart-router/{route.level.value}",
            "session_id": session_id,
            "turn": pin.turn_count if pin else 0,
            "classification_source": route.classification.source.value,
            "fallback_used": fallback_used,
        }
        if include_metadata:
            yield f"data: {json.dumps({'router': metadata})}\n\n"

        # Privacy middleware: re-hydrate IP placeholders in streamed content.
        # A placeholder token may be split across SSE chunks, so each chunk
        # passes through a carry buffer that holds back a trailing partial
        # "[ipaddress-..." tail until it can no longer complete a match.
        rehydrate_engine = getattr(request.app.state, "ip_redaction", None)
        guardrail_engine: GuardrailEngine | None = getattr(
            request.app.state, "guardrails", None
        )

        def _split_carry(text: str) -> tuple[str, str]:
            """Split text into (flushable, carry) around a possible partial token tail."""
            # Longest possible tail: "[ipaddress-99]" = 14 chars; be generous.
            for keep in range(min(len(text), 20), 0, -1):
                tail = text[-keep:]
                # Hold back if the tail is a plausible token prefix: starts
                # with '[' or '[`' and contains only token-ish characters.
                if tail.lstrip("`").startswith("[") and re.fullmatch(
                    r"[\[`]?ipaddress\s*-\s*\d{0,2}\]?", tail, re.IGNORECASE
                ):
                    return text[:-keep], tail
            # Guardrails: hold back a trailing partial secret (e.g. a key
            # streamed token-by-token) until it completes or proves benign.
            if _guardrail_mask_active:
                idx = secret_carry_split(text)
                if idx < len(text):
                    return text[:idx], text[idx:]
            return text, ""

        import re as _re

        from app.guardrails.streaming import secret_carry_split
        _PARTIAL_TAIL_RE = _re.compile(r"[\[`]?ipaddress\s*-\s*\d{0,2}\]?$", _re.IGNORECASE)

        def _rehydrate_chunk(payload_text: str, carry: str) -> tuple[str, str]:
            text = carry + payload_text
            if rehydrate_engine is not None and redaction_key and "ipaddress" in text:
                text = rehydrate_engine.rehydrate_text_sync(text, redaction_key)
            # Guardrails: split FIRST — hold any plausible partial-secret tail
            # in the carry — then mask only the flushable part. Masking before
            # the split fires at minimum regex length mid-growth, destroying
            # the marker and leaking the remaining body as plaintext.
            flush, carry = _split_carry(text)
            if guardrail_engine is not None and guardrail_engine.config.output_action == "mask":
                flush, fs = guardrail_engine.mask_secrets(flush)
                for f in fs:
                    router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                # PII masking (email, phone, SSN, credit card)
                if hasattr(guardrail_engine.config, "pii_masking_enabled") and guardrail_engine.config.pii_masking_enabled:
                    flush, pii_fs = guardrail_engine.mask_pii(flush)
                    for f in pii_fs:
                        router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                # Malicious URL masking
                if hasattr(guardrail_engine.config, "malicious_url_detection") and guardrail_engine.config.malicious_url_detection:
                    flush, url_fs = guardrail_engine.mask_malicious_urls(flush)
                    for f in url_fs:
                        router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                # System prompt leak masking
                if hasattr(guardrail_engine.config, "system_prompt_leak_detection") and guardrail_engine.config.system_prompt_leak_detection:
                    flush, spleak_fs = guardrail_engine.mask_system_prompt_leak(flush)
                    for f in spleak_fs:
                        router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
            return flush, carry

        _guardrail_mask_active = (
            guardrail_engine is not None
            and guardrail_engine.config.output_enabled
            and guardrail_engine.config.output_action == "mask"
        )

        async def _rehydrate_line(line: str, carry: str):
            """Re-hydrate content inside a data: SSE line; returns (line, carry)."""
            if not line.startswith("data: "):
                return line, carry
            # Fast-path: skip JSON parse when neither IP rehydration nor
            # guardrail masking applies and no carry buffer is pending.
            if not _guardrail_mask_active and "ipaddress" not in line and not carry:
                return line, carry
            try:
                data = json.loads(line[6:])
            except (ValueError, TypeError):
                return line, carry
            mutated = False
            for choice in data.get("choices", []):
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    flushed, carry = _rehydrate_chunk(delta["content"], carry)
                    if flushed != delta["content"]:
                        delta["content"] = flushed
                        mutated = True
            if not mutated:
                return line, carry
            return f"data: {json.dumps(data)}", carry

        try:
            carry = ""
            async for line in stream_resp.aiter_lines():
                if line.strip() == "data: [DONE]":
                    # Flush any carried partial-token text before finishing.
                    # Mask first: the carry may hold a partial secret that
                    # completed (or an interleaved one) only at stream end.
                    if carry:
                        if _guardrail_mask_active and guardrail_engine is not None:
                            carry, _fs = guardrail_engine.mask_secrets(carry)
                            for f in _fs:
                                router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                            # PII + malicious URL masking on carry flush
                            if hasattr(guardrail_engine.config, "pii_masking_enabled") and guardrail_engine.config.pii_masking_enabled:
                                carry, pii_fs = guardrail_engine.mask_pii(carry)
                                for f in pii_fs:
                                    router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                            if hasattr(guardrail_engine.config, "malicious_url_detection") and guardrail_engine.config.malicious_url_detection:
                                carry, url_fs = guardrail_engine.mask_malicious_urls(carry)
                                for f in url_fs:
                                    router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                            if hasattr(guardrail_engine.config, "system_prompt_leak_detection") and guardrail_engine.config.system_prompt_leak_detection:
                                carry, spleak_fs = guardrail_engine.mask_system_prompt_leak(carry)
                                for f in spleak_fs:
                                    router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
                        flush_event = {"choices": [{"delta": {"content": carry}}]}
                        yield f"data: {json.dumps(flush_event)}\n\n"
                        carry = ""
                    # Postfix must precede [DONE]: OpenAI-compatible clients
                    # stop reading the stream at [DONE] and silently discard
                    # any event emitted after it.
                    postfix_event = {"choices": [{"delta": {"content": f"\n\n[smart-router/{route.level.value}]"}}]}
                    yield f"data: {json.dumps(postfix_event)}\n\n"
                    yield f"{line}\n"
                    break
                line, carry = await _rehydrate_line(line, carry)
                yield f"{line}\n"
        except Exception as e:
            # Stream broke mid-flight (e.g. provider timeout on long-context
            # calls). Log it, count it accurately, and emit a coded error
            # event so clients can distinguish timeout from other breaks.
            upstream_ms = int((time.monotonic() - upstream_start) * 1000)
            is_timeout = isinstance(e, (httpx.TimeoutException, TimeoutError))
            error_kind = "upstream_timeout" if is_timeout else "stream_interrupted"
            logger.error(
                "router.stream.error",
                level=route.level.value,
                model=model_used,
                session_id=session_id,
                error_kind=error_kind,
                error=str(e),
                elapsed_s=round(upstream_ms / 1000, 1),
            )
            router_stream_errors_total.labels(
                level=route.level.value, model=model_used, kind=error_kind,
            ).inc()
            router_requests_total.labels(
                level=route.level.value, model=model_used,
                source=route.classification.source.value,
                status=504 if is_timeout else 502,
            ).inc()
            error_data = {"error": {
                "message": f"stream interrupted: {e!s}",
                "type": "upstream_error",
                "code": f"router_{error_kind}",
            }}
            yield f"data: {json.dumps(error_data)}\n\n"
        else:
            router_requests_total.labels(
                level=route.level.value, model=model_used,
                source=route.classification.source.value, status=200,
            ).inc()
        finally:
            await stream_resp.aclose()

            # Update metrics
            upstream_ms = int((time.monotonic() - upstream_start) * 1000)
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
    response.headers["X-Router-Model"] = f"smart-router/{route.level.value}"
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


def _add_model_postfix(json_resp: dict[str, Any], model_used: str, route: RouteDecision) -> None:
    """Append a compact model marker to assistant content for user visibility."""
    marker = f"[smart-router/{route.level.value}]"
    for choice in json_resp.get("choices", []):
        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            continue
        content = message.get("content")
        if content is None or content == "":
            message["content"] = marker
        elif isinstance(content, str) and marker not in content:
            message["content"] = f"{content.rstrip()}\n\n{marker}"


_MODEL_POSTFIX_RE = re.compile(r"(?:\r?\n){1,2}\[(?:LLM: )?[^\]\r\n]+\]\s*\Z")


def _strip_model_postfix_from_messages(messages: list[dict[str, Any]]) -> None:
    """Remove router-added model markers before messages reach an external LLM."""
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _MODEL_POSTFIX_RE.sub("", content).rstrip()
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    block["text"] = _MODEL_POSTFIX_RE.sub("", block["text"]).rstrip()


async def _classify_only(request, body, task_text, router_opts):
    """Return classification result only, no downstream call."""
    classifier = request.app.state.classifier
    result, _digest_info = await classifier.classify(
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


async def _passthrough(request, body, model, redaction_key=None):
    """Forward as-is to OpenRouter."""
    provider = request.app.state.provider
    config = request.app.state.config.get()
    payload = body.model_dump(exclude={"router"}, exclude_none=True)
    _strip_model_postfix_from_messages(payload.get("messages", []))
    payload["model"] = model
    apply_prompt_cache_features(payload, None, config)

    if body.stream:
        _, stream_resp, _model_used, _, error = await provider.chat_completion(payload, stream=True)
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
        json_resp, _, _model_used, _, error = await provider.chat_completion(payload)
        if error and json_resp is None:
            return JSONResponse(status_code=502, content={"error": {"message": error, "type": "upstream_error"}})
        _rehydrate_response_content(
            getattr(request.app.state, "ip_redaction", None), json_resp, redaction_key,
        )
        _guardrail_process_output(request, json_resp)
        return JSONResponse(content=json_resp)


async def _stateless_classify_and_forward(
    request, body, config, routing_engine, task_text, max_level, min_level, forced_model, bypass_cache,
):
    """Classify in isolation, no session pinning."""
    classifier = request.app.state.classifier

    classification, _digest_info = await classifier.classify(
        body.messages, body.tools, body.response_format,
        task_text=task_text, bypass_cache=bypass_cache,
    )

    level = classification.level or Level.from_str(config.classification.default_level)
    route = routing_engine.resolve(level, classification, max_level=max_level, min_level=min_level, forced_model=forced_model)

    return await _forward_to_provider(
        request, body, route, None, None, None, False, time.monotonic(),
        redaction_key=None,
    )
