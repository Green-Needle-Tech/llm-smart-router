"""POST /v1/chat/completions — the primary endpoint."""
from __future__ import annotations

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
from app.guardrails.streaming import secret_carry_split
from app.middleware.auth import check_router_auth, unauthorized_response
from app.middleware.request_limits import validate_request_bounds
from app.privacy.ip_redaction import IPRedactionEngine
from app.providers.prompt_cache import apply_prompt_cache_features, extract_cache_usage
from app.routing.policy import PolicyViolation, enforce_route_policy
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
from app.session.resolver import _get_api_key_identity, resolve_session_id
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
    router_tier_prefix_pins_total,
    router_tokens_total,
    router_upstream_duration_seconds,
)
from app.telemetry.token_tracker import (
    accumulate as accumulate_tokens,
)
from app.telemetry.token_tracker import (
    build_postfix as build_token_postfix,
)
from app.telemetry.token_tracker import (
    extract_tokens,
)
from app.temporal_awareness.engine import TemporalAwarenessEngine

_SEV_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

router = APIRouter()
logger = get_logger("chat")


def _parse_router_opts(body, request, config) -> dict | JSONResponse:
    """Parse router options from body and headers. Returns opts dict or error response."""
    router_opts = body.router or {}
    task_text = router_opts.get("task_text")
    try:
        max_level = Level.from_str(router_opts["max_level"]) if router_opts.get("max_level") else None
        min_level = Level.from_str(router_opts["min_level"]) if router_opts.get("min_level") else None
        forced_level = Level.from_str(router_opts["level"]) if router_opts.get("level") else None
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": f"Invalid level in router options: {e!s}",
                "type": "invalid_request_error",
                "param": "router",
                "code": "invalid_level",
            }},
        )
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
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": f"Invalid level in X-Router-Level header: {header_level}",
                    "type": "invalid_request_error",
                    "param": "X-Router-Level",
                    "code": "invalid_level",
                }},
            )
    header_model = request.headers.get("X-Router-Model")
    if header_model:
        forced_model = header_model

    return {
        "task_text": task_text,
        "max_level": max_level,
        "min_level": min_level,
        "forced_level": forced_level,
        "forced_model": forced_model,
        "reclassify": reclassify,
        "repin": repin,
        "stateless": stateless,
        "bypass_cache": bypass_cache,
        "include_metadata": include_metadata,
        "router_opts": router_opts,
    }


async def _preprocess_request(request, body, config) -> JSONResponse | None:
    """Run auth, bounds validation, and guardrails.

    Returns a JSONResponse error if any check fails, None if all pass.
    """
    if config.auth.enabled:
        auth_header = request.headers.get("Authorization", "")
        if not check_router_auth(auth_header, config):
            return unauthorized_response()

    bounds_error = validate_request_bounds(body)
    if bounds_error is not None:
        return bounds_error

    guardrail_block = _guardrail_scan_input(request, body)
    if guardrail_block is not None:
        return guardrail_block

    return None


@router.post("/v1/chat/completions", responses={400: {"description": "Bad request"}})
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Handle a chat completion request with session-pinned routing."""
    start = time.monotonic()
    config = request.app.state.config.get()

    preproc_error = await _preprocess_request(request, body, config)
    if preproc_error is not None:
        return preproc_error

    redaction_key = await _redact_incoming(request, body)
    _process_temporal_awareness(request, body)

    routing_engine = request.app.state.routing_engine
    directive = routing_engine.parse_model_directive(body.model)

    opts = _parse_router_opts(body, request, config)
    if isinstance(opts, JSONResponse):
        return opts

    forced_level = opts["forced_level"]
    forced_model = opts["forced_model"]
    max_level = opts["max_level"]
    min_level = opts["min_level"]

    try:
        policy = enforce_route_policy(
            forced_level, forced_model,
            max_level=max_level, min_level=min_level,
            settings=config, allow_overrides=config.routing.allow_client_overrides,
        )
    except PolicyViolation as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {
                "message": e.message, "type": "invalid_request_error",
                "param": None, "code": "policy_violation",
            }},
        )
    forced_level = policy.effective_level if policy.clamped or policy.overridden else forced_level
    if not config.routing.allow_client_overrides:
        forced_model = None

    mode = directive["mode"]
    if mode == "classify_only":
        return await _classify_only(request, body, opts["task_text"], opts["router_opts"])
    if mode == "passthrough":
        if not routing_engine.is_passthrough_allowed(directive["model"]):
            raise HTTPException(status_code=400, detail=f"Passthrough not allowed for model: {directive['model']}")
        return await _passthrough(request, body, directive["model"], redaction_key=redaction_key)
    if mode == "stateless" or opts["stateless"] or not config.session.enabled:
        return await _stateless_classify_and_forward(
            request, body, config, routing_engine, opts["task_text"],
            max_level, min_level, forced_model, opts["bypass_cache"],
        )
    if mode == "level":
        forced_level = directive["level"]

    return await _session_pinned_route(
        request, body, config, routing_engine,
        directive, forced_level, forced_model, max_level, min_level,
        opts["reclassify"], opts["repin"], opts["task_text"],
        opts["bypass_cache"], opts["include_metadata"], start,
        redaction_key=redaction_key,
    )


def _build_guardrail_engine(cfg) -> GuardrailEngine:
    """Create a per-request GuardrailEngine with its own config snapshot."""
    return GuardrailEngine(GuardrailConfig(
        input_enabled=cfg.input_enabled,
        input_action=cfg.input_action,
        block_on_severity=cfg.block_on_severity,
        output_enabled=cfg.output_enabled,
        output_action=cfg.output_action,
        invisible_text_detection=cfg.invisible_text_detection,
        pii_masking_enabled=cfg.pii_masking_enabled,
        input_pii_masking_enabled=getattr(cfg, "input_pii_masking_enabled", True),
        banned_substrings=cfg.banned_substrings,
        refusal_detection=cfg.refusal_detection,
        malicious_url_detection=cfg.malicious_url_detection,
        system_prompt_leak_detection=getattr(cfg, "system_prompt_leak_detection", False),
        system_prompt_fragments=getattr(cfg, "system_prompt_fragments", []),
        system_prompt_leak_threshold=getattr(cfg, "system_prompt_leak_threshold", 0.85),
        homoglyph_normalization=getattr(cfg, "homoglyph_normalization", True),
        obfuscation_detection=getattr(cfg, "obfuscation_detection", True),
        entropy_threshold=getattr(cfg, "entropy_threshold", 4.5),
    ))


def _scan_message_contents(messages: list, scanner_fn) -> list:
    """Apply a scanner function to all message content (str or block list)."""
    findings = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            findings.extend(scanner_fn(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    findings.extend(scanner_fn(block["text"]))
    return findings


def _log_findings(findings: list, log_event: str, direction: str = "input", **extra) -> None:
    """Log guardrail findings and increment metrics."""
    for f in findings:
        router_guardrail_findings_total.labels(
            rule_id=f.rule_id, severity=f.severity, direction=direction,
        ).inc()
        log_method = logger.info if "masked" in log_event else logger.warning
        log_method(log_event, rule=f.rule_id, severity=f.severity, **extra)


def _scan_invisible_text(messages, engine, cfg) -> list:
    """Detect and strip invisible/zero-width characters from messages."""
    if not cfg.invisible_text_detection:
        return []
    findings = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            ifs = engine.scan_invisible_text(content)
            if ifs:
                findings.extend(ifs)
                msg["content"] = strip_invisible_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    ifs = engine.scan_invisible_text(block["text"])
                    if ifs:
                        findings.extend(ifs)
                        block["text"] = strip_invisible_text(block["text"])
    return findings


def _scan_input_pii(messages, engine, cfg) -> list:
    """Mask PII and secrets in input messages before forwarding upstream."""
    if not getattr(cfg, "input_pii_masking_enabled", True):
        return []
    findings = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            masked, pii_fs = engine.mask_input_sensitive(content)
            if masked != content:
                msg["content"] = masked
                findings.extend(pii_fs)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    masked, pii_fs = engine.mask_input_sensitive(block["text"])
                    if masked != block["text"]:
                        block["text"] = masked
                        findings.extend(pii_fs)
    return findings


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
    engine = _build_guardrail_engine(cfg)
    messages = [
        (m.model_dump() if hasattr(m, "model_dump") else m) for m in body.messages
    ]

    # Invisible text detection + stripping
    invisible_findings = _scan_invisible_text(messages, engine, cfg)
    _log_findings(invisible_findings, "router.guardrail.invisible_text")

    # Input PII + secret masking
    input_pii_findings = _scan_input_pii(messages, engine, cfg)
    _log_findings(input_pii_findings, "router.guardrail.input_sensitive_masked")

    # Write normalized content back onto the original pydantic body.messages
    for orig, dumped in zip(body.messages, messages, strict=False):
        new_content = dumped.get("content")
        if orig.content != new_content:
            orig.content = new_content

    # Banned substrings detection
    banned_findings = _scan_message_contents(
        messages, engine.scan_banned_substrings
    ) if cfg.banned_substrings else []
    _log_findings(banned_findings, "router.guardrail.banned_substring", action=cfg.input_action)

    # Obfuscation and high-entropy detection
    obfuscation_findings = _scan_message_contents(
        messages, engine.scan_obfuscation
    ) if getattr(cfg, "obfuscation_detection", True) else []
    _log_findings(obfuscation_findings, "router.guardrail.obfuscation", action=cfg.input_action)

    # Standard injection scan
    result = engine.scan_messages(messages)
    _log_findings(result.findings, "router.guardrail.input_finding", action=cfg.input_action)

    # Combine all findings for block decision
    all_findings = result.findings + banned_findings + obfuscation_findings
    return _check_guardrail_block(cfg, all_findings)


def _check_guardrail_block(cfg, all_findings):
    """Return a block response if configured to block and findings exceed threshold."""
    if cfg.input_action != "block" or not all_findings:
        return None
    threshold = _SEV_ORDER.get(cfg.block_on_severity, 2)
    if not any(_SEV_ORDER.get(f.severity, 0) >= threshold for f in all_findings):
        return None
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
def _process_guardrail_finding(f, engine) -> None:
    """Log and count a single output guardrail finding."""
    if f.rule_id.startswith("refusal-"):
        router_guardrail_findings_total.labels(
            rule_id=f.rule_id, severity=f.severity, direction="output",
        ).inc()
        logger.info("router.guardrail.refusal_detected", rule=f.rule_id)
        return
    _NON_SECRET_PREFIXES = ("pii-",)
    _NON_SECRET_IDS = {"malicious-url", "output-system-prompt-leak"}
    is_non_secret = f.rule_id.startswith(_NON_SECRET_PREFIXES) or f.rule_id in _NON_SECRET_IDS
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
    # Reference is_non_secret to avoid unused-variable warnings; the
    # distinction is preserved for future per-type metric splitting.
    _ = is_non_secret


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
            _process_guardrail_finding(f, engine)


async def _redact_incoming(request, body) -> str | None:
    """Run IP redaction over the incoming messages if enabled.

    Returns the re-hydration key (session id), or None when the privacy
    module is disabled. The key mirrors the routing session resolution so
    re-hydration on the response side looks up the same mapping bucket.

    The session key is namespaced by API key identity to prevent cross-tenant
    IP mapping leakage.
    """
    engine: IPRedactionEngine | None = getattr(
        request.app.state, "ip_redaction", None
    )
    if engine is None:
        return None
    headers_dict = {k.lower(): v for k, v in request.headers.items()}
    config = request.app.state.config.get()
    auth_header = headers_dict.get("authorization", "")
    key_identity = _get_api_key_identity(auth_header)
    session_id, _ = resolve_session_id(
        body, headers_dict, config, config.session.fingerprint_salt,
        api_key_identity=key_identity,
        namespace=getattr(config.session, "namespace_by_api_key", False),
    )
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


def _rehydrate_content_value(engine, content, key: str):
    """Re-hydrate a message content value (str or block list) in place."""
    if isinstance(content, str):
        return engine.rehydrate_text_sync(content, key)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"] = engine.rehydrate_text_sync(block["text"], key)
    return content


def _rehydrate_response_content(engine, json_resp, key: str | None) -> None:
    """Re-hydrate placeholders in a non-streaming response in place (sync path)."""
    if engine is None or key is None:
        return
    for choice in json_resp.get("choices", []):
        message = choice.get("message")
        if isinstance(message, dict):
            message["content"] = _rehydrate_content_value(
                engine, message.get("content"), key
            )


def _detect_tier_prefix(body, config) -> Level | None:
    """Detect a tier label (L1–L5) at the start of the first user prompt.

    When the user's opening message begins with a tier identifier like
    "L4 explain quantum computing", the session is pinned directly to
    that tier, bypassing the classifier LLM entirely.

    If strip_prefix is enabled and meaningful content remains after the
    prefix, the prefix is removed from the message content before
    forwarding upstream.

    Returns the detected Level, or None.
    """
    tier_cfg = getattr(config.classification, "tier_prefix", None)
    if tier_cfg is None or not getattr(tier_cfg, "enabled", False):
        return None

    pattern_str = getattr(tier_cfg, "pattern", r"^(L[1-5])[\s:.\-]")
    strip_prefix = getattr(tier_cfg, "strip_prefix", True)

    # Find the first user message
    first_user_msg = None
    for msg in body.messages:
        if msg.role == "user":
            first_user_msg = msg
            break

    if first_user_msg is None:
        return None

    content = first_user_msg.content
    if not isinstance(content, str) or not content.strip():
        return None

    match = re.match(pattern_str, content.strip(), re.IGNORECASE)
    if match is None:
        return None

    level_str = match.group(1)
    try:
        level = Level.from_str(level_str)
    except ValueError:
        return None

    # Strip the prefix from the message if configured and content remains
    if strip_prefix:
        stripped = content.strip()[match.end():].strip()
        if stripped:
            first_user_msg.content = stripped

    router_tier_prefix_pins_total.labels(level=level.value).inc()
    logger.info(
        "router.tier_prefix.detected",
        level=level.value,
        strip_prefix=strip_prefix,
    )
    return level


async def _handle_session_hit(request, body, pin, config, routing_engine, forced_level, forced_model, max_level, min_level, repin, include_metadata, start, session_id, session_source, redaction_key):
    """Handle a session pin hit — use the pinned model with overrides."""
    store = request.app.state.session_store
    pin.turn_count += 1
    pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)

    effective_level = pin.level
    effective_model = _resolve_effective_model(pin, routing_engine, config)

    if forced_level and repin:
        pin.level = forced_level
        pin.model = routing_engine.resolve_model_for_level(forced_level)
        effective_level = forced_level
        effective_model = pin.model
    elif forced_level:
        effective_level = forced_level
        effective_model = routing_engine.resolve_model_for_level(forced_level)
    if forced_model:
        effective_model = forced_model
    if max_level and effective_level > max_level:
        effective_level = max_level
        effective_model = routing_engine.resolve_model_for_level(effective_level)
    if min_level and effective_level < min_level:
        effective_level = min_level
        effective_model = routing_engine.resolve_model_for_level(effective_level)

    await store.put(pin)

    classification = ClassificationResult(
        level=effective_level, confidence=1.0, reason="session pin",
        source=ClassificationSource.SESSION, latency_ms=0,
    )

    if config.session.escalation.enabled and config.session.escalation.free_signals_enabled:
        escalation_result = _check_escalation_signals(body, pin, config)
        if escalation_result:
            effective_level, effective_model = escalation_result
            await store.put(pin)

    route = RouteDecision(
        level=effective_level, model=effective_model,
        params=pin.params, classification=classification,
    )
    return await _forward_to_provider(
        request, body, route, session_id, session_source, pin, include_metadata, start,
        redaction_key=redaction_key,
    )


async def _classify_and_pin(request, body, config, classifier, routing_engine, forced_level, task_text, bypass_cache, start, session_id):
    """Run classification (forced level, tier-prefix, cache, or LLM) and return (level, classification, cache_key)."""
    if forced_level:
        return forced_level, ClassificationResult(
            level=forced_level, confidence=1.0, reason="forced override",
            source=ClassificationSource.OVERRIDE,
            latency_ms=int((time.monotonic() - start) * 1000),
        ), None

    prefix_level = _detect_tier_prefix(body, config)
    if prefix_level is not None:
        return prefix_level, ClassificationResult(
            level=prefix_level, confidence=1.0, reason="tier-prefix pin",
            source=ClassificationSource.OVERRIDE,
            latency_ms=int((time.monotonic() - start) * 1000),
        ), None

    cache = request.app.state.classification_cache
    cache_key = None
    cached = None
    if not bypass_cache and config.classification.cache.enabled:
        digest_info = classifier.digest_builder.build(
            messages=body.messages, tools=body.tools,
            response_format=body.response_format, task_text=task_text,
        )
        cache_key = classifier.digest_builder.digest_hash(
            digest_info["digest"], config.classification.model,
            config.classification.rubric_version,
        )
        cached = await cache.get(cache_key)
        if cached:
            router_cache_events_total.labels(result="hit").inc()
            level = Level.from_str(cached["level"])
            return level, ClassificationResult(
                level=level, confidence=cached.get("confidence", 1.0),
                reason=cached.get("reason", ""),
                source=ClassificationSource.CACHE,
                latency_ms=int((time.monotonic() - start) * 1000),
            ), cache_key
        router_cache_events_total.labels(result="miss").inc()
    else:
        router_cache_events_total.labels(result="bypass").inc()

    classification, _digest = await classifier.classify(
        body.messages, body.tools, body.response_format,
        task_text=task_text, bypass_cache=bypass_cache,
    )
    level = classification.level or Level.from_str(config.classification.default_level)
    return level, classification, cache_key


async def _handle_lock_race_won(
    store, session_id, existing_pin, config, routing_engine,
    request, body, include_metadata, start, redaction_key,
    *, session_source=None,
):
    """Handle the case where we lost the lock but another session has a pin."""
    pin = existing_pin
    pin.turn_count += 1
    pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)
    await store.put(pin)
    classification = ClassificationResult(
        level=pin.level, confidence=1.0, reason="session pin (race resolved)",
        source=ClassificationSource.SESSION, latency_ms=0,
    )
    route = RouteDecision(
        level=pin.level, model=pin.model, params=pin.params, classification=classification,
    )
    return await _forward_to_provider(
        request, body, route, session_id, session_source, pin, include_metadata, start,
        redaction_key=redaction_key,
    )


async def _handle_lock_timeout(
    store, session_id, config, routing_engine,
    request, body, max_level, min_level, include_metadata, start, redaction_key,
):
    """Handle the case where we lost the lock and no existing pin was found."""
    default_level = Level.from_str(config.classification.default_level)
    classification = ClassificationResult(
        level=default_level, confidence=0.0, reason="lock timeout",
        source=ClassificationSource.DEFAULT,
        latency_ms=int((time.monotonic() - start) * 1000),
    )
    route = routing_engine.resolve(default_level, classification, max_level=max_level, min_level=min_level)
    return await _forward_to_provider(
        request, body, route, session_id, None, include_metadata, start,
        redaction_key=redaction_key,
    )


async def _create_and_pin_session(
    request, body, config, classifier, routing_engine, store,
    forced_level, task_text, bypass_cache, start, session_id, session_source,
    max_level, min_level, include_metadata, redaction_key,
):
    """Classify, create a new session pin, and forward to provider."""
    level, classification, cache_key = await _classify_and_pin(
        request, body, config, classifier, routing_engine, forced_level, task_text, bypass_cache, start, session_id,
    )

    # Write to cache if we ran the classifier
    if cache_key and config.classification.cache.enabled and classification.source != ClassificationSource.CACHE:
        cache = request.app.state.classification_cache
        await cache.put(cache_key, {
            "level": level.value,
            "confidence": classification.confidence,
            "reason": classification.reason,
            "source": classification.source.value,
        })

    effective_level = level
    if max_level and effective_level > max_level:
        effective_level = max_level
    if min_level and effective_level < min_level:
        effective_level = min_level

    model = routing_engine.resolve_model_for_level(effective_level)
    params = config.routing.get_params(effective_level.value)

    is_unknown = classification.reason and "UNKNOWN" in classification.reason.upper()
    status = SessionStatus.PROVISIONAL if is_unknown else SessionStatus.PINNED

    new_pin = SessionPin(
        session_id=session_id, level=effective_level, model=model,
        params=params, status=status, classification=classification, turn_count=1,
    )
    if not is_unknown:
        new_pin.escalation.original_level = effective_level
    new_pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)

    await store.release(session_id)
    await store.put(new_pin)

    router_sessions_created_total.labels(level=effective_level.value, id_source=session_source.value).inc()
    router_sessions_active.labels(level=effective_level.value).inc()

    route = RouteDecision(
        level=effective_level, model=model, params=params, classification=classification,
    )
    return await _forward_to_provider(
        request, body, route, session_id, session_source, new_pin, include_metadata, start,
        redaction_key=redaction_key,
    )


def _resolve_session(request, body, config):
    """Resolve session ID from request."""
    headers_dict = {k.lower(): v for k, v in request.headers.items()}
    auth_header = headers_dict.get("authorization", "")
    key_identity = _get_api_key_identity(auth_header)
    return resolve_session_id(
        body, headers_dict, config, config.session.fingerprint_salt,
        api_key_identity=key_identity,
        namespace=getattr(config.session, "namespace_by_api_key", False),
    )


async def _lookup_pin(store, session_id, config):
    """Look up and validate a session pin. Returns pin or None."""
    pin = await store.get(session_id)
    if pin is not None and pin.status != SessionStatus.CLASSIFYING:
        expired_reason = check_expiry(pin)
        if expired_reason or check_turn_cap(pin, config.session.max_turns):
            await store.delete(session_id)
            pin = None
            router_session_lookups_total.labels(result="miss").inc()
        else:
            router_session_lookups_total.labels(result="hit").inc()
    return pin


async def _session_pinned_route(
    request, body, config, routing_engine,
    directive, forced_level, forced_model, max_level, min_level,
    reclassify, repin, task_text, bypass_cache, include_metadata, start,
    redaction_key=None,
):
    """The main session-pinned routing path."""
    store = request.app.state.session_store
    classifier = request.app.state.classifier

    session_id, session_source = _resolve_session(request, body, config)
    if session_id is None:
        router_session_lookups_total.labels(result="disabled").inc()
        return await _stateless_classify_and_forward(
            request, body, config, routing_engine, task_text, max_level, min_level, forced_model, bypass_cache
        )

    pin = await _lookup_pin(store, session_id, config)
    if pin is not None and not reclassify:
        return await _handle_session_hit(
            request, body, pin, config, routing_engine, forced_level, forced_model,
            max_level, min_level, repin, include_metadata, start,
            session_id, session_source, redaction_key,
        )

    router_session_lookups_total.labels(result="miss").inc()

    won, existing_pin = await acquire_or_wait(
        store, session_id,
        ttl_seconds=config.session.lock_reservation_seconds,
        wait_ms=config.session.lock_wait_ms,
    )

    if not won and existing_pin is not None:
        return await _handle_lock_race_won(
            store, session_id, existing_pin, config, routing_engine,
            request, body, include_metadata, start, redaction_key,
            session_source=session_source,
        )

    if not won and existing_pin is None:
        return await _handle_lock_timeout(
            store, session_id, config, routing_engine,
            request, body, max_level, min_level, include_metadata, start, redaction_key,
        )

    return await _create_and_pin_session(
        request, body, config, classifier, routing_engine, store,
        forced_level, task_text, bypass_cache, start, session_id, session_source,
        max_level, min_level, include_metadata, redaction_key,
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


_REPAIR_LANGUAGE_RE = re.compile(
    r"\b(no,|that's wrong|still (failing|broken)|doesn't work|try again|not what I|incorrect|you missed)\b",
    re.IGNORECASE,
)
_DEEP_KEYWORDS_RE = re.compile(
    r"\b(architect|design a system|prove|derive|refactor the|threat model|optimize the algorithm)\b",
    re.IGNORECASE,
)
_STRONG_ESCALATION_SIGNALS = {"repair_language", "tool_error_loop"}


def _detect_escalation_signals(raw_user_text, pin, esc_cfg) -> list[tuple[str, int]]:
    """Detect escalation signals from user text and return (signal, weight) pairs."""
    signals = []
    if _REPAIR_LANGUAGE_RE.search(raw_user_text):
        signals.append(("repair_language", esc_cfg.signal_weights.get("repair_language", 3)))
    if _DEEP_KEYWORDS_RE.search(raw_user_text):
        signals.append(("deep_keywords", esc_cfg.signal_weights.get("deep_keywords", 2)))
    if pin.turn_count >= esc_cfg.escalate_after_turns:
        signals.append(("turn_depth", esc_cfg.signal_weights.get("turn_depth", 1)))
    return [(s, w) for s, w in signals if w > 0]


def _apply_escalation(pin, signals_fired, esc_cfg, config) -> tuple[Level, str] | None:
    """Apply escalation if threshold/cooldown conditions are met."""
    if not (pin.escalation.score >= esc_cfg.threshold and
            pin.escalation.count < esc_cfg.max_escalations_per_session and
            pin.turn_count >= pin.escalation.cooldown_until_turn):
        return None
    new_level = Level.from_numeric(pin.level.numeric + 1)
    if new_level > Level.from_str(config.routing.global_max_level):
        return None
    # L5 guard: require a strong signal for the final tier
    has_strong = any(s in _STRONG_ESCALATION_SIGNALS for s, _ in signals_fired)
    if new_level == Level.L5 and not has_strong:
        return None
    old_level = pin.level
    pin.level = new_level
    pin.model = config.routing.get_model(new_level.value)
    pin.params = config.routing.get_params(new_level.value)
    pin.escalation.count += 1
    pin.escalation.last_escalated_turn = pin.turn_count
    pin.escalation.cooldown_until_turn = pin.turn_count + esc_cfg.cooldown_turns
    pin.escalation.score = 0
    pin.escalation.last_trigger = [s for s, _ in signals_fired]
    if pin.escalation.original_level is None:
        pin.escalation.original_level = old_level
    router_escalation_turn.observe(float(pin.turn_count))
    router_escalations_total.labels(
        from_level=old_level.value, to_level=new_level.value,
        trigger="free_signal", layer="free_signal",
    ).inc()
    return (new_level, pin.model)


def _check_escalation_signals(body, pin, config) -> tuple[Level, str] | None:
    """Check free-signal escalation. Returns (new_level, new_model) if escalated."""
    esc_cfg = config.session.escalation
    if not esc_cfg.enabled:
        return None
    raw_user_text = _extract_raw_user_text(body)
    signals_fired = _detect_escalation_signals(raw_user_text, pin, esc_cfg)
    if not signals_fired:
        pin.escalation.score = max(0, pin.escalation.score - esc_cfg.decay_per_turn)
        return None
    for signal, weight in signals_fired:
        pin.escalation.score += weight
        router_escalation_signals_total.labels(signal=signal).inc()
    return _apply_escalation(pin, signals_fired, esc_cfg, config)


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


async def _check_budget(request, body, route, session_id, config) -> JSONResponse | tuple[RouteDecision, None]:
    """Pre-request budget check. Returns error response or (possibly-downgraded) route."""
    budget_mgr = getattr(request.app.state, "budget_manager", None)
    if budget_mgr is None or not config.budget.enabled:
        return None
    tier_max_cost = config.routing.get_tier(route.level.value).max_cost_per_request_usd
    decision = await budget_mgr.check_and_reserve(
        session_id=session_id,
        model=route.model,
        messages=body.messages,
        tier_max_cost_usd=tier_max_cost,
        daily_limit_usd=config.budget.daily_limit_usd,
        on_exceeded=config.budget.on_exceeded,
        downgrade_to=config.budget.downgrade_to,
    )
    if not decision.allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {
                "message": f"Budget limit exceeded: {decision.reason}",
                "type": "budget_exceeded",
                "code": "budget_exceeded",
            }},
        )
    if decision.downgrade_level:
        from app.schemas.router import Level as _L
        try:
            dl = _L.from_str(decision.downgrade_level)
            return RouteDecision(
                level=dl,
                model=config.routing.get_model(dl.value),
                params=config.routing.get_params(dl.value),
                classification=route.classification,
            )
        except ValueError:
            pass
    return None


def _build_upstream_payload(body, route, session_id, config, provider) -> dict:
    """Build the upstream provider payload with tier params and max_tokens."""
    payload = body.model_dump(exclude={"router"}, exclude_none=True)
    _strip_model_postfix_from_messages(payload.get("messages", []))
    payload["model"] = route.model
    apply_prompt_cache_features(payload, session_id, config)
    for key, val in route.params.items():
        if key not in payload or payload.get(key) is None:
            payload[key] = val
    _apply_max_tokens(payload, route, config, provider)
    if body.stream:
        payload.setdefault("stream_options", {})["include_usage"] = True
    return payload


def _apply_max_tokens(payload, route, config, provider) -> None:
    """Resolve max_tokens: 'auto' uses OpenRouter-detected value; int uses as-is."""
    tier_max_tokens = config.routing.get_max_tokens(route.level.value)
    if isinstance(tier_max_tokens, str) and tier_max_tokens == "auto":
        detected = provider.get_max_completion_tokens(route.model)
        if detected is not None and detected > 0:
            payload["max_tokens"] = detected
    elif isinstance(tier_max_tokens, int) and tier_max_tokens > 0:
        payload["max_tokens"] = tier_max_tokens


async def _forward_to_provider(
    request, body, route, session_id, session_source, pin, include_metadata, start,
    redaction_key=None,
):
    """Forward the request to the provider and return the response."""
    config = request.app.state.config.get()
    provider = request.app.state.provider

    budget_result = await _check_budget(request, body, route, session_id, config)
    if budget_result is not None:
        if isinstance(budget_result, JSONResponse):
            return budget_result
        route = budget_result  # downgraded route

    payload = _build_upstream_payload(body, route, session_id, config, provider)
    fallbacks = config.routing.get_fallbacks(route.level.value)
    tier_base_url, tier_api_key = _resolve_tier_provider(config, route.level.value)

    router_active_requests.inc()
    try:
        if body.stream:
            return await _handle_stream(
                request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
                redaction_key=redaction_key, tier_base_url=tier_base_url, tier_api_key=tier_api_key,
            )
        return await _handle_non_stream(
            request, payload, route, fallbacks, session_id, session_source, pin, include_metadata, start,
            redaction_key=redaction_key, tier_base_url=tier_base_url, tier_api_key=tier_api_key,
        )
    finally:
        router_active_requests.dec()


def _record_cache_usage(json_resp, route, model_used) -> None:
    """Record upstream prompt-cache usage metrics."""
    cached_tokens, cache_written = extract_cache_usage(json_resp)
    if not (cached_tokens or cache_written):
        return
    router_prompt_cached_tokens_total.labels(level=route.level.value, model=model_used).inc(cached_tokens)
    router_prompt_cache_writes_total.labels(level=route.level.value, model=model_used).inc(cache_written)
    prompt_tokens_cache = (json_resp.get("usage") or {}).get("prompt_tokens") or 0
    if prompt_tokens_cache:
        ratio = cached_tokens / prompt_tokens_cache
        router_prompt_cache_hit_ratio.labels(level=route.level.value, model=model_used).set(ratio)


def _update_pin_metrics(pin, provider, route, model_used, prompt_tokens, completion_tokens) -> None:
    """Update pin cost and token metrics."""
    if not pin:
        return
    cost = provider.estimate_cost(model_used, prompt_tokens, completion_tokens)
    if cost is not None:
        pin.cost_usd_total += cost
        router_cost_usd_total.labels(level=route.level.value, model=model_used).inc(cost)
    router_tokens_total.labels(level=route.level.value, model=model_used, kind="prompt").inc(prompt_tokens)
    router_tokens_total.labels(level=route.level.value, model=model_used, kind="completion").inc(completion_tokens)


def _record_request_metrics(route, model_used, fallback_used, upstream_ms) -> None:
    """Record request-level Prometheus metrics."""
    router_requests_total.labels(
        level=route.level.value, model=model_used,
        source=route.classification.source.value, status=200,
    ).inc()
    router_upstream_duration_seconds.labels(level=route.level.value, model=model_used).observe(upstream_ms / 1000)
    if fallback_used:
        router_fallbacks_total.labels(
            level=route.level.value, from_model=route.model, to_model=model_used, reason="fallback",
        ).inc()


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
        status = 502 if "exhausted" in error else 504
        error_type = "upstream_error" if status == 502 else "upstream_timeout"
        return JSONResponse(
            status_code=status,
            content={"error": {
                "message": error, "type": error_type, "param": None, "code": f"router_upstream_{status}",
            }},
        )

    json_resp["model"] = f"smart-router/{route.level.value}"

    config = request.app.state.config.get()
    prompt_tokens, completion_tokens = _apply_token_tracking(
        json_resp, pin, route, model_used, config)

    await _post_process_response(
        request, json_resp, route, model_used, pin, provider,
        prompt_tokens, completion_tokens, session_id, redaction_key, upstream_ms, fallback_used)

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

    response = JSONResponse(content=json_resp)
    _add_router_headers(response, route, session_id, session_source, pin, total_ms, fallback_used)
    return response


def _apply_token_tracking(json_resp, pin, route, model_used, config):
    """Apply token tracking postfix to response. Returns (prompt_tokens, completion_tokens)."""
    tt_cfg = getattr(getattr(config, "telemetry", None), "token_tracking", None)
    tt_enabled = tt_cfg is not None and getattr(tt_cfg, "enabled", True)
    tt_show = tt_cfg is not None and getattr(tt_cfg, "show_in_postfix", True)
    usage = json_resp.get("usage", {})
    prompt_tokens, completion_tokens = extract_tokens(usage)
    if tt_enabled:
        if pin is not None:
            accumulate_tokens(pin.token_usage, route.level.value, prompt_tokens, completion_tokens)
        token_usage_for_postfix = pin.token_usage if pin is not None else {
            route.level.value: {"prompt": prompt_tokens, "completion": completion_tokens}
        }
        _add_model_postfix(json_resp, model_used, route, token_usage_for_postfix, tt_show)
    else:
        _add_model_postfix(json_resp, model_used, route)
    return prompt_tokens, completion_tokens


async def _post_process_response(
    request, json_resp, route, model_used, pin, provider,
    prompt_tokens, completion_tokens, session_id, redaction_key, upstream_ms, fallback_used,
):
    """Run all post-response processing: rehydration, guardrails, metrics, budget."""
    _rehydrate_response_content(
        getattr(request.app.state, "ip_redaction", None), json_resp, redaction_key,
    )
    _guardrail_process_output(request, json_resp)
    _record_cache_usage(json_resp, route, model_used)
    _update_pin_metrics(pin, provider, route, model_used, prompt_tokens, completion_tokens)
    if pin:
        await request.app.state.session_store.put(pin)
    budget_mgr = getattr(request.app.state, "budget_manager", None)
    if budget_mgr is not None:
        await budget_mgr.reconcile(session_id, model_used, prompt_tokens, completion_tokens)
    _record_request_metrics(route, model_used, fallback_used, upstream_ms)


def _mask_carry_flush(carry, guardrail_engine) -> str:
    """Mask secrets, PII, URLs, and system-prompt leaks in the carry buffer at stream end."""
    if not carry:
        return carry
    if guardrail_engine is None:
        return carry
    carry, _fs = guardrail_engine.mask_secrets(carry)
    for f in _fs:
        router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
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
    return carry


def _process_stream_chunk(chunk, usage_holder, tool_call_holder) -> None:
    """Extract usage and tool-call signals from a stream chunk."""
    if not isinstance(chunk, dict):
        return
    if chunk.get("usage"):
        usage_holder["usage"] = chunk["usage"]
    for ch in chunk.get("choices", []):
        delta = ch.get("delta") or {}
        if delta.get("tool_calls") or ch.get("finish_reason") == "tool_calls":
            tool_call_holder["had_tool_calls"] = True


def _mask_stream_flush(flush, guardrail_engine) -> str:
    """Apply all guardrail masks to the flushable portion of a stream chunk."""
    if guardrail_engine is None or guardrail_engine.config.output_action != "mask":
        return flush
    flush, fs = guardrail_engine.mask_secrets(flush)
    for f in fs:
        router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
    if hasattr(guardrail_engine.config, "pii_masking_enabled") and guardrail_engine.config.pii_masking_enabled:
        flush, pii_fs = guardrail_engine.mask_pii(flush)
        for f in pii_fs:
            router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
    if hasattr(guardrail_engine.config, "malicious_url_detection") and guardrail_engine.config.malicious_url_detection:
        flush, url_fs = guardrail_engine.mask_malicious_urls(flush)
        for f in url_fs:
            router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
    if hasattr(guardrail_engine.config, "system_prompt_leak_detection") and guardrail_engine.config.system_prompt_leak_detection:
        flush, spleak_fs = guardrail_engine.mask_system_prompt_leak(flush)
        for f in spleak_fs:
            router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
    return flush


def _handle_stream_error(e, route, model_used, session_id, upstream_start) -> dict:
    """Handle a mid-stream exception: log, increment metrics, return error data."""
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
    return {"error": {
        "message": f"stream interrupted: {e!s}",
        "type": "upstream_error",
        "code": f"router_{error_kind}",
    }}


def _build_stream_postfix(route, pin, _stream_usage, _tt_enabled, _tt_show) -> str:
    """Build the postfix text for the end of a stream."""
    if _tt_enabled and _stream_usage is not None:
        s_prompt, s_completion = extract_tokens(_stream_usage)
        token_usage_for_postfix = pin.token_usage if pin is not None else {
            route.level.value: {"prompt": s_prompt, "completion": s_completion}
        }
        return build_token_postfix(route.level.value, token_usage_for_postfix, _tt_show)
    return f"[smart-router/{route.level.value}]"


async def _finalize_stream_token_tracking(
    pin, route, model_used, provider, _stream_usage, request,
) -> None:
    """Accumulate token usage and update pin cost/metrics at stream end."""
    if pin is None or _stream_usage is None:
        return
    s_prompt, s_completion = extract_tokens(_stream_usage)
    accumulate_tokens(pin.token_usage, route.level.value, s_prompt, s_completion)
    cost = provider.estimate_cost(model_used, s_prompt, s_completion)
    if cost is not None:
        pin.cost_usd_total += cost
        router_cost_usd_total.labels(level=route.level.value, model=model_used).inc(cost)
    router_tokens_total.labels(level=route.level.value, model=model_used, kind="prompt").inc(s_prompt)
    router_tokens_total.labels(level=route.level.value, model=model_used, kind="completion").inc(s_completion)
    await request.app.state.session_store.put(pin)


def _record_stream_success(route, model_used) -> None:
    """Record successful stream completion metrics."""
    router_requests_total.labels(
        level=route.level.value, model=model_used,
        source=route.classification.source.value, status=200,
    ).inc()


def _record_stream_finally(route, model_used, upstream_start, fallback_used) -> None:
    """Record final upstream duration and fallback metrics."""
    upstream_ms = int((time.monotonic() - upstream_start) * 1000)
    router_upstream_duration_seconds.labels(level=route.level.value, model=model_used).observe(upstream_ms / 1000)
    if fallback_used:
        router_fallbacks_total.labels(
            level=route.level.value, from_model=route.model, to_model=model_used, reason="fallback",
        ).inc()


def _process_data_line(line, stream_usage, had_tool_calls) -> tuple:
    """Extract usage and tool-call signals from a data SSE line."""
    try:
        _chunk = json.loads(line[6:])
        _holder = {"usage": None, "had_tool_calls": had_tool_calls}
        _process_stream_chunk(_chunk, _holder, _holder)
        if _holder.get("usage"):
            stream_usage = _holder["usage"]
        if _holder.get("had_tool_calls"):
            had_tool_calls = True
    except (ValueError, TypeError):
        pass
    return stream_usage, had_tool_calls



_PARTIAL_TAIL_RE = re.compile(r"[\[`]?ipaddress\s*-\s*\d{0,2}\]?$", re.IGNORECASE)


def _split_carry(text: str, guardrail_mask_active: bool) -> tuple[str, str]:
    """Split text into (flushable, carry) around a possible partial token tail."""
    for keep in range(min(len(text), 20), 0, -1):
        tail = text[-keep:]
        if tail.lstrip("`").startswith("[") and re.fullmatch(
            r"[\[`]?ipaddress\s*-\s*\d{0,2}\]?", tail, re.IGNORECASE
        ):
            return text[:-keep], tail
    if guardrail_mask_active:
        idx = secret_carry_split(text)
        if idx < len(text):
            return text[:idx], text[idx:]
    return text, ""


def _rehydrate_chunk(
    payload_text: str, carry: str,
    rehydrate_engine, redaction_key, guardrail_engine, guardrail_mask_active,
) -> tuple[str, str]:
    """Re-hydrate and mask a chunk of streaming text."""
    text = carry + payload_text
    if rehydrate_engine is not None and redaction_key and "ipaddress" in text:
        text = rehydrate_engine.rehydrate_text_sync(text, redaction_key)
    flush, carry = _split_carry(text, guardrail_mask_active)
    flush = _mask_stream_flush(flush, guardrail_engine) if guardrail_engine else flush
    return flush, carry


async def _rehydrate_line(
    line: str, carry: str,
    rehydrate_engine, redaction_key, guardrail_engine, guardrail_mask_active,
):
    """Re-hydrate content inside a data: SSE line; returns (line, carry)."""
    if not line.startswith("data: "):
        return line, carry
    if not guardrail_mask_active and "ipaddress" not in line and not carry:
        return line, carry
    try:
        data = json.loads(line[6:])
    except (ValueError, TypeError):
        return line, carry
    mutated = False
    for choice in data.get("choices", []):
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            flushed, carry = _rehydrate_chunk(
                delta["content"], carry,
                rehydrate_engine, redaction_key, guardrail_engine, guardrail_mask_active,
            )
            if flushed != delta["content"]:
                delta["content"] = flushed
                mutated = True
    if not mutated:
        return line, carry
    return f"data: {json.dumps(data)}", carry


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

        rehydrate_engine = getattr(request.app.state, "ip_redaction", None)
        guardrail_engine: GuardrailEngine | None = getattr(
            request.app.state, "guardrails", None
        )

        # Stream processing helpers (module-level for reduced nesting)
        rehydrate_engine = getattr(request.app.state, "ip_redaction", None)
        guardrail_engine: GuardrailEngine | None = getattr(
            request.app.state, "guardrails", None
        )
        _guardrail_mask_active = (
            guardrail_engine is not None
            and guardrail_engine.config.output_enabled
            and guardrail_engine.config.output_action == "mask"
        )

        _config = request.app.state.config.get()
        _tt_cfg = getattr(getattr(_config, "telemetry", None), "token_tracking", None)
        _tt_enabled = _tt_cfg is not None and getattr(_tt_cfg, "enabled", True)
        _tt_show = _tt_cfg is not None and getattr(_tt_cfg, "show_in_postfix", True)
        _stream_usage: dict[str, Any] | None = None
        _stream_had_tool_calls = False

        try:
            carry = ""
            async for line in stream_resp.aiter_lines():
                if line.strip() == "data: [DONE]":
                    if carry:
                        if _guardrail_mask_active and guardrail_engine is not None:
                            carry = _mask_carry_flush(carry, guardrail_engine)
                        flush_event = {"choices": [{"delta": {"content": carry}}]}
                        yield f"data: {json.dumps(flush_event)}\n\n"
                        carry = ""

                    if _tt_enabled and _stream_usage is not None:
                        await _finalize_stream_token_tracking(
                            pin, route, model_used, provider, _stream_usage, request,
                        )
                    postfix_text = _build_stream_postfix(
                        route, pin, _stream_usage, _tt_enabled, _tt_show,
                    )
                    if not _stream_had_tool_calls:
                        postfix_event = {"choices": [{"delta": {"content": f"\n\n{postfix_text}"}}]}
                        yield f"data: {json.dumps(postfix_event)}\n\n"
                    yield f"{line}\n"
                    break
                if line.startswith("data: "):
                    _stream_usage, _stream_had_tool_calls = _process_data_line(
                        line, _stream_usage, _stream_had_tool_calls,
                    )
                line, carry = await _rehydrate_line(
                    line, carry,
                    rehydrate_engine, redaction_key, guardrail_engine, _guardrail_mask_active,
                )
                yield f"{line}\n"
        except Exception as e:
            error_data = _handle_stream_error(e, route, model_used, session_id, upstream_start)
            yield f"data: {json.dumps(error_data)}\n\n"
        else:
            _record_stream_success(route, model_used)
        finally:
            await stream_resp.aclose()
            _record_stream_finally(route, model_used, upstream_start, fallback_used)

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


def _add_model_postfix(
    json_resp: dict[str, Any],
    model_used: str,
    route: RouteDecision,
    token_usage: dict[str, dict[str, int]] | None = None,
    show_tokens: bool = True,
) -> None:
    """Append a compact model marker to assistant content for user visibility.

    When ``show_tokens`` is True and ``token_usage`` contains data, the
    marker includes cumulative per-tier token usage::

        [smart-router/L1-In:3032|Out:1000, L2-In:10021|Out:6054]

    Otherwise falls back to the classic format::

        [smart-router/L1]
    """
    marker = build_token_postfix(route.level.value, token_usage, show_tokens)
    for choice in json_resp.get("choices", []):
        # Skip tool-call responses: the postfix is a user-facing visibility
        # marker for final text answers, not for intermediate tool-call
        # turns.  Appending it here injects [smart-router/Ln-In:…|Out:…]
        # into the content of a tool-call response, which the gateway then
        # surfaces to the user mid-task.
        if choice.get("finish_reason") == "tool_calls":
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            continue
        # Also skip when the message carries tool_calls (some providers set
        # finish_reason to "stop" even when tool_calls are present).
        if message.get("tool_calls"):
            continue
        content = message.get("content")
        if content is None or content == "":
            message["content"] = marker
        elif isinstance(content, str) and marker not in content:
            message["content"] = f"{content.rstrip()}\n\n{marker}"


# Matches both the classic format [smart-router/L1] and the token-tracking
# format [smart-router/L1-In:3032|Out:1000, L2-In:10021|Out:6054].
# Also matches legacy [LLM: model/name] markers.
_MODEL_POSTFIX_RE = re.compile(
    r"(?:\r?\n){1,2}\[(?:LLM: )?[^\]\r\n]+\]\s*\Z"
)


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
