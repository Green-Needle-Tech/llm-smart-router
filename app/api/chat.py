"""POST /v1/chat/completions — the primary endpoint."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
import uuid
from functools import lru_cache
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.guardrails.custom import (
    CustomGuardrailEngine,
    CustomGuardrailSettings,
    build_payload_text,
    is_short_followup,
)
from app.guardrails.rules import strip_invisible_text
from app.guardrails.scanner import GuardrailConfig, GuardrailEngine
from app.guardrails.streaming import secret_carry_split
from app.middleware.auth import check_router_auth, unauthorized_response
from app.middleware.request_limits import validate_request_bounds
from app.privacy.ip_redaction import IPRedactionEngine
from app.providers.prompt_cache import (
    apply_prompt_cache_features,
    extract_cache_usage,
    needs_explicit_cache_control,
    strip_cache_control_from_messages,
)
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
from app.session.lifecycle import (
    check_expiry,
    check_turn_cap,
    should_reclassify_on_turn,
)
from app.session.locks import acquire_or_wait
from app.session.resolver import _get_api_key_identity, resolve_session_id
from app.telemetry import langfuse_tracing
from app.telemetry.logging import get_logger
from app.telemetry.metrics import (
    router_active_requests,
    router_cache_events_total,
    router_cost_usd_total,
    router_custom_guardrail_blocks_total,
    router_custom_guardrail_evals_total,
    router_custom_guardrail_skipped_total,
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
    router_retry_on_failure_total,
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

    custom_block = await _custom_guardrail_check_input(request, body, config)
    if custom_block is not None:
        return custom_block

    return None


# --- Opt-in custom guardrail (TypeSafe Noul, input only) --------------------


def _custom_guardrail_settings(config) -> CustomGuardrailSettings | None:
    """Snapshot the live custom-guardrail config for this request."""
    cfg = getattr(getattr(config, "telemetry", None), "guardrails", None)
    cfg = getattr(cfg, "custom", None) if cfg is not None else None
    if cfg is None:
        return None
    if hasattr(cfg, "model_dump"):
        return CustomGuardrailSettings(**cfg.model_dump())
    return CustomGuardrailSettings(**vars(cfg))


def _guardrail_rejection_marker() -> str:
    """Version-only postfix for guardrail rejections (no tier was routed)."""
    from app.version import APPLICATION_VERSION

    return f"[smart-router/v{APPLICATION_VERSION}]"


def _custom_guardrail_rejection(reason: str, settings: CustomGuardrailSettings | None = None, body=None):
    """Build the rejection response for the client.

    Message is configurable via telemetry.guardrails.custom.rejection_message
    (supports a "{reason}" placeholder); falls back to the built-in default.

    Delivery mode (telemetry.guardrails.custom.rejection_delivery):
      - "reply" (default): a normal HTTP 200 chat completion whose content is
        the rejection message + router version postfix. Downstream clients
        (e.g. Hermes gateway) render it as a single assistant message instead
        of stacking a provider error message on top of a failed-turn notice.
      - "error": legacy HTTP 400 guardrail_violation envelope.
    """
    template = (
        settings.rejection_message
        if settings and settings.rejection_message.strip()
        else "Request rejected by custom guardrail: {reason}"
    )
    message = template.format(reason=reason) if "{reason}" in template else template

    delivery = getattr(settings, "rejection_delivery", "reply") if settings else "reply"
    if delivery == "error":
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": message,
                "type": "guardrail_violation",
                "param": None,
                "code": "router_custom_guardrail_rejected",
                "guardrail": {"decision": "no", "reason": reason},
            }},
        )

    content = f"{message.rstrip()}\n\n{_guardrail_rejection_marker()}"
    model_name = getattr(body, "model", None) or "smart-router"
    completion_id = f"chatcmpl-guardrail-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    guardrail_meta = {
        "decision": "no",
        "reason": reason,
        "code": "router_custom_guardrail_rejected",
    }

    if body is not None and getattr(body, "stream", False):
        async def _reject_stream():
            chunks = [
                {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                },
                {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                },
                {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "guardrail": guardrail_meta,
                },
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_reject_stream(), media_type="text/event-stream")

    return JSONResponse(
        status_code=200,
        content={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "guardrail": guardrail_meta,
        },
    )


def _shallow_message_dicts(messages) -> list[dict[str, Any]]:
    """Build lightweight {role, content} dicts without full Pydantic model_dump."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content")
            if isinstance(content, list):
                content = [dict(b) if isinstance(b, dict) else b for b in content]
            out.append({"role": m.get("role"), "content": content})
        else:
            content = m.content
            if isinstance(content, list):
                content = [dict(b) if isinstance(b, dict) else b for b in content]
            out.append({"role": m.role, "content": content})
    return out


async def _custom_guardrail_check_input(request, body, config) -> JSONResponse | None:
    """Opt-in custom guardrail (input only): evaluate the request payload with
    a TypeSafe Noul probability-of-yes call, thresholded into a binary
    decision. None = pass (or disabled); JSONResponse = reject."""
    settings = _custom_guardrail_settings(config)
    if settings is None or not settings.is_enabled():
        return None
    messages = _shallow_message_dicts(body.messages)
    # Short follow-up bypass (v2.30.0): confirmations and short instructions
    # ("yes", "go ahead", "make it darker") carry no topic signal of their own
    # and were wrongly rejected by topic-scope guardrails. When prior
    # non-system conversation history exists, skip the decision call entirely.
    if (
        settings.skip_short_followups
        and is_short_followup(messages, settings.short_followup_max_chars)
    ):
        router_custom_guardrail_skipped_total.labels(reason="short_followup").inc()
        return None
    if settings.payload_scope == "last_user":
        user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        if user_msgs:
            messages = [user_msgs[-1]]
    elif settings.payload_scope == "conversation":
        # Keep user + assistant history, drop the (large) system prompt.
        non_system = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
        if non_system:
            messages = non_system
    payload_text = build_payload_text(messages, settings.max_payload_chars)
    if not payload_text:
        return None
    app_state = getattr(getattr(request, "app", None), "state", None)
    shared_http = getattr(app_state, "custom_guardrail_http", None) if app_state is not None else None
    engine = (
        CustomGuardrailEngine(settings, http_client=shared_http)
        if shared_http is not None
        else CustomGuardrailEngine(settings)
    )
    decision = await engine.evaluate(payload_text)
    router_custom_guardrail_evals_total.labels(
        decision=decision.decision, source=decision.source,
    ).inc()
    if decision.decision == "yes":
        return None  # pass — request proceeds untouched
    router_custom_guardrail_blocks_total.inc()
    logger.warning(
        "router.custom_guardrail.rejected",
        reason=decision.reason,
    )
    return _custom_guardrail_rejection(decision.reason, settings, body)


@router.post("/v1/chat/completions", responses={400: {"description": "Bad request"}})
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Handle a chat completion request with session-pinned routing."""
    start = time.monotonic()
    config = request.app.state.config.get()

    preproc_error = await _preprocess_request(request, body, config)
    if preproc_error is not None:
        return preproc_error

    # Resolve session once BEFORE IP redaction & temporal normalization mutate messages,
    # guaranteeing identical session keys for IP redaction/rehydration and fingerprint fallback.
    resolved_session = _resolve_session(request, body, config) if config.session.enabled else None
    resolved_sid = resolved_session[0] if resolved_session else None

    redaction_key = await _redact_incoming(request, body, session_id=resolved_sid)
    _process_temporal_awareness(request, body)

    # Langfuse: set an explicit, sanitized trace input only when tracing is enabled.
    if langfuse_tracing.enabled():
        langfuse_tracing.note_trace_input({
            "model_directive": body.model,
            "stream": bool(body.stream),
            "messages": [
                {
                    "role": getattr(m, "role", None),
                    "content": langfuse_tracing._truncate(
                        m.content if isinstance(getattr(m, "content", None), str) else "[non-text content]"
                    ),
                }
                for m in body.messages[-12:]
            ],
        })

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
            redaction_key=redaction_key,
        )
    if mode == "level":
        forced_level = directive["level"]

    return await _session_pinned_route(
        request, body, config, routing_engine,
        directive, forced_level, forced_model, max_level, min_level,
        opts["reclassify"], opts["repin"], opts["task_text"],
        opts["bypass_cache"], opts["include_metadata"], start,
        redaction_key=redaction_key,
        resolved_session=resolved_session,
    )


def _guardrail_cfg_key(cfg) -> tuple:
    """Hashable snapshot key for detecting live guardrail config changes."""
    return (
        cfg.input_enabled,
        cfg.input_action,
        cfg.block_on_severity,
        cfg.output_enabled,
        cfg.output_action,
        cfg.invisible_text_detection,
        cfg.pii_masking_enabled,
        getattr(cfg, "input_pii_masking_enabled", True),
        tuple(cfg.banned_substrings or ()),
        cfg.refusal_detection,
        cfg.malicious_url_detection,
        getattr(cfg, "system_prompt_leak_detection", False),
        tuple(getattr(cfg, "system_prompt_fragments", ()) or ()),
        getattr(cfg, "system_prompt_leak_threshold", 0.85),
        getattr(cfg, "homoglyph_normalization", True),
        getattr(cfg, "obfuscation_detection", True),
    )


def _build_guardrail_engine(cfg) -> GuardrailEngine:
    """Create a GuardrailEngine with a config snapshot."""
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
    ))


def _get_live_guardrail_engine(request, cfg) -> GuardrailEngine:
    """Return a cached GuardrailEngine synced to the current live config."""
    key = _guardrail_cfg_key(cfg)
    if getattr(request.app.state, "_guardrails_cfg_key", None) == key:
        cached = getattr(request.app.state, "_guardrails_cached_engine", None)
        if cached is not None:
            return cached
    engine = _build_guardrail_engine(cfg)
    request.app.state._guardrails_cfg_key = key
    request.app.state._guardrails_cached_engine = engine
    request.app.state.guardrails = engine
    return engine


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
    if getattr(request.app.state, "guardrails", None) is None:
        return None
    config = request.app.state.config.get()
    cfg = config.telemetry.guardrails
    engine = _get_live_guardrail_engine(request, cfg)
    # Langfuse: real `with` block so the OTEL context nests the span.
    with langfuse_tracing.span_cm("guardrail-input-scan"):
        return _guardrail_scan_input_impl(request, body, engine, cfg)


def _guardrail_scan_input_impl(request, body, engine, cfg):
    messages = _shallow_message_dicts(body.messages)

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
    if engine.config.output_action in ("mask", "block"):
        router_guardrail_secret_masks_total.labels(rule_id=f.rule_id).inc()
    else:
        router_guardrail_findings_total.labels(
            rule_id=f.rule_id, severity=f.severity, direction="output",
        ).inc()
    logger.warning(
        "router.guardrail.output_finding", rule=f.rule_id,
        action=engine.config.output_action,
    )


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


async def _redact_incoming(request, body, session_id: str | None = None) -> str | None:
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
    if session_id is None:
        config = request.app.state.config.get()
        session_id, _ = _resolve_session(request, body, config)
    key = session_id or f"req-{uuid.uuid4()}"
    messages = _shallow_message_dicts(body.messages)
    changed = await engine.redact_messages(messages, key)
    if changed:
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
    if engine is None or not engine.config.enabled:
        return
    messages = _shallow_message_dicts(body.messages)
    processed = engine.process_messages(messages)
    for orig, proc in zip(body.messages, processed, strict=False):
        content = proc.get("content")
        if orig.content != content:
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


@lru_cache(maxsize=16)
def _compile_tier_prefix(pattern_str: str) -> re.Pattern | None:
    try:
        return re.compile(pattern_str, re.IGNORECASE)
    except re.error:
        return None


def _detect_tier_prefix(body, config) -> Level | None:
    """Detect a tier label (L1-L5) at the start of the first user prompt."""
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
    if not isinstance(content, str):
        return None
    stripped_input = content.strip()
    if not stripped_input:
        return None

    compiled = _compile_tier_prefix(pattern_str)
    if compiled is None:
        return None
    match = compiled.match(stripped_input)
    if match is None:
        return None

    level_str = match.group(1)
    try:
        level = Level.from_str(level_str)
    except ValueError:
        return None

    # Strip the prefix from the message if configured and content remains
    if strip_prefix:
        stripped = stripped_input[match.end():].strip()
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

    prev_level = pin.level
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

    escalated = False
    escalated_from: Level | None = None
    if config.session.escalation.enabled and config.session.escalation.free_signals_enabled:
        escalation_result = _check_escalation_signals(body, pin, config)
        if escalation_result:
            effective_level, effective_model = escalation_result
            escalated = True
            escalated_from = prev_level

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

    route = RouteDecision(
        level=effective_level, model=effective_model,
        params=pin.params, classification=classification,
        escalated=escalated, escalated_from=escalated_from,
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
        request, body, route, session_id, None, None, include_metadata, start,
        redaction_key=redaction_key,
    )


async def _create_and_pin_session(
    request, body, config, classifier, routing_engine, store,
    forced_level, task_text, bypass_cache, start, session_id, session_source,
    max_level, min_level, include_metadata, redaction_key,
):
    """Classify, create a new session pin, and forward to provider."""
    try:
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

        is_unknown = bool(classification.reason and "UNKNOWN" in classification.reason.upper())
        status = SessionStatus.PROVISIONAL if is_unknown else SessionStatus.PINNED

        new_pin = SessionPin(
            session_id=session_id, level=effective_level, model=model,
            params=params, status=status, classification=classification, turn_count=1,
            provisional_turns=1 if is_unknown else 0,
        )
        if not is_unknown:
            new_pin.escalation.original_level = effective_level
        new_pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)

        await store.put(new_pin)
    finally:
        await store.release(session_id)

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


async def _handle_provisional_turn(
    request, body, pin, config, classifier, routing_engine, store,
    forced_level, forced_model, max_level, min_level, task_text,
    bypass_cache, include_metadata, start, session_id, session_source, redaction_key,
):
    """Re-classify a PROVISIONAL session (e.g. after a 'hi' opener) using the latest user message."""
    effective_task = task_text or _extract_raw_user_text(body)
    level, classification, _ = await _classify_and_pin(
        request, body, config, classifier, routing_engine,
        forced_level, effective_task, bypass_cache, start, session_id,
    )
    still_unknown = bool(classification.reason and "UNKNOWN" in classification.reason.upper())
    effective_level = level
    if max_level and effective_level > max_level:
        effective_level = max_level
    if min_level and effective_level < min_level:
        effective_level = min_level

    pin.turn_count += 1
    pin.provisional_turns += 1
    pin.level = effective_level
    pin.model = forced_model or routing_engine.resolve_model_for_level(effective_level)
    pin.params = config.routing.get_params(effective_level.value)
    pin.classification = classification
    max_prov = getattr(config.session, "max_provisional_turns", 3)
    if not still_unknown or pin.provisional_turns >= max_prov:
        pin.status = SessionStatus.PINNED
        pin.escalation.original_level = effective_level
    pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)
    await store.put(pin)

    route = RouteDecision(
        level=effective_level, model=pin.model,
        params=pin.params, classification=classification,
    )
    return await _forward_to_provider(
        request, body, route, session_id, session_source, pin, include_metadata, start,
        redaction_key=redaction_key,
    )


async def _session_pinned_route(
    request, body, config, routing_engine,
    directive, forced_level, forced_model, max_level, min_level,
    reclassify, repin, task_text, bypass_cache, include_metadata, start,
    redaction_key=None,
    resolved_session=None,
):
    """The main session-pinned routing path."""
    store = request.app.state.session_store
    classifier = request.app.state.classifier

    session_id, session_source = (
        resolved_session if resolved_session is not None else _resolve_session(request, body, config)
    )
    if session_id is None:
        router_session_lookups_total.labels(result="disabled").inc()
        return await _stateless_classify_and_forward(
            request, body, config, routing_engine, task_text, max_level, min_level,
            forced_model, bypass_cache, redaction_key=redaction_key,
        )

    pin = await _lookup_pin(store, session_id, config)
    if pin is not None and not reclassify:
        max_prov = getattr(config.session, "max_provisional_turns", 3)
        if should_reclassify_on_turn(pin, max_prov) and not forced_level:
            return await _handle_provisional_turn(
                request, body, pin, config, classifier, routing_engine, store,
                forced_level, forced_model, max_level, min_level, task_text,
                bypass_cache, include_metadata, start, session_id, session_source, redaction_key,
            )
        return await _handle_session_hit(
            request, body, pin, config, routing_engine, forced_level, forced_model,
            max_level, min_level, repin, include_metadata, start,
            session_id, session_source, redaction_key,
        )

    if pin is not None and reclassify:
        await store.delete(session_id)

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


_INJECTED_CONTEXT_RE = re.compile(
    r"<(?:memory|skill|system|context|soul|persona|user_profile)[-_]?context>"
    r".*?"
    r"</(?:memory|skill|system|context|soul|persona|user_profile)[-_]?context>",
    re.DOTALL | re.IGNORECASE,
)


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

    if "<" in last_user_text:
        last_user_text = _INJECTED_CONTEXT_RE.sub("", last_user_text)
    return last_user_text.strip()


_REPAIR_LANGUAGE_RE = re.compile(
    r"\b(no,|that's wrong|still (failing|broken)|doesn't work|try again|not what I|incorrect|you missed)\b",
    re.IGNORECASE,
)
_DEEP_KEYWORDS_RE = re.compile(
    r"\b(architect|design a system|prove|derive|refactor the|threat model|optimize the algorithm)\b",
    re.IGNORECASE,
)
_STRONG_ESCALATION_SIGNALS = {"repair_language", "tool_error_loop", "upstream_failure"}


def _detect_escalation_signals(raw_user_text, pin, esc_cfg) -> list[tuple[str, int]]:
    """Detect escalation signals from user text and return (signal, weight) pairs."""
    signals = []
    if _REPAIR_LANGUAGE_RE.search(raw_user_text):
        signals.append(("repair_language", esc_cfg.signal_weights.get("repair_language", 3)))
    if _DEEP_KEYWORDS_RE.search(raw_user_text):
        signals.append(("deep_keywords", esc_cfg.signal_weights.get("deep_keywords", 2)))
    if pin.turn_count >= esc_cfg.escalate_after_turns:
        signals.append(("turn_depth", esc_cfg.signal_weights.get("turn_depth", 1)))
    # Upstream failure signal: previous turn(s) had upstream errors/timeouts
    if pin.escalation.consecutive_failures > 0:
        weight = esc_cfg.signal_weights.get("upstream_failure", 3)
        if weight > 0:
            signals.append(("upstream_failure", weight))
    return [(s, w) for s, w in signals if w > 0]


def _apply_escalation(pin, signals_fired, esc_cfg, config) -> tuple[Level, str] | None:
    """Apply escalation if threshold/cooldown conditions are met."""
    if not (pin.escalation.score >= esc_cfg.threshold and
            pin.escalation.count < esc_cfg.max_escalations_per_session and
            pin.turn_count >= pin.escalation.cooldown_until_turn):
        return None
    new_level = Level.from_numeric(pin.level.numeric + 1)
    if new_level == pin.level or new_level > Level.from_str(config.routing.global_max_level):
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


def _mark_upstream_failure(pin, config, store=None) -> None:
    """Mark a pin as having experienced an upstream failure (timeout/error).

    Increments consecutive_failures and last_failure_turn so the next
    user turn triggers the upstream_failure escalation signal.
    Also triggers an immediate tier bump if retry_on_failure is enabled
    and the session hasn't exhausted its retry budget.
    """
    if pin is None:
        return
    pin.escalation.consecutive_failures += 1
    pin.escalation.last_failure_turn = pin.turn_count
    logger.warning(
        "router.upstream.failure_marked",
        session_id=pin.session_id,
        level=pin.level.value,
        consecutive_failures=pin.escalation.consecutive_failures,
        turn=pin.turn_count,
    )


def _clear_upstream_failure(pin) -> None:
    """Reset consecutive_failures after a successful response."""
    if pin is None:
        return
    if pin.escalation.consecutive_failures > 0:
        pin.escalation.consecutive_failures = 0


def _collect_cross_tier_emergency_models(
    config,
    current_level: Level,
    tried_models: set[str],
) -> list[tuple[str, str | None, str | None]]:
    """Collect remaining untried models configured across OTHER tiers.

    Ordered closest-tier-first (higher tiers up to global_max_level first,
    then lower tiers down to L1). Used as a last-resort within-router failover
    so a long task stays on llm-smart-router whenever ANY configured upstream
    model is healthy, without re-pinning the session.
    """
    routing = getattr(config, "routing", None)
    if routing is None or not hasattr(routing, "L1") or not hasattr(routing, "get_tier"):
        return []
    try:
        global_max = Level.from_str(getattr(routing, "global_max_level", "L5"))
    except (ValueError, AttributeError):
        global_max = Level.L5

    cur_num = current_level.numeric
    # Prefer higher tiers first (closest above), then lower tiers (closest below)
    candidate_nums = [n for n in range(cur_num + 1, global_max.numeric + 1)] + [
        n for n in range(cur_num - 1, 0, -1)
    ]

    result: list[tuple[str, str | None, str | None]] = []
    seen: set[str] = set(tried_models)
    for num in candidate_nums:
        lvl = Level.from_numeric(num)
        try:
            tier = routing.get_tier(lvl.value)
        except Exception:
            continue
        t_base_url, t_api_key = _resolve_tier_provider(config, lvl.value)
        for m in [getattr(tier, "model", ""), *(getattr(tier, "fallbacks", None) or [])]:
            if m and m not in seen:
                seen.add(m)
                result.append((m, t_base_url, t_api_key))
    return result


async def _retry_with_higher_tier(
    request, body, route, pin, config, routing_engine, include_metadata, start,
    session_id, session_source, redaction_key, store,
    *, tried_models: set[str] | None = None, deadline: float | None = None,
):
    """Retry a failed request with the next tier up.

    Called when retry_on_failure is enabled and the upstream call failed.
    Bumps the pin to the next tier that has untried models, updates the store,
    and re-forwards while skipping already-failed models. Returns the response
    from the higher tier, or None if no higher tier is available or retry
    budget is exhausted.
    """
    esc_cfg = config.session.escalation
    if pin is None:
        return None
    if pin.escalation.retry_count >= esc_cfg.retry_on_failure_max_per_session:
        logger.info(
            "router.retry.exhausted",
            session_id=session_id,
            retry_count=pin.escalation.retry_count,
        )
        return None

    global_max = Level.from_str(config.routing.global_max_level)
    skip = tried_models if tried_models is not None else set()

    # Find the next higher tier that actually has at least one untried model
    # (prevents L3 -> L4 from retrying the exact same failed model when
    # adjacent tiers share a primary model).
    new_level: Level | None = None
    probe_num = pin.level.numeric + 1
    while probe_num <= global_max.numeric:
        cand = Level.from_numeric(probe_num)
        if cand == pin.level:
            break
        cand_primary = routing_engine.resolve_model_for_level(cand)
        cand_fbs = config.routing.get_fallbacks(cand.value)
        cand_models = [m for m in [cand_primary, *cand_fbs] if m and m not in skip]
        if cand_models or not skip:
            new_level = cand
            break
        probe_num += 1

    if new_level is None or new_level == pin.level or new_level > global_max:
        logger.info(
            "router.retry.max_tier_reached",
            session_id=session_id,
            current_level=pin.level.value,
        )
        return None

    old_level = pin.level
    old_model = pin.model
    old_params = pin.params
    pin.escalation.retry_count += 1
    pin.level = new_level
    pin.model = routing_engine.resolve_model_for_level(new_level)
    pin.params = config.routing.get_params(new_level.value)
    if pin.escalation.original_level is None:
        pin.escalation.original_level = old_level

    router_retry_on_failure_total.labels(
        from_level=old_level.value, outcome="escalated",
    ).inc()
    logger.info(
        "router.retry.escalating",
        session_id=session_id,
        from_level=old_level.value,
        to_level=new_level.value,
        retry_count=pin.escalation.retry_count,
    )

    await store.put(pin)

    new_route = RouteDecision(
        level=new_level, model=pin.model, params=pin.params,
        classification=ClassificationResult(
            level=new_level, confidence=1.0, reason="retry_on_failure",
            source=ClassificationSource.SESSION, latency_ms=0,
        ),
        escalated=True, escalated_from=old_level,
    )
    resp = await _forward_to_provider(
        request, body, new_route, session_id, session_source, pin,
        include_metadata, start, redaction_key=redaction_key,
        tried_models=skip, deadline=deadline,
    )
    if getattr(resp, "status_code", 200) >= 500:
        # Escalated tier also failed completely; restore pin level so callers
        # or subsequent emergency fallbacks see consistent state.
        pin.level = old_level
        pin.model = old_model
        pin.params = old_params
        await store.put(pin)
        return None
    return resp


def _resolve_tier_provider(config, level: str) -> tuple[str | None, str | None]:
    """Resolve per-tier provider overrides for a given level.

    Returns (base_url, api_key). Both are None when the tier uses the
    global provider configuration.
    """
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


def _strip_reasoning_disable_for_model(payload: dict, model: str, config) -> None:
    """Remove reasoning-disable directives for reasoning-mandatory models."""
    mandatory = getattr(
        getattr(config, "provider", None), "reasoning_mandatory_models", None
    ) or []
    if model not in mandatory:
        return
    reasoning = payload.get("reasoning")
    # A disable arrives in two wire shapes: ``{"enabled": false}`` (direct)
    # and ``{"effort": "none"}`` (Hermes aux client translates enabled=false
    # to effort:"none" on Responses-API routes). Both hard-400 on
    # reasoning-mandatory models.
    if isinstance(reasoning, dict) and (
        reasoning.get("enabled") is False or reasoning.get("effort") == "none"
    ):
        del payload["reasoning"]


def _strip_reasoning_disable(payload: dict, route, config) -> None:
    """Remove `reasoning: {"enabled": false}` for reasoning-mandatory models.

    Some upstream models (e.g. z-ai/glm-5.3) hard-reject a reasoning disable
    with HTTP 400 "Reasoning is mandatory for this endpoint and cannot be
    disabled".  Auxiliary clients (title generation, compression, ...) send
    `reasoning.enabled=false` by design; stripping it for these models avoids
    a deterministic 400, 3 wasted retries, and an unnecessary fallback.
    """
    _strip_reasoning_disable_for_model(payload, route.model, config)


def _estimate_prompt_tokens(payload: dict) -> int:
    """Conservative token count estimate for messages + tools (~3 chars/token)."""
    total_chars = 0
    for msg in payload.get("messages") or ():
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict):
                    txt = blk.get("text")
                    if isinstance(txt, str):
                        total_chars += len(txt)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            with contextlib.suppress(TypeError, ValueError):
                total_chars += len(json.dumps(tool_calls))
    tools = payload.get("tools")
    if tools:
        with contextlib.suppress(TypeError, ValueError):
            total_chars += len(json.dumps(tools))
    return (total_chars // 3) + 256


def _apply_max_tokens(
    payload, route, config, provider, *, model: str | None = None,
) -> None:
    """Resolve max_tokens: 'auto' uses OpenRouter-detected value clamped to context."""
    target_model = model or route.model
    routing = getattr(config, "routing", None)
    tier_max_tokens = (
        routing.get_max_tokens(route.level.value)
        if routing is not None and hasattr(routing, "get_max_tokens")
        else None
    )
    resolved: int | None = None
    if isinstance(tier_max_tokens, str) and tier_max_tokens == "auto":
        get_mct = getattr(provider, "get_max_completion_tokens", None)
        detected = get_mct(target_model) if callable(get_mct) else None
        if detected is not None and detected > 0:
            resolved = detected
    elif isinstance(tier_max_tokens, int) and tier_max_tokens > 0:
        resolved = tier_max_tokens

    if resolved is None:
        return

    # Guard against long-conversation context overflow: never request
    # prompt_tokens + max_tokens > model_context_length.
    get_ctx = getattr(provider, "get_context_length", None)
    ctx_len = get_ctx(target_model) if callable(get_ctx) else None
    if isinstance(ctx_len, int) and ctx_len > 0:
        est_prompt = _estimate_prompt_tokens(payload)
        headroom = ctx_len - est_prompt - 512
        if headroom < 256:
            # Prompt is near or over this model's context window — drop
            # max_tokens so the provider can fit or reject cleanly to fallback.
            payload.pop("max_tokens", None)
            return
        if resolved > headroom:
            resolved = headroom

    payload["max_tokens"] = resolved


def _adapt_payload_for_model(
    base_payload: dict[str, Any],
    model: str,
    route,
    session_id: str | None,
    config,
    provider,
) -> dict[str, Any]:
    """Build a model-adapted copy of base_payload for primary or fallback attempts.

    Ensures per-model reasoning rules, prompt-cache cache_control headers, and
    context-safe max_tokens are always valid for the specific candidate model.
    """
    adapted = dict(base_payload)
    adapted["model"] = model
    _strip_reasoning_disable_for_model(adapted, model, config)

    # Adapt prompt cache_control: strip Anthropic/Qwen cache_control when
    # falling back to a provider that caches implicitly (and may reject ttl).
    if "messages" in adapted and isinstance(adapted["messages"], list):
        if needs_explicit_cache_control(model):
            # Clone message list shallowly before injecting cache_control
            adapted["messages"] = [dict(m) if isinstance(m, dict) else m for m in adapted["messages"]]
            apply_prompt_cache_features(adapted, session_id, config)
        else:
            adapted["messages"] = strip_cache_control_from_messages(adapted["messages"])

    _apply_max_tokens(adapted, route, config, provider, model=model)
    return adapted


def _build_upstream_payload(body, route, session_id, config, provider) -> dict:
    """Build the upstream provider payload with tier params and max_tokens."""
    payload = body.model_dump(exclude={"router"}, exclude_none=True)
    _strip_model_postfix_from_messages(payload.get("messages", []))
    _strip_reasoning_disable(payload, route, config)
    payload["model"] = route.model
    apply_prompt_cache_features(payload, session_id, config)
    for key, val in route.params.items():
        if key not in payload or payload.get(key) is None:
            payload[key] = val
    _apply_max_tokens(payload, route, config, provider)
    if body.stream:
        payload.setdefault("stream_options", {})["include_usage"] = True
    return payload


async def _forward_to_provider(
    request, body, route, session_id, session_source, pin, include_metadata, start,
    redaction_key=None,
    *, tried_models: set[str] | None = None, deadline: float | None = None,
):
    """Forward the request to the provider and return the response."""
    config = request.app.state.config.get()
    provider = request.app.state.provider

    # Langfuse: attach session/route context to the trace (propagated to
    # all spans, including the upstream generation observations).
    # propagate_attributes is a context manager — entered with `with` so
    # all spans created below (guardrails, generations) carry the attrs.
    with langfuse_tracing.note_route(
        session_id=session_id,
        level=getattr(getattr(route, "level", None), "value", None),
        model=getattr(route, "model", None),
        metadata={
            "classification_source": getattr(getattr(route, "classification", None), "source", None) and route.classification.source.value,
            "session_source": session_source.value if session_source else None,
            "router_version": getattr(request.app.state, "version", None),
        },
    ):
        budget_result = await _check_budget(request, body, route, session_id, config)
        if budget_result is not None:
            if isinstance(budget_result, JSONResponse):
                return budget_result
            route = budget_result  # downgraded route

        payload = _build_upstream_payload(body, route, session_id, config, provider)
        fallbacks = config.routing.get_fallbacks(route.level.value)
        tier_base_url, tier_api_key = _resolve_tier_provider(config, route.level.value)
        if deadline is None:
            deadline = _provider_deadline(config)

        router_active_requests.inc()
        try:
            if body.stream:
                return await _handle_stream(
                    request, payload, body, route, fallbacks, session_id, session_source, pin, include_metadata, start,
                    redaction_key=redaction_key, tier_base_url=tier_base_url, tier_api_key=tier_api_key,
                    deadline=deadline, tried_models=tried_models,
                )
            return await _handle_non_stream(
                request, payload, body, route, fallbacks, session_id, session_source, pin, include_metadata, start,
                redaction_key=redaction_key, tier_base_url=tier_base_url, tier_api_key=tier_api_key,
                deadline=deadline, tried_models=tried_models,
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
    request, payload, body, route, fallbacks, session_id, session_source, pin, include_metadata, start,
    redaction_key=None, tier_base_url=None, tier_api_key=None, deadline=None,
    tried_models: set[str] | None = None,
):
    """Handle non-streaming request."""
    provider = request.app.state.provider
    config = request.app.state.config.get()
    if deadline is None:
        deadline = _provider_deadline(config)

    def adapter(m, p):
        return _adapt_payload_for_model(p, m, route, session_id, config, provider)
    seen_models: set[str] = set(tried_models) if tried_models is not None else set()

    upstream_start = time.monotonic()
    json_resp, _, model_used, fallback_used, error = await provider.chat_completion(
        payload, fallback_models=fallbacks, stream=False,
        base_url=tier_base_url, api_key=tier_api_key, deadline=deadline,
        payload_adapter=adapter, skip_models=seen_models or None,
    )
    upstream_ms = int((time.monotonic() - upstream_start) * 1000)
    total_ms = int((time.monotonic() - start) * 1000)

    if error and json_resp is None:
        for m in [route.model, *list(fallbacks or [])]:
            if m:
                seen_models.add(m)
        # Mark the pin as having an upstream failure for next-turn escalation
        _mark_upstream_failure(pin, config)

        esc_cfg = getattr(getattr(config, "session", None), "escalation", None)
        retry_cfg = getattr(esc_cfg, "retry_on_failure", False) if esc_cfg else False

        # Attempt immediate retry with a higher tier if enabled
        if retry_cfg and pin is not None:
            routing_engine = request.app.state.routing_engine
            store = request.app.state.session_store
            retry_resp = await _retry_with_higher_tier(
                request, body, route, pin, config,
                routing_engine, include_metadata, start, session_id,
                session_source, redaction_key, store,
                tried_models=seen_models, deadline=deadline,
            )
            if retry_resp is not None:
                return retry_resp

        # Last-resort cross-tier emergency fallback (when retry_on_failure is
        # enabled and outer call has remaining budget): try any remaining
        # untried configured models across other tiers without re-pinning.
        if (
            retry_cfg
            and tried_models is None
            and (deadline is None or (deadline - time.monotonic()) > 2)
        ):
            emergency = _collect_cross_tier_emergency_models(config, route.level, seen_models)
            for em_model, em_base_url, em_api_key in emergency:
                if deadline is not None and (deadline - time.monotonic()) <= 1:
                    break
                seen_models.add(em_model)
                em_payload = _adapt_payload_for_model(
                    payload, em_model, route, session_id, config, provider,
                )
                em_resp, _, em_used, _, em_err = await provider.chat_completion(
                    em_payload, fallback_models=[], stream=False,
                    base_url=em_base_url, api_key=em_api_key, deadline=deadline,
                )
                if em_resp is not None and not em_err:
                    logger.warning(
                        "router.emergency_cross_tier_fallback.succeeded",
                        session_id=session_id,
                        level=route.level.value,
                        from_model=route.model,
                        to_model=em_used,
                    )
                    json_resp = em_resp
                    model_used = em_used
                    fallback_used = True
                    error = None
                    upstream_ms = int((time.monotonic() - upstream_start) * 1000)
                    total_ms = int((time.monotonic() - start) * 1000)
                    break

    if error and json_resp is None:
        status = 502 if "exhausted" in error else 504
        error_type = "upstream_error" if status == 502 else "upstream_timeout"
        return JSONResponse(
            status_code=status,
            content={"error": {
                "message": error, "type": error_type, "param": None, "code": f"router_upstream_{status}",
            }},
        )

    # Success — clear any previous failure state
    _clear_upstream_failure(pin)

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
    # Context-window limit from provider config (default 1M)
    provider_cfg = getattr(config, "provider", None)
    ctx_window = getattr(provider_cfg, "context_window", 1_000_000) if provider_cfg else 1_000_000
    if tt_enabled:
        if pin is not None:
            accumulate_tokens(pin.token_usage, route.level.value, prompt_tokens, completion_tokens)
        token_usage_for_postfix = pin.token_usage if pin is not None else {
            route.level.value: {"prompt": prompt_tokens, "completion": completion_tokens}
        }
        _add_model_postfix(
            json_resp, model_used, route, token_usage_for_postfix, tt_show,
            last_ctx_tokens=prompt_tokens, context_window=ctx_window,
        )
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
    if guardrail_engine is None or guardrail_engine.config.output_action not in ("mask", "block"):
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


# --- Stall detection (v2.19.0) -------------------------------------------

def _provider_deadline(config) -> float | None:
    """Absolute time.monotonic() deadline for one client request, or None."""
    provider_cfg = getattr(config, "provider", None)
    secs = getattr(provider_cfg, "request_deadline_seconds", 0) if provider_cfg else 0
    try:
        secs = int(secs or 0)
    except (TypeError, ValueError):
        return None
    return time.monotonic() + secs if secs > 0 else None


def _stall_timeouts(config) -> tuple[float | None, float | None]:
    """Return (first_token_timeout, idle_timeout) in seconds; None = disabled."""
    provider_cfg = getattr(config, "provider", None)
    if provider_cfg is None:
        return None, None

    def _norm(name: str) -> float | None:
        try:
            v = float(getattr(provider_cfg, name, 0) or 0)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    return _norm("stream_first_token_timeout_seconds"), _norm("stream_idle_timeout_seconds")


class _StreamStalled(Exception):
    """Upstream stopped producing SSE data within the allowed window.

    ``first_token`` is True when no meaningful SSE data had arrived yet,
    which means the stall is still transparently recoverable by restreaming
    from a fallback model or higher tier.
    """

    def __init__(self, waited: float, *, first_token: bool):
        self.waited = waited
        self.first_token = first_token
        stage = "first token" if first_token else "chunk"
        super().__init__(f"upstream stalled: no {stage} for {waited:.0f}s")


class _StreamUpstreamError(Exception):
    """Recoverable upstream stream failure before any content reached client.

    Covers OpenRouter HTTP-200 SSE-embedded error events (``data: {"error": ...}``),
    ``finish_reason: "error"``, and empty/abruptly closed upstream streams.
    """


def _is_sse_keepalive_or_blank(line: str) -> bool:
    """True for SSE comment lines (e.g. `: OPENROUTER PROCESSING`) and blanks."""
    stripped = line.strip()
    return not stripped or stripped.startswith(":")


def _inspect_sse_data_line(line: str) -> tuple[str, str | None]:
    """Classify a `data: ...` SSE line (excluding `[DONE]`).

    Returns `(kind, detail)` where `kind` is:
      - ``"error"``   — SSE-embedded error or error finish_reason with no content
      - ``"content"`` — real client-visible content/reasoning/tool_calls chunk
      - ``"meta"``    — role-only opener, usage-only, or non-content metadata chunk
    """
    try:
        data = json.loads(line[6:])
    except (ValueError, TypeError):
        return "content", None
    if not isinstance(data, dict):
        return "meta", None
    choices = data.get("choices")
    if "error" in data and not choices:
        return "error", str(data["error"])[:300]
    if isinstance(choices, list) and choices:
        has_visible = False
        has_error_finish = False
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta") or {}
            if isinstance(delta, dict) and (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or delta.get("tool_calls")
                or delta.get("function_call")
            ):
                has_visible = True
            if ch.get("finish_reason") == "error":
                has_error_finish = True
        if has_visible:
            return "content", None
        if has_error_finish:
            return "error", "upstream stream ended with finish_reason=error"
    return "meta", None


async def _aiter_lines_with_stall_guard(stream_resp, first_timeout, idle_timeout):
    """Yield SSE lines, raising _StreamStalled when upstream goes silent.

    httpx's own read timeout resets on every byte, and OpenRouter emits
    ``: OPENROUTER PROCESSING`` SSE comment keepalives while queued.
    Comments and blank lines are yielded so the client connection stays warm,
    but they do NOT reset the wall-clock first-token deadline or the
    inter-data idle window — preventing infinite keepalive hangs and ensuring
    first-token recovery stays armed until real ``data:`` arrives.
    """
    iterator = stream_resp.aiter_lines().__aiter__()
    seen_data = False
    first_deadline = (
        time.monotonic() + first_timeout if first_timeout is not None else None
    )
    last_data_at: float | None = None
    while True:
        if not seen_data:
            if first_deadline is not None:
                timeout = first_deadline - time.monotonic()
                if timeout <= 0:
                    raise _StreamStalled(first_timeout or 0, first_token=True)
            else:
                timeout = None
        else:
            if idle_timeout is not None:
                elapsed = time.monotonic() - (last_data_at or time.monotonic())
                timeout = idle_timeout - elapsed
                if timeout <= 0:
                    raise _StreamStalled(idle_timeout, first_token=False)
            else:
                timeout = None

        try:
            if timeout is None:
                line = await iterator.__anext__()
            else:
                line = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            waited = idle_timeout if seen_data else first_timeout
            raise _StreamStalled(waited or 0, first_token=not seen_data) from exc

        if not _is_sse_keepalive_or_blank(line):
            seen_data = True
            last_data_at = time.monotonic()
        yield line


def _next_tier_route(route, pin, config, routing_engine, *, skip_models: set[str] | None = None):
    """Build a RouteDecision for the next higher tier with untried models, or None."""
    if routing_engine is None:
        return None
    try:
        global_max = Level.from_str(config.routing.global_max_level)
    except (ValueError, AttributeError):
        return None
    skip = skip_models or set()

    new_level: Level | None = None
    model: str | None = None
    params: dict[str, Any] = {}
    try:
        probe_num = route.level.numeric + 1
    except AttributeError:
        return None

    while probe_num <= global_max.numeric:
        cand = Level.from_numeric(probe_num)
        if cand == route.level:
            break
        try:
            cand_model = routing_engine.resolve_model_for_level(cand)
            cand_params = config.routing.get_params(cand.value)
            cand_fbs = config.routing.get_fallbacks(cand.value)
        except (AttributeError, KeyError, ValueError):
            break
        untried = [m for m in [cand_model, *(cand_fbs or [])] if m and m not in skip]
        if untried or not skip:
            new_level = cand
            model = untried[0] if untried else cand_model
            params = cand_params
            break
        probe_num += 1

    if new_level is None or model is None:
        return None

    if pin is not None:
        pin.level = new_level
        pin.model = routing_engine.resolve_model_for_level(new_level)
        pin.params = params
        if getattr(pin.escalation, "original_level", None) is None:
            pin.escalation.original_level = route.level
        pin.escalation.retry_count += 1
    return RouteDecision(
        level=new_level, model=model, params=params,
        classification=ClassificationResult(
            level=new_level, confidence=1.0, reason="stream_stall_recovery",
            source=ClassificationSource.SESSION, latency_ms=0,
        ),
        escalated=True, escalated_from=route.level,
    )


def _stall_retry_budget_left(pin, config) -> bool:
    """True when the session may still burn a stall-recovery escalation."""
    if pin is None:
        return False
    esc_cfg = getattr(getattr(config, "session", None), "escalation", None)
    cap = getattr(esc_cfg, "retry_on_failure_max_per_session", 2) if esc_cfg else 2
    return pin.escalation.retry_count < cap


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


def _build_stream_postfix(route, pin, _stream_usage, _tt_enabled, _tt_show, _ctx_window=0) -> str:
    """Build the postfix text for the end of a stream."""
    if _tt_enabled and _stream_usage is not None:
        s_prompt, s_completion = extract_tokens(_stream_usage)
        token_usage_for_postfix = pin.token_usage if pin is not None else {
            route.level.value: {"prompt": s_prompt, "completion": s_completion}
        }
        return build_token_postfix(
            route.level.value, token_usage_for_postfix, _tt_show,
            last_ctx_tokens=s_prompt, context_window=_ctx_window,
        )
    return build_token_postfix(route.level.value, None, _tt_show)


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


# Matches any trailing prefix of `[ipaddress-XX]` (e.g. `[`, `[ip`, `[ipaddr`, `[ipaddress-0`)
_IP_WORD_PREFIXES = "|".join("ipaddress"[:k] for k in range(len("ipaddress"), 0, -1))
_PARTIAL_TAIL_RE = re.compile(
    rf"\[\s*(?:ipaddress\s*-\s*\d{{0,2}}\s*\]?|ipaddress\s*|{_IP_WORD_PREFIXES})?\Z",
    re.IGNORECASE,
)


def _split_carry(text: str, guardrail_mask_active: bool) -> tuple[str, str]:
    """Split text into (flushable, carry) around a possible partial token tail."""
    tail_window = text[-24:]
    if "[" in tail_window:
        m = _PARTIAL_TAIL_RE.search(tail_window)
        if m:
            split_at = len(text) - len(tail_window) + m.start()
            return text[:split_at], text[split_at:]
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
    if rehydrate_engine is not None and redaction_key and "ipaddress" in text.lower():
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
    rehydrate_active = rehydrate_engine is not None and bool(redaction_key)
    if (
        not guardrail_mask_active
        and not carry
        and not (rehydrate_active and ("[" in line or "ipaddress" in line.lower()))
    ):
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
    request, payload, body, route, fallbacks, session_id, session_source, pin, include_metadata, start,
    redaction_key=None, tier_base_url=None, tier_api_key=None, deadline=None,
    tried_models: set[str] | None = None,
):
    """Handle streaming request — pass through SSE chunks with pre-content recovery."""
    provider = request.app.state.provider
    config = request.app.state.config.get()
    if deadline is None:
        deadline = _provider_deadline(config)

    def adapter(m, p):
        return _adapt_payload_for_model(p, m, route, session_id, config, provider)
    seen_models: set[str] = set(tried_models) if tried_models is not None else set()

    upstream_start = time.monotonic()
    _, stream_resp, model_used, fallback_used, error = await provider.chat_completion(
        payload, fallback_models=fallbacks, stream=True,
        base_url=tier_base_url, api_key=tier_api_key, deadline=deadline,
        payload_adapter=adapter, skip_models=seen_models or None,
    )

    if error and stream_resp is None:
        for m in [route.model, *list(fallbacks or [])]:
            if m:
                seen_models.add(m)
        # Mark the pin for next-turn escalation
        _mark_upstream_failure(pin, config)

        esc_cfg = getattr(getattr(config, "session", None), "escalation", None)
        retry_cfg = getattr(esc_cfg, "retry_on_failure", False) if esc_cfg else False

        # Attempt immediate retry with a higher tier if enabled
        if retry_cfg and pin is not None:
            routing_engine = request.app.state.routing_engine
            store = request.app.state.session_store
            retry_resp = await _retry_with_higher_tier(
                request, body, route, pin, config,
                routing_engine, include_metadata, start, session_id,
                session_source, redaction_key, store,
                tried_models=seen_models, deadline=deadline,
            )
            if retry_resp is not None:
                return retry_resp

        # Last-resort cross-tier emergency fallback (without re-pinning)
        if (
            retry_cfg
            and tried_models is None
            and (deadline is None or (deadline - time.monotonic()) > 2)
        ):
            emergency = _collect_cross_tier_emergency_models(config, route.level, seen_models)
            for em_model, em_base_url, em_api_key in emergency:
                if deadline is not None and (deadline - time.monotonic()) <= 1:
                    break
                seen_models.add(em_model)
                em_payload = _adapt_payload_for_model(
                    payload, em_model, route, session_id, config, provider,
                )
                _, em_stream, em_used, _, em_err = await provider.chat_completion(
                    em_payload, fallback_models=[], stream=True,
                    base_url=em_base_url, api_key=em_api_key, deadline=deadline,
                )
                if em_stream is not None and not em_err:
                    logger.warning(
                        "router.stream.emergency_cross_tier_fallback.succeeded",
                        session_id=session_id,
                        level=route.level.value,
                        from_model=route.model,
                        to_model=em_used,
                    )
                    stream_resp = em_stream
                    model_used = em_used
                    fallback_used = True
                    error = None
                    break

    if error and stream_resp is None:
        status = 502 if "exhausted" in error else 504
        return JSONResponse(
            status_code=status,
            content={"error": {
                "message": error, "type": "upstream_error", "param": None, "code": f"router_upstream_{status}",
            }},
        )

    total_ms = int((time.monotonic() - start) * 1000)
    initial_tried = set(seen_models)
    if model_used:
        initial_tried.add(model_used)
    # If initial FallbackExecutor already landed on a fallback model, earlier
    # models in the chain have already failed — record them so we don't retry them.
    if fallback_used and route.model:
        initial_tried.add(route.model)
        for fb in fallbacks or []:
            if fb == model_used:
                break
            initial_tried.add(fb)

    async def stream_generator():
        """Pass through SSE chunks from upstream with transparent pre-content recovery.

        Handles four upstream failure classes before any visible content has
        reached the client:
          1. First-token stalls (including endless `: OPENROUTER PROCESSING` keepalives)
          2. HTTP-200 SSE-embedded upstream errors (`data: {"error": ...}`)
          3. Pre-content transport/read drops (`RemoteProtocolError`, `ReadError`, ...)
          4. Empty upstream streams (EOF / `[DONE]` with zero content or tool calls)

        Recovery order (respecting the request wall-clock deadline):
          1. Untried fallback models in the current tier (no re-pin)
          2. Escalation to the next higher tier with untried models
          3. Emergency cross-tier fallback across other configured tiers (no re-pin)
        """
        nonlocal stream_resp, model_used, route, fallback_used

        rehydrate_engine = getattr(request.app.state, "ip_redaction", None)
        guardrail_engine: GuardrailEngine | None = getattr(
            request.app.state, "guardrails", None
        )
        _guardrail_mask_active = (
            guardrail_engine is not None
            and guardrail_engine.config.output_enabled
            and guardrail_engine.config.output_action in ("mask", "block")
        )

        _config = request.app.state.config.get()
        _tt_cfg = getattr(getattr(_config, "telemetry", None), "token_tracking", None)
        _tt_enabled = _tt_cfg is not None and getattr(_tt_cfg, "enabled", True)
        _tt_show = _tt_cfg is not None and getattr(_tt_cfg, "show_in_postfix", True)
        _provider_cfg = getattr(_config, "provider", None)
        _ctx_window = getattr(_provider_cfg, "context_window", 1_000_000) if _provider_cfg else 1_000_000
        _first_timeout, _idle_timeout = _stall_timeouts(_config)
        _routing_engine = getattr(request.app.state, "routing_engine", None)

        _stream_usage: dict[str, Any] | None = None
        _stream_had_tool_calls = False
        _emitted_to_client = False
        metadata_sent = False
        _tried_models: set[str] = set(initial_tried)
        _current_fallbacks: list[str] = list(fallbacks or [])
        _current_tier_base_url = tier_base_url
        _current_tier_api_key = tier_api_key

        def _can_try_emergency() -> bool:
            esc = getattr(getattr(_config, "session", None), "escalation", None)
            if esc is None or not getattr(esc, "retry_on_failure", False):
                return False
            cap = getattr(esc, "retry_on_failure_max_per_session", 2)
            return cap > 0

        def _has_any_recovery_candidate() -> bool:
            if any(m and m not in _tried_models for m in _current_fallbacks):
                return True
            if _stall_retry_budget_left(pin, _config) and _routing_engine is not None:
                try:
                    gmax = Level.from_str(_config.routing.global_max_level)
                    if route.level.numeric < gmax.numeric:
                        return True
                except (ValueError, AttributeError):
                    pass
            return bool(
                _can_try_emergency()
                and _collect_cross_tier_emergency_models(_config, route.level, _tried_models)
            )

        async def _open_next_recovery_stream(failure_exc: Exception) -> bool:
            """Attempt to open a healthy replacement stream. Returns True on success."""
            nonlocal stream_resp, model_used, route, fallback_used
            nonlocal _current_fallbacks, _current_tier_base_url, _current_tier_api_key

            while True:
                if deadline is not None and (deadline - time.monotonic()) <= 5:
                    return False

                # Priority 1: remaining untried fallback models in the CURRENT tier
                # (preserves session pin and does not burn escalation budget).
                rem_tier_fbs = [m for m in _current_fallbacks if m and m not in _tried_models]
                if rem_tier_fbs:
                    cand_primary = rem_tier_fbs[0]
                    cand_fbs = rem_tier_fbs[1:]

                    def cand_adapter(m, p, _r=route):
                        return _adapt_payload_for_model(
                            p, m, _r, session_id, _config, provider,
                        )

                    cand_payload = _adapt_payload_for_model(
                        payload, cand_primary, route, session_id, _config, provider,
                    )
                    logger.info(
                        "router.stream.fallback_recovery",
                        session_id=session_id,
                        level=route.level.value,
                        failed_model=model_used,
                        next_model=cand_primary,
                        reason=str(failure_exc)[:200],
                    )
                    _, new_resp, new_model, _, retry_error = await provider.chat_completion(
                        cand_payload, fallback_models=cand_fbs, stream=True,
                        base_url=_current_tier_base_url, api_key=_current_tier_api_key,
                        deadline=deadline, payload_adapter=cand_adapter,
                        skip_models=_tried_models,
                    )
                    if retry_error or new_resp is None:
                        _tried_models.update(rem_tier_fbs)
                        continue
                    # Mark any skipped fallback models before new_model as tried too
                    for m in rem_tier_fbs:
                        _tried_models.add(m)
                        if m == new_model:
                            break
                    stream_resp = new_resp
                    model_used = new_model
                    fallback_used = True
                    return True

                # Priority 2: escalate to next higher tier with untried models.
                if _stall_retry_budget_left(pin, _config):
                    old_level = route.level
                    old_pin_state = (
                        (pin.level, pin.model, pin.params) if pin is not None else None
                    )
                    new_route = _next_tier_route(
                        route, pin, _config, _routing_engine, skip_models=_tried_models,
                    )
                    if new_route is not None:
                        router_retry_on_failure_total.labels(
                            from_level=old_level.value, outcome="stall_escalated",
                        ).inc()
                        logger.info(
                            "router.stream.stall_escalated",
                            session_id=session_id,
                            from_level=old_level.value, to_level=new_route.level.value,
                            model=new_route.model,
                        )
                        retry_fallbacks = list(_config.routing.get_fallbacks(new_route.level.value) or [])
                        r_base_url, r_api_key = _resolve_tier_provider(_config, new_route.level.value)

                        def esc_adapter(m, p, _r=new_route):
                            return _adapt_payload_for_model(
                                p, m, _r, session_id, _config, provider,
                            )

                        retry_payload = _adapt_payload_for_model(
                            payload, new_route.model, new_route, session_id, _config, provider,
                        )
                        _, new_resp, new_model, new_fb, retry_error = await provider.chat_completion(
                            retry_payload, fallback_models=retry_fallbacks, stream=True,
                            base_url=r_base_url, api_key=r_api_key, deadline=deadline,
                            payload_adapter=esc_adapter, skip_models=_tried_models,
                        )
                        if retry_error or new_resp is None:
                            _tried_models.add(new_route.model)
                            _tried_models.update(retry_fallbacks)
                            if pin is not None and old_pin_state is not None:
                                pin.level, pin.model, pin.params = old_pin_state
                            # Continue loop to try next tier / emergency pool
                            route = new_route
                            continue
                        route = new_route
                        _current_fallbacks = retry_fallbacks
                        _current_tier_base_url = r_base_url
                        _current_tier_api_key = r_api_key
                        _tried_models.add(new_model)
                        stream_resp = new_resp
                        model_used = new_model
                        fallback_used = new_fb
                        if pin is not None:
                            await request.app.state.session_store.put(pin)
                        return True

                # Priority 3: emergency cross-tier fallback (without re-pinning)
                if _can_try_emergency():
                    emergency = _collect_cross_tier_emergency_models(
                        _config, route.level, _tried_models,
                    )
                    if emergency:
                        em_model, em_base_url, em_api_key = emergency[0]
                        _tried_models.add(em_model)
                        em_payload = _adapt_payload_for_model(
                            payload, em_model, route, session_id, _config, provider,
                        )
                        logger.warning(
                            "router.stream.emergency_cross_tier_fallback",
                            session_id=session_id,
                            level=route.level.value,
                            failed_model=model_used,
                            next_model=em_model,
                        )
                        _, new_resp, new_model, _, retry_error = await provider.chat_completion(
                            em_payload, fallback_models=[], stream=True,
                            base_url=em_base_url, api_key=em_api_key, deadline=deadline,
                        )
                        if retry_error or new_resp is None:
                            continue
                        stream_resp = new_resp
                        model_used = new_model
                        fallback_used = True
                        return True

                return False

        router_active_requests.inc()
        try:
            while True:
                carry = ""
                recoverable_exc: Exception | None = None
                seen_done = False
                try:
                    async for line in _aiter_lines_with_stall_guard(
                        stream_resp, _first_timeout, _idle_timeout,
                    ):
                        # Pass through SSE comments (`: OPENROUTER PROCESSING`) and
                        # blank lines without locking out pre-content recovery.
                        if _is_sse_keepalive_or_blank(line):
                            yield f"{line}\n"
                            continue

                        if line.strip() == "data: [DONE]":
                            seen_done = True
                            if (
                                not _emitted_to_client
                                and not carry
                                and _has_any_recovery_candidate()
                            ):
                                # Upstream returned an empty stream (zero choices/content/tools).
                                raise _StreamUpstreamError(
                                    f"upstream {model_used} returned empty stream before [DONE]"
                                )
                            if carry:
                                if rehydrate_engine is not None and redaction_key and "ipaddress" in carry.lower():
                                    carry = rehydrate_engine.rehydrate_text_sync(carry, redaction_key)
                                if _guardrail_mask_active and guardrail_engine is not None:
                                    carry = _mask_carry_flush(carry, guardrail_engine)
                                flush_event = {"choices": [{"delta": {"content": carry}}]}
                                yield f"data: {json.dumps(flush_event)}\n\n"
                                _emitted_to_client = True
                                carry = ""

                            if _tt_enabled and _stream_usage is not None:
                                await _finalize_stream_token_tracking(
                                    pin, route, model_used, provider, _stream_usage, request,
                                )
                            postfix_text = _build_stream_postfix(
                                route, pin, _stream_usage, _tt_enabled, _tt_show, _ctx_window,
                            )
                            if not _stream_had_tool_calls and _emitted_to_client:
                                postfix_event = {"choices": [{"delta": {"content": f"\n\n{postfix_text}"}}]}
                                yield f"data: {json.dumps(postfix_event)}\n\n"
                            yield f"{line}\n"
                            break

                        line_kind = "content"
                        if line.startswith("data: "):
                            line_kind, err_detail = _inspect_sse_data_line(line)
                            if line_kind == "error" and not _emitted_to_client:
                                raise _StreamUpstreamError(
                                    f"upstream SSE error from {model_used}: {err_detail}"
                                )
                            _stream_usage, _stream_had_tool_calls = _process_data_line(
                                line, _stream_usage, _stream_had_tool_calls,
                            )

                        if not metadata_sent:
                            metadata_sent = True
                            if include_metadata:
                                _meta = {"router": {
                                    "level": route.level.value,
                                    "model": f"smart-router/{route.level.value}",
                                    "session_id": session_id,
                                    "turn": pin.turn_count if pin else 0,
                                    "classification_source": route.classification.source.value,
                                    "fallback_used": fallback_used,
                                }}
                                yield f"data: {json.dumps(_meta)}\n\n"

                        line, carry = await _rehydrate_line(
                            line, carry,
                            rehydrate_engine, redaction_key, guardrail_engine, _guardrail_mask_active,
                        )
                        if line_kind == "content":
                            _emitted_to_client = True
                        yield f"{line}\n"
                except _StreamStalled as e:
                    if e.first_token and not _emitted_to_client:
                        recoverable_exc = e
                    else:
                        await _safe_aclose(stream_resp)
                        stream_resp = None
                        _record_stream_finally(route, model_used, upstream_start, fallback_used)
                        _mark_upstream_failure(pin, _config)
                        router_stream_errors_total.labels(
                            level=route.level.value, model=model_used, kind="upstream_stalled",
                        ).inc()
                        logger.warning(
                            "router.stream.stalled",
                            level=route.level.value, model=model_used, session_id=session_id,
                            waited_s=e.waited, first_token=e.first_token,
                        )
                        error_data = _handle_stream_error(
                            TimeoutError(str(e)), route, model_used, session_id, upstream_start,
                        )
                        yield f"data: {json.dumps(error_data)}\n\n"
                        return
                except _StreamUpstreamError as e:
                    recoverable_exc = e
                except Exception as e:
                    if not _emitted_to_client:
                        recoverable_exc = e
                    else:
                        error_data = _handle_stream_error(e, route, model_used, session_id, upstream_start)
                        _mark_upstream_failure(pin, request.app.state.config.get())
                        yield f"data: {json.dumps(error_data)}\n\n"
                        await _safe_aclose(stream_resp)
                        stream_resp = None
                        _record_stream_finally(route, model_used, upstream_start, fallback_used)
                        return
                else:
                    if not seen_done and not _emitted_to_client and _has_any_recovery_candidate():
                        recoverable_exc = _StreamUpstreamError(
                            f"upstream {model_used} closed stream before first token"
                        )
                    else:
                        _record_stream_success(route, model_used)
                        _clear_upstream_failure(pin)
                        await _safe_aclose(stream_resp)
                        stream_resp = None
                        _record_stream_finally(route, model_used, upstream_start, fallback_used)
                        return

                # --- recoverable pre-content failure -------------------------
                await _safe_aclose(stream_resp)
                stream_resp = None
                _record_stream_finally(route, model_used, upstream_start, fallback_used)
                _mark_upstream_failure(pin, _config)
                if model_used:
                    _tried_models.add(model_used)

                is_stall = isinstance(recoverable_exc, _StreamStalled)
                if is_stall:
                    router_stream_errors_total.labels(
                        level=route.level.value, model=model_used, kind="upstream_stalled",
                    ).inc()
                    logger.warning(
                        "router.stream.stalled",
                        level=route.level.value, model=model_used, session_id=session_id,
                        waited_s=recoverable_exc.waited, first_token=recoverable_exc.first_token,
                    )
                else:
                    logger.warning(
                        "router.stream.pre_content_failure",
                        level=route.level.value, model=model_used, session_id=session_id,
                        error=str(recoverable_exc)[:300],
                    )

                # Emit SSE keepalive comment while switching upstream so client stays alive
                yield ": smart-router recovering\n\n"
                recovered = await _open_next_recovery_stream(recoverable_exc)
                if not recovered:
                    final_exc = (
                        TimeoutError(str(recoverable_exc))
                        if is_stall
                        else recoverable_exc
                    )
                    error_data = _handle_stream_error(
                        final_exc, route, model_used, session_id, upstream_start,
                    )
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                _stream_usage = None
                _stream_had_tool_calls = False
        finally:
            router_active_requests.dec()
            await _safe_aclose(stream_resp)

    response = StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
    )
    _add_router_headers(response, route, session_id, session_source, pin, total_ms, fallback_used)
    return response


async def _safe_aclose(resp) -> None:
    """Close an upstream streaming response, ignoring teardown errors."""
    if resp is None:
        return
    try:
        await resp.aclose()
    except Exception:  # teardown must never mask the real error
        logger.debug("router.stream.aclose_failed")


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
    last_ctx_tokens: int = 0,
    context_window: int = 0,
) -> None:
    """Append a compact model marker to assistant content for user visibility.

    When ``show_tokens`` is True and ``token_usage`` contains data, the
    marker includes cumulative per-tier token usage plus the last call's
    context-window consumption and the configured limit::

        [smart-router/L1-In:3032|Out:1000, L2-In:10021|Out:6054/Ctx:6100/1M]

    Otherwise falls back to the classic format::

        [smart-router/L1]
    """
    marker = build_token_postfix(
        route.level.value, token_usage, show_tokens,
        last_ctx_tokens, context_window,
    )
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

        rehydrate_engine = getattr(request.app.state, "ip_redaction", None)
        guardrail_engine: GuardrailEngine | None = getattr(request.app.state, "guardrails", None)
        guardrail_mask_active = (
            guardrail_engine is not None
            and guardrail_engine.config.output_enabled
            and guardrail_engine.config.output_action in ("mask", "block")
        )

        async def gen():
            carry = ""
            try:
                async for line in stream_resp.aiter_lines():
                    if line.strip() == "data: [DONE]":
                        if carry:
                            if rehydrate_engine is not None and redaction_key and "ipaddress" in carry.lower():
                                carry = rehydrate_engine.rehydrate_text_sync(carry, redaction_key)
                            if guardrail_mask_active and guardrail_engine is not None:
                                carry = _mask_carry_flush(carry, guardrail_engine)
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': carry}}]})}\n\n"
                        yield f"{line}\n"
                        break
                    line, carry = await _rehydrate_line(
                        line, carry,
                        rehydrate_engine, redaction_key, guardrail_engine, guardrail_mask_active,
                    )
                    yield f"{line}\n"
            finally:
                await _safe_aclose(stream_resp)

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
    *, redaction_key=None,
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
        redaction_key=redaction_key,
    )
