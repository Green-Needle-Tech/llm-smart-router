"""Regression tests for long-task resilience in Hermes Agent integrations.

Covers the failure modes that caused llm-smart-router to fail mid-task and
trigger Hermes Agent's client-side fallback_providers:
  1. OpenRouter `: OPENROUTER PROCESSING` SSE comments defeating first-token stall recovery
  2. HTTP-200 SSE-embedded error events (`data: {"error": ...}`) passed through without fallback
  3. Empty / abruptly closed upstream streams and pre-content transport errors
  4. Stream stall skipping same-tier fallbacks and re-trying duplicate models across L3->L4
  5. L5->L5 self-escalation bug + cross-tier emergency failover without re-pinning
  6. Deterministic 4xx fast-advance and HTTP 400 max_tokens/context overflow auto-recovery
  7. Context-aware max_tokens clamping and per-model payload adaptation (reasoning & cache_control)
  8. Long-session request bounds (>500 messages, >256KB tool output, >50 tools)
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest
import respx

from app.api.chat import (
    _adapt_payload_for_model,
    _apply_escalation,
    _apply_max_tokens,
    _handle_non_stream,
    _handle_stream,
)
from app.config.loader import ConfigManager
from app.config.schema import Settings
from app.middleware.request_limits import validate_request_bounds
from app.routing.engine import RoutingEngine
from app.routing.fallback import FallbackExecutor
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.schemas.router import (
    ClassificationResult,
    ClassificationSource,
    EscalationState,
    Level,
    RouteDecision,
    SessionPin,
)

# ---------------------------------------------------------------------------
# Helpers & Fakes
# ---------------------------------------------------------------------------


class _KeepaliveThenHangStream:
    """Emits `: OPENROUTER PROCESSING` keepalive comments forever (simulating queued upstream)."""

    def __init__(self, interval: float = 0.05):
        self._interval = interval
        self.closed = False

    def aiter_lines(self):
        async def gen():
            while True:
                yield ": OPENROUTER PROCESSING"
                yield ""
                await asyncio.sleep(self._interval)
        return gen()

    async def aclose(self):
        self.closed = True


class _LinesStream:
    """Emits a fixed sequence of SSE lines, optionally raising an exception."""

    def __init__(self, lines: list[str], exc: Exception | None = None, delay: float = 0.0):
        self._lines = list(lines)
        self._exc = exc
        self._delay = delay
        self.closed = False

    def aiter_lines(self):
        async def gen():
            for line in self._lines:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield line
            if self._exc is not None:
                raise self._exc
        return gen()

    async def aclose(self):
        self.closed = True


class _ModelRouterProvider:
    """Fake provider that dispatches by requested model name."""

    def __init__(self, responses: dict[str, list]):
        self._responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[dict] = []

    async def chat_completion(
        self,
        payload,
        fallback_models=None,
        *,
        stream=False,
        base_url=None,
        api_key=None,
        deadline=None,
        payload_adapter=None,
        skip_models=None,
    ):
        skip = skip_models or set()
        candidates = []
        for m in [payload.get("model"), *(fallback_models or [])]:
            if m and m not in candidates and m not in skip:
                candidates.append(m)

        last_err = None
        for m in candidates:
            eff_payload = payload_adapter(m, payload) if payload_adapter else {**payload, "model": m}
            self.calls.append({"model": m, "payload": eff_payload, "stream": stream})
            queue = self._responses.get(m, [])
            if not queue:
                last_err = f"no mock response for {m}"
                continue
            item = queue.pop(0)
            if isinstance(item, str) and item.startswith("ERR:"):
                last_err = item[4:]
                continue
            if stream:
                return None, item, m, m != payload.get("model"), None
            return item, None, m, m != payload.get("model"), None

        return None, None, payload.get("model", ""), False, last_err or "all fallbacks exhausted"

    def get_max_completion_tokens(self, model: str) -> int | None:
        return 65536

    def get_context_length(self, model: str) -> int | None:
        return 200_000

    def estimate_cost(self, *a, **k):
        return None


class _FakeStore:
    def __init__(self):
        self.pins: dict[str, SessionPin] = {}

    async def put(self, pin: SessionPin):
        self.pins[pin.session_id] = pin


def _settings_with_tiers() -> Settings:
    return Settings.model_validate({
        "version": 1,
        "provider": {
            "timeout_seconds": 120,
            "max_retries": 0,
            "retry_backoff_seconds": 0.0,
            "stream_first_token_timeout_seconds": 1,
            "stream_idle_timeout_seconds": 2,
            "request_deadline_seconds": 900,
            "reasoning_mandatory_models": ["z-ai/glm-5.3", "z-ai/glm-5.3-flash"],
        },
        "session": {
            "escalation": {
                "enabled": True,
                "retry_on_failure": True,
                "retry_on_failure_max_per_session": 3,
            }
        },
        "routing": {
            "global_max_level": "L5",
            "global_min_level": "L1",
            "L1": {"label": "trivial", "model": "z-ai/glm-5.3-flash", "fallbacks": ["google/gemini-2.5-flash"]},
            "L2": {"label": "routine", "model": "z-ai/glm-5.3-flash", "fallbacks": ["openai/gpt-5.6-luna"]},
            "L3": {"label": "intermediate", "model": "z-ai/glm-5.3", "fallbacks": ["openai/gpt-5.6-luna", "z-ai/glm-5.2"]},
            "L4": {"label": "advanced", "model": "z-ai/glm-5.3", "fallbacks": ["z-ai/glm-5.2", "anthropic/claude-opus-5"]},
            "L5": {"label": "expert", "model": "anthropic/claude-opus-5", "fallbacks": ["z-ai/glm-5.3", "openai/gpt-5.6-sol"]},
        },
        "telemetry": {
            "token_tracking": {"enabled": False, "show_in_postfix": False},
        },
    })


def _make_request(provider, cfg: Settings, store: _FakeStore | None = None):
    cm = SimpleNamespace(get=lambda: cfg)
    state = SimpleNamespace(
        provider=provider,
        ip_redaction=None,
        guardrails=None,
        config=cm,
        routing_engine=RoutingEngine(cm),
        session_store=store or _FakeStore(),
        budget_manager=None,
        version="2.31.1",
    )
    return SimpleNamespace(app=SimpleNamespace(state=state, url=None))


def _route(level: Level, model: str) -> RouteDecision:
    return RouteDecision(
        level=level,
        model=model,
        params={},
        classification=ClassificationResult(
            level=level, source=ClassificationSource.SESSION,
        ),
    )


def _pin(level: Level, model: str) -> SessionPin:
    pin = SessionPin(session_id="sess-long", level=level, model=model)
    pin.escalation = EscalationState()
    return pin


def _ok_sse(text: str = "healthy answer") -> _LinesStream:
    return _LinesStream([
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}',
        "",
        "data: [DONE]",
        "",
    ])


def _ok_json(model: str, text: str = "ok") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


async def _drain(resp) -> str:
    return "".join([chunk async for chunk in resp.body_iterator])


# ---------------------------------------------------------------------------
# 1. Endless `: OPENROUTER PROCESSING` keepalives must NOT defeat stall recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keepalive_comments_do_not_defeat_first_token_stall_recovery():
    """OpenRouter keepalive comments must not reset first_timeout or lock out recovery."""
    cfg = _settings_with_tiers()
    # Shorten first-token timeout for fast test
    cfg.provider.stream_first_token_timeout_seconds = 1  # int field, override below via SimpleNamespace wrapper

    fast_provider_cfg = SimpleNamespace(
        **cfg.provider.model_dump(),
    )
    fast_provider_cfg.stream_first_token_timeout_seconds = 0.25
    fast_provider_cfg.stream_idle_timeout_seconds = 2.0
    cfg_ns = SimpleNamespace(
        provider=fast_provider_cfg,
        routing=cfg.routing,
        session=cfg.session,
        telemetry=cfg.telemetry,
        budget=cfg.budget,
    )

    hanging = _KeepaliveThenHangStream(interval=0.04)
    fallback_ok = _ok_sse("recovered via L3 fallback")
    provider = _ModelRouterProvider({
        "z-ai/glm-5.3": [hanging],
        "openai/gpt-5.6-luna": [fallback_ok],
    })
    store = _FakeStore()
    req = _make_request(provider, cfg_ns, store)
    pin = _pin(Level.L3, "z-ai/glm-5.3")

    resp = await _handle_stream(
        req,
        {"model": "z-ai/glm-5.3", "messages": [{"role": "user", "content": "long task"}]},
        ChatCompletionRequest(model="smart-router", messages=[ChatMessage(role="user", content="long task")], stream=True),
        _route(Level.L3, "z-ai/glm-5.3"),
        ["openai/gpt-5.6-luna", "z-ai/glm-5.2"],
        "sess-long",
        None,
        pin,
        False,
        time.monotonic(),
    )
    body = await _drain(resp)

    assert "recovered via L3 fallback" in body
    assert '"error"' not in body
    assert "data: [DONE]" in body
    assert hanging.closed
    # Recovered within same tier — session pin stayed at L3 and retry_count stayed 0!
    assert pin.level == Level.L3
    assert pin.escalation.retry_count == 0
    assert [c["model"] for c in provider.calls] == ["z-ai/glm-5.3", "openai/gpt-5.6-luna"]


# ---------------------------------------------------------------------------
# 2. HTTP-200 SSE-embedded error recovers transparently without exposing error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_embedded_error_recovers_transparently():
    """OpenRouter HTTP-200 stream carrying `data: {"error": ...}` must fail over."""
    cfg = _settings_with_tiers()
    sse_err = _LinesStream([
        ": OPENROUTER PROCESSING",
        "",
        'data: {"error":{"code":502,"message":"Provider returned error"}}',
        "",
    ])
    provider = _ModelRouterProvider({
        "z-ai/glm-5.3": [sse_err],
        "openai/gpt-5.6-luna": [_ok_sse("clean fallback response")],
    })
    req = _make_request(provider, cfg)
    pin = _pin(Level.L3, "z-ai/glm-5.3")

    resp = await _handle_stream(
        req,
        {"model": "z-ai/glm-5.3", "messages": [{"role": "user", "content": "step 42"}]},
        ChatCompletionRequest(model="smart-router", messages=[ChatMessage(role="user", content="step 42")], stream=True),
        _route(Level.L3, "z-ai/glm-5.3"),
        ["openai/gpt-5.6-luna", "z-ai/glm-5.2"],
        "sess-long",
        None,
        pin,
        False,
        time.monotonic(),
    )
    body = await _drain(resp)

    assert "clean fallback response" in body
    assert "Provider returned error" not in body
    assert '"error"' not in body
    assert "data: [DONE]" in body
    assert sse_err.closed


# ---------------------------------------------------------------------------
# 3. Empty upstream stream & pre-content transport drop recover transparently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_stream_and_pre_content_transport_drop_recover():
    """Empty stream (only keepalives then EOF) and pre-content ReadError both fail over."""
    cfg = _settings_with_tiers()
    empty_stream = _LinesStream([": OPENROUTER PROCESSING", ""])
    drop_stream = _LinesStream(
        [": OPENROUTER PROCESSING"],
        exc=httpx.RemoteProtocolError("Server disconnected without sending a response"),
    )
    provider = _ModelRouterProvider({
        "z-ai/glm-5.3": [empty_stream],
        "openai/gpt-5.6-luna": [drop_stream],
        "z-ai/glm-5.2": [_ok_sse("third model succeeded")],
    })
    req = _make_request(provider, cfg)
    pin = _pin(Level.L3, "z-ai/glm-5.3")

    resp = await _handle_stream(
        req,
        {"model": "z-ai/glm-5.3", "messages": [{"role": "user", "content": "continue"}]},
        ChatCompletionRequest(model="smart-router", messages=[ChatMessage(role="user", content="continue")], stream=True),
        _route(Level.L3, "z-ai/glm-5.3"),
        ["openai/gpt-5.6-luna", "z-ai/glm-5.2"],
        "sess-long",
        None,
        pin,
        False,
        time.monotonic(),
    )
    body = await _drain(resp)

    assert "third model succeeded" in body
    assert '"error"' not in body
    assert [c["model"] for c in provider.calls] == [
        "z-ai/glm-5.3",
        "openai/gpt-5.6-luna",
        "z-ai/glm-5.2",
    ]


# ---------------------------------------------------------------------------
# 4. L3 -> L4 escalation skips duplicate models shared across tiers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_skips_already_failed_models_shared_by_adjacent_tiers():
    """When L3 (glm-5.3, gpt-5.6-luna, glm-5.2) fails, L4 skips glm-5.3 & glm-5.2."""
    cfg = _settings_with_tiers()
    provider = _ModelRouterProvider({
        "z-ai/glm-5.3": ["ERR:upstream 503"],
        "openai/gpt-5.6-luna": ["ERR:upstream 503"],
        "z-ai/glm-5.2": ["ERR:upstream 503"],
        "anthropic/claude-opus-5": [_ok_json("anthropic/claude-opus-5", "rescued at L4")],
    })
    store = _FakeStore()
    req = _make_request(provider, cfg, store)
    pin = _pin(Level.L3, "z-ai/glm-5.3")
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="user", content="hard debugging")],
        stream=False,
    )

    resp = await _handle_non_stream(
        req,
        {"model": "z-ai/glm-5.3", "messages": [{"role": "user", "content": "hard debugging"}]},
        body,
        _route(Level.L3, "z-ai/glm-5.3"),
        ["openai/gpt-5.6-luna", "z-ai/glm-5.2"],
        "sess-long",
        None,
        pin,
        False,
        time.monotonic(),
    )

    assert resp.status_code == 200
    assert "rescued at L4" in resp.body.decode()
    called_models = [c["model"] for c in provider.calls]
    # glm-5.3 and glm-5.2 must each be tried ONLY ONCE (not re-tried at L4!)
    assert called_models == [
        "z-ai/glm-5.3",
        "openai/gpt-5.6-luna",
        "z-ai/glm-5.2",
        "anthropic/claude-opus-5",
    ]


# ---------------------------------------------------------------------------
# 5. L5 never self-escalates L5->L5 & cross-tier emergency fallback rescues turn
# ---------------------------------------------------------------------------


def test_apply_escalation_does_not_self_escalate_at_l5():
    """_apply_escalation at L5 with global_max_level=L5 must return None."""
    cfg = _settings_with_tiers()
    pin = _pin(Level.L5, "anthropic/claude-opus-5")
    pin.escalation.score = 10
    pin.turn_count = 20

    result = _apply_escalation(pin, [("repair_language", 3)], cfg.session.escalation, cfg)
    assert result is None
    assert pin.level == Level.L5
    assert pin.escalation.count == 0


@pytest.mark.asyncio
async def test_l5_exhausted_uses_cross_tier_emergency_fallback_without_repinning():
    """When all L5 models fail, router tries untried models from other tiers instead of 502."""
    cfg = _settings_with_tiers()
    provider = _ModelRouterProvider({
        "anthropic/claude-opus-5": ["ERR:upstream 529"],
        "z-ai/glm-5.3": ["ERR:upstream 503"],
        "openai/gpt-5.6-sol": ["ERR:upstream 503"],
        # L4/L3 fallback model is healthy!
        "z-ai/glm-5.2": [_ok_json("z-ai/glm-5.2", "saved by cross-tier emergency")],
    })
    store = _FakeStore()
    req = _make_request(provider, cfg, store)
    pin = _pin(Level.L5, "anthropic/claude-opus-5")
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="user", content="long task step 99")],
        stream=False,
    )

    resp = await _handle_non_stream(
        req,
        {"model": "anthropic/claude-opus-5", "messages": [{"role": "user", "content": "long task step 99"}]},
        body,
        _route(Level.L5, "anthropic/claude-opus-5"),
        ["z-ai/glm-5.3", "openai/gpt-5.6-sol"],
        "sess-long",
        None,
        pin,
        False,
        time.monotonic(),
    )

    assert resp.status_code == 200
    assert "saved by cross-tier emergency" in resp.body.decode()
    assert resp.headers["x-router-fallback-used"] == "true"
    # Pin must remain at L5 (emergency fallback does NOT re-pin!)
    assert pin.level == Level.L5
    assert pin.escalation.retry_count == 0
    # L5 models were NOT retried via L5->L5 self-escalation
    called = [c["model"] for c in provider.calls]
    assert called == [
        "anthropic/claude-opus-5",
        "z-ai/glm-5.3",
        "openai/gpt-5.6-sol",
        "z-ai/glm-5.2",
    ]


# ---------------------------------------------------------------------------
# 6. FallbackExecutor: deterministic 4xx fast-advance + 400 max_tokens recovery
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_deterministic_404_advances_immediately_without_retries():
    """A 404 or non-max_tokens 400 must not burn max_retries on the same model."""
    cfg = ConfigManager(settings_path="/nonexistent").load()
    cfg.provider.max_retries = 2
    cfg.provider.retry_backoff_seconds = 0.0

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(404, json={"error": {"message": "model not found"}}),
            httpx.Response(200, json=_ok_json("fallback/m1")),
        ]
    )

    async with httpx.AsyncClient() as client:
        executor = FallbackExecutor(cfg, client)
        body, _, model_used, fb, err = await executor.execute_with_fallback(
            "bad/model", ["fallback/m1"], {"messages": [{"role": "user", "content": "hi"}]}, {},
        )

    assert err is None
    assert fb is True
    assert model_used == "fallback/m1"
    assert body is not None
    assert route.call_count == 2  # 1 failed 404 + 1 fallback success (no wasted retries)


@respx.mock
@pytest.mark.asyncio
async def test_400_max_tokens_overflow_retries_once_without_max_tokens():
    """HTTP 400 context/max_tokens overflow strips max_tokens and recovers on same model."""
    cfg = ConfigManager(settings_path="/nonexistent").load()
    cfg.provider.max_retries = 2
    cfg.provider.retry_backoff_seconds = 0.0

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                400,
                json={"error": {"message": "This endpoint's maximum context length is 200000 tokens. You requested 240000 (max_tokens too large)."}},
            ),
            httpx.Response(200, json=_ok_json("z-ai/glm-5.3", "fit without max_tokens")),
        ]
    )

    async with httpx.AsyncClient() as client:
        executor = FallbackExecutor(cfg, client)
        body, _, model_used, fb, err = await executor.execute_with_fallback(
            "z-ai/glm-5.3",
            ["z-ai/glm-5.2"],
            {"messages": [{"role": "user", "content": "long prompt"}], "max_tokens": 131072},
            {},
        )

    assert err is None
    assert fb is False
    assert model_used == "z-ai/glm-5.3"
    assert body["choices"][0]["message"]["content"] == "fit without max_tokens"
    assert route.call_count == 2
    second_payload = json.loads(route.calls[1].request.content)
    assert "max_tokens" not in second_payload


# ---------------------------------------------------------------------------
# 7. Context-aware max_tokens clamping & per-model payload adaptation
# ---------------------------------------------------------------------------


def test_apply_max_tokens_clamps_to_model_context_headroom():
    """Long prompts must clamp auto max_tokens so prompt + max_tokens <= context_length."""
    cfg = _settings_with_tiers()
    provider = SimpleNamespace(
        get_max_completion_tokens=lambda m: 131072,
        get_context_length=lambda m: 200_000,
    )
    # ~150,000 tokens (450,000 chars / 3)
    huge_payload = {
        "model": "z-ai/glm-5.3",
        "messages": [{"role": "user", "content": "a" * 450_000}],
    }
    _apply_max_tokens(huge_payload, _route(Level.L3, "z-ai/glm-5.3"), cfg, provider)
    assert "max_tokens" in huge_payload
    assert huge_payload["max_tokens"] < 55_000
    assert huge_payload["max_tokens"] >= 40_000


def test_adapt_payload_for_fallback_strips_anthropic_cache_control_and_reasoning_disable():
    """Falling back from Anthropic to a reasoning-mandatory model cleans incompatible fields."""
    cfg = _settings_with_tiers()
    provider = SimpleNamespace(
        get_max_completion_tokens=lambda m: 32768 if m == "z-ai/glm-5.3" else 65536,
        get_context_length=lambda m: 200_000,
    )
    base_payload = {
        "model": "anthropic/claude-opus-5",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Hermes.",
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    }
                ],
            },
            {"role": "user", "content": "hello"},
        ],
        "reasoning": {"enabled": False},
        "max_tokens": 65536,
    }
    adapted = _adapt_payload_for_model(
        base_payload, "z-ai/glm-5.3", _route(Level.L5, "anthropic/claude-opus-5"),
        "sess-1", cfg, provider,
    )

    assert adapted["model"] == "z-ai/glm-5.3"
    assert "reasoning" not in adapted
    assert adapted["max_tokens"] == 32768
    # Anthropic cache_control must be stripped for non-Anthropic fallback
    assert adapted["messages"][0]["content"] == "You are Hermes."


# ---------------------------------------------------------------------------
# 8. Long Hermes session request bounds (messages, tool outputs, MCP tools)
# ---------------------------------------------------------------------------


def test_long_hermes_session_passes_request_bounds():
    """Long tasks with >500 messages, >256KB tool output, and >50 tools must pass."""
    cfg = _settings_with_tiers()
    messages = [ChatMessage(role="user", content="start")]
    messages.extend(ChatMessage(role="tool", content="ok") for _ in range(800))
    messages.append(ChatMessage(role="tool", content="x" * 500_000))  # 500 KB tool result
    tools = [
        {"type": "function", "function": {"name": f"mcp_tool_{i}", "parameters": {"type": "object"}}}
        for i in range(100)
    ]
    req = ChatCompletionRequest(model="smart-router", messages=messages, tools=tools)

    assert validate_request_bounds(req, cfg) is None


@pytest.mark.asyncio
async def test_streaming_l5_exhausted_recovers_via_cross_tier_emergency_pool():
    """When all L5 models fail on stream (SSE error / transport drop), router recovers via L4/L3 pool."""
    cfg = _settings_with_tiers()
    provider = _ModelRouterProvider({
        "anthropic/claude-opus-5": [_LinesStream(['data: {"error":{"code":529,"message":"overloaded"}}'])],
        "z-ai/glm-5.3": [_LinesStream([], exc=httpx.ReadError("reset"))],
        "openai/gpt-5.6-sol": [_LinesStream([": OPENROUTER PROCESSING", ""])],
        "z-ai/glm-5.2": [_ok_sse("stream rescued by cross-tier pool")],
    })
    store = _FakeStore()
    req = _make_request(provider, cfg, store)
    pin = _pin(Level.L5, "anthropic/claude-opus-5")
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="user", content="long streaming task")],
        stream=True,
    )

    resp = await _handle_stream(
        req,
        {"model": "anthropic/claude-opus-5", "messages": [{"role": "user", "content": "long streaming task"}]},
        body,
        _route(Level.L5, "anthropic/claude-opus-5"),
        ["z-ai/glm-5.3", "openai/gpt-5.6-sol"],
        "sess-long",
        None,
        pin,
        False,
        time.monotonic(),
    )
    out = await _drain(resp)

    assert "stream rescued by cross-tier pool" in out
    assert '"error"' not in out
    assert "data: [DONE]" in out
    assert pin.level == Level.L5
    assert [c["model"] for c in provider.calls] == [
        "anthropic/claude-opus-5",
        "z-ai/glm-5.3",
        "openai/gpt-5.6-sol",
        "z-ai/glm-5.2",
    ]

