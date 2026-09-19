"""Regression tests for stream stall detection and auto-escalation (v2.19.0).

httpx's ``timeout`` is a PER-OPERATION read timeout whose clock resets on
every byte received.  An upstream model that trickles data slowly (or that
accepts the connection and then never sends a first token) therefore never
trips it, and the request hangs indefinitely.

Covers:
  1. First-token stall, recoverable   -> transparently restreamed from the
     next tier up; the client sees a complete answer, never an error.
  2. First-token stall, no budget     -> coded error event, no hang.
  3. Mid-stream stall after content   -> coded error (cannot restream, bytes
     already went to the client).
  4. Fallback chain wall-clock budget -> chain stops at the deadline instead
     of burning timeout_seconds per model.
  5. Slow-but-alive stream            -> NOT treated as a stall.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from app.api.chat import _handle_stream
from app.routing.fallback import FallbackExecutor
from app.schemas.router import (
    ClassificationResult,
    ClassificationSource,
    EscalationState,
    Level,
    RouteDecision,
    SessionPin,
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class _StallingStream:
    """Stream that emits ``lines``, sleeping ``stall_after`` seconds first.

    ``stall_at`` is the index at which the long sleep happens; the sleep is
    long enough that the stall guard must fire.
    """

    def __init__(self, lines, stall_at=0, stall_seconds=30.0):
        self._lines = lines
        self._stall_at = stall_at
        self._stall_seconds = stall_seconds
        self.closed = False

    def aiter_lines(self):
        async def gen():
            for idx, line in enumerate(self._lines):
                if idx == self._stall_at:
                    await asyncio.sleep(self._stall_seconds)
                yield line
        return gen()

    async def aclose(self):
        self.closed = True


class _HealthyStream:
    def __init__(self, lines, per_line_delay=0.0):
        self._lines = lines
        self._delay = per_line_delay
        self.closed = False

    def aiter_lines(self):
        async def gen():
            for line in self._lines:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield line
        return gen()

    async def aclose(self):
        self.closed = True


class _SequenceProvider:
    """Returns each queued stream in order, recording the models requested."""

    def __init__(self, streams, error_after=None):
        self._streams = list(streams)
        self._error_after = error_after
        self.calls = []

    async def chat_completion(self, payload, fallback_models=None, *, stream=False,
                              base_url=None, api_key=None, deadline=None, **kwargs):
        self.calls.append(payload.get("model"))
        if not self._streams:
            return None, None, payload.get("model"), False, "all fallbacks exhausted"
        resp = self._streams.pop(0)
        return None, resp, payload.get("model"), False, None

    def estimate_cost(self, *a, **k):
        return None


class _FakeRoutingEngine:
    def __init__(self, mapping):
        self._mapping = mapping

    def resolve_model_for_level(self, level):
        return self._mapping.get(level.value, "fallback/model")


class _FakeStore:
    def __init__(self):
        self.puts = 0

    async def put(self, pin):
        self.puts += 1


def _config(first_timeout=1.0, idle_timeout=1.0, deadline_seconds=900,
            max_level="L5", retry_cap=2):
    routing = SimpleNamespace(
        global_max_level=max_level,
        get_fallbacks=lambda level: [],
        get_params=lambda level: {},
        get_tier=lambda level: SimpleNamespace(base_url=None, api_key_env=None),
    )
    return SimpleNamespace(
        provider=SimpleNamespace(
            stream_first_token_timeout_seconds=first_timeout,
            stream_idle_timeout_seconds=idle_timeout,
            request_deadline_seconds=deadline_seconds,
            context_window=1_000_000,
            timeout_seconds=300,
            max_retries=2,
            retry_backoff_seconds=0.0,
            base_url="https://example.invalid/v1",
            retry_on_status=[429, 500, 502, 503, 504],
        ),
        routing=routing,
        session=SimpleNamespace(
            escalation=SimpleNamespace(
                retry_on_failure=True,
                retry_on_failure_max_per_session=retry_cap,
            ),
        ),
        telemetry=SimpleNamespace(token_tracking=SimpleNamespace(
            enabled=False, show_in_postfix=False)),
    )


class _FakeRequest:
    def __init__(self, provider, config, routing_engine=None, store=None):
        state = SimpleNamespace(
            provider=provider,
            ip_redaction=None,
            guardrails=None,
            config=SimpleNamespace(get=lambda: config),
            routing_engine=routing_engine,
            session_store=store or _FakeStore(),
        )
        self.app = SimpleNamespace(state=state, url=None)


def _route(level=Level.L3, model="tier/model"):
    return RouteDecision(
        level=level, model=model, params={},
        classification=ClassificationResult(
            level=level, source=ClassificationSource.SESSION,
        ),
    )


def _pin(level=Level.L3, retry_count=0):
    pin = SessionPin(
        session_id="sess-stall", level=level, model="tier/model",
    )
    pin.escalation = EscalationState()
    pin.escalation.retry_count = retry_count
    return pin


def _lines(text="Hello"):
    return [
        'data: {"choices":[{"delta":{"content":"%s"}}]}' % text,
        "",
        "data: [DONE]",
        "",
    ]


async def _drain(response):
    return "".join([c async for c in response.body_iterator])


# --------------------------------------------------------------------------
# 1. First-token stall is transparently recovered from a higher tier
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_token_stall_auto_escalates_and_completes():
    stalled = _StallingStream(_lines("never"), stall_at=0, stall_seconds=30)
    healthy = _HealthyStream(_lines("recovered answer"))
    provider = _SequenceProvider([stalled, healthy])
    config = _config(first_timeout=0.3)
    engine = _FakeRoutingEngine({"L4": "better/model"})
    store = _FakeStore()
    request = _FakeRequest(provider, config, engine, store)
    pin = _pin(Level.L3)

    response = await _handle_stream(
        request, {"model": "tier/model"}, {}, _route(), [], "sess-stall",
        None, pin, False, time.monotonic(),
    )
    body = await _drain(response)

    # The client got a real answer, not an error, despite the first upstream
    # never producing a token.
    assert "recovered answer" in body
    assert '"error"' not in body
    assert "data: [DONE]" in body
    # The stalled upstream was closed and a second, higher tier was called.
    assert stalled.closed
    assert len(provider.calls) == 2
    assert provider.calls[1] == "better/model"
    # The pin was escalated L3 -> L4 and persisted.
    assert pin.level == Level.L4
    assert pin.escalation.retry_count == 1
    assert store.puts >= 1


# --------------------------------------------------------------------------
# 2. First-token stall with no retry budget -> clean coded error, no hang
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_token_stall_without_budget_emits_error():
    stalled = _StallingStream(_lines(), stall_at=0, stall_seconds=30)
    provider = _SequenceProvider([stalled])
    config = _config(first_timeout=0.3, retry_cap=0)  # no budget
    request = _FakeRequest(provider, config, _FakeRoutingEngine({}), _FakeStore())
    pin = _pin(Level.L3)

    started = time.monotonic()
    response = await _handle_stream(
        request, {"model": "tier/model"}, {}, _route(), [], "sess-stall",
        None, pin, False, time.monotonic(),
    )
    body = await _drain(response)
    elapsed = time.monotonic() - started

    assert '"error"' in body
    assert "router_upstream_timeout" in body
    # The whole thing resolved in ~the stall timeout, NOT the 300s httpx read
    # timeout — this is the actual hang being fixed.
    assert elapsed < 5, f"took {elapsed}s — stall guard did not fire"
    assert stalled.closed
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------
# 3. Stall AFTER content was emitted cannot be restreamed -> coded error
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_midstream_stall_after_content_emits_error_not_restream():
    lines = [
        'data: {"choices":[{"delta":{"content":"partial"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":"never arrives"}}]}',
        "",
        "data: [DONE]",
        "",
    ]
    stalled = _StallingStream(lines, stall_at=2, stall_seconds=30)
    healthy = _HealthyStream(_lines("should not be used"))
    provider = _SequenceProvider([stalled, healthy])
    config = _config(first_timeout=5.0, idle_timeout=0.3)
    request = _FakeRequest(
        provider, config, _FakeRoutingEngine({"L4": "better/model"}), _FakeStore())
    pin = _pin(Level.L3)

    response = await _handle_stream(
        request, {"model": "tier/model"}, {}, _route(), [], "sess-stall",
        None, pin, False, time.monotonic(),
    )
    body = await _drain(response)

    assert "partial" in body
    assert '"error"' in body
    # Must NOT restream — the client already received bytes, replaying would
    # duplicate content.
    assert "should not be used" not in body
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------
# 4. Fallback chain respects the wall-clock deadline
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_chain_stops_at_wall_clock_deadline():
    attempts = []

    class _SlowClient:
        async def post(self, url, **kwargs):
            attempts.append(kwargs["json"]["model"])
            await asyncio.sleep(0.4)
            raise TimeoutError("slow")

        def build_request(self, *a, **k):
            raise AssertionError("stream path not used")

    config = _config()
    executor = FallbackExecutor(config, _SlowClient())

    # Budget only allows roughly one attempt, though 4 models are queued.
    deadline = time.monotonic() + 0.5
    started = time.monotonic()
    _, _, _, _, error = await executor.execute_with_fallback(
        primary_model="m1",
        fallback_models=["m2", "m3", "m4"],
        payload={}, headers={}, stream=False, deadline=deadline,
    )
    elapsed = time.monotonic() - started

    assert error is not None
    assert "deadline exceeded" in error
    # Without the deadline all 4 models would be tried (~1.6s).
    assert len(attempts) < 4, f"tried every model despite deadline: {attempts}"
    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_attempt_timeout_is_clamped_to_remaining_budget():
    config = _config()  # provider.timeout_seconds = 300
    executor = FallbackExecutor(config, object())

    # No deadline -> full provider timeout.
    assert executor._attempt_timeout(None) == 300.0
    # Tight deadline -> clamped well below 300s so one model cannot eat the
    # entire request budget.
    clamped = executor._attempt_timeout(time.monotonic() + 10)
    assert 1.0 <= clamped <= 10.5


# --------------------------------------------------------------------------
# 5. A slow-but-progressing stream is not falsely flagged as stalled
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slow_but_progressing_stream_is_not_flagged_as_stalled():
    # Each line arrives well inside the idle window, though the total stream
    # duration exceeds it.
    lines = []
    for i in range(6):
        lines.append('data: {"choices":[{"delta":{"content":"tok%d"}}]}' % i)
        lines.append("")
    lines += ["data: [DONE]", ""]

    healthy = _HealthyStream(lines, per_line_delay=0.05)
    provider = _SequenceProvider([healthy])
    config = _config(first_timeout=2.0, idle_timeout=0.5)
    request = _FakeRequest(provider, config, _FakeRoutingEngine({}), _FakeStore())

    response = await _handle_stream(
        request, {"model": "tier/model"}, {}, _route(), [], "sess-ok",
        None, _pin(), False, time.monotonic(),
    )
    body = await _drain(response)

    assert '"error"' not in body
    assert "tok0" in body and "tok5" in body
    assert "data: [DONE]" in body
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------
# 6. Stall guard can be disabled (0 = off) — preserves old behavior
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stall_guard_disabled_when_timeouts_zero():
    from app.api.chat import _stall_timeouts

    cfg = _config(first_timeout=0, idle_timeout=0)
    first, idle = _stall_timeouts(cfg)
    assert first is None
    assert idle is None


@pytest.mark.asyncio
async def test_provider_deadline_none_when_disabled():
    from app.api.chat import _provider_deadline

    assert _provider_deadline(_config(deadline_seconds=0)) is None
    d = _provider_deadline(_config(deadline_seconds=900))
    assert d is not None and d > time.monotonic()


# --------------------------------------------------------------------------
# 7. L5 (top tier) stall has nowhere to escalate -> coded error, no re-stream
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_top_tier_stall_does_not_restream_same_tier():
    stalled = _StallingStream(_lines(), stall_at=0, stall_seconds=30)
    other = _HealthyStream(_lines("should not be used"))
    provider = _SequenceProvider([stalled, other])
    config = _config(first_timeout=0.3)
    # routing engine would happily hand back a model, but L5 is the ceiling
    engine = _FakeRoutingEngine({"L5": "anthropic/claude-opus-5"})
    request = _FakeRequest(provider, config, engine, _FakeStore())
    pin = _pin(Level.L5)

    response = await _handle_stream(
        request, {"model": "tier/model"}, {}, _route(Level.L5), [], "sess-stall",
        None, pin, False, time.monotonic(),
    )
    body = await _drain(response)

    assert '"error"' in body
    assert "should not be used" not in body
    assert len(provider.calls) == 1
    assert pin.level == Level.L5
