"""Integration tests for the fallback chain executor.

Covers spec goal:
  G5  — "Never hard-fail on router error": any upstream failure degrades
        through the chain and returns a structured error, never an exception.

G6 / §2.2 step 10 ("a fallback must not re-pin the session") is NOT tested
here — FallbackExecutor has no session store, so such a test would be
vacuous. See tests/integration/test_fallback_session_invariant.py.

OpenRouter is mocked at the HTTP layer with respx, per §2.3 (Tests).
"""
from __future__ import annotations

import httpx
import pytest
import respx

from app.config.loader import ConfigManager
from app.routing.fallback import FallbackExecutor


UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"

PRIMARY = "anthropic/claude-sonnet-4.5"
FALLBACK_1 = "openai/gpt-4.1"
FALLBACK_2 = "meta-llama/llama-3.3-70b-instruct"


def _completion(model: str) -> dict:
    """A minimal OpenAI-shaped chat completion body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


@pytest.fixture
def config():
    return ConfigManager(settings_path="/nonexistent").load()


@pytest.fixture
async def http_client():
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def executor(config, http_client):
    return FallbackExecutor(config, http_client)


@pytest.fixture
def payload():
    return {
        "model": PRIMARY,
        "messages": [{"role": "user", "content": "hello"}],
    }


# ── Happy path ────────────────────────────────────────────────────────────


@respx.mock
async def test_primary_success_no_fallback(executor, payload):
    """Primary responds 200: chain stops immediately, fallback_used is False."""
    route = respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, json=_completion(PRIMARY))
    )

    body, raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1, FALLBACK_2], payload, {}
    )

    assert error is None
    assert fallback_used is False
    assert model_used == PRIMARY
    assert body["model"] == PRIMARY
    assert raw is None
    assert route.call_count == 1, "fallbacks must not be called when primary succeeds"


@respx.mock
async def test_primary_model_is_injected_into_payload(executor, payload):
    """The executor overrides payload['model'] with the model it is trying."""
    route = respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, json=_completion(PRIMARY))
    )
    payload["model"] = "smart-router"  # routing directive must never reach upstream

    await executor.execute_with_fallback(PRIMARY, [], payload, {})

    sent = route.calls[0].request
    import json as _json

    assert _json.loads(sent.content)["model"] == PRIMARY


# ── Degradation through the chain ─────────────────────────────────────────


@respx.mock
async def test_retryable_status_advances_to_fallback(executor, payload):
    """429 is in provider.retry_on_status: chain advances, fallback_used is True."""
    respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(200, json=_completion(FALLBACK_1)),
        ]
    )

    body, _raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1, FALLBACK_2], payload, {}
    )

    assert error is None
    assert fallback_used is True
    assert model_used == FALLBACK_1
    assert body["model"] == FALLBACK_1


@respx.mock
async def test_chain_tries_models_in_declared_order(executor, payload):
    """Two failures in a row must land on the SECOND fallback, in order."""
    route = respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(500),
            httpx.Response(200, json=_completion(FALLBACK_2)),
        ]
    )

    _body, _raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1, FALLBACK_2], payload, {}
    )

    assert error is None
    assert fallback_used is True
    assert model_used == FALLBACK_2

    import json as _json

    attempted = [_json.loads(c.request.content)["model"] for c in route.calls]
    assert attempted == [PRIMARY, FALLBACK_1, FALLBACK_2]


@respx.mock
async def test_timeout_advances_to_fallback(executor, payload):
    """A transport timeout is caught, not raised, and advances the chain."""
    respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.TimeoutException("upstream timed out"),
            httpx.Response(200, json=_completion(FALLBACK_1)),
        ]
    )

    _body, _raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}
    )

    assert error is None
    assert fallback_used is True
    assert model_used == FALLBACK_1


@respx.mock
async def test_non_retryable_status_still_degrades(executor, payload):
    """A 400 is not in retry_on_status, but raise_for_status must not escape."""
    respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(400, json={"error": "bad request"}),
            httpx.Response(200, json=_completion(FALLBACK_1)),
        ]
    )

    _body, _raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}
    )

    assert error is None, "G5: an upstream 4xx must never surface as an exception"
    assert fallback_used is True
    assert model_used == FALLBACK_1


# ── Exhaustion (G5) ───────────────────────────────────────────────────────


@respx.mock
async def test_all_fallbacks_exhausted_returns_error_not_exception(executor, payload):
    """Every model fails: return a structured error tuple, never raise."""
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(502))

    body, raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1, FALLBACK_2], payload, {}
    )

    assert body is None
    assert raw is None
    assert error is not None, "G5: exhaustion must be reported, not raised"
    assert "502" in error
    assert model_used == PRIMARY, "reported model falls back to the primary slug"
    assert fallback_used is False
    assert route.call_count == 3, "all three models must be attempted"


@respx.mock
async def test_exhaustion_reports_the_last_error(executor, payload):
    """The returned error describes the final attempt, not the first."""
    respx.post(UPSTREAM).mock(
        side_effect=[httpx.Response(429), httpx.Response(504)]
    )

    _body, _raw, _model, _fb, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}
    )

    assert error is not None
    assert "504" in error
    assert FALLBACK_1 in error


@respx.mock
async def test_empty_fallback_list_attempts_primary_only(executor, payload):
    """With no fallbacks configured, exactly one upstream call is made."""
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(500))

    _body, _raw, _model, _fb, error = await executor.execute_with_fallback(
        PRIMARY, [], payload, {}
    )

    assert route.call_count == 1
    assert error is not None


# ── Streaming ─────────────────────────────────────────────────────────────


@respx.mock
async def test_streaming_success_returns_open_response(executor, payload):
    """On stream=True the raw response is handed back unread for pass-through."""
    respx.post(UPSTREAM).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
        )
    )

    body, raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}, stream=True
    )

    assert error is None
    assert body is None, "streaming must not buffer a JSON body"
    assert raw is not None
    assert model_used == PRIMARY
    assert fallback_used is False

    chunks = [c async for c in raw.aiter_bytes()]
    assert b"[DONE]" in b"".join(chunks)
    await raw.aclose()


@respx.mock
async def test_streaming_retryable_status_advances_and_closes(executor, payload):
    """A retryable status on the streaming path drains and advances the chain."""
    respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(200, content=b"data: [DONE]\n\n"),
        ]
    )

    _body, raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}, stream=True
    )

    assert error is None
    assert fallback_used is True
    assert model_used == FALLBACK_1
    await raw.aclose()


@respx.mock
async def test_streaming_exhaustion_returns_no_response(executor, payload):
    """Streaming exhaustion returns (None, None, ...) with an error string."""
    respx.post(UPSTREAM).mock(return_value=httpx.Response(503))

    body, raw, _model, _fb, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}, stream=True
    )

    assert body is None
    assert raw is None
    assert error is not None


@respx.mock
async def test_streaming_non_retryable_status_degrades(executor, payload):
    """A 400 is NOT in retry_on_status, so it reaches raise_for_status().

    On the streaming path that lands in the inner `except Exception:` which
    closes the response and re-raises; the outer HTTPStatusError handler must
    then catch it and advance the chain rather than let it escape (G5).
    """
    respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(400, json={"error": "bad request"}),
            httpx.Response(200, content=b"data: [DONE]\n\n"),
        ]
    )

    _body, raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}, stream=True
    )

    assert error is None, "G5: a streaming 4xx must never surface as an exception"
    assert fallback_used is True
    assert model_used == FALLBACK_1
    assert raw is not None
    await raw.aclose()


@respx.mock
async def test_streaming_non_retryable_exhaustion_reports_status(executor, payload):
    """A 401 with no working fallback returns a structured error, never raises."""
    respx.post(UPSTREAM).mock(return_value=httpx.Response(401, json={"error": "nope"}))

    body, raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}, stream=True
    )

    assert body is None
    assert raw is None, "the failed streaming response must be closed, not leaked"
    assert error is not None
    assert "401" in error
    assert model_used == PRIMARY
    assert fallback_used is False


@respx.mock
async def test_streaming_connect_error_advances_chain(executor, payload):
    """A transport-level connect failure on the streaming path is caught."""
    respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.Response(200, content=b"data: [DONE]\n\n"),
        ]
    )

    _body, raw, model_used, fallback_used, error = await executor.execute_with_fallback(
        PRIMARY, [FALLBACK_1], payload, {}, stream=True
    )

    assert error is None
    assert fallback_used is True
    assert model_used == FALLBACK_1
    await raw.aclose()


class _ClosureSpyTransport(httpx.AsyncBaseTransport):
    """Transport that records aclose() on each streaming response body.

    respx buffers its mock responses, so it cannot distinguish "the executor
    closed the failed stream" from "nobody ever opened it". This transport
    hands back a real unbuffered stream per attempt and appends the attempt
    number to `closed` when that attempt's body is closed.
    """

    def __init__(self, statuses: list[int]):
        self.statuses = list(statuses)
        self.closed: list[int] = []
        self._i = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        status = self.statuses[self._i]
        self._i += 1
        attempt = self._i
        recorder = self.closed

        class _SpyStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"data: [DONE]\n\n"

            async def aclose(self) -> None:
                recorder.append(attempt)

        return httpx.Response(
            status,
            stream=_SpyStream(),
            headers={"content-type": "text/event-stream"},
        )


async def test_streaming_failed_attempt_is_closed_not_leaked(config, payload):
    """A non-retryable streaming failure must close its response body.

    Without the `except Exception: await resp.aclose(); raise` guard, the
    401 attempt's connection is never released back to the pool.
    """
    transport = _ClosureSpyTransport([401, 200])
    async with httpx.AsyncClient(transport=transport) as client:
        executor = FallbackExecutor(config, client)

        _body, raw, model_used, fallback_used, error = await executor.execute_with_fallback(
            PRIMARY, [FALLBACK_1], payload, {}, stream=True
        )

    assert error is None
    assert fallback_used is True
    assert model_used == FALLBACK_1
    assert 1 in transport.closed, (
        "the failed 401 streaming attempt was never closed — connection leak"
    )
    assert 2 not in transport.closed, "the successful stream must stay open for the caller"

    await raw.aclose()
    assert 2 in transport.closed, "caller close must reach the underlying stream"


async def test_streaming_retryable_attempt_is_closed(config, payload):
    """The retryable (429) streaming branch must also release its response."""
    transport = _ClosureSpyTransport([429, 200])
    async with httpx.AsyncClient(transport=transport) as client:
        executor = FallbackExecutor(config, client)

        _body, raw, _model, fallback_used, error = await executor.execute_with_fallback(
            PRIMARY, [FALLBACK_1], payload, {}, stream=True
        )

    assert error is None
    assert fallback_used is True
    assert 1 in transport.closed, "the drained 429 attempt must be closed"

    await raw.aclose()


# ── Session invariant: see tests/integration/test_fallback_session_invariant.py ──
#
# The G6 / §2.2-step-10 invariant ("a fallback must not re-pin the session")
# CANNOT be tested here. FallbackExecutor holds only `config` and `http`; it
# has no session store and never reads or writes a SessionPin. Seeding a store
# in this file and asserting the pin is unchanged is vacuous — a control store
# the executor provably never saw yields identical assertions.
#
# That invariant is owned by app/api/chat.py, which resolves the pin, calls the
# provider, and writes the pin back. It is tested at the API layer in
# tests/integration/test_fallback_session_invariant.py, where injecting a real
# re-pin bug does make the tests fail.


# ── Header propagation ────────────────────────────────────────────────────


@respx.mock
async def test_headers_are_forwarded_on_every_attempt(executor, payload):
    """Auth/provider headers must be present on the fallback attempt too."""
    route = respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=_completion(FALLBACK_1)),
        ]
    )
    headers = {"Authorization": "Bearer test-key", "X-Title": "Hermes Smart Router"}

    await executor.execute_with_fallback(PRIMARY, [FALLBACK_1], payload, headers)

    assert route.call_count == 2
    for call in route.calls:
        assert call.request.headers["authorization"] == "Bearer test-key"
        assert call.request.headers["x-title"] == "Hermes Smart Router"
