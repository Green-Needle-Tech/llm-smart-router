"""API-level tests for the fallback / session-pin invariant.

These live at the API layer on purpose. `FallbackExecutor` holds only
`config` and `http` — it has no session store and structurally cannot
touch a `SessionPin`. Asserting the pin is unchanged against the executor
directly is vacuous: a store the executor never saw behaves identically.

The layer that actually owns the invariant is `app/api/chat.py`, which
resolves the pin, calls the provider, and writes the pin back
(`_handle_non_stream` / `_handle_stream`). A regression that re-pinned a
session on fallback would live there, so that is where it is tested.

Covers spec goals:
  G5  — "Never hard-fail on router error".
  G6  — "Stable model per session"; §2.2 step 10: a fallback serves ONE
        turn from a different model and must NOT rewrite the pin.
"""
from __future__ import annotations

import os

import httpx
import pytest
import respx
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("ROUTER_API_KEY", "test-router-key")

from app.main import create_app  # noqa: E402
from app.schemas.router import Level, SessionPin, SessionStatus  # noqa: E402


UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

L3_MODEL = "anthropic/claude-sonnet-4.5"
L3_FALLBACK = "openai/gpt-4.1"
L3_FALLBACK_2 = "meta-llama/llama-3.3-70b-instruct"

AUTH = {"Authorization": "Bearer test-router-key"}


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
async def app_client():
    """Boot the real ASGI app with upstream mocked. Yields (app, client, respx_mock)."""
    app = create_app()
    with respx.mock(assert_all_called=False) as rmock:
        # startup calls list_models() for pricing
        rmock.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                yield app, ac, rmock


async def _pin_l3(app, session_id: str) -> SessionPin:
    """Seed a PINNED L3 session so the request takes the session-hit path."""
    pin = SessionPin(
        session_id=session_id,
        level=Level.L3,
        model=L3_MODEL,
        status=SessionStatus.PINNED,
        turn_count=1,
    )
    pin.touch(7200, 86400)
    await app.state.session_store.put(pin)
    return pin


def _with_l3_fallbacks(app, fallbacks: list[str]) -> None:
    """Configure the L3 tier's fallback chain for this test."""
    app.state.config.get().routing.L3.fallbacks = fallbacks


# ── The invariant: a fallback serves one turn, it does not move the pin ────


async def test_fallback_serves_turn_without_repinning(app_client):
    """429 on L3 -> fallback answers, but the pin still says L3 / L3_MODEL."""
    app, ac, rmock = app_client
    session_id = "sess-no-repin"
    await _pin_l3(app, session_id)
    _with_l3_fallbacks(app, [L3_FALLBACK])

    rmock.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(200, json=_completion(L3_FALLBACK)),
        ]
    )

    resp = await ac.post(
        "/v1/chat/completions",
        headers={**AUTH, "X-Session-Id": session_id},
        json={"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200
    # the turn was served by the fallback
    assert resp.json()["model"] == L3_FALLBACK
    assert resp.headers["x-router-fallback-used"] == "true"
    # ...but the ROUTE still reports the pinned tier and model
    assert resp.headers["x-router-level"] == "L3"
    assert resp.headers["x-router-model"] == L3_MODEL

    after = await app.state.session_store.get(session_id)
    assert after is not None
    assert after.level == Level.L3, "G6: the pinned tier must not move on a fallback"
    assert after.model == L3_MODEL, "§2.2 step 10: the pinned model must not be rewritten"
    assert after.status == SessionStatus.PINNED
    # proof the request actually reached the store (not a vacuous assertion)
    assert after.turn_count == 2, "the pin must have been written back this turn"


async def test_pin_survives_two_consecutive_fallback_turns(app_client):
    """Two turns both served by a fallback still leave the pin on L3."""
    app, ac, rmock = app_client
    session_id = "sess-no-repin-twice"
    await _pin_l3(app, session_id)
    _with_l3_fallbacks(app, [L3_FALLBACK])

    rmock.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_completion(L3_FALLBACK)),
            httpx.Response(503),
            httpx.Response(200, json=_completion(L3_FALLBACK)),
        ]
    )

    for _ in range(2):
        resp = await ac.post(
            "/v1/chat/completions",
            headers={**AUTH, "X-Session-Id": session_id},
            json={
                "model": "smart-router",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["x-router-fallback-used"] == "true"

    after = await app.state.session_store.get(session_id)
    assert after.level == Level.L3
    assert after.model == L3_MODEL
    assert after.turn_count == 3, "both turns were counted against the same pin"


async def test_deep_fallback_does_not_repin(app_client):
    """Landing on the SECOND fallback still leaves the pin untouched."""
    app, ac, rmock = app_client
    session_id = "sess-deep-fallback"
    await _pin_l3(app, session_id)
    _with_l3_fallbacks(app, [L3_FALLBACK, L3_FALLBACK_2])

    rmock.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(500),
            httpx.Response(200, json=_completion(L3_FALLBACK_2)),
        ]
    )

    resp = await ac.post(
        "/v1/chat/completions",
        headers={**AUTH, "X-Session-Id": session_id},
        json={"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200
    assert resp.json()["model"] == L3_FALLBACK_2

    after = await app.state.session_store.get(session_id)
    assert after.level == Level.L3
    assert after.model == L3_MODEL


async def test_exhausted_chain_returns_502_and_keeps_pin(app_client):
    """G5: total upstream failure is a structured 502, and the pin is intact."""
    app, ac, rmock = app_client
    session_id = "sess-exhausted"
    await _pin_l3(app, session_id)
    _with_l3_fallbacks(app, [L3_FALLBACK])

    rmock.post(UPSTREAM).mock(return_value=httpx.Response(502))

    resp = await ac.post(
        "/v1/chat/completions",
        headers={**AUTH, "X-Session-Id": session_id},
        json={"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code in (502, 504), "G5: degrade to a structured error, never 500"
    assert "error" in resp.json()

    after = await app.state.session_store.get(session_id)
    assert after is not None, "a failed upstream must not destroy the session"
    assert after.level == Level.L3
    assert after.model == L3_MODEL


# ── Streaming path ────────────────────────────────────────────────────────


async def test_streaming_fallback_does_not_repin(app_client):
    """The streaming path must honour the same invariant as the JSON path."""
    app, ac, rmock = app_client
    session_id = "sess-stream-no-repin"
    await _pin_l3(app, session_id)
    _with_l3_fallbacks(app, [L3_FALLBACK])

    rmock.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
            ),
        ]
    )

    resp = await ac.post(
        "/v1/chat/completions",
        headers={**AUTH, "X-Session-Id": session_id},
        json={
            "model": "smart-router",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert resp.status_code == 200
    assert resp.headers["x-router-fallback-used"] == "true"
    assert resp.headers["x-router-level"] == "L3"
    assert b"[DONE]" in resp.content

    after = await app.state.session_store.get(session_id)
    assert after.level == Level.L3, "G6 holds on the streaming path too"
    assert after.model == L3_MODEL


async def test_streaming_exhaustion_returns_error_and_keeps_pin(app_client):
    """Streaming exhaustion is a JSON error response, not a broken stream."""
    app, ac, rmock = app_client
    session_id = "sess-stream-exhausted"
    await _pin_l3(app, session_id)
    _with_l3_fallbacks(app, [L3_FALLBACK])

    rmock.post(UPSTREAM).mock(return_value=httpx.Response(503))

    resp = await ac.post(
        "/v1/chat/completions",
        headers={**AUTH, "X-Session-Id": session_id},
        json={
            "model": "smart-router",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert resp.status_code in (502, 504)
    assert "error" in resp.json()

    after = await app.state.session_store.get(session_id)
    assert after is not None
    assert after.level == Level.L3
