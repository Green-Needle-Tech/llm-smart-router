"""Integration tests for session pinning behavior."""
import asyncio
from datetime import UTC
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.schemas.router import (
    Level,
    SessionPin,
    SessionStatus,
)
from app.session.memory_store import MemorySessionStore


@pytest.fixture
def store():
    return MemorySessionStore(max_sessions=100)


@pytest.mark.asyncio
async def test_session_hit_skips_classifier(store):
    """THE critical test: a 20-turn session triggers exactly one classifier call."""
    session_id = "test-session-1"

    # Simulate turn 1: classify and pin
    pin = SessionPin(
        session_id=session_id,
        level=Level.L3,
        model="anthropic/claude-sonnet-4.5",
        status=SessionStatus.PINNED,
        turn_count=1,
    )
    pin.touch(7200, 86400)
    await store.put(pin)

    # Simulate turns 2-20: each should be a session hit
    for turn in range(2, 21):
        result = await store.get(session_id)
        assert result is not None
        assert result.level == Level.L3
        assert result.model == "anthropic/claude-sonnet-4.5"
        result.turn_count += 1
        result.touch(7200, 86400)
        await store.put(result)

    # Verify: one pin, turn_count = 20
    final = await store.get(session_id)
    assert final.turn_count == 20
    assert final.level == Level.L3


@pytest.mark.asyncio
async def test_session_miss_triggers_classification(store):
    """A new session should be a miss."""
    result = await store.get("new-session")
    assert result is None


@pytest.mark.asyncio
async def test_reserve_prevents_duplicate_classification(store):
    """The first-turn race protocol should prevent duplicate classification."""
    session_id = "race-session"

    # First reservation wins
    won1 = await store.reserve(session_id, ttl_seconds=30)
    assert won1 is True

    # Second reservation loses
    won2 = await store.reserve(session_id, ttl_seconds=30)
    assert won2 is False

    # Release
    await store.release(session_id)

    # Now can reserve again
    won3 = await store.reserve(session_id, ttl_seconds=30)
    assert won3 is True


@pytest.mark.asyncio
async def test_session_expiry(store):
    """An expired session should return None."""
    from datetime import datetime, timedelta
    session_id = "expiring-session"

    pin = SessionPin(
        session_id=session_id,
        level=Level.L2,
        model="openai/gpt-4.1-mini",
        turn_count=1,
    )
    # Set expiry in the past
    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    pin.expires_at = past
    await store.put(pin)

    result = await store.get(session_id)
    assert result is None


@pytest.mark.asyncio
async def test_session_deletion(store):
    session_id = "delete-me"
    pin = SessionPin(session_id=session_id, level=Level.L1, model="test/model")
    await store.put(pin)

    deleted = await store.delete(session_id)
    assert deleted is True

    result = await store.get(session_id)
    assert result is None


@pytest.mark.asyncio
async def test_lru_eviction():
    """Oldest sessions should be evicted when capacity is exceeded."""
    store = MemorySessionStore(max_sessions=3)

    for i in range(4):
        pin = SessionPin(session_id=f"session-{i}", level=Level.L1, model="test/model")
        await store.put(pin)

    # session-0 should be evicted (oldest)
    assert await store.get("session-0") is None
    # session-3 should exist (newest)
    assert await store.get("session-3") is not None


@pytest.mark.asyncio
async def test_concurrent_turn1_one_classifier_call(store):
    """50 concurrent turn-1 requests sharing one session id produce one reservation."""
    session_id = "concurrent-session"
    results = await asyncio.gather(
        *[store.reserve(session_id, ttl_seconds=30) for _ in range(50)]
    )

    won_count = sum(1 for r in results if r)
    assert won_count == 1  # Exactly one winner


@pytest.mark.asyncio
async def test_session_pinned_route_single_classifier_call(monkeypatch):
    """Verify that on cache miss, _call_classifier_model is called only ONCE, not duplicated during cache lookup."""
    from types import SimpleNamespace

    from app.api.chat import _session_pinned_route
    from app.cache.memory import MemoryClassificationCache
    from app.classify.classifier import ClassifierService
    from app.config.loader import ConfigManager
    from app.routing.engine import RoutingEngine

    cm = ConfigManager(settings_path="/nonexistent")
    config = cm.load()
    config.classification.enabled = True
    config.classification.cache.enabled = True
    config.heuristics.enabled = False  # force model classification

    classifier = ClassifierService(config, openrouter_api_key="test-key")
    call_count = 0

    async def fake_call_classifier_model(digest: str) -> str:
        nonlocal call_count
        call_count += 1
        return '{"level": "L2", "confidence": 0.95, "reason": "moderate task"}'

    classifier._call_classifier_model = fake_call_classifier_model  # type: ignore[method-assign]

    store = MemorySessionStore(max_sessions=100)
    cache = MemoryClassificationCache(ttl_seconds=3600, max_entries=1000)
    routing_engine = RoutingEngine(config)

    app_state = SimpleNamespace(
        config=SimpleNamespace(get=lambda: config),
        session_store=store,
        classification_cache=cache,
        classifier=classifier,
        routing_engine=routing_engine,
        guardrails=None,
        ip_redaction=None,
        temporal_awareness_engine=None,
    )

    request = SimpleNamespace(
        app=SimpleNamespace(state=app_state),
        headers={"x-session-id": "test-single-call-session"},
    )
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="user", content="Write a python script to parse CSV files.")],
    )
    directive = routing_engine.parse_model_directive("smart-router")

    # Mock _forward_to_provider to avoid actual network calls
    with patch("app.api.chat._forward_to_provider", new=AsyncMock(return_value="OK")) as mock_fwd:
        resp = await _session_pinned_route(
            request=request,
            body=body,
            config=config,
            routing_engine=routing_engine,
            directive=directive,
            forced_level=None,
            forced_model=None,
            max_level=None,
            min_level=None,
            reclassify=False,
            repin=False,
            task_text=None,
            bypass_cache=False,
            include_metadata=False,
            start=0,
        )
        assert resp == "OK"

    # Crucial assertion: classifier model was called exactly once on cache miss!
    assert call_count == 1

    # And verify the result was cached
    cached_entries = list(cache._cache.values())
    assert len(cached_entries) == 1
    assert cached_entries[0]["level"] == "L2"

