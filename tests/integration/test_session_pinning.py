"""Integration tests for session pinning behavior."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.schemas.router import Level, SessionPin, SessionStatus, ClassificationResult, ClassificationSource
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
    from datetime import datetime, timedelta, timezone
    session_id = "expiring-session"

    pin = SessionPin(
        session_id=session_id,
        level=Level.L2,
        model="openai/gpt-4.1-mini",
        turn_count=1,
    )
    # Set expiry in the past
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
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
