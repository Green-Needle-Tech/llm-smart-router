"""Tests for tier-prefix session pinning (bypasses classifier on turn 1)."""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.schemas.router import Level, ClassificationSource
from app.config.loader import ConfigManager
from app.api.chat import _detect_tier_prefix


def _make_config(tier_prefix_enabled=True, strip_prefix=True, pattern=None):
    """Build a config with tier_prefix settings."""
    cm = ConfigManager(settings_path="/nonexistent")
    config = cm.load()
    config.classification.tier_prefix.enabled = tier_prefix_enabled
    config.classification.tier_prefix.strip_prefix = strip_prefix
    if pattern is not None:
        config.classification.tier_prefix.pattern = pattern
    return config


def _make_body(content: str, role: str = "user") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role=role, content=content)],
    )


# ─── _detect_tier_prefix unit tests ───


def test_detect_l4_prefix():
    """L4 at start of prompt is detected and prefix is stripped."""
    config = _make_config()
    body = _make_body("L4 explain quantum computing")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L4
    assert body.messages[0].content == "explain quantum computing"


def test_detect_l1_prefix():
    """L1 at start of prompt is detected."""
    config = _make_config()
    body = _make_body("L1 say hello")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L1
    assert body.messages[0].content == "say hello"


def test_detect_l5_prefix():
    """L5 at start of prompt is detected."""
    config = _make_config()
    body = _make_body("L5 design a distributed database")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L5
    assert body.messages[0].content == "design a distributed database"


def test_detect_all_tiers():
    """All five tier prefixes are detected correctly."""
    config = _make_config()
    for tier_num in range(1, 6):
        tier = f"L{tier_num}"
        body = _make_body(f"{tier} do something")
        level = _detect_tier_prefix(body, config)
        assert level == Level.from_str(tier), f"Failed for {tier}"


def test_no_prefix_returns_none():
    """A prompt without a tier prefix returns None."""
    config = _make_config()
    body = _make_body("explain quantum computing")
    level = _detect_tier_prefix(body, config)
    assert level is None


def test_prefix_with_colon_delimiter():
    """L3: prompt format is detected."""
    config = _make_config()
    body = _make_body("L3: analyze this data")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L3
    assert body.messages[0].content == "analyze this data"


def test_prefix_with_dash_delimiter():
    """L2 - prompt format is detected."""
    config = _make_config()
    body = _make_body("L2 - summarize this")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L2
    assert body.messages[0].content == "summarize this"


def test_prefix_with_dot_delimiter():
    """L1. prompt format is detected."""
    config = _make_config()
    body = _make_body("L1. write a haiku")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L1
    assert body.messages[0].content == "write a haiku"


def test_disabled_returns_none():
    """When tier_prefix is disabled, no detection occurs."""
    config = _make_config(tier_prefix_enabled=False)
    body = _make_body("L4 explain quantum computing")
    level = _detect_tier_prefix(body, config)
    assert level is None
    # Content should NOT be stripped
    assert body.messages[0].content == "L4 explain quantum computing"


def test_strip_disabled_keeps_prefix():
    """When strip_prefix is False, the prefix remains in the message."""
    config = _make_config(strip_prefix=False)
    body = _make_body("L4 explain quantum computing")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L4
    # Content should still have the prefix
    assert body.messages[0].content == "L4 explain quantum computing"


def test_prefix_only_no_content_keeps_message():
    """A message that is just 'L4' with nothing after is detected but not stripped."""
    config = _make_config()
    body = _make_body("L4 ")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L4
    # Not stripped because remaining content is empty
    assert body.messages[0].content == "L4 "


def test_case_insensitive_prefix():
    """Lowercase 'l4' should match (Level.from_str uppercases)."""
    config = _make_config()
    body = _make_body("l4 explain something")
    level = _detect_tier_prefix(body, config)
    assert level == Level.L4


def test_prefix_not_at_start_no_match():
    """Tier label not at the start should not match."""
    config = _make_config()
    body = _make_body("please use L4 for this")
    level = _detect_tier_prefix(body, config)
    assert level is None


def test_no_user_message_returns_none():
    """No user message in the body returns None."""
    config = _make_config()
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="system", content="You are helpful.")],
    )
    level = _detect_tier_prefix(body, config)
    assert level is None


def test_first_user_message_used():
    """When multiple messages exist, the first user message is checked."""
    config = _make_config()
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[
            ChatMessage(role="system", content="System prompt"),
            ChatMessage(role="user", content="L4 do the thing"),
            ChatMessage(role="assistant", content="OK"),
            ChatMessage(role="user", content="now explain"),
        ],
    )
    level = _detect_tier_prefix(body, config)
    assert level == Level.L4
    assert body.messages[1].content == "do the thing"


def test_custom_pattern():
    """A custom pattern can be configured."""
    config = _make_config(pattern=r"^(TIER[1-5])\s+")
    body = _make_body("TIER3 analyze data")
    level = _detect_tier_prefix(body, config)
    # Level.from_str("TIER3") will fail, so returns None
    assert level is None


def test_non_string_content_returns_none():
    """Non-string content (e.g. multimodal list) returns None."""
    config = _make_config()
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="user", content=None)],
    )
    level = _detect_tier_prefix(body, config)
    assert level is None


def test_empty_string_returns_none():
    """Empty string content returns None."""
    config = _make_config()
    body = _make_body("")
    level = _detect_tier_prefix(body, config)
    assert level is None


# ─── Integration test: full session-miss flow ───


@pytest.mark.asyncio
async def test_tier_prefix_bypasses_classifier():
    """A tier-prefix in the first prompt pins the session without calling the classifier LLM."""
    from app.api.chat import _session_pinned_route
    from app.classify.classifier import ClassifierService
    from app.cache.memory import MemoryClassificationCache
    from app.routing.engine import RoutingEngine
    from app.session.memory_store import MemorySessionStore

    config = _make_config()
    config.classification.enabled = True
    config.classification.cache.enabled = True
    config.heuristics.enabled = False  # force model path if classifier is called

    classifier = ClassifierService(config, openrouter_api_key="test-key")
    call_count = 0

    async def fake_call_classifier_model(digest: str) -> str:
        nonlocal call_count
        call_count += 1
        return '{"level": "L2", "confidence": 0.95, "reason": "test"}'

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
        headers={"x-session-id": "tier-prefix-test-session"},
    )
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="user", content="L4 explain quantum entanglement")],
    )
    directive = routing_engine.parse_model_directive("smart-router")

    with patch("app.api.chat._forward_to_provider", new=AsyncMock(return_value="OK")):
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

    # Critical: classifier LLM was NOT called
    assert call_count == 0

    # Verify the session was pinned to L4
    pin = await store.get("tier-prefix-test-session")
    assert pin is not None
    assert pin.level == Level.L4
    assert pin.classification is not None
    assert pin.classification.source == ClassificationSource.OVERRIDE
    assert "tier-prefix" in pin.classification.reason

    # Verify the prefix was stripped from the message
    assert body.messages[0].content == "explain quantum entanglement"


@pytest.mark.asyncio
async def test_tier_prefix_does_not_fire_on_session_hit():
    """On a session hit (turn 2+), the tier prefix is NOT re-evaluated."""
    from app.api.chat import _session_pinned_route
    from app.classify.classifier import ClassifierService
    from app.cache.memory import MemoryClassificationCache
    from app.routing.engine import RoutingEngine
    from app.session.memory_store import MemorySessionStore
    from app.schemas.router import SessionPin, SessionStatus

    config = _make_config()
    config.classification.enabled = True
    config.heuristics.enabled = False

    classifier = ClassifierService(config, openrouter_api_key="test-key")
    call_count = 0

    async def fake_call_classifier_model(digest: str) -> str:
        nonlocal call_count
        call_count += 1
        return '{"level": "L2", "confidence": 0.95, "reason": "test"}'

    classifier._call_classifier_model = fake_call_classifier_model  # type: ignore[method-assign]

    store = MemorySessionStore(max_sessions=100)
    cache = MemoryClassificationCache(ttl_seconds=3600, max_entries=1000)
    routing_engine = RoutingEngine(config)

    # Pre-pin a session to L2
    session_id = "pre-pinned-session"
    pin = SessionPin(
        session_id=session_id,
        level=Level.L2,
        model=config.routing.get_model("L2"),
        status=SessionStatus.PINNED,
        turn_count=1,
    )
    pin.touch(7200, 86400)
    await store.put(pin)

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
        headers={"x-session-id": session_id},
    )
    # Turn 2: message starts with "L4" but session is already pinned to L2
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="user", content="L4 actually I want more detail")],
    )
    directive = routing_engine.parse_model_directive("smart-router")

    with patch("app.api.chat._forward_to_provider", new=AsyncMock(return_value="OK")):
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

    # Classifier was NOT called (session hit, not miss)
    assert call_count == 0

    # Session stays at L2 (tier prefix only fires on session miss)
    pin_after = await store.get(session_id)
    assert pin_after.level == Level.L2


@pytest.mark.asyncio
async def test_forced_level_takes_precedence_over_prefix():
    """When forced_level is already set (e.g. via smart-router/L4 directive),
    the tier-prefix detection is skipped."""
    from app.api.chat import _session_pinned_route
    from app.classify.classifier import ClassifierService
    from app.cache.memory import MemoryClassificationCache
    from app.routing.engine import RoutingEngine
    from app.session.memory_store import MemorySessionStore

    config = _make_config()

    classifier = ClassifierService(config, openrouter_api_key="test-key")
    call_count = 0

    async def fake_call_classifier_model(digest: str) -> str:
        nonlocal call_count
        call_count += 1
        return '{"level": "L2", "confidence": 0.95, "reason": "test"}'

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
        headers={"x-session-id": "forced-level-test-session"},
    )
    # Message has "L3" prefix, but forced_level=L5 takes precedence
    body = ChatCompletionRequest(
        model="smart-router",
        messages=[ChatMessage(role="user", content="L3 explain something")],
    )
    directive = routing_engine.parse_model_directive("smart-router")

    with patch("app.api.chat._forward_to_provider", new=AsyncMock(return_value="OK")):
        resp = await _session_pinned_route(
            request=request,
            body=body,
            config=config,
            routing_engine=routing_engine,
            directive=directive,
            forced_level=Level.L5,  # Explicit forced level
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

    # Classifier was NOT called
    assert call_count == 0

    # Session pinned to L5 (forced), not L3 (prefix)
    pin = await store.get("forced-level-test-session")
    assert pin.level == Level.L5
    # Content should NOT be stripped (forced_level path, not prefix path)
    assert body.messages[0].content == "L3 explain something"


# ─── Config tests ───


def test_tier_prefix_config_defaults():
    """TierPrefixConfig has correct defaults."""
    from app.config.schema import TierPrefixConfig
    cfg = TierPrefixConfig()
    assert cfg.enabled is True
    assert cfg.strip_prefix is True
    assert "L[1-5]" in cfg.pattern


def test_tier_prefix_config_in_settings():
    """ConfigManager loads tier_prefix from defaults."""
    cm = ConfigManager(settings_path="/nonexistent")
    config = cm.load()
    assert hasattr(config.classification, "tier_prefix")
    assert config.classification.tier_prefix.enabled is True
    assert config.classification.tier_prefix.strip_prefix is True
