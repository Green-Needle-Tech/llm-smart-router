"""Tests for per-session cumulative token tracking and postfix display."""
import json
from types import SimpleNamespace

import pytest

from app.api.chat import _add_model_postfix, _strip_model_postfix_from_messages
from app.schemas.router import (
    ClassificationResult,
    ClassificationSource,
    Level,
    RouteDecision,
    SessionPin,
)
from app.telemetry.token_tracker import (
    accumulate,
    build_postfix,
    extract_tokens,
    render_postfix,
)


# ---------------------------------------------------------------------------
# extract_tokens
# ---------------------------------------------------------------------------

def test_extract_tokens_normal():
    assert extract_tokens({"prompt_tokens": 100, "completion_tokens": 50}) == (100, 50)


def test_extract_tokens_missing_keys():
    assert extract_tokens({}) == (0, 0)


def test_extract_tokens_none():
    assert extract_tokens(None) == (0, 0)


def test_extract_tokens_string_values():
    assert extract_tokens({"prompt_tokens": "100", "completion_tokens": "50"}) == (100, 50)


def test_extract_tokens_invalid_values():
    assert extract_tokens({"prompt_tokens": "abc", "completion_tokens": None}) == (0, 0)


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------

def test_accumulate_first_call():
    usage = {}
    accumulate(usage, "L1", 100, 50)
    assert usage == {"L1": {"prompt": 100, "completion": 50}}


def test_accumulate_adds_to_existing():
    usage = {"L1": {"prompt": 100, "completion": 50}}
    accumulate(usage, "L1", 200, 100)
    assert usage == {"L1": {"prompt": 300, "completion": 150}}


def test_accumulate_multiple_tiers():
    usage = {"L1": {"prompt": 100, "completion": 50}}
    accumulate(usage, "L2", 500, 200)
    assert usage == {
        "L1": {"prompt": 100, "completion": 50},
        "L2": {"prompt": 500, "completion": 200},
    }


def test_accumulate_zero_tokens():
    usage = {}
    accumulate(usage, "L1", 0, 0)
    assert usage == {"L1": {"prompt": 0, "completion": 0}}


# ---------------------------------------------------------------------------
# render_postfix
# ---------------------------------------------------------------------------

def test_render_postfix_empty():
    assert render_postfix({}) == ""
    assert render_postfix(None) == ""


def test_render_postfix_single_tier():
    usage = {"L1": {"prompt": 3032, "completion": 1000}}
    assert render_postfix(usage) == "L1-In:3032|Out:1000"


def test_render_postfix_multiple_tiers_sorted():
    usage = {
        "L2": {"prompt": 10021, "completion": 6054},
        "L1": {"prompt": 3032, "completion": 1000},
    }
    assert render_postfix(usage) == "L1-In:3032|Out:1000, L2-In:10021|Out:6054"


def test_render_postfix_skips_zero_usage():
    usage = {
        "L1": {"prompt": 100, "completion": 50},
        "L2": {"prompt": 0, "completion": 0},
    }
    assert render_postfix(usage) == "L1-In:100|Out:50"


def test_render_postfix_all_zero():
    usage = {"L1": {"prompt": 0, "completion": 0}}
    assert render_postfix(usage) == ""


# ---------------------------------------------------------------------------
# build_postfix
# ---------------------------------------------------------------------------

def test_build_postfix_with_tokens():
    usage = {"L1": {"prompt": 3032, "completion": 1000}}
    assert build_postfix("L1", usage) == "[smart-router/L1/L1-In:3032|Out:1000]"


def test_build_postfix_multi_tier():
    usage = {
        "L1": {"prompt": 3032, "completion": 1000},
        "L2": {"prompt": 10021, "completion": 6054},
    }
    result = build_postfix("L2", usage)
    assert result == "[smart-router/L2/L1-In:3032|Out:1000, L2-In:10021|Out:6054]"


def test_build_postfix_no_usage_falls_back():
    assert build_postfix("L1", None) == "[smart-router/L1]"
    assert build_postfix("L1", {}) == "[smart-router/L1]"


def test_build_postfix_show_in_postfix_false():
    usage = {"L1": {"prompt": 100, "completion": 50}}
    assert build_postfix("L1", usage, show_in_postfix=False) == "[smart-router/L1]"


def test_build_postfix_all_zero_falls_back():
    usage = {"L1": {"prompt": 0, "completion": 0}}
    assert build_postfix("L1", usage) == "[smart-router/L1]"


# ---------------------------------------------------------------------------
# _add_model_postfix with token tracking
# ---------------------------------------------------------------------------

def _route(level: Level = Level.L1) -> RouteDecision:
    return RouteDecision(
        level=level,
        model="google/gemini-2.5-flash",
        params={},
        classification=ClassificationResult(
            level=level,
            confidence=1.0,
            reason="test",
            source=ClassificationSource.OVERRIDE,
        ),
    )


def test_add_model_postfix_with_token_usage():
    body = {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
    usage = {"L1": {"prompt": 3032, "completion": 1000}}
    _add_model_postfix(body, "model/test", _route(), token_usage=usage)
    assert body["choices"][0]["message"]["content"] == (
        "Hello\n\n[smart-router/L1/L1-In:3032|Out:1000]"
    )


def test_add_model_postfix_multi_tier_usage():
    body = {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
    usage = {
        "L1": {"prompt": 3032, "completion": 1000},
        "L2": {"prompt": 10021, "completion": 6054},
    }
    _add_model_postfix(body, "model/test", _route(Level.L2), token_usage=usage)
    assert body["choices"][0]["message"]["content"] == (
        "Hello\n\n[smart-router/L2/L1-In:3032|Out:1000, L2-In:10021|Out:6054]"
    )


def test_add_model_postfix_no_token_usage_falls_back():
    body = {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
    _add_model_postfix(body, "model/test", _route(), token_usage=None)
    assert body["choices"][0]["message"]["content"] == "Hello\n\n[smart-router/L1]"


def test_add_model_postfix_show_tokens_false():
    body = {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
    usage = {"L1": {"prompt": 100, "completion": 50}}
    _add_model_postfix(body, "model/test", _route(), token_usage=usage, show_tokens=False)
    assert body["choices"][0]["message"]["content"] == "Hello\n\n[smart-router/L1]"


def test_add_model_postfix_null_content_with_tokens():
    body = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    usage = {"L1": {"prompt": 100, "completion": 50}}
    _add_model_postfix(body, "model/test", _route(), token_usage=usage)
    assert body["choices"][0]["message"]["content"] == "[smart-router/L1/L1-In:100|Out:50]"


# ---------------------------------------------------------------------------
# _strip_model_postfix_from_messages with token-tracking format
# ---------------------------------------------------------------------------

def test_strip_model_postfix_removes_token_tracking_format():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer\n\n[smart-router/L1/L1-In:3032|Out:1000]"},
        {"role": "user", "content": "Follow-up"},
    ]
    _strip_model_postfix_from_messages(messages)
    assert messages[1]["content"] == "First answer"


def test_strip_model_postfix_removes_multi_tier_format():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer\n\n[smart-router/L2/L1-In:3032|Out:1000, L2-In:10021|Out:6054]"},
        {"role": "user", "content": "Follow-up"},
    ]
    _strip_model_postfix_from_messages(messages)
    assert messages[1]["content"] == "First answer"


def test_strip_model_postfix_removes_classic_format():
    """Ensure the classic format still strips correctly."""
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer\n\n[smart-router/L1]"},
        {"role": "user", "content": "Follow-up"},
    ]
    _strip_model_postfix_from_messages(messages)
    assert messages[1]["content"] == "First answer"


def test_strip_model_postfix_preserves_inline_user_text_with_tokens():
    messages = [{"role": "user", "content": "Explain [smart-router/L1/L1-In:100|Out:50] syntax"}]
    _strip_model_postfix_from_messages(messages)
    # User messages are not stripped — only assistant messages
    assert messages[0]["content"] == "Explain [smart-router/L1/L1-In:100|Out:50] syntax"


# ---------------------------------------------------------------------------
# SessionPin token_usage field
# ---------------------------------------------------------------------------

def test_session_pin_has_token_usage_default():
    pin = SessionPin(
        session_id="test-1",
        level=Level.L1,
        model="test/model",
    )
    assert pin.token_usage == {}


def test_session_pin_token_usage_accumulation():
    pin = SessionPin(
        session_id="test-1",
        level=Level.L1,
        model="test/model",
    )
    accumulate(pin.token_usage, "L1", 100, 50)
    accumulate(pin.token_usage, "L1", 200, 100)
    assert pin.token_usage == {"L1": {"prompt": 300, "completion": 150}}


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

def test_token_tracking_config_defaults():
    from app.config.schema import TokenTrackingConfig
    cfg = TokenTrackingConfig()
    assert cfg.enabled is True
    assert cfg.show_in_postfix is True


def test_token_tracking_config_disabled():
    from app.config.schema import TokenTrackingConfig
    cfg = TokenTrackingConfig(enabled=False, show_in_postfix=False)
    assert cfg.enabled is False
    assert cfg.show_in_postfix is False


def test_settings_has_token_tracking():
    from app.config.schema import Settings
    s = Settings()
    assert hasattr(s.telemetry, "token_tracking")
    assert s.telemetry.token_tracking.enabled is True
