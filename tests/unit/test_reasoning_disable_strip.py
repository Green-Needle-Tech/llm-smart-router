"""Tests for stripping `reasoning: {"enabled": false}` on reasoning-mandatory models.

z-ai/glm-5.3 rejects a reasoning disable with HTTP 400 ("Reasoning is
mandatory for this endpoint and cannot be disabled").  The router strips
the disable for models listed in provider.reasoning_mandatory_models so
auxiliary calls (title generation etc.) don't burn retries and fall back.
"""
from types import SimpleNamespace

from app.api.chat import _build_upstream_payload, _strip_reasoning_disable
from app.schemas.openai import ChatCompletionRequest
from app.schemas.router import (
    ClassificationResult,
    ClassificationSource,
    Level,
    RouteDecision,
)


def _route(model: str = "z-ai/glm-5.3") -> RouteDecision:
    return RouteDecision(
        level=Level.L3,
        model=model,
        params={},
        classification=ClassificationResult(
            level=Level.L3,
            confidence=1.0,
            reason="test",
            source=ClassificationSource.OVERRIDE,
        ),
    )


def _config(mandatory=("z-ai/glm-5.3",)) -> SimpleNamespace:
    return SimpleNamespace(
        provider=SimpleNamespace(reasoning_mandatory_models=list(mandatory)),
        routing=SimpleNamespace(get_max_tokens=lambda level: None),
    )


def test_strip_removes_disabled_reasoning_for_mandatory_model():
    payload = {"reasoning": {"enabled": False}}
    _strip_reasoning_disable(payload, _route(), _config())
    assert "reasoning" not in payload


def test_strip_keeps_enabled_reasoning():
    payload = {"reasoning": {"enabled": True, "effort": "low"}}
    _strip_reasoning_disable(payload, _route(), _config())
    assert payload["reasoning"] == {"enabled": True, "effort": "low"}


def test_strip_noop_for_other_models():
    payload = {"reasoning": {"enabled": False}}
    _strip_reasoning_disable(payload, _route("openai/gpt-5.6-luna"), _config())
    assert payload["reasoning"] == {"enabled": False}


def test_strip_noop_when_list_empty():
    payload = {"reasoning": {"enabled": False}}
    _strip_reasoning_disable(payload, _route(), _config(mandatory=()))
    assert payload["reasoning"] == {"enabled": False}


def test_strip_noop_for_non_dict_reasoning():
    payload = {"reasoning": "low"}
    _strip_reasoning_disable(payload, _route(), _config())
    assert payload["reasoning"] == "low"


def test_build_upstream_payload_strips_disable():
    body = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"enabled": False},
        }
    )
    provider = SimpleNamespace(get_max_completion_tokens=lambda m: None)
    payload = _build_upstream_payload(body, _route(), "sess-1", _config(), provider)
    assert payload["model"] == "z-ai/glm-5.3"
    assert "reasoning" not in payload
