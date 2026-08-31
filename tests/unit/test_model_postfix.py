"""Tests for response model-identification postfix behavior."""
import json
from types import SimpleNamespace

import pytest

from app.api.chat import _add_model_postfix, _strip_model_postfix_from_messages
from app.api.chat import _forward_to_provider
from app.schemas.router import ClassificationResult, ClassificationSource, Level, RouteDecision


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


def test_add_model_postfix_appends_upstream_model():
    body = {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}

    _add_model_postfix(body, "google/gemini-2.5-flash", _route())

    assert body["choices"][0]["message"]["content"] == (
        "Hello\n\n[smart-router/L1]"
    )


def test_add_model_postfix_handles_null_content():
    body = {"choices": [{"message": {"role": "assistant", "content": None}}]}

    _add_model_postfix(body, "z-ai/glm-5.2", _route())

    assert body["choices"][0]["message"]["content"] == "[smart-router/L1]"


def test_add_model_postfix_does_not_mutate_non_assistant_choices():
    body = {"choices": [{"delta": {"content": "Hello"}}]}

    _add_model_postfix(body, "model/test", _route())

    assert body == {"choices": [{"delta": {"content": "Hello"}}]}


def test_add_model_postfix_skips_tool_call_finish_reason():
    """Postfix must not appear on tool-call responses (finish_reason=tool_calls)."""
    body = {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "tool_calls"}]}

    _add_model_postfix(body, "z-ai/glm-5.2", _route())

    assert body["choices"][0]["message"]["content"] is None


def test_add_model_postfix_skips_message_with_tool_calls():
    """Postfix must not appear when message has tool_calls (even if finish_reason=stop)."""
    body = {"choices": [{"message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}],
    }, "finish_reason": "stop"}]}

    _add_model_postfix(body, "z-ai/glm-5.2", _route())

    assert body["choices"][0]["message"]["content"] is None


def test_add_model_postfix_appends_to_text_response_with_tool_calls_in_other_choices():
    """If one choice is text and another is tool_calls, only the text choice gets postfix."""
    body = {"choices": [
        {"message": {"role": "assistant", "content": "Here is the answer"}, "finish_reason": "stop"},
        {"message": {"role": "assistant", "content": None, "tool_calls": []}, "finish_reason": "tool_calls"},
    ]}

    _add_model_postfix(body, "z-ai/glm-5.2", _route())

    assert body["choices"][0]["message"]["content"] == "Here is the answer\n\n[smart-router/L1]"
    assert body["choices"][1]["message"]["content"] is None


def test_strip_model_postfix_removes_new_format_before_forwarding():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer\n\n[smart-router/L1]"},
        {"role": "user", "content": "Follow-up"},
    ]

    _strip_model_postfix_from_messages(messages)

    assert messages[1]["content"] == "First answer"


def test_strip_model_postfix_removes_legacy_format_before_forwarding():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer\n\n[LLM: google/gemini-2.5-flash]"},
        {"role": "user", "content": "Follow-up"},
    ]

    _strip_model_postfix_from_messages(messages)

    assert messages[1]["content"] == "First answer"


def test_strip_model_postfix_preserves_inline_user_text():
    messages = [{"role": "user", "content": "Explain [LLM: example/model] syntax"}]

    _strip_model_postfix_from_messages(messages)

    assert messages[0]["content"] == "Explain [LLM: example/model] syntax"


def test_strip_model_postfix_handles_structured_assistant_content():
    messages = [{
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Answer\n\n[smart-router/L1]"},
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
        ],
    }]

    _strip_model_postfix_from_messages(messages)

    assert messages[0]["content"][0]["text"] == "Answer"


@pytest.mark.asyncio
async def test_forwarding_payload_excludes_model_postfix():
    captured = {}

    async def fake_non_stream(request, payload, *args, **kwargs):
        captured["payload"] = payload
        return payload

    provider = SimpleNamespace(get_max_completion_tokens=lambda model: 100)
    routing = SimpleNamespace(
        get_max_tokens=lambda level: "auto",
        get_fallbacks=lambda level: [],
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        config=SimpleNamespace(get=lambda: SimpleNamespace(routing=routing)),
        provider=provider,
    )))
    body = SimpleNamespace(
        stream=False,
        model_dump=lambda **kwargs: {
            "messages": [{
                "role": "assistant",
                "content": "Previous answer\n\n[LLM: model/private]",
            }],
        },
    )
    route = RouteDecision(
        level=Level.L1,
        model="model/external",
        params={},
        classification=ClassificationResult(
            level=Level.L1,
            confidence=1.0,
            reason="test",
            source=ClassificationSource.OVERRIDE,
        ),
    )

    from app.api import chat
    original = chat._handle_non_stream
    chat._handle_non_stream = fake_non_stream
    try:
        await _forward_to_provider(request, body, route, None, None, None, False, 0)
    finally:
        chat._handle_non_stream = original

    assert captured["payload"]["messages"][0]["content"] == "Previous answer"
