"""Tests for response model-identification postfix behavior."""
import json
from types import SimpleNamespace

import pytest

from app.api.chat import _add_model_postfix, _strip_model_postfix_from_messages
from app.api.chat import _forward_to_provider
from app.schemas.router import ClassificationResult, ClassificationSource, Level, RouteDecision


def test_add_model_postfix_appends_upstream_model():
    body = {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}

    _add_model_postfix(body, "google/gemini-2.5-flash")

    assert body["choices"][0]["message"]["content"] == (
        "Hello\n\n[LLM: google/gemini-2.5-flash]"
    )


def test_add_model_postfix_handles_null_content():
    body = {"choices": [{"message": {"role": "assistant", "content": None}}]}

    _add_model_postfix(body, "z-ai/glm-5.2")

    assert body["choices"][0]["message"]["content"] == "[LLM: z-ai/glm-5.2]"


def test_add_model_postfix_does_not_mutate_non_assistant_choices():
    body = {"choices": [{"delta": {"content": "Hello"}}]}

    _add_model_postfix(body, "model/test")

    assert body == {"choices": [{"delta": {"content": "Hello"}}]}


def test_strip_model_postfix_removes_it_before_forwarding():
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
            {"type": "text", "text": "Answer\n\n[LLM: z-ai/glm-5.2]"},
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
        ],
    }]

    _strip_model_postfix_from_messages(messages)

    assert messages[0]["content"][0]["text"] == "Answer"


@pytest.mark.asyncio
async def test_forwarding_payload_excludes_model_postfix():
    captured = {}

    async def fake_non_stream(request, payload, *args):
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
