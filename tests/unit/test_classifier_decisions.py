"""Unit tests for the decisions-mode classifier (v2.20.0)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.classify.classifier import ClassifierService
from app.classify.parser import parse_classifier_output
from app.schemas.openai import ChatMessage
from app.schemas.router import ClassificationSource


def _make_config(provider_mode="decisions", tier_criteria=None):
    classification = SimpleNamespace(
        enabled=True,
        model="typesafe/jev-1.13",
        temperature=0,
        max_tokens=60,
        timeout_seconds=8,
        default_level="L3",
        unknown_level="L1",
        min_confidence=0.5,
        low_confidence_action="escalate",
        prompt_file="/app/config/prompts/classifier.txt",
        rubric_version="v1",
        base_url="https://openrouter.ai/api/v1",
        api_key_env=None,
        provider_mode=provider_mode,
        tier_criteria=tier_criteria or {},
        digest=SimpleNamespace(
            system_chars=500,
            tail_chars=2000,
            include_tool_names=True,
            include_context_summary=True,
            strip_scaffolding=True,
            learn_common_prefix=True,
            prefix_samples=20,
            min_prefix_chars=200,
            strip_sections_enabled=False,
            strip_sections=[],
            keep_sections=[],
            delimit_untrusted=True,
            injection_guard=True,
        ),
        cache=SimpleNamespace(enabled=False, ttl_seconds=3600, max_entries=10000),
        tier_prefix=SimpleNamespace(enabled=False),
    )
    provider = SimpleNamespace(base_url="https://openrouter.ai/api/v1", headers={})
    heuristics = SimpleNamespace(enabled=False, measure="task_payload", huge_context_tokens=100000, rules=None)
    return SimpleNamespace(classification=classification, provider=provider, heuristics=heuristics)


def _decisions_response(choice="L2", confidence=0.9):
    return {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "tier": {
                "type": "choice",
                "choice": choice,
                "probabilities": {"L1": 0.02, "L2": 0.9, "L3": 0.05, "L4": 0.02, "L5": 0.01},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 393, "output_tokens": 58},
    }


@pytest.mark.asyncio
async def test_decisions_mode_builds_correct_payload_and_parses():
    config = _make_config()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=_decisions_response("L2", 0.9))
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    svc = ClassifierService(config, openrouter_api_key="test-key", http_client=mock_http)
    messages = [ChatMessage(role="user", content="Summarize this article")]
    result, _ = await svc.classify(messages)

    # Payload shape
    call = mock_http.post.call_args
    # /v1 is stripped: decisions API lives at /api/alpha/decisions
    assert call.args[0] == "https://openrouter.ai/api/alpha/decisions"
    payload = call.kwargs["json"]
    assert payload["model"] == "typesafe/jev-1.13"
    assert payload["questions"]["tier"]["type"] == "choice"
    assert set(payload["questions"]["tier"]["criteria"]) == {"L1", "L2", "L3", "L4", "L5"}

    # Result
    assert result.level is not None and result.level.value == "L2"
    assert result.confidence == pytest.approx(0.9)
    assert result.source == ClassificationSource.MODEL


@pytest.mark.asyncio
async def test_decisions_mode_custom_criteria():
    custom = {"L1": "trivial", "L2": "easy", "L3": "medium", "L4": "hard", "L5": "expert"}
    config = _make_config(tier_criteria=custom)
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=_decisions_response("L4", 0.8))
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    svc = ClassifierService(config, openrouter_api_key="test-key", http_client=mock_http)
    messages = [ChatMessage(role="user", content="Design a rate limiter")]
    result, _ = await svc.classify(messages)

    payload = mock_http.post.call_args.kwargs["json"]
    assert payload["questions"]["tier"]["criteria"] == custom
    assert result.level.value == "L4"


@pytest.mark.asyncio
async def test_decisions_mode_invalid_level_falls_back_to_default():
    config = _make_config()
    bad = _decisions_response("L9", 0.5)
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=bad)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    svc = ClassifierService(config, openrouter_api_key="test-key", http_client=mock_http)
    messages = [ChatMessage(role="user", content="hello")]
    result, _ = await svc.classify(messages)

    # ValueError -> caught by classify() -> default result
    assert result.source == ClassificationSource.DEFAULT
    assert result.level.value == "L3"


@pytest.mark.asyncio
async def test_decisions_mode_http_error_falls_back_to_default():
    import httpx

    config = _make_config()
    mock_http = MagicMock()
    mock_http.post = AsyncMock(side_effect=httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock(status_code=500)))

    svc = ClassifierService(config, openrouter_api_key="test-key", http_client=mock_http)
    messages = [ChatMessage(role="user", content="hello")]
    result, _ = await svc.classify(messages)

    assert result.source == ClassificationSource.DEFAULT


def test_decisions_json_output_parses_via_existing_parser():
    raw = json.dumps({"level": "L3", "confidence": 0.72, "reason": "decision model (typesafe/jev-1.13)"})
    result = parse_classifier_output(raw, source=ClassificationSource.MODEL)
    assert result.level.value == "L3"
    assert result.confidence == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_chat_mode_unchanged_when_provider_mode_chat():
    config = _make_config(provider_mode="chat")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": json.dumps({"level": "L1", "confidence": 0.99, "reason": "greeting"})}}]
    })
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    svc = ClassifierService(config, openrouter_api_key="test-key", http_client=mock_http)
    messages = [ChatMessage(role="user", content="Hey thanks!")]
    result, _ = await svc.classify(messages)

    call = mock_http.post.call_args
    assert call.args[0].endswith("/chat/completions")
    assert result.level.value == "L1"
