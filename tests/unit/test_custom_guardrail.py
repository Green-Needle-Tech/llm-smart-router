"""Unit tests for the opt-in custom guardrail (typesafe yes/no, v2.23.x)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.guardrails.custom import (
    CustomGuardrailDecision,
    CustomGuardrailEngine,
    CustomGuardrailSettings,
    build_payload_text,
)


def _systemone_response(choice="yes", reason="looks fine"):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "guardrail": {
                "type": "choice",
                "choice": choice,
                "reason": reason,
            }
        },
    }


def _mock_http(response_dict):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=response_dict)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.aclose = AsyncMock()
    return mock_http


# --- payload builder ---------------------------------------------------------


class TestBuildPayloadText:
    def test_role_tagging_and_blocks(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        out = build_payload_text(messages)
        assert "[system] You are helpful." in out
        assert "[user] hello" in out

    def test_truncation(self):
        messages = [{"role": "user", "content": "x" * 100}]
        out = build_payload_text(messages, max_chars=10)
        assert len(out) < 40
        assert "[truncated]" in out


# --- strict decision parsing --------------------------------------------------


class TestParseDecision:
    @pytest.mark.asyncio
    async def test_raw_yes(self):
        s = CustomGuardrailSettings(enabled=True, base_url="https://x/v1")
        engine = CustomGuardrailEngine(s, http_client=_mock_http({"choices": [{"message": {"content": "yes"}}]}))
        decision = await engine.evaluate("payload")
        assert decision.decision == "yes"

    @pytest.mark.asyncio
    async def test_systemone_no(self):
        s = CustomGuardrailSettings(enabled=True)
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_systemone_response("no", "policy violation")))
        decision = await engine.evaluate("payload")
        assert decision.decision == "no"
        assert decision.reason == "policy violation"
        assert decision.source == "model"

    @pytest.mark.asyncio
    async def test_case_insensitive_and_quotes(self):
        s = CustomGuardrailSettings(enabled=True)
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_systemone_response('"YES"')))
        decision = await engine.evaluate("payload")
        assert decision.decision == "yes"

    @pytest.mark.asyncio
    async def test_invalid_choice_is_parse_error_not_crash(self):
        s = CustomGuardrailSettings(enabled=True, on_error="pass")
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_systemone_response("maybe")))
        decision = await engine.evaluate("payload")
        assert decision.decision == "yes"  # fail-open
        assert decision.source == "error"
        assert "error" in decision.reason

    @pytest.mark.asyncio
    async def test_on_error_reject_fail_closed(self):
        s = CustomGuardrailSettings(enabled=True, on_error="reject")
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_systemone_response("banana")))
        decision = await engine.evaluate("payload")
        assert decision.decision == "no"
        assert decision.source == "error"

    @pytest.mark.asyncio
    async def test_http_error_fail_open(self):
        mock_http = MagicMock()
        mock_http.post = AsyncMock(side_effect=RuntimeError("boom"))
        mock_http.aclose = AsyncMock()
        s = CustomGuardrailSettings(enabled=True)
        engine = CustomGuardrailEngine(s, http_client=mock_http)
        decision = await engine.evaluate("payload")
        assert decision.decision == "yes"
        assert decision.source == "error"

    def test_pydantic_literal_enforced(self):
        with pytest.raises(Exception):
            CustomGuardrailDecision(decision="maybe")
        assert CustomGuardrailDecision(decision="yes").decision == "yes"


# --- global toggle & prompt customization ------------------------------------


class TestToggleAndPrompt:
    def test_disabled_by_default(self):
        s = CustomGuardrailSettings()
        assert s.enabled is False
        assert not s.is_enabled()

    def test_enabled_global_toggle(self):
        assert CustomGuardrailSettings(enabled=True).is_enabled()

    def test_apply_on_phase(self):
        assert CustomGuardrailSettings(apply_on="input").applies_to_phase("input")
        assert not CustomGuardrailSettings(apply_on="input").applies_to_phase("output")
        assert CustomGuardrailSettings(apply_on="both").applies_to_phase("output")

    def test_prompt_file_override(self, tmp_path):
        f = tmp_path / "policy.txt"
        f.write_text("FILE POLICY")
        s = CustomGuardrailSettings(enabled=True, prompt="INLINE", prompt_file=str(f))
        assert s.resolve_prompt() == "FILE POLICY"
        # unreadable file falls back to inline prompt
        s2 = CustomGuardrailSettings(enabled=True, prompt="INLINE", prompt_file="/nonexistent/x.txt")
        assert s2.resolve_prompt() == "INLINE"

    def test_inline_prompt_and_default(self):
        s = CustomGuardrailSettings(enabled=True, prompt="MY POLICY")
        assert s.resolve_prompt() == "MY POLICY"
        assert CustomGuardrailSettings().resolve_prompt()  # built-in default non-empty

    @pytest.mark.asyncio
    async def test_disabled_returns_pass_without_http_call(self):
        s = CustomGuardrailSettings(enabled=False)
        mock_http = MagicMock()
        mock_http.post = AsyncMock()
        engine = CustomGuardrailEngine(s, http_client=mock_http)
        decision = await engine.evaluate("payload")
        assert decision.decision == "yes"
        assert decision.source == "disabled"
        mock_http.post.assert_not_awaited()


# --- request payload shape -----------------------------------------------------


class TestPayloadShape:
    @pytest.mark.asyncio
    async def test_systemone_payload_structure(self):
        mock_http = _mock_http(_systemone_response("yes"))
        s = CustomGuardrailSettings(enabled=True, prompt="MY CUSTOM PROMPT")
        engine = CustomGuardrailEngine(s, http_client=mock_http)
        decision = await engine.evaluate("the payload text")
        assert decision.decision == "yes"
        url = mock_http.post.await_args.args[0]
        payload = mock_http.post.await_args.kwargs["json"]
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert payload["model"] == "jev-1.13.0"
        assert payload["state"] == "the payload text"
        q = payload["questions"]["guardrail"]
        assert q["type"] == "choice"
        assert q["instructions"] == "MY CUSTOM PROMPT"
        assert set(q["criteria"]) == {"yes", "no"}

    @pytest.mark.asyncio
    async def test_generic_chat_fallback_endpoint(self):
        mock_http = _mock_http({"choices": [{"message": {"content": json.dumps({"decision": "no", "reason": "bad"})}}]})
        s = CustomGuardrailSettings(enabled=True, base_url="https://openrouter.ai/api/v1", model="m/x")
        engine = CustomGuardrailEngine(s, http_client=mock_http)
        decision = await engine.evaluate("payload")
        assert decision.decision == "no"
        assert decision.reason == "bad"
        url = mock_http.post.await_args.args[0]
        assert url == "https://openrouter.ai/api/v1/chat/completions"
