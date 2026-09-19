"""Unit tests for the opt-in custom guardrail (TypeSafe Noul, input only, v2.24.0)."""
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


def _noul_response(noul=0.98):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "guardrail": {
                "type": "noul",
                "noul": noul,
            }
        },
        "usage": {"input_tokens": 300, "output_tokens": 30},
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

    def test_last_user_message_survives_head_truncation(self):
        # A large system prompt must not push the user question out of budget.
        messages = [
            {"role": "system", "content": "s" * 20_000},
            {"role": "user", "content": "What is last week sales?"},
        ]
        out = build_payload_text(messages, max_chars=8_000)
        assert "What is last week sales?" in out
        assert "[truncated]" in out
        assert len(out) <= 8_000 + 100

    def test_last_user_message_preferred_over_earlier_user(self):
        # In multi-turn traffic only the LAST user message is guaranteed.
        messages = [
            {"role": "user", "content": "first question " + "a" * 9_000},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second question"},
        ]
        out = build_payload_text(messages, max_chars=500)
        assert "second question" in out

    def test_oversized_user_message_tail_truncated(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "y" * 9_000 + " THE QUESTION"},
        ]
        out = build_payload_text(messages, max_chars=200)
        assert "THE QUESTION" in out
        assert out.startswith("...[truncated]")


# --- Noul probability parsing & thresholding -----------------------------------


class TestNoulDecision:
    @pytest.mark.asyncio
    async def test_high_probability_passes(self):
        s = CustomGuardrailSettings(enabled=True)
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_noul_response(0.97)))
        d = await engine.evaluate("payload")
        assert d.decision == "yes"
        assert d.probability_yes == 0.97
        assert d.source == "model"
        assert "0.97" in d.reason

    @pytest.mark.asyncio
    async def test_low_probability_rejects(self):
        s = CustomGuardrailSettings(enabled=True)
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_noul_response(0.05)))
        d = await engine.evaluate("payload")
        assert d.decision == "no"
        assert d.probability_yes == 0.05
        assert "0.05" in d.reason

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        # strict threshold: 0.7 P(yes) fails a 0.9 threshold but passes 0.5
        s = CustomGuardrailSettings(enabled=True, yes_threshold=0.9)
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_noul_response(0.7)))
        d = await engine.evaluate("payload")
        assert d.decision == "no"
        s2 = CustomGuardrailSettings(enabled=True, yes_threshold=0.5)
        engine2 = CustomGuardrailEngine(s2, http_client=_mock_http(_noul_response(0.7)))
        d2 = await engine2.evaluate("payload")
        assert d2.decision == "yes"

    @pytest.mark.asyncio
    async def test_noul_out_of_range_is_error(self):
        s = CustomGuardrailSettings(enabled=True, on_error="pass")
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_noul_response(1.5)))
        d = await engine.evaluate("payload")
        assert d.source == "error"
        assert d.decision == "yes"  # fail-open

    @pytest.mark.asyncio
    async def test_noul_not_a_number_is_error(self):
        s = CustomGuardrailSettings(enabled=True, on_error="pass")
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_noul_response("high")))
        d = await engine.evaluate("payload")
        assert d.source == "error"

    @pytest.mark.asyncio
    async def test_on_error_reject_fail_closed(self):
        s = CustomGuardrailSettings(enabled=True, on_error="reject")
        engine = CustomGuardrailEngine(s, http_client=_mock_http(_noul_response(None)))
        d = await engine.evaluate("payload")
        assert d.decision == "no"
        assert d.source == "error"

    @pytest.mark.asyncio
    async def test_http_error_fail_open(self):
        mock_http = MagicMock()
        mock_http.post = AsyncMock(side_effect=RuntimeError("boom"))
        mock_http.aclose = AsyncMock()
        s = CustomGuardrailSettings(enabled=True)
        engine = CustomGuardrailEngine(s, http_client=mock_http)
        d = await engine.evaluate("payload")
        assert d.decision == "yes"
        assert d.source == "error"

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

    def test_prompt_file_override(self, tmp_path):
        f = tmp_path / "policy.txt"
        f.write_text("FILE POLICY QUESTION?")
        s = CustomGuardrailSettings(enabled=True, prompt="INLINE", prompt_file=str(f))
        assert s.resolve_prompt() == "FILE POLICY QUESTION?"
        # unreadable file falls back to inline prompt
        s2 = CustomGuardrailSettings(enabled=True, prompt="INLINE", prompt_file="/nonexistent/x.txt")
        assert s2.resolve_prompt() == "INLINE"

    def test_inline_prompt_and_default(self):
        s = CustomGuardrailSettings(enabled=True, prompt="MY POLICY QUESTION?")
        assert s.resolve_prompt() == "MY POLICY QUESTION?"
        # built-in default is a yes/no question (Noul best practice)
        default = CustomGuardrailSettings().resolve_prompt()
        assert default.strip().endswith("?")

    @pytest.mark.asyncio
    async def test_disabled_returns_pass_without_http_call(self):
        s = CustomGuardrailSettings(enabled=False)
        mock_http = MagicMock()
        mock_http.post = AsyncMock()
        engine = CustomGuardrailEngine(s, http_client=mock_http)
        d = await engine.evaluate("payload")
        assert d.decision == "yes"
        assert d.source == "disabled"
        mock_http.post.assert_not_awaited()


# --- request payload shape -----------------------------------------------------


class TestPayloadShape:
    @pytest.mark.asyncio
    async def test_systemone_noul_payload_structure(self):
        mock_http = _mock_http(_noul_response(0.9))
        s = CustomGuardrailSettings(enabled=True, prompt="Does the payload comply with my policy?")
        engine = CustomGuardrailEngine(s, http_client=mock_http)
        d = await engine.evaluate("the payload text")
        assert d.decision == "yes"
        url = mock_http.post.await_args.args[0]
        payload = mock_http.post.await_args.kwargs["json"]
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert payload["model"] == "jev-1.13.0"
        assert payload["state"] == "the payload text"
        q = payload["questions"]["guardrail"]
        assert q["type"] == "noul"
        assert q["instructions"] == "Does the payload comply with my policy?"
        assert set(q["criteria"]) == {"true", "false"}

    @pytest.mark.asyncio
    async def test_generic_chat_fallback_endpoint(self):
        mock_http = _mock_http({"choices": [{"message": {"content": json.dumps({"decision": "no", "reason": "bad"})}}]})
        s = CustomGuardrailSettings(enabled=True, base_url="https://openrouter.ai/api/v1", model="m/x")
        engine = CustomGuardrailEngine(s, http_client=mock_http)
        d = await engine.evaluate("payload")
        assert d.decision == "no"
        assert d.reason == "bad"
        url = mock_http.post.await_args.args[0]
        assert url == "https://openrouter.ai/api/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_raw_yes_string_fallback(self):
        s = CustomGuardrailSettings(enabled=True, base_url="https://x/v1")
        engine = CustomGuardrailEngine(s, http_client=_mock_http({"choices": [{"message": {"content": "yes"}}]}))
        d = await engine.evaluate("payload")
        assert d.decision == "yes"
        assert d.probability_yes == 1.0


class TestRejectionDelivery:
    """rejection_delivery="reply" must yield a single 200 completion with the
    router version postfix; "error" keeps the legacy 400 envelope."""

    def _settings(self, delivery: str):
        from app.guardrails.custom import CustomGuardrailSettings
        return CustomGuardrailSettings(enabled=True, rejection_message="Out of scope ({reason})", rejection_delivery=delivery)

    def _body(self, stream: bool = False):
        body = MagicMock()
        body.stream = stream
        body.model = "smart-router"
        return body

    def test_reply_delivery_returns_200_completion_with_version_postfix(self):
        from fastapi.responses import JSONResponse
        from app.api.chat import _custom_guardrail_rejection
        from app.version import APPLICATION_VERSION

        resp = _custom_guardrail_rejection("policy 0.38", self._settings("reply"), self._body())
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 200
        content = resp.body.decode() if isinstance(resp.body, bytes) else str(resp.body)
        payload = json.loads(content)
        msg = payload["choices"][0]["message"]["content"]
        assert msg.startswith("Out of scope (policy 0.38)")
        assert msg.endswith(f"[smart-router/v{APPLICATION_VERSION}]")
        assert payload["guardrail"]["code"] == "router_custom_guardrail_rejected"
        assert payload["choices"][0]["finish_reason"] == "stop"

    def test_error_delivery_keeps_legacy_400_envelope(self):
        from fastapi.responses import JSONResponse
        from app.api.chat import _custom_guardrail_rejection

        resp = _custom_guardrail_rejection("policy 0.38", self._settings("error"), self._body())
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        payload = json.loads(resp.body.decode() if isinstance(resp.body, bytes) else str(resp.body))
        assert payload["error"]["code"] == "router_custom_guardrail_rejected"

    def test_streaming_reply_yields_sse_chunks(self):
        import asyncio
        from fastapi.responses import StreamingResponse
        from app.api.chat import _custom_guardrail_rejection
        from app.version import APPLICATION_VERSION

        resp = _custom_guardrail_rejection("policy 0.38", self._settings("reply"), self._body(stream=True))
        assert isinstance(resp, StreamingResponse)

        async def collect():
            chunks = []
            async for part in resp.body_iterator:
                chunks.append(part)
            return "".join(chunks)

        raw = asyncio.run(collect())
        assert f"[smart-router/v{APPLICATION_VERSION}]" in raw
        assert "Out of scope (policy 0.38)" in raw
        assert raw.rstrip().endswith("data: [DONE]")

    def test_default_delivery_is_reply(self):
        from app.guardrails.custom import CustomGuardrailSettings
        assert CustomGuardrailSettings().rejection_delivery == "reply"
