"""Tests for upstream prompt-cache (KV cache) optimization."""
from __future__ import annotations

import pytest

from app.config.schema import Settings
from app.providers.prompt_cache import (
    apply_prompt_cache_features,
    extract_cache_usage,
    needs_explicit_cache_control,
)


@pytest.fixture
def config():
    return Settings()


class TestNeedsExplicitCacheControl:
    def test_anthropic(self):
        assert needs_explicit_cache_control("anthropic/claude-opus-4.5")

    def test_qwen(self):
        assert needs_explicit_cache_control("qwen/qwen3-235b-a22b")

    def test_openai_implicit(self):
        assert not needs_explicit_cache_control("openai/gpt-5.2")

    def test_deepseek_implicit(self):
        assert not needs_explicit_cache_control("deepseek/deepseek-chat")

    def test_gemini_implicit(self):
        assert not needs_explicit_cache_control("google/gemini-2.5-pro")


class TestSessionIdPassthrough:
    def test_session_id_added(self, config):
        payload = {"model": "openai/gpt-5.2", "messages": []}
        apply_prompt_cache_features(payload, "sess-abc", config)
        assert payload["session_id"] == "sess-abc"

    def test_client_session_id_preserved(self, config):
        payload = {"model": "openai/gpt-5.2", "messages": [], "session_id": "client-sid"}
        apply_prompt_cache_features(payload, "router-sid", config)
        assert payload["session_id"] == "client-sid"

    def test_no_session_id_noop(self, config):
        payload = {"model": "openai/gpt-5.2", "messages": []}
        apply_prompt_cache_features(payload, None, config)
        assert "session_id" not in payload

    def test_truncated_to_256(self, config):
        payload = {"model": "openai/gpt-5.2", "messages": []}
        apply_prompt_cache_features(payload, "x" * 300, config)
        assert len(payload["session_id"]) == 256

    def test_disabled_noop(self, config):
        config.provider.prompt_caching.enabled = False
        payload = {"model": "anthropic/claude-opus-4.5", "messages": [{"role": "system", "content": "x" * 8000}]}
        apply_prompt_cache_features(payload, "sess", config)
        assert "session_id" not in payload
        assert "cache_control" not in str(payload["messages"])


class TestCacheControlInjection:
    def _system_payload(self, model, text="You are a helpful assistant. " * 200):
        return {"model": model, "messages": [{"role": "system", "content": text},
                                             {"role": "user", "content": "hi"}]}

    def test_anthropic_gets_breakpoint(self, config):
        payload = self._system_payload("anthropic/claude-opus-4.5")
        apply_prompt_cache_features(payload, "s", config)
        blocks = payload["messages"][0]["content"]
        assert blocks[0]["cache_control"]["type"] == "ephemeral"
        assert blocks[0]["cache_control"]["ttl"] == "5m"

    def test_anthropic_1h_ttl(self, config):
        config.provider.prompt_caching.anthropic_ttl = "1h"
        payload = self._system_payload("anthropic/claude-opus-4.5")
        apply_prompt_cache_features(payload, "s", config)
        assert payload["messages"][0]["content"][0]["cache_control"]["ttl"] == "1h"

    def test_openai_not_injected(self, config):
        payload = self._system_payload("openai/gpt-5.2")
        apply_prompt_cache_features(payload, "s", config)
        assert "cache_control" not in str(payload["messages"])

    def test_short_prompt_below_floor(self, config):
        payload = self._system_payload("anthropic/claude-opus-4.5", text="too short")
        apply_prompt_cache_features(payload, "s", config)
        assert payload["messages"][0]["content"] == "too short"

    def test_existing_cache_control_not_overwritten(self, config):
        payload = self._system_payload("anthropic/claude-opus-4.5")
        payload["messages"][0]["content"] = [
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
        ]
        apply_prompt_cache_features(payload, "s", config)
        assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_injection_disabled(self, config):
        config.provider.prompt_caching.inject_cache_control = False
        payload = self._system_payload("anthropic/claude-opus-4.5")
        apply_prompt_cache_features(payload, "s", config)
        assert "cache_control" not in str(payload["messages"])
        # session_id still forwarded
        assert payload["session_id"] == "s"


class TestExtractCacheUsage:
    def test_full_usage(self):
        resp = {"usage": {"prompt_tokens": 10339,
                          "prompt_tokens_details": {"cached_tokens": 10318, "cache_write_tokens": 0}}}
        assert extract_cache_usage(resp) == (10318, 0)

    def test_missing_details(self):
        assert extract_cache_usage({"usage": {"prompt_tokens": 100}}) == (0, 0)

    def test_none(self):
        assert extract_cache_usage(None) == (0, 0)

    def test_garbage_values(self):
        resp = {"usage": {"prompt_tokens_details": {"cached_tokens": "abc", "cache_write_tokens": None}}}
        assert extract_cache_usage(resp) == (0, 0)
