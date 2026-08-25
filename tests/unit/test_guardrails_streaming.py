"""Tests for guardrail secret masking across streamed chunks.

Secrets streamed token-by-token arrive split across SSE chunks; no single
chunk contains a complete key, so per-chunk mask_secrets() cannot match.
The stream handler's carry buffer (app/guardrails/streaming.py) holds back
a trailing partial secret until it completes (and gets masked) or proves
benign. These tests exercise that logic directly, including the full
chunked pipeline: mask -> carry-split -> accumulate.
"""
from __future__ import annotations

from app.guardrails.scanner import GuardrailConfig, GuardrailEngine
from app.guardrails.streaming import secret_carry_split


def _engine() -> GuardrailEngine:
    return GuardrailEngine(GuardrailConfig())


def _b64ish(n: int) -> str:
    """Deterministic alphanumeric body of length n (avoids terminal redaction)."""
    import string
    alpha = string.ascii_letters + string.digits
    return "".join(alpha[i % len(alpha)] for i in range(n))


def _stream_through_carry(engine: GuardrailEngine, chunks: list[str]) -> str:
    """Simulate the stream handler: per-chunk mask + carry split."""
    out = ""
    carry = ""
    for c in chunks:
        text = carry + c
        text, _fs = engine.mask_secrets(text)
        idx = secret_carry_split(text)
        out += text[:idx]
        carry = text[idx:]
    # [DONE] flush
    if carry:
        carry, _fs = engine.mask_secrets(carry)
        out += carry
    return out


class TestSecretCarrySplit:
    def test_full_marker_short_body_held(self):
        # sk-or-v1- body must be 16+; 8 chars -> incomplete, hold from marker
        text = "prefix sk-or-v1-" + _b64ish(8)
        idx = secret_carry_split(text)
        assert idx == len("prefix ")

    def test_full_marker_complete_body_released(self):
        # A complete key is masked by mask_secrets BEFORE the carry split,
        # so the split sees no marker at all and releases everything.
        e = _engine()
        key = "sk-or-v1-" + _b64ish(24)
        text, _fs = e.mask_secrets(f"prefix {key}")
        assert "***REDACTED***" in text
        assert secret_carry_split(text) == len(text)

    def test_unmasked_marker_with_long_body_releases_marker(self):
        # Direct split (no masking): body >= threshold+margin means a real
        # key would already be masked, so only a marker-prefix tail may hold.
        text = "prefix sk-or-v1-" + _b64ish(24)
        idx = secret_carry_split(text)
        # The marker+body must NOT be held (only a <=16-char prefix tail may be)
        assert idx > len("prefix sk-or-v1-")

    def test_partial_marker_prefix_held(self):
        # "sk-or" is a prefix of "sk-or-v1-" and may complete next chunk
        text = "hello sk-or"
        idx = secret_carry_split(text)
        assert idx == len("hello ")

    def test_plain_text_released(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert secret_carry_split(text) == len(text)

    def test_numbers_only_not_held(self):
        # A short digit run (1-7 digits) is not a telegram token start
        text = "value 12345 done"
        assert secret_carry_split(text) == len(text)

    def test_telegram_partial_held(self):
        text = "token 123456789:AA" + _b64ish(10)
        idx = secret_carry_split(text)
        assert idx == len("token ")

    def test_akia_partial_held(self):
        text = "key AKIA" + _b64ish(5).upper()
        idx = secret_carry_split(text)
        assert idx == len("key ")

    def test_pem_partial_held(self):
        text = "cert -----BEGIN RSA PRIV"
        idx = secret_carry_split(text)
        assert idx == len("cert ")


class TestChunkedStreamingMaskPipeline:
    def test_openrouter_key_token_by_token(self):
        key = "sk-or-v1-" + _b64ish(28)
        chunks = ["Here is ", "the key: ", key[:12], key[12:25], key[25:]]
        out = _stream_through_carry(_engine(), chunks)
        assert key not in out
        assert "***REDACTED***" in out

    def test_openrouter_key_single_chunk(self):
        key = "sk-or-v1-" + _b64ish(28)
        out = _stream_through_carry(_engine(), [f"key: {key}"])
        assert key not in out and "***REDACTED***" in out

    def test_openrouter_key_char_by_char(self):
        key = "sk-or-v1-" + _b64ish(28)
        text = f"key: {key}"
        chunks = [text[i:i+3] for i in range(0, len(text), 3)]
        out = _stream_through_carry(_engine(), chunks)
        assert key not in out
        assert "***REDACTED***" in out

    def test_github_token_token_by_token(self):
        key = "ghp_" + _b64ish(30)
        chunks = ["secret ", key[:10], key[10:22], key[22:]]
        out = _stream_through_carry(_engine(), chunks)
        assert key not in out and "***REDACTED***" in out

    def test_aws_key_token_by_token(self):
        key = "AKIA" + _b64ish(16).upper()
        chunks = ["aws ", key[:8], key[8:]]
        out = _stream_through_carry(_engine(), chunks)
        assert key not in out and "***REDACTED***" in out

    def test_telegram_bot_token_chunked(self):
        key = "123456789:AA" + _b64ish(32)
        chunks = ["bot ", key[:15], key[15:]]
        out = _stream_through_carry(_engine(), chunks)
        assert key not in out and "***REDACTED***" in out

    def test_benign_text_untouched(self):
        chunks = ["The capital of ", "France is Paris", ". It is lovely."]
        out = _stream_through_carry(_engine(), chunks)
        assert out == "The capital of France is Paris. It is lovely."

    def test_benign_words_with_sk_prefix_not_mangled(self):
        # "sk" as a word fragment must not trigger infinite holds or mangling
        chunks = ["He said sk", "ip the stone", " across the lake."]
        out = _stream_through_carry(_engine(), chunks)
        assert out == "He said skip the stone across the lake."

    def test_benign_numbers_not_mangled(self):
        chunks = ["Order 12345", " shipped on 2026-08-25."]
        out = _stream_through_carry(_engine(), chunks)
        assert out == "Order 12345 shipped on 2026-08-25."

    def test_pem_block_chunked(self):
        # NOTE: the PEM rule masks the -----BEGIN header only; the base64
        # body after it is NOT covered by the current rule set (same in
        # non-streaming mode). This test pins the current behavior so a
        # future rule improvement shows up as a deliberate change.
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBALKZB0E\n-----END RSA PRIVATE KEY-----"
        chunks = ["leaked:\n", pem[:30], pem[30:60], pem[60:]]
        out = _stream_through_carry(_engine(), chunks)
        assert "-----BEGIN RSA PRIVATE KEY-----" not in out
        assert out.count("***REDACTED***") >= 1
