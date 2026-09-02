"""Tests for the IP Redaction & Re-Hydration privacy module."""
from __future__ import annotations

import pytest

from app.privacy.ip_redaction import (
    IP_RE,
    IPRedactionEngine,
    IPRedactionStore,
)


@pytest.fixture
def engine(tmp_path):
    store = IPRedactionStore(str(tmp_path / "ip.db"))
    yield IPRedactionEngine(store)
    store.close()


# --- Detection ---------------------------------------------------------------

class TestDetection:
    def test_ipv4_matched(self):
        assert IP_RE.search("connect to 192.168.1.10 now")

    def test_ipv4_invalid_octet_not_matched(self):
        assert not IP_RE.search("version 256.300.1.999 here")

    def test_port_not_part_of_match(self):
        m = IP_RE.search("10.0.0.1:8080")
        assert m and m.group(0) == "10.0.0.1"

    def test_cidr_clean(self):
        m = IP_RE.search("subnet 10.0.0.0/24 range")
        assert m and m.group(0) == "10.0.0.0"

    def test_ipv6_full_matched(self):
        assert IP_RE.search("host 2001:0db8:85a3:0000:0000:8a2e:0370:7334 there")

    def test_ipv6_compressed_matched(self):
        assert IP_RE.search("addr ::1 loopback")

    def test_version_string_not_ip(self):
        # Built at runtime: octet 999 is invalid, so no IPv4 should match.
        fake = ".".join(["v1", "2", "3"]) + " and " + ".".join(["1", "2", "999", "5"])
        assert not IP_RE.search(fake)


# --- Redaction ---------------------------------------------------------------

class TestRedaction:
    @pytest.mark.asyncio
    async def test_single_ipv4(self, engine):
        out = await engine.redact_text("connect to 192.168.1.10 please", "s1")
        assert out == "connect to [ipaddress-01] please"

    @pytest.mark.asyncio
    async def test_multiple_unique_ips_sequential(self, engine):
        out = await engine.redact_text("from 10.0.0.1 to 10.0.0.2", "s1")
        assert out == "from [ipaddress-01] to [ipaddress-02]"

    @pytest.mark.asyncio
    async def test_duplicate_ip_same_prompt_same_token(self, engine):
        out = await engine.redact_text("10.0.0.1 then 10.0.0.1 again", "s1")
        assert out == "[ipaddress-01] then [ipaddress-01] again"

    @pytest.mark.asyncio
    async def test_duplicate_ip_across_turns_same_token(self, engine):
        await engine.redact_text("ping 10.0.0.1", "s1")
        out = await engine.redact_text("and 10.0.0.5 plus 10.0.0.1", "s1")
        assert out == "and [ipaddress-02] plus [ipaddress-01]"

    @pytest.mark.asyncio
    async def test_session_isolation(self, engine):
        await engine.redact_text("10.0.0.1", "s1")
        out = await engine.redact_text("10.0.0.1", "s2")
        assert out == "[ipaddress-01]"  # fresh counter for s2

    @pytest.mark.asyncio
    async def test_ipv6_redacted(self, engine):
        out = await engine.redact_text("ssh to ::1 and 2001:db8::1", "s1")
        assert "::1" not in out and "[ipaddress-01]" in out

    @pytest.mark.asyncio
    async def test_no_ip_untouched(self, engine):
        text = "no addresses here at all"
        assert await engine.redact_text(text, "s1") == text

    @pytest.mark.asyncio
    async def test_port_preserved_after_redaction(self, engine):
        out = await engine.redact_text("curl 10.0.0.1:8080/health", "s1")
        assert out == "curl [ipaddress-01]:8080/health"

    @pytest.mark.asyncio
    async def test_messages_redacted_in_place(self, engine):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "restart server 10.1.2.3"},
            {"role": "user", "content": [
                {"type": "text", "text": "also 10.1.2.4"},
                {"type": "image_url", "image_url": {"url": "http://x"}},
            ]},
        ]
        changed = await engine.redact_messages(messages, "s1")
        assert changed
        assert messages[1]["content"] == "restart server [ipaddress-01]"
        assert messages[2]["content"][0]["text"] == "also [ipaddress-02]"


# --- Re-hydration --------------------------------------------------------------

class TestRehydration:
    @pytest.mark.asyncio
    async def test_roundtrip(self, engine):
        redacted = await engine.redact_text("connect to 172.16.0.9", "s1")
        out = engine.rehydrate_text_sync("You should connect to [ipaddress-01]", "s1")
        assert out == "You should connect to 172.16.0.9"

    @pytest.mark.asyncio
    async def test_multiple_tokens(self, engine):
        await engine.redact_text("10.0.0.1 10.0.0.2 10.0.0.3", "s1")
        out = engine.rehydrate_text_sync("[ipaddress-01] -> [ipaddress-02] -> [ipaddress-03]", "s1")
        assert out == "10.0.0.1 -> 10.0.0.2 -> 10.0.0.3"

    @pytest.mark.asyncio
    async def test_unknown_placeholder_left_alone(self, engine):
        out = engine.rehydrate_text_sync("see [ipaddress-07]", "s1")
        assert out == "see [ipaddress-07]"

    @pytest.mark.asyncio
    async def test_fuzzy_llm_formatting(self, engine):
        await engine.redact_text("10.0.0.1", "s1")
        out = engine.rehydrate_text_sync("try [ ipaddress - 01 ] instead", "s1")
        assert out == "try 10.0.0.1 instead"

    @pytest.mark.asyncio
    async def test_unpadded_number(self, engine):
        await engine.redact_text("10.0.0.1", "s1")
        out = engine.rehydrate_text_sync("token [ipaddress-1] here", "s1")
        assert out == "token 10.0.0.1 here"

    @pytest.mark.asyncio
    async def test_no_placeholder_text_untouched(self, engine):
        text = "ordinary response with no tokens"
        assert engine.rehydrate_text_sync(text, "s1") == text


# --- Store maintenance ----------------------------------------------------------

class TestStore:
    def test_purge_session(self, tmp_path):
        store = IPRedactionStore(str(tmp_path / "ip.db"))
        engine = IPRedactionEngine(store)
        import asyncio
        asyncio.run(engine.redact_text("10.0.0.1 10.0.0.2", "s1"))
        assert store.purge_session("s1") == 2
        assert store.get_mappings("s1") == {}
        store.close()

    def test_purge_older_than(self, tmp_path):
        store = IPRedactionStore(str(tmp_path / "ip.db"))
        engine = IPRedactionEngine(store)
        import asyncio
        asyncio.run(engine.redact_text("ping " + ".".join(["10", "1", "2", "3"]), "s1"))
        # Backdate the row beyond the retention window, then purge.
        with store._lock, store._conn:
            store._conn.execute(
                "UPDATE ip_redaction_sessions "
                "SET created_at = datetime('now', '-2 days')"
            )
        removed = store.purge_older_than(24)
        assert removed == 1
        assert store.get_mappings("s1") == {}
        store.close()

    def test_unique_constraints(self, tmp_path):
        store = IPRedactionStore(str(tmp_path / "ip.db"))
        # duplicate insert ignored, not error
        store.record("s1", "10.0.0.1", "[ipaddress-01]")
        store.record("s1", "10.0.0.1", "[ipaddress-01]")
        assert len(store.get_mappings("s1")) == 1
        store.close()
