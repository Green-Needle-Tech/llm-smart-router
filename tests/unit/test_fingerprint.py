"""Tests for fingerprint derivation."""
from app.session.fingerprint import derive_fingerprint


def test_stable_across_appended_turns():
    """Fingerprint should not change when conversation grows."""
    sys1 = "You are a helpful coding assistant."
    user1 = "Fix the typo in line 42"
    tools = ["execute_code", "read_file"]

    fp1 = derive_fingerprint(sys1, user1, tools, "key123", "salt")
    # Same first user message, same system -> same fingerprint
    fp2 = derive_fingerprint(sys1, user1, tools, "key123", "salt")
    assert fp1 == fp2


def test_different_across_distinct_openers():
    """Different first user messages should produce different fingerprints."""
    sys = "You are a coding assistant."
    fp1 = derive_fingerprint(sys, "Write a function", [], "key", "salt")
    fp2 = derive_fingerprint(sys, "Design a system", [], "key", "salt")
    assert fp1 != fp2


def test_strip_patterns():
    """Timestamp patterns should be stripped before hashing."""
    sys1 = "System prompt with 2026-08-16T09:00:00 timestamp"
    sys2 = "System prompt with 2026-08-16T11:30:00 timestamp"
    patterns = [r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"]

    fp1 = derive_fingerprint(sys1, "task", [], "key", "salt", strip_patterns=patterns)
    fp2 = derive_fingerprint(sys2, "task", [], "key", "salt", strip_patterns=patterns)
    assert fp1 == fp2


def test_different_api_keys():
    """Different API keys should produce different fingerprints."""
    fp1 = derive_fingerprint("sys", "task", [], "key1", "salt")
    fp2 = derive_fingerprint("sys", "task", [], "key2", "salt")
    assert fp1 != fp2


def test_different_salts():
    fp1 = derive_fingerprint("sys", "task", [], "key", "salt1")
    fp2 = derive_fingerprint("sys", "task", [], "key", "salt2")
    assert fp1 != fp2


def test_different_tool_orders():
    """Tool order should not matter (sorted internally)."""
    fp1 = derive_fingerprint("sys", "task", ["tool_a", "tool_b"], "key", "salt")
    fp2 = derive_fingerprint("sys", "task", ["tool_b", "tool_a"], "key", "salt")
    assert fp1 == fp2
