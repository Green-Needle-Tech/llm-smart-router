"""Tests for heuristic rules."""
from app.classify.heuristics import evaluate_heuristics


def test_tiny_prompt():
    result = evaluate_heuristics("yes", has_code=False, prompt_tokens=5)
    assert result is not None
    level, stop, name = result
    assert stop is True
    assert level.value == "L1"


def test_json_reshape():
    result = evaluate_heuristics(
        "Return as JSON: {name, age}",
        json_mode=True,
        prompt_tokens=50,
    )
    assert result is not None
    level, stop, name = result
    assert stop is True
    assert level.value == "L1"


def test_huge_context():
    result = evaluate_heuristics(
        "task" * 100,
        prompt_tokens=40000,
        huge_context_tokens=32000,
    )
    assert result is not None
    level, stop, name = result
    assert stop is True
    assert level.value == "L4"


def test_code_heavy_floor():
    result = evaluate_heuristics(
        "task with code",
        has_code=True,
        code_fences=4,
        prompt_tokens=500,
    )
    assert result is not None
    level, stop, name = result
    # floor, not stop
    assert stop is False
    assert level.value == "L3"


def test_deep_keywords():
    result = evaluate_heuristics(
        "Architect a distributed system with proper design",
        prompt_tokens=200,
    )
    assert result is not None
    # Deep keywords should set L4 floor
    level, stop, name = result
    assert level.value == "L4"


def test_no_match():
    result = evaluate_heuristics(
        "Write a summary of this article about AI",
        has_code=False,
        code_fences=0,
        prompt_tokens=300,
    )
    # No rules match
    assert result is None


# ── Regression: tiny_prompt must measure the task payload, not the wrapper ──
#
# The digest is wrapped with <<<UNTRUSTED_INPUT_BEGIN>>>/<<<..._END>>> plus a
# "[conversation: N messages, ~T task tokens]" line — roughly 95 chars of
# boilerplate. Before the fix the rule read `len(digest) < 40`, which could
# never be true for a wrapped digest, so every trivial prompt fell through to
# the LLM classifier and (via UNKNOWN -> default_level) landed on L3.

WRAPPED = (
    "<<<UNTRUSTED_INPUT_BEGIN>>>\n"
    "[conversation: 1 messages, ~1 task tokens]\n"
    "{body}\n"
    "<<<UNTRUSTED_INPUT_END>>>"
)


def test_tiny_prompt_fires_on_wrapped_digest():
    """A 6-char prompt inside a ~100-char wrapper must still hit tiny_prompt."""
    digest = WRAPPED.format(body="Say hi")
    assert len(digest) > 40, "fixture must exceed the old len(digest) threshold"

    result = evaluate_heuristics(
        digest,
        has_code=False,
        code_fences=0,
        prompt_tokens=1,
        task_chars=len("Say hi"),
    )
    assert result is not None, "tiny_prompt should fire on the task payload"
    level, stop, name = result
    assert name == "tiny_prompt"
    assert stop is True
    assert level.value == "L1"


def test_tiny_prompt_does_not_fire_on_long_task_in_wrapper():
    """A long task must not be misread as tiny just because task_chars is set."""
    body = "Design a distributed consensus algorithm handling Byzantine faults."
    digest = WRAPPED.format(body=body)

    result = evaluate_heuristics(
        digest,
        has_code=False,
        code_fences=0,
        prompt_tokens=20,
        task_chars=len(body),
    )
    # tiny_prompt must not stop here; deep-keyword floor may still apply.
    if result is not None:
        _level, stop, name = result
        assert name != "tiny_prompt"
        assert stop is False


def test_task_chars_defaults_to_len_digest():
    """Omitting task_chars preserves the original len(digest) behaviour."""
    result = evaluate_heuristics("yes", has_code=False, prompt_tokens=5)
    assert result is not None
    level, stop, name = result
    assert name == "tiny_prompt"
    assert stop is True
    assert level.value == "L1"


def test_tiny_prompt_skipped_when_code_present():
    """A short prompt that carries code is not trivial."""
    digest = WRAPPED.format(body="```py\nx=1\n```")
    result = evaluate_heuristics(
        digest,
        has_code=True,
        code_fences=2,
        prompt_tokens=8,
        task_chars=13,
    )
    if result is not None:
        _level, _stop, name = result
        assert name != "tiny_prompt"
