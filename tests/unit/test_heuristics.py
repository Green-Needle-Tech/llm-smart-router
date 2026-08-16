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
