"""Regression tests for escalation tuning fixes.

Covers two bugs found on 2026-08-17:
  1. A zero-weight signal still landed in signals_fired, which suppressed the
     decay branch forever once turn_depth started firing.
  2. escalation.score was never reset after an escalation, so a session chained
     straight into the next tier as soon as cooldown expired.
"""
from types import SimpleNamespace

import pytest

from app.api.chat import _check_escalation_signals
from app.schemas.router import SessionPin, Level


def _cfg(turn_depth_weight=0, threshold=5, escalate_after_turns=25,
         max_escalations=2, cooldown=5, global_max="L5",
         deep_keywords_weight=0):
    esc = SimpleNamespace(
        enabled=True,
        threshold=threshold,
        decay_per_turn=1,
        cooldown_turns=cooldown,
        max_escalations_per_session=max_escalations,
        never_downgrade=True,
        respect_max_level=True,
        explicit_signals_enabled=True,
        free_signals_enabled=True,
        signal_weights={
            "repair_language": 3,
            "tool_error_loop": 3,
            "deep_keywords": deep_keywords_weight,
            "context_growth": 2,
            "truncation": 2,
            "degenerate_response": 2,
            "turn_depth": turn_depth_weight,
            "code_volume_growth": 1,
        },
        escalate_after_turns=escalate_after_turns,
    )

    class Routing:
        global_max_level = global_max

        def get_model(self, level):
            return {"L1": "deepseek/deepseek-v4-flash",
                    "L2": "google/gemini-2.5-flash",
                    "L3": "z-ai/glm-5.2",
                    "L4": "anthropic/claude-sonnet-5",
                    "L5": "anthropic/claude-opus-5"}[level]

        def get_params(self, level):
            return {"temperature": 0.6}

    return SimpleNamespace(
        session=SimpleNamespace(escalation=esc),
        routing=Routing(),
    )


def _body(text):
    return SimpleNamespace(
        messages=[SimpleNamespace(role="user", content=text)]
    )


def _pin(level="L3", turn_count=1, score=0):
    pin = SessionPin(
        session_id="test-session",
        level=Level.from_str(level),
        model="z-ai/glm-5.2",
        params={"temperature": 0.6},
    )
    pin.turn_count = turn_count
    pin.escalation.score = score
    return pin


def test_turn_depth_zero_weight_does_not_block_decay():
    """Bug 1: a deep but benign session must still decay toward zero."""
    cfg = _cfg(turn_depth_weight=0, escalate_after_turns=25)
    pin = _pin(turn_count=40, score=3)

    for _ in range(3):
        result = _check_escalation_signals(_body("thanks, looks good"), pin, cfg)
        assert result is None

    assert pin.escalation.score == 0, (
        "score must decay when the only firing signal has zero weight"
    )


def test_long_benign_session_never_escalates():
    """A 60-turn chatty session with no difficulty signals stays put."""
    cfg = _cfg(turn_depth_weight=0)
    pin = _pin(level="L3", turn_count=1)

    for turn in range(1, 61):
        pin.turn_count = turn
        _check_escalation_signals(_body("ok, and what about the next one?"), pin, cfg)

    assert pin.level == Level.from_str("L3")
    assert pin.escalation.count == 0


def test_score_resets_after_escalation():
    """Bug 2: escalating must consume the accumulated evidence."""
    cfg = _cfg(threshold=5, deep_keywords_weight=2)
    # repair_language (3) + deep_keywords (2) = 5 -> escalate
    pin = _pin(level="L3", turn_count=10)

    result = _check_escalation_signals(
        _body("no, that's wrong. please architect a better approach"), pin, cfg
    )

    assert result is not None, "expected escalation at threshold"
    assert pin.level == Level.from_str("L4")
    assert pin.escalation.count == 1
    assert pin.escalation.score == 0, "score must reset after escalation"


def test_no_immediate_chain_to_next_tier():
    """After escalating, one more genuine signal must not instantly re-escalate."""
    cfg = _cfg(threshold=5, cooldown=5, deep_keywords_weight=2)
    pin = _pin(level="L3", turn_count=10)

    # First escalation L3 -> L4
    _check_escalation_signals(
        _body("no, that's wrong. please architect a better approach"), pin, cfg
    )
    assert pin.level == Level.from_str("L4")

    # A single weaker signal past cooldown must not reach threshold alone
    pin.turn_count = 20
    result = _check_escalation_signals(_body("please derive the bound"), pin, cfg)

    assert result is None, "must not chain into L5 on partial evidence"
    assert pin.level == Level.from_str("L4")
    assert pin.escalation.count == 1


def test_genuine_difficulty_still_escalates():
    """Guard against over-tuning: real repair loops must still escalate."""
    cfg = _cfg(threshold=5)
    pin = _pin(level="L3", turn_count=5)

    # repair_language fires (3), below threshold 5
    _check_escalation_signals(_body("no, that's wrong"), pin, cfg)
    assert pin.level == Level.from_str("L3")

    # second repair turn: 3 + 3 = 6 >= 5 -> escalate
    _check_escalation_signals(_body("still broken, doesn't work"), pin, cfg)
    assert pin.level == Level.from_str("L4"), (
        "sustained repair language must still escalate"
    )


# ---------------------------------------------------------------------------
# Fix 3: deep_keywords must only scan the raw user text, not the full message
# payload (which includes injected system/skill/memory context).
# ---------------------------------------------------------------------------

def test_deep_keywords_ignored_in_injected_context():
    """Bug 3: 'architect' appearing in injected memory-context within the user
    message must NOT trigger deep_keywords escalation — only the user's actual
    typed text should be scanned."""
    cfg = _cfg(deep_keywords_weight=2)

    # Simulate what Hermes sends: user types a simple instruction, but the
    # message payload includes <memory-context> with deep keywords
    body = _body(
        "except for images, .js and .css must be self contain in html\n\n"
        "<memory-context>\n[System note: recalled memory.]\n"
        "- David is an AWS architect. Prove your answers.\n"
        "- Derive measurable criteria from requirements.\n"
        "</memory-context>"
    )
    pin = _pin(level="L3", turn_count=10)

    for _ in range(5):
        _check_escalation_signals(body, pin, cfg)

    assert pin.level == Level.from_str("L3"), (
        "deep_keywords in injected memory context must not escalate a simple user task"
    )
    assert pin.escalation.score == 0, "score must stay at 0 with no genuine signals"


def test_deep_keywords_still_fire_on_raw_user_text():
    """Guard: when the user actually types deep keywords, escalation must fire."""
    cfg = _cfg(threshold=5, deep_keywords_weight=2)
    pin = _pin(level="L3", turn_count=10)

    # Turn 1: deep_keywords fires (weight 2), score = 2
    _check_escalation_signals(
        _body("please architect a system"), pin, cfg
    )
    assert pin.level == Level.from_str("L3")
    assert pin.escalation.score == 2

    # Turn 2: score = 4, still below threshold 5
    _check_escalation_signals(
        _body("now derive the proof"), pin, cfg
    )
    assert pin.level == Level.from_str("L3")
    assert pin.escalation.score == 4

    # Turn 3: score = 6 >= 5 — escalate
    result = _check_escalation_signals(
        _body("refactor the algorithm"), pin, cfg
    )
    assert result is not None
    assert pin.level == Level.from_str("L4")


# ---------------------------------------------------------------------------
# Fix 4: L5 must require stronger evidence than deep_keywords alone.
# ---------------------------------------------------------------------------

def test_l5_not_reachable_by_keywords_alone():
    """L5 must not be reachable via deep_keywords alone. Even sustained
    keyword hits should cap at L4 without repair_language or tool_error_loop."""
    cfg = _cfg(threshold=5, cooldown=5, deep_keywords_weight=2)
    pin = _pin(level="L4", turn_count=10)

    # Hit deep_keywords repeatedly past cooldown
    for turn in range(10, 40):
        pin.turn_count = turn
        _check_escalation_signals(
            _body("architect a system, derive the proof, optimize the algorithm"), pin, cfg
        )

    assert pin.level == Level.from_str("L4"), (
        "keywords alone must not escalate from L4 to L5"
    )
    assert pin.escalation.count == 0, "no escalation should have fired"


def test_l5_reachable_with_repair_language():
    """Guard: genuine repair language at L4 must still escalate to L5."""
    cfg = _cfg(threshold=5, cooldown=5, deep_keywords_weight=2)
    pin = _pin(level="L4", turn_count=10)

    # First repair_language (3) + deep_keywords (2) = 5 -> escalate L4->L5
    result = _check_escalation_signals(
        _body("no, that's wrong. architect a better approach"), pin, cfg
    )
    assert result is not None
    assert pin.level == Level.from_str("L5")


# ---------------------------------------------------------------------------
# Fix 5: escalation diagnostics — last_trigger and turn must be recorded.
# ---------------------------------------------------------------------------

def test_escalation_records_last_trigger():
    """When escalation fires, pin.escalation.last_trigger must list the signals."""
    cfg = _cfg(threshold=5, deep_keywords_weight=2)
    pin = _pin(level="L3", turn_count=10)

    _check_escalation_signals(
        _body("no, that's wrong. please architect a better approach"), pin, cfg
    )

    assert pin.level == Level.from_str("L4")
    assert pin.escalation.count == 1
    assert len(pin.escalation.last_trigger) > 0, (
        "last_trigger must be populated with the signal names that fired"
    )
    assert "repair_language" in pin.escalation.last_trigger
    assert "deep_keywords" in pin.escalation.last_trigger


def test_escalation_turn_histogram_recorded():
    """The router_escalation_turn histogram must observe the turn number."""
    # This test verifies the metric is called; we check via the collector.
    from prometheus_client import REGISTRY

    cfg = _cfg(threshold=5, deep_keywords_weight=2)
    pin = _pin(level="L3", turn_count=42)

    _check_escalation_signals(
        _body("no, that's wrong. please architect a better approach"), pin, cfg
    )

    # The histogram should have observed turn 42
    metric = REGISTRY._names_to_collectors.get("router_escalation_turn")
    assert metric is not None, "router_escalation_turn metric must exist"
    # Access the histogram's sum — it should include turn 42
    samples = list(metric.collect()[0].samples)
    count_sample = [s for s in samples if s.name == "router_escalation_turn_count"]
    assert count_sample, "histogram must have a count sample"
    assert count_sample[0].value > 0, "histogram must have observed at least one escalation"



