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
         max_escalations=2, cooldown=3, global_max="L5"):
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
            "deep_keywords": 2,
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
    cfg = _cfg(threshold=5)
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
    cfg = _cfg(threshold=5, cooldown=3)
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
