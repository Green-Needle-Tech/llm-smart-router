"""Unit tests for the settings-driven custom guardrail (question type/id/criteria).

No network: these drive the parser and settings resolution only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.guardrails.custom import (  # noqa: E402
    CustomGuardrailDecision,
    CustomGuardrailEngine,
    CustomGuardrailSettings,
    NOUL_CRITERIA,
    _stringify_systemone_answer,
)


def _engine(**overrides) -> CustomGuardrailEngine:
    return CustomGuardrailEngine(CustomGuardrailSettings(**overrides))


def test_defaults_preserve_previous_behavior():
    s = CustomGuardrailSettings()
    assert s.question_id == "guardrail"
    assert s.question_type == "noul"
    assert s.resolve_criteria() == NOUL_CRITERIA
    assert s.build_question() == {
        "type": "noul",
        "instructions": s.resolve_prompt(),
        "criteria": NOUL_CRITERIA,
    }


def test_fully_custom_noul_question_from_settings():
    s = CustomGuardrailSettings(
        question_id="eat_scope", question_type="noul",
        prompt="Is this about EAT?", criteria={"true": "EAT", "false": "other"},
    )
    q = s.build_question()
    assert q["type"] == "noul" and q["instructions"] == "Is this about EAT?"
    assert q["criteria"]["true"] == "EAT"


def test_choice_question_from_settings():
    s = CustomGuardrailSettings(
        question_type="choice", prompt="Proceed?",
        criteria={"yes": "allowed", "no": "blocked"},
    )
    q = s.build_question()
    assert q["type"] == "choice" and q["criteria"]["yes"] == "allowed"


def test_score_question_from_settings():
    s = CustomGuardrailSettings(question_type="score", prompt="How compliant?",
                                criteria=["low", "medium", "high"])
    assert s.build_question()["criteria"] == ["low", "medium", "high"]


def test_parser_noul():
    d = _engine()._parse_decision(json.dumps({"noul": 0.92}), threshold=0.5)
    assert d.decision == "yes" and abs(d.probability_yes - 0.92) < 1e-9
    d = _engine()._parse_decision(json.dumps({"noul": 0.10}), threshold=0.5)
    assert d.decision == "no"


def test_parser_choice_with_yes_probability():
    raw = json.dumps({"choice": "no", "probabilities": {"yes": 0.12, "no": 0.88}})
    d = _engine()._parse_decision(raw, threshold=0.5)
    assert d.decision == "no" and abs(d.probability_yes - 0.12) < 1e-9


def test_parser_choice_without_probabilities():
    d = _engine()._parse_decision(json.dumps({"choice": "yes"}), threshold=0.5)
    assert d.decision == "yes" and d.probability_yes == 1.0


def test_parser_score_normalizes_by_legend():
    raw = json.dumps({"score": 2.0, "legend": {"0": "l", "1": "m", "2": "h"}})
    d = _engine()._parse_decision(raw, threshold=0.5)
    assert d.decision == "yes" and abs(d.probability_yes - 1.0) < 1e-9
    d = _engine()._parse_decision(json.dumps({"score": 0.0, "legend": {"0": "l", "1": "m", "2": "h"}}), threshold=0.5)
    assert d.decision == "no" and d.probability_yes == 0.0


def test_parser_rejects_garbage():
    for bad in ('{"noul": "high"}', '{"noul": 1.5}', "maybe", "{}"):
        try:
            _engine()._parse_decision(bad, threshold=0.5)
            raised = False
        except ValueError:
            raised = True
        if bad == "{}":  # falls through to strict yes/no validation -> ValueError
            assert raised
        else:
            assert raised, bad


def test_stringify_uses_question_id_and_drops_type():
    resp = {"answers": {"eat_scope": {"type": "noul", "noul": 0.7}}}
    out = json.loads(_stringify_systemone_answer(resp, "eat_scope"))
    assert out == {"noul": 0.7}
    # unknown id -> empty answer -> parser error path (never a silent pass)
    assert _stringify_systemone_answer(resp, "other") == "{}"


def test_disabled_guardrail_short_circuits():
    import asyncio
    d = asyncio.run(_engine(enabled=False).evaluate("anything"))
    assert d.decision == "yes" and d.source == "disabled"


def test_payload_scope_default_and_values():
    s = CustomGuardrailSettings(enabled=True)
    assert s.payload_scope == "all"
    s = CustomGuardrailSettings(enabled=True, payload_scope="last_user")
    assert s.payload_scope == "last_user"
    try:
        CustomGuardrailSettings(enabled=True, payload_scope="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_last_user_scope_selects_final_user_message():
    from app.guardrails.custom import build_payload_text

    messages = [
        {"role": "system", "content": "s" * 20_000},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "What is last week sales?"},
    ]
    user_msgs = [m for m in messages if m.get("role") == "user"]
    scoped = [user_msgs[-1]]
    out = build_payload_text(scoped, 8000)
    assert "What is last week sales?" in out
    assert "old question" not in out
    assert "system" not in out


def test_conversation_scope_keeps_history_drops_system():
    from app.guardrails.custom import build_payload_text

    messages = [
        {"role": "system", "content": "s" * 20_000},
        {"role": "user", "content": "What is last week's inventory details"},
        {"role": "assistant", "content": "Historical inventory details aren't available."},
        {"role": "user", "content": "Yes"},
    ]
    non_system = [m for m in messages if m.get("role") != "system"]
    out = build_payload_text(non_system, 8000)
    assert "Yes" in out
    assert "inventory" in out  # conversation topic visible for the follow-up
    assert "system" not in out  # large agent system prompt excluded


def test_conversation_scope_budget_preserves_last_user():
    from app.guardrails.custom import build_payload_text

    messages = [
        {"role": "system", "content": "s" * 20_000},
        {"role": "user", "content": "u" * 9000},
        {"role": "assistant", "content": "a" * 9000},
        {"role": "user", "content": "Generate html"},
    ]
    non_system = [m for m in messages if m.get("role") != "system"]
    out = build_payload_text(non_system, 8000)
    assert "Generate html" in out
    assert len(out) <= 8000 + len("\n...[truncated]")


# --- short follow-up bypass (v2.30.0) ---------------------------------------


def test_short_followup_bypass_defaults():
    s = CustomGuardrailSettings()
    assert s.skip_short_followups is True
    assert s.short_followup_max_chars == 120


def test_is_short_followup_true_for_confirmation_with_history():
    from app.guardrails.custom import is_short_followup

    messages = [
        {"role": "system", "content": "s" * 500},
        {"role": "user", "content": "What is last week's revenue?"},
        {"role": "assistant", "content": "It was $12,300."},
        {"role": "user", "content": "yes"},
    ]
    assert is_short_followup(messages, 120)


def test_is_short_followup_true_for_short_instruction():
    from app.guardrails.custom import is_short_followup

    messages = [
        {"role": "user", "content": "Summarize the sales report"},
        {"role": "assistant", "content": "Sales were up 5%."},
        {"role": "user", "content": "make it darker"},
    ]
    assert is_short_followup(messages, 120)


def test_is_short_followup_true_for_content_blocks():
    from app.guardrails.custom import is_short_followup

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": [{"type": "text", "text": "go ahead"}]},
    ]
    assert is_short_followup(messages, 120)


def test_is_short_followup_false_for_first_turn():
    from app.guardrails.custom import is_short_followup

    # System-only prefix does not count as history: the first user turn is always evaluated.
    messages = [
        {"role": "system", "content": "s" * 500},
        {"role": "user", "content": "hi"},
    ]
    assert not is_short_followup(messages, 120)


def test_is_short_followup_false_for_long_followup():
    from app.guardrails.custom import is_short_followup

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "x" * 121},
    ]
    assert not is_short_followup(messages, 120)
    assert is_short_followup(messages, 200)


def test_is_short_followup_false_for_empty_last_user():
    from app.guardrails.custom import is_short_followup

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "   "},
    ]
    assert not is_short_followup(messages, 120)


def test_is_short_followup_false_without_user_message():
    from app.guardrails.custom import is_short_followup

    assert not is_short_followup([{"role": "assistant", "content": "x"}], 120)
    assert not is_short_followup([], 120)


def _hook_config(settings):
    from types import SimpleNamespace

    return SimpleNamespace(
        telemetry=SimpleNamespace(guardrails=SimpleNamespace(custom=settings))
    )


def _hook_body(messages):
    from app.schemas.openai import ChatCompletionRequest

    return ChatCompletionRequest.model_validate(
        {"model": "smart-router/L1", "messages": messages}
    )


def test_hook_skips_short_followup(monkeypatch):
    import asyncio
    from app.api import chat as chat_mod

    class ExplodingEngine:
        def __init__(self, settings):
            raise AssertionError("decision model must not be called for short follow-ups")

    monkeypatch.setattr(chat_mod, "CustomGuardrailEngine", ExplodingEngine)
    messages = [
        {"role": "user", "content": "What is last week's revenue?"},
        {"role": "assistant", "content": "It was $12,300."},
        {"role": "user", "content": "yes"},
    ]
    result = asyncio.run(
        chat_mod._custom_guardrail_check_input(
            None, _hook_body(messages), _hook_config(CustomGuardrailSettings(enabled=True))
        )
    )
    assert result is None


def test_hook_evaluates_first_turn_short_message(monkeypatch):
    import asyncio
    from app.api import chat as chat_mod

    calls = []

    class FakeEngine:
        def __init__(self, settings):
            self.settings = settings

        async def evaluate(self, payload_text):
            calls.append(payload_text)
            return CustomGuardrailDecision(decision="yes")

    monkeypatch.setattr(chat_mod, "CustomGuardrailEngine", FakeEngine)
    messages = [
        {"role": "system", "content": "s" * 500},
        {"role": "user", "content": "hi"},
    ]
    result = asyncio.run(
        chat_mod._custom_guardrail_check_input(
            None, _hook_body(messages), _hook_config(CustomGuardrailSettings(enabled=True))
        )
    )
    assert result is None
    assert len(calls) == 1 and "hi" in calls[0]


def test_hook_evaluates_followup_when_bypass_disabled(monkeypatch):
    import asyncio
    from app.api import chat as chat_mod

    calls = []

    class FakeEngine:
        def __init__(self, settings):
            self.settings = settings

        async def evaluate(self, payload_text):
            calls.append(payload_text)
            return CustomGuardrailDecision(decision="yes")

    monkeypatch.setattr(chat_mod, "CustomGuardrailEngine", FakeEngine)
    messages = [
        {"role": "user", "content": "What is last week's revenue?"},
        {"role": "assistant", "content": "It was $12,300."},
        {"role": "user", "content": "yes"},
    ]
    settings = CustomGuardrailSettings(enabled=True, skip_short_followups=False)
    result = asyncio.run(
        chat_mod._custom_guardrail_check_input(
            None, _hook_body(messages), _hook_config(settings)
        )
    )
    assert result is None
    assert len(calls) == 1


def test_hook_evaluates_long_followup(monkeypatch):
    import asyncio
    from app.api import chat as chat_mod

    calls = []

    class FakeEngine:
        def __init__(self, settings):
            self.settings = settings

        async def evaluate(self, payload_text):
            calls.append(payload_text)
            return CustomGuardrailDecision(decision="yes")

    monkeypatch.setattr(chat_mod, "CustomGuardrailEngine", FakeEngine)
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "Also, while you are at it, please analyse the " + "x" * 150},
    ]
    result = asyncio.run(
        chat_mod._custom_guardrail_check_input(
            None, _hook_body(messages), _hook_config(CustomGuardrailSettings(enabled=True))
        )
    )
    assert result is None
    assert len(calls) == 1
