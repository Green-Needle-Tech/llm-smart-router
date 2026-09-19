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
