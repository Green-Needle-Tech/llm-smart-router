"""Regression tests for UNKNOWN classifier output -> tier mapping.

Bug: the classifier prompt explicitly instructs the model to emit
``{"level":"UNKNOWN"}`` for greetings, bare acknowledgements, and
too-vague openers ("hi", "thanks", "let's start"). The model complied,
but ``ClassifierService.classify`` then folded UNKNOWN into
``classification.default_level`` (L3), so "Say hi" was routed to the
most expensive medium-tier model instead of the cheapest one.

UNKNOWN now maps to ``classification.unknown_level`` (default L1),
while genuine parse failures / timeouts keep using ``default_level``
because conservatism is correct when we truly don't know.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.classify.classifier import ClassifierService
from app.schemas.openai import ChatMessage
from app.schemas.router import ClassificationSource


def _config(unknown_level="L1", default_level="L3", heuristics_enabled=False):
    """Minimal duck-typed config for ClassifierService."""
    digest = SimpleNamespace(
        system_chars=500,
        tail_chars=2000,
        include_tool_names=True,
        include_context_summary=True,
        strip_scaffolding=True,
        learn_common_prefix=False,
        prefix_samples=20,
        min_prefix_chars=200,
        strip_sections_enabled=False,
        strip_sections=[],
        keep_sections=[],
        delimit_untrusted=True,
        injection_guard=False,
    )
    classification = SimpleNamespace(
        enabled=True,
        model="test/classifier",
        temperature=0,
        max_tokens=60,
        timeout_seconds=8,
        default_level=default_level,
        unknown_level=unknown_level,
        min_confidence=0.5,
        low_confidence_action="escalate",
        prompt_file="/nonexistent/classifier.txt",  # forces inline fallback
        rubric_version="v1",
        digest=digest,
        cache=SimpleNamespace(enabled=False, ttl_seconds=3600, max_entries=100),
    )
    heuristics = SimpleNamespace(
        enabled=heuristics_enabled,
        measure="task_payload",
        huge_context_tokens=32000,
        rules=[],
    )
    return SimpleNamespace(classification=classification, heuristics=heuristics)


def _classify(raw_output, config):
    """Run classify() with the classifier model stubbed to return raw_output."""
    svc = ClassifierService(config, openrouter_api_key="test-key")

    async def _fake_call(_digest):
        return raw_output

    svc._call_classifier_model = _fake_call  # type: ignore[method-assign]

    result, _digest_info = asyncio.run(
        svc.classify(messages=[ChatMessage(role="user", content="Say hi")])
    )
    return result


@pytest.mark.parametrize(
    "raw",
    [
        '{"level":"UNKNOWN","confidence":1.0,"reason":"greeting, no task content"}',
        '{"level":"UNKNOWN","confidence":0.9,"reason":"bare acknowledgement"}',
        '```json\n{"level":"UNKNOWN","confidence":1.0,"reason":"too vague"}\n```',
    ],
)
def test_unknown_maps_to_unknown_level_not_default(raw):
    """UNKNOWN must route to L1, not the L3 conservative default."""
    result = _classify(raw, _config(unknown_level="L1", default_level="L3"))
    assert result.level.value == "L1"
    assert result.source == ClassificationSource.MODEL


def test_unknown_level_is_configurable():
    """Operators can point UNKNOWN at a different tier."""
    result = _classify(
        '{"level":"UNKNOWN","confidence":1.0,"reason":"greeting"}',
        _config(unknown_level="L2", default_level="L4"),
    )
    assert result.level.value == "L2"


def test_unknown_skips_low_confidence_escalation():
    """A low-confidence UNKNOWN must not be escalated a tier.

    UNKNOWN is a definite judgement ("there is no task here"), so the
    low_confidence_action=escalate policy must not bump L1 -> L2.
    """
    result = _classify(
        '{"level":"UNKNOWN","confidence":0.05,"reason":"greeting"}',
        _config(unknown_level="L1", default_level="L3"),
    )
    assert result.level.value == "L1"


def test_parse_failure_still_uses_default_level():
    """Unparseable output is a real unknown-unknown: stay conservative."""
    result = _classify("total garbage, no json, no level token", _config())
    assert result.level.value == "L3"
    assert result.source == ClassificationSource.DEFAULT


def test_normal_levels_are_unaffected():
    """The UNKNOWN branch must not disturb ordinary classification."""
    for level in ("L1", "L2", "L3", "L4"):
        result = _classify(
            f'{{"level":"{level}","confidence":0.95,"reason":"test"}}',
            _config(),
        )
        assert result.level.value == level
        assert result.source == ClassificationSource.MODEL
