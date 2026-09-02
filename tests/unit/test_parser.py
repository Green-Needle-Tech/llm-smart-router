"""Tests for the classifier output parser."""
from app.classify.parser import parse_classifier_output
from app.schemas.router import Level


def test_valid_json_l1():
    raw = '{"level":"L1","confidence":0.95,"reason":"formatting task"}'
    result = parse_classifier_output(raw)
    assert result.level == Level.L1
    assert result.confidence == 0.95
    assert result.reason == "formatting task"


def test_valid_json_l4():
    raw = '{"level":"L4","confidence":0.88,"reason":"system design"}'
    result = parse_classifier_output(raw)
    assert result.level == Level.L4


def test_fenced_json():
    raw = '```json\n{"level":"L2","confidence":0.8,"reason":"summary"}\n```'
    result = parse_classifier_output(raw)
    assert result.level == Level.L2
    assert result.confidence == 0.8


def test_bare_level():
    raw = "L3"
    result = parse_classifier_output(raw)
    assert result.level == Level.L3
    assert result.confidence == 0.5
    assert result.reason == "regex fallback"


def test_unknown_label():
    raw = '{"level":"UNKNOWN","confidence":0.1,"reason":"greeting"}'
    result = parse_classifier_output(raw)
    assert result.level is None
    assert result.reason == "greeting"


def test_prose_fallback():
    raw = "I think this is a level L2 task because it involves summarization."
    result = parse_classifier_output(raw)
    assert result.level == Level.L2


def test_empty_string():
    result = parse_classifier_output("")
    assert result.level is None
    assert result.reason == "parse failure"


def test_garbage():
    result = parse_classifier_output("xyzabc 12345")
    assert result.level is None
    assert result.reason == "parse failure"


def test_unknown_in_text():
    result = parse_classifier_output("UNKNOWN")
    assert result.level is None


def test_all_levels():
    for level in Level:
        raw = f'{{"level":"{level.value}","confidence":0.9,"reason":"test"}}'
        result = parse_classifier_output(raw)
        assert result.level == level
