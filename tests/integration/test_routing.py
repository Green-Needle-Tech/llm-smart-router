"""Integration tests for the routing engine."""
import pytest

from app.config.loader import ConfigManager
from app.routing.engine import RoutingEngine
from app.schemas.router import ClassificationResult, ClassificationSource, Level


@pytest.fixture
def config():
    cm = ConfigManager(settings_path="/nonexistent")
    return cm.load()


@pytest.fixture
def engine(config):
    return RoutingEngine(config)


def test_parse_auto(engine):
    d = engine.parse_model_directive("smart-router")
    assert d["mode"] == "auto"


def test_parse_level(engine):
    d = engine.parse_model_directive("smart-router/L3")
    assert d["mode"] == "level"
    assert d["level"] == Level.L3


def test_parse_classify_only(engine):
    d = engine.parse_model_directive("smart-router/classify-only")
    assert d["mode"] == "classify_only"


def test_parse_stateless(engine):
    d = engine.parse_model_directive("smart-router/stateless")
    assert d["mode"] == "stateless"


def test_parse_passthrough(engine):
    d = engine.parse_model_directive("anthropic/claude-sonnet-4.5")
    assert d["mode"] == "passthrough"
    assert d["model"] == "anthropic/claude-sonnet-4.5"


def test_resolve_level_to_model(engine, config):
    cls = ClassificationResult(level=Level.L3, source=ClassificationSource.MODEL)
    route = engine.resolve(Level.L3, cls)
    assert route.model == config.routing.L3.model


def test_resolve_with_max_level(engine):
    cls = ClassificationResult(level=Level.L4, source=ClassificationSource.MODEL)
    route = engine.resolve(Level.L4, cls, max_level=Level.L2)
    assert route.level == Level.L2


def test_resolve_with_min_level(engine):
    cls = ClassificationResult(level=Level.L1, source=ClassificationSource.MODEL)
    route = engine.resolve(Level.L1, cls, min_level=Level.L3)
    assert route.level == Level.L3


def test_resolve_global_max(engine):
    cls = ClassificationResult(level=Level.L4, source=ClassificationSource.MODEL)
    route = engine.resolve(Level.L4, cls)
    # Global max is L4, so L4 is allowed
    assert route.level == Level.L4
