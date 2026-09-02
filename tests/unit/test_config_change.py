"""Tests for session-hit model re-resolution on config change (on_config_change)."""
from __future__ import annotations

from types import SimpleNamespace

from app.api.chat import _resolve_effective_model
from app.schemas.router import Level, SessionPin


def _pin(model="z-ai/glm-5.2", level=Level.L3) -> SessionPin:
    return SessionPin(
        session_id="s1", level=level, model=model, params={},
        turn_count=3,
    )


def _routing(new_model: str):
    return SimpleNamespace(resolve_model_for_level=lambda level: new_model)


def _config(on_config_change: str):
    return SimpleNamespace(session=SimpleNamespace(on_config_change=on_config_change))


class TestKeepLevelDefault:
    def test_model_re_resolved_from_live_config(self):
        pin = _pin(model="z-ai/glm-5.2")
        engine = _routing("z-ai/glm-5.3")  # tier changed via hot-reload
        out = _resolve_effective_model(pin, engine, _config("keep_level"))
        assert out == "z-ai/glm-5.3"

    def test_pin_mutated_to_new_model(self):
        pin = _pin(model="z-ai/glm-5.2")
        engine = _routing("z-ai/glm-5.3")
        _resolve_effective_model(pin, engine, _config("keep_level"))
        assert pin.model == "z-ai/glm-5.3"

    def test_unchanged_model_is_noop(self):
        pin = _pin(model="z-ai/glm-5.2")
        engine = _routing("z-ai/glm-5.2")
        out = _resolve_effective_model(pin, engine, _config("keep_level"))
        assert out == "z-ai/glm-5.2"


class TestKeepPin:
    def test_frozen_model_kept(self):
        pin = _pin(model="z-ai/glm-5.2")
        engine = _routing("z-ai/glm-5.3")
        out = _resolve_effective_model(pin, engine, _config("keep_pin"))
        assert out == "z-ai/glm-5.2"
        assert pin.model == "z-ai/glm-5.2"  # untouched

    def test_empty_resolution_keeps_pin_model(self):
        pin = _pin(model="z-ai/glm-5.2")
        engine = _routing("")  # tier misconfigured to empty model
        out = _resolve_effective_model(pin, engine, _config("keep_level"))
        assert out == "z-ai/glm-5.2"
