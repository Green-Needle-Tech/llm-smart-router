"""Tests for config loading and validation."""
import os
import json
import tempfile
import pytest
from app.config.loader import ConfigManager, _deep_merge


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"d": 4, "e": 5}, "f": 6}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 2, "d": 4, "e": 5}, "f": 6}


def test_load_defaults():
    cm = ConfigManager(settings_path="/nonexistent/path")
    settings = cm.load()
    assert settings.version == 1
    assert settings.server.port == 8080
    assert settings.auth.enabled is True


def test_load_with_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "version": 1,
            "server": {"port": 9090},
            "routing": {
                "L1": {"label": "trivial", "model": "test/model-1"},
            },
        }, f)
        f.flush()

        cm = ConfigManager(settings_path=f.name)
        settings = cm.load()
        assert settings.server.port == 9090
        assert settings.routing.L1.model == "test/model-1"
        # Defaults preserved
        assert settings.server.request_timeout_seconds == 600

    os.unlink(f.name)


def test_rejects_api_keys_in_settings():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "version": 1,
            "provider": {"base_url": "sk-or-abc12345"},
        }, f)
        f.flush()

        cm = ConfigManager(settings_path=f.name)
        with pytest.raises(Exception):
            cm.load()

    os.unlink(f.name)


def test_memory_workers_guard():
    os.environ["WORKERS"] = "2"
    cm = ConfigManager(settings_path="/nonexistent")
    cm.load()
    with pytest.raises(RuntimeError, match="memory.*WORKERS"):
        cm.validate_startup()
    del os.environ["WORKERS"]


def test_context_window_default():
    """ProviderConfig defaults context_window to 1M tokens."""
    cm = ConfigManager(settings_path="/nonexistent")
    settings = cm.load()
    assert settings.provider.context_window == 1_000_000


def test_context_window_custom():
    """context_window can be overridden via settings.json."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "version": 1,
            "provider": {"context_window": 200_000},
        }, f)
        f.flush()

        cm = ConfigManager(settings_path=f.name)
        settings = cm.load()
        assert settings.provider.context_window == 200_000

    os.unlink(f.name)
