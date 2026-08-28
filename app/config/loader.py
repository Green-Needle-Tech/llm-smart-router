"""Config loader: load, validate, deep-merge with defaults, hot-reload."""
from __future__ import annotations

import contextlib
import copy
import json
import os
import threading
import time
from typing import Any

from .defaults import DEFAULTS
from .schema import Settings


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _apply_env_overrides(cfg: dict) -> dict:
    """Apply environment variable overrides on top of config."""
    if os.environ.get("LOG_LEVEL"):
        cfg.setdefault("telemetry", {})["log_level"] = os.environ["LOG_LEVEL"]
    if os.environ.get("SESSION_ENABLED"):
        cfg.setdefault("session", {})["enabled"] = os.environ["SESSION_ENABLED"].lower() == "true"
    if os.environ.get("CLASSIFICATION_ENABLED"):
        cfg.setdefault("classification", {})["enabled"] = os.environ["CLASSIFICATION_ENABLED"].lower() == "true"
    if os.environ.get("SESSION_IDLE_TTL_SECONDS"):
        cfg.setdefault("session", {})["idle_ttl_seconds"] = int(os.environ["SESSION_IDLE_TTL_SECONDS"])
    if os.environ.get("SESSION_FINGERPRINT_SALT"):
        cfg.setdefault("session", {})["fingerprint_salt"] = os.environ["SESSION_FINGERPRINT_SALT"]
    if os.environ.get("CACHE_BACKEND"):
        cfg.setdefault("session", {})["backend"] = os.environ["CACHE_BACKEND"]
    if os.environ.get("PORT"):
        cfg.setdefault("server", {})["port"] = int(os.environ["PORT"])
    return cfg


class ConfigManager:
    """Manages loading, validating, and hot-reloading settings."""

    def __init__(self, settings_path: str | None = None):
        self.settings_path = settings_path or os.environ.get(
            "SETTINGS_PATH", "/app/config/settings.json"
        )
        self._settings: Settings | None = None
        self._lock = threading.Lock()
        self._last_mtime: float = 0
        self._reload_callbacks: list = []

    def load(self) -> Settings:
        """Load settings from file, merge with defaults, validate."""
        with self._lock:
            cfg = copy.deepcopy(DEFAULTS)

            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    file_cfg = json.load(f)
                cfg = _deep_merge(cfg, file_cfg)
                with contextlib.suppress(OSError):
                    self._last_mtime = os.path.getmtime(self.settings_path)

            cfg = _apply_env_overrides(cfg)
            self._settings = Settings.model_validate(cfg)
            return self._settings

    def get(self) -> Settings:
        """Get current settings, loading if necessary."""
        if self._settings is None:
            return self.load()
        return self._settings

    def reload(self) -> Settings:
        """Force reload from file. Returns new settings or raises."""
        return self.load()

    def check_and_reload(self) -> bool:
        """Check if file changed, reload if so. Returns True if reloaded."""
        try:
            mtime = os.path.getmtime(self.settings_path)
            if mtime != self._last_mtime:
                self.load()
                self._notify_callbacks()
                return True
        except OSError:
            pass
        return False

    def on_reload(self, callback):
        """Register a callback called after successful reload."""
        self._reload_callbacks.append(callback)

    def _notify_callbacks(self):
        for cb in self._reload_callbacks:
            with contextlib.suppress(Exception):
                cb(self._settings)

    def start_watcher(self, interval: float = 5.0):
        """Start a background thread polling for config changes."""
        def _watch():
            while True:
                time.sleep(interval)
                with contextlib.suppress(Exception):
                    self.check_and_reload()

        t = threading.Thread(target=_watch, daemon=True, name="config-watcher")
        t.start()

    def validate_startup(self) -> None:
        """Startup-only validation guards."""
        s = self.get()
        workers = int(os.environ.get("WORKERS", "1"))
        if s.session.backend == "memory" and workers > 1:
            raise RuntimeError(
                "FATAL: session.backend='memory' requires WORKERS=1. "
                "Set CACHE_BACKEND=redis to use multiple workers."
            )

    def redacted_dict(self) -> dict[str, Any]:
        """Return settings as dict with any API keys redacted."""
        d = self.get().model_dump()
        return d
