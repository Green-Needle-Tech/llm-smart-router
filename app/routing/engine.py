"""Routing engine: maps level → model, applies overrides, floors/ceilings."""
from __future__ import annotations

from app.schemas.router import ClassificationResult, Level, RouteDecision


class RoutingEngine:
    """Maps classification level to a model and parameters."""

    def __init__(self, config_manager):
        self._config_manager = config_manager

    @property
    def config(self):
        # Support both ConfigManager (has .get()) and raw Settings objects
        # (used by unit tests that pass SimpleNamespace/Settings fakes)
        if hasattr(self._config_manager, 'get'):
            return self._config_manager.get()
        return self._config_manager

    def resolve(
        self,
        level: Level,
        classification: ClassificationResult,
        *,
        max_level: Level | None = None,
        min_level: Level | None = None,
        forced_model: str | None = None,
    ) -> RouteDecision:
        """Resolve a level to a RouteDecision with model, params, and cost estimate."""
        # Apply ceilings and floors
        effective_level = level

        global_max = Level.from_str(self.config.routing.global_max_level)
        global_min = Level.from_str(self.config.routing.global_min_level)

        if max_level is not None:
            effective_level = min(effective_level, max_level, global_max) if max_level < global_max else min(effective_level, global_max)
        else:
            effective_level = min(effective_level, global_max)

        if min_level is not None:
            effective_level = max(effective_level, min_level, global_min) if min_level > global_min else max(effective_level, global_min)
        else:
            effective_level = max(effective_level, global_min)

        # Get tier config
        tier = self.config.routing.get_tier(effective_level.value)
        model = forced_model or tier.model
        params = dict(tier.params)

        return RouteDecision(
            level=effective_level,
            model=model,
            params=params,
            classification=classification,
            estimated_cost_usd=None,
        )

    def resolve_model_for_level(self, level: Level) -> str:
        """Get the model slug for a level from current config."""
        return self.config.routing.get_model(level.value)

    def get_fallbacks(self, level: Level) -> list[str]:
        """Get the fallback model list for a level."""
        return self.config.routing.get_fallbacks(level.value)

    def is_passthrough_allowed(self, model: str) -> bool:
        """Check if a model slug is allowed for passthrough."""
        if not self.config.routing.allow_passthrough:
            return False
        # Check if it's a configured tier model or in a tier's fallbacks
        for level in ["L1", "L2", "L3", "L4", "L5"]:
            tier = self.config.routing.get_tier(level)
            if tier.model == model:
                return True
            if model in tier.fallbacks:
                return True
        # Check explicit allowlist
        allowlist = getattr(self.config.routing, "passthrough_model_allowlist", [])
        if model in allowlist:
            return True
        # Default deny — do not allow arbitrary model slugs
        return False

    def parse_model_directive(self, model: str) -> dict:
        """Parse the model field as a routing directive.

        Returns dict with:
          - mode: "auto" | "level" | "classify_only" | "stateless" | "passthrough"
          - level: Optional[Level]
          - model: Optional[str] (for passthrough)
        """
        model = model.strip()

        if model in ("smart-router", "auto"):
            return {"mode": "auto", "level": None, "model": None}

        for level in Level:
            if model == f"smart-router/{level.value}":
                return {"mode": "level", "level": level, "model": None}

        if model == "smart-router/classify-only":
            return {"mode": "classify_only", "level": None, "model": None}

        if model == "smart-router/stateless":
            return {"mode": "stateless", "level": None, "model": None}

        # Passthrough: any string with "/" that looks like a model slug
        if "/" in model and not model.startswith("smart-router/"):
            return {"mode": "passthrough", "level": None, "model": model}

        # Default to auto
        return {"mode": "auto", "level": None, "model": None}
