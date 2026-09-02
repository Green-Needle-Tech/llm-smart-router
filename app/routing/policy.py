"""Central route policy enforcement: tier clamping, model allowlist, overrides."""
from __future__ import annotations

from dataclasses import dataclass

from app.schemas.router import Level


@dataclass
class EffectiveRoute:
    """Result of policy enforcement for a routing decision."""

    effective_level: Level
    effective_model: str
    requested_level: Level | None
    requested_model: str | None
    overridden: bool
    clamped: bool


class PolicyViolation(Exception):
    """Raised when a route policy check fails."""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _get_allowed_models(settings) -> set[str]:
    """Collect all models allowed for routing from config (tier + fallbacks)."""
    allowed: set[str] = set()
    for level in ["L1", "L2", "L3", "L4", "L5"]:
        tier = settings.routing.get_tier(level)
        if tier.model:
            allowed.add(tier.model)
        for fb in tier.fallbacks:
            allowed.add(fb)
    # Add explicit passthrough allowlist
    for m in getattr(settings.routing, "passthrough_model_allowlist", []):
        allowed.add(m)
    return allowed


def enforce_route_policy(
    requested_level: Level | None,
    requested_model: str | None,
    *,
    max_level: Level | None = None,
    min_level: Level | None = None,
    settings=None,
    allow_overrides: bool = False,
) -> EffectiveRoute:
    """Enforce global tier limits and model allowlist on a routing decision.

    This is the single chokepoint through which all routes must pass.

    - Clamps level to global_max_level and global_min_level.
    - Rejects conflicting min_level > max_level.
    - Restricts forced models to the configured allowlist unless allow_overrides.
    - Returns the effective level and model after enforcement.
    """
    global_max = Level.from_str(settings.routing.global_max_level)
    global_min = Level.from_str(settings.routing.global_min_level)

    # Check for conflicting min/max
    if max_level is not None and min_level is not None and min_level > max_level:
        raise PolicyViolation(
            f"min_level ({min_level.value}) cannot exceed max_level ({max_level.value})",
            status_code=422,
        )

    # Start with requested level or default
    effective_level = requested_level or Level.from_str(settings.classification.default_level)

    # Apply per-request max/min
    if max_level is not None:
        effective_level = min(effective_level, max_level)
    if min_level is not None:
        effective_level = max(effective_level, min_level)

    # Apply global clamps
    clamped = False
    if effective_level > global_max:
        effective_level = global_max
        clamped = True
    if effective_level < global_min:
        effective_level = global_min
        clamped = True

    # Resolve model
    overridden = False
    if requested_model and not allow_overrides:
        # Overrides disabled — ignore requested model
        effective_model = settings.routing.get_model(effective_level.value)
    elif requested_model and allow_overrides:
        allowed = _get_allowed_models(settings)
        if requested_model not in allowed:
            raise PolicyViolation(
                f"Model '{requested_model}' is not in the allowed model list",
                status_code=400,
            )
        effective_model = requested_model
        overridden = True
    else:
        effective_model = settings.routing.get_model(effective_level.value)

    return EffectiveRoute(
        effective_level=effective_level,
        effective_model=effective_model,
        requested_level=requested_level,
        requested_model=requested_model,
        overridden=overridden,
        clamped=clamped,
    )
