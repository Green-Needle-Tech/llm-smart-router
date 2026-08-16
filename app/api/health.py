"""Health check endpoints."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
async def healthz():
    """Liveness probe. Always 200 if process is up."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    """Readiness probe. 200 if config valid and OpenRouter credentials resolve."""
    from app.config.loader import ConfigManager
    cm = ConfigManager()
    try:
        settings = cm.get()
        # Basic checks
        if not settings.provider.base_url:
            return {"status": "not_ready", "reason": "no provider base_url"}
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}
