"""Admin endpoints: stats, sessions list, settings management."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.middleware.auth import check_admin_auth

router = APIRouter(prefix="/admin")


def _check_admin(request: Request):
    config = request.app.state.config.get()
    auth_header = request.headers.get("Authorization", "")
    if not check_admin_auth(auth_header, config):
        raise HTTPException(status_code=401, detail="Invalid admin key")


@router.get("/sessions")
async def list_sessions(
    request: Request,
    level: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Paginated list of live pins."""
    _check_admin(request)
    store = request.app.state.session_store
    pins = await store.list_sessions(level=level, offset=offset, limit=limit)
    return {
        "sessions": [p.model_dump() for p in pins],
        "total": await store.count(),
    }


@router.delete("/sessions")
async def flush_sessions(request: Request):
    """Flush all pins."""
    _check_admin(request)
    store = request.app.state.session_store
    count = await store.delete_all()
    return {"flushed": count}


@router.get("/settings")
async def get_settings(request: Request):
    """Return active settings (API keys redacted)."""
    _check_admin(request)
    cm = request.app.state.config
    return cm.redacted_dict()


@router.post("/settings/reload")
async def reload_settings(request: Request):
    """Re-read and validate settings.json."""
    _check_admin(request)
    try:
        cm = request.app.state.config
        settings = cm.reload()
        return {"status": "ok", "version": settings.version}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Config validation failed: {e!s}") from e


@router.get("/stats")
async def get_stats(request: Request):
    """Rolling counters and summary stats."""
    _check_admin(request)
    store = request.app.state.session_store

    # Collect from Prometheus registry

    # Get metrics as text then parse key stats
    from prometheus_client import generate_latest
    metrics_text = generate_latest().decode()

    stats = {
        "active_sessions": await store.count(),
        "metrics": metrics_text[:5000],  # truncated raw metrics for now
    }
    return stats
