"""Health check endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/healthz")
async def healthz():
    """Liveness probe. Always 200 if process is up."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness probe. Returns 503 when required dependencies are unavailable.

    Checks:
    - app.state.config exists and has provider base_url
    - provider adapter is initialized
    - session store is accessible
    - IP redaction store is accessible (if enabled)
    """
    checks = {}
    all_ok = True

    # Check config
    cm = getattr(request.app.state, "config", None)
    if cm is None:
        checks["config"] = "missing"
        all_ok = False
    else:
        try:
            settings = cm.get()
            if not settings.provider.base_url:
                checks["config"] = "no provider base_url"
                all_ok = False
            else:
                checks["config"] = "ok"
        except Exception as e:
            checks["config"] = f"error: {e}"
            all_ok = False

    # Check provider
    provider = getattr(request.app.state, "provider", None)
    if provider is None:
        checks["provider"] = "missing"
        all_ok = False
    else:
        checks["provider"] = "ok"

    # Check session store
    session_store = getattr(request.app.state, "session_store", None)
    if session_store is None:
        checks["session_store"] = "missing"
        all_ok = False
    else:
        # Try a count operation
        try:
            await session_store.count()
            checks["session_store"] = "ok"
        except Exception as e:
            checks["session_store"] = f"error: {e}"
            all_ok = False

    # Check IP redaction store (if enabled)
    ip_redaction = getattr(request.app.state, "ip_redaction", None)
    if ip_redaction is not None:
        try:
            # Simple check: store connection is alive
            ip_redaction.store.get_mappings("__health_check__")
            checks["ip_redaction"] = "ok"
        except Exception as e:
            checks["ip_redaction"] = f"error: {e}"
            all_ok = False

    status = "ready" if all_ok else "not_ready"
    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": status, "checks": checks},
    )
