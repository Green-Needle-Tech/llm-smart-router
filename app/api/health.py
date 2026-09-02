"""Health check endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/healthz")
async def healthz():
    """Liveness probe. Always 200 if process is up."""
    return {"status": "ok"}


def _check_config(state) -> tuple[str, bool]:
    cm = getattr(state, "config", None)
    if cm is None:
        return "missing", False
    try:
        settings = cm.get()
        if not settings.provider.base_url:
            return "no provider base_url", False
        return "ok", True
    except Exception as e:
        return f"error: {e}", False


def _check_provider(state) -> tuple[str, bool]:
    provider = getattr(state, "provider", None)
    if provider is None:
        return "missing", False
    return "ok", True


async def _check_session_store(state) -> tuple[str, bool]:
    session_store = getattr(state, "session_store", None)
    if session_store is None:
        return "missing", False
    try:
        await session_store.count()
        return "ok", True
    except Exception as e:
        return f"error: {e}", False


def _check_ip_redaction(state) -> tuple[str, bool]:
    ip_redaction = getattr(state, "ip_redaction", None)
    if ip_redaction is None:
        return None, True  # Not enabled, skip
    try:
        ip_redaction.store.get_mappings("__health_check__")
        return "ok", True
    except Exception as e:
        return f"error: {e}", False


@router.get("/readyz", responses={503: {"description": "Not ready"}})
async def readyz(request: Request):
    """Readiness probe. Returns 503 when required dependencies are unavailable."""
    state = request.app.state
    checks = {}
    all_ok = True

    for name, result in [
        ("config", _check_config(state)),
        ("provider", _check_provider(state)),
    ]:
        status, ok = result
        if status is not None:
            checks[name] = status
        if not ok:
            all_ok = False

    status, ok = await _check_session_store(state)
    checks["session_store"] = status
    if not ok:
        all_ok = False

    status, ok = _check_ip_redaction(state)
    if status is not None:
        checks["ip_redaction"] = status
    if not ok:
        all_ok = False

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )
