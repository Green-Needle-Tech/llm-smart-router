"""Session management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.schemas.router import Level, SessionPin, SessionStatus

router = APIRouter(prefix="/v1/router/sessions")


class SetPinRequest(BaseModel):
    level: str
    reason: str = ""


class SignalRequest(BaseModel):
    signal: str
    weight: int = 1
    detail: str = ""


@router.get("/{session_id}")
async def get_session(session_id: str, request: Request):
    """Inspect a session pin."""
    store = request.app.state.session_store
    pin = await store.get(session_id)
    if pin is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return pin.model_dump()


@router.put("/{session_id}")
async def set_session(session_id: str, body: SetPinRequest, request: Request):
    """Manually set or move a pin."""
    config = request.app.state.config.get()
    store = request.app.state.session_store

    try:
        level = Level.from_str(body.level)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid level: {body.level}") from None

    model = config.routing.get_model(level.value)
    params = config.routing.get_params(level.value)

    existing = await store.get(session_id)
    if existing:
        existing.level = level
        existing.model = model
        existing.params = params
        existing.status = SessionStatus.PINNED
        existing.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)
        await store.put(existing)
        return existing.model_dump()

    pin = SessionPin(
        session_id=session_id,
        level=level,
        model=model,
        params=params,
        status=SessionStatus.PINNED,
        turn_count=0,
    )
    pin.touch(config.session.idle_ttl_seconds, config.session.max_ttl_seconds)
    await store.put(pin)
    return pin.model_dump()


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Drop a pin."""
    store = request.app.state.session_store
    deleted = await store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"deleted": True, "session_id": session_id}


@router.post("/{session_id}/signal")
async def signal_session(session_id: str, body: SignalRequest, request: Request):
    """Report difficulty evidence for escalation scoring."""
    config = request.app.state.config.get()
    store = request.app.state.session_store
    esc_cfg = config.session.escalation

    if not esc_cfg.enabled:
        return {"score": 0, "escalated": False, "reason": "escalation disabled"}

    pin = await store.get(session_id)
    if pin is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    weights = esc_cfg.signal_weights
    weight = body.weight if body.weight else weights.get(body.signal, 1)

    # Add to score
    pin.escalation.score += weight
    pin.escalation.last_trigger = [body.signal]

    escalated = False
    from_level = pin.level

    # Check threshold
    if (
        pin.escalation.score >= esc_cfg.threshold
        and pin.escalation.count < esc_cfg.max_escalations_per_session
        and pin.turn_count >= pin.escalation.cooldown_until_turn
    ):
        new_level = Level.from_numeric(pin.level.numeric + 1)
        if new_level <= Level.from_str(config.routing.global_max_level):
            pin.level = new_level
            pin.model = config.routing.get_model(new_level.value)
            pin.params = config.routing.get_params(new_level.value)
            pin.escalation.count += 1
            pin.escalation.last_escalated_turn = pin.turn_count
            pin.escalation.cooldown_until_turn = pin.turn_count + esc_cfg.cooldown_turns
            pin.escalation.original_level = pin.escalation.original_level or from_level
            escalated = True

    await store.put(pin)
    return {
        "score": pin.escalation.score,
        "escalated": escalated,
        "level": pin.level.value,
        "model": pin.model,
    }
