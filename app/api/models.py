"""GET /v1/models endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request):
    """List virtual router models and optionally upstream tier models."""
    config = request.app.state.config.get()

    models = []
    # Virtual router models
    models.append({
        "id": "smart-router",
        "object": "model",
        "owned_by": "router",
    })
    for level in ["L1", "L2", "L3", "L4"]:
        models.append({
            "id": f"smart-router/{level}",
            "object": "model",
            "owned_by": "router",
        })
    models.append({"id": "smart-router/classify-only", "object": "model", "owned_by": "router"})
    models.append({"id": "smart-router/stateless", "object": "model", "owned_by": "router"})

    # Upstream tier models
    if config.routing.expose_upstream_models:
        for level in ["L1", "L2", "L3", "L4"]:
            tier = config.routing.get_tier(level)
            if tier.model:
                models.append({
                    "id": tier.model,
                    "object": "model",
                    "owned_by": "router",
                    "tier": level,
                })
            for fb in tier.fallbacks:
                models.append({
                    "id": fb,
                    "object": "model",
                    "owned_by": "router",
                    "tier": level,
                    "fallback": True,
                })

    return {"object": "list", "data": models}
