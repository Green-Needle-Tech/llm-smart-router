"""Debug endpoint for classification."""
from __future__ import annotations

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse

from app.schemas.openai import ChatCompletionRequest

router = APIRouter(prefix="/v1/router")


@router.post("/classify")
async def classify(request: Request, body: ChatCompletionRequest, debug: str = Query(default="")):
    """Debug: classify a prompt without forwarding or pinning."""
    classifier = request.app.state.classifier
    config = request.app.state.config.get()

    # Get router overrides
    task_text = None
    ignore_system = False
    if body.router and isinstance(body.router, dict):
        task_text = body.router.get("task_text")
        ignore_system = body.router.get("ignore_system", False) or body.router.get("scaffolding_system_blocks") is not None

    result, digest_info = await classifier.classify(
        messages=body.messages,
        tools=body.tools,
        response_format=body.response_format,
        task_text=task_text,
        ignore_system=ignore_system,
        bypass_cache=body.router.get("bypass_cache", False) if body.router else False,
    )

    response = {
        "level": result.level.value if result.level else "UNKNOWN",
        "confidence": result.confidence,
        "reason": result.reason,
        "source": result.source.value,
        "latency_ms": result.latency_ms,
    }

    if debug == "digest":
        response["digest"] = digest_info.get("digest", "")
        response["scaffolding_stripped_chars"] = digest_info.get("scaffolding_stripped_chars", 0)
        response["stripped_by"] = digest_info.get("stripped_by", [])
        response["task_tokens"] = digest_info.get("task_tokens", 0)
        response["total_tokens"] = digest_info.get("total_tokens", 0)

    return response
