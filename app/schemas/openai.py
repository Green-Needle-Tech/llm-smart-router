"""OpenAI-compatible API schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "smart-router"
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stop: Any | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    n: int | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None

    # Non-standard router extension (stripped before forwarding)
    router: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: dict[str, Any] | None = None
    finish_reason: str | None = None
    logprobs: Any | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
    router: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
