"""OpenAI-compatible API schemas."""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model: str = "smart-router"
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Any] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    n: Optional[int] = None
    user: Optional[str] = None
    stream_options: Optional[dict[str, Any]] = None

    # Non-standard router extension (stripped before forwarding)
    router: Optional[dict[str, Any]] = None

    model_config = {"extra": "allow"}


class Choice(BaseModel):
    index: int = 0
    message: Optional[ChatMessage] = None
    delta: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


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
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None
    router: Optional[dict[str, Any]] = None

    model_config = {"extra": "allow"}


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
