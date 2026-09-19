"""Langfuse tracing integration (SDK v4, OpenTelemetry-based).

Design goals:
- Zero impact when Langfuse is not configured: every helper is a no-op.
- Never break the router: all Langfuse calls are wrapped in try/except.
- Correct OTEL context: all helpers return real context managers, used with
  `with` blocks, so child observations nest under the request trace.
  (Manually calling `cm.__enter__()` does NOT attach the OTEL context —
  verified empirically against SDK 4.15.4 — so it is never done here.)
- Best practices per https://langfuse.com/docs/observability/best-practices:
  one trace per /v1/chat/completions request (root span ended when the
  response body fully completes, including streaming), a `generation`
  observation per upstream model attempt, descriptive names, explicit
  sanitized trace input instead of raw function args, session_id/tags
  propagated to all spans via `propagate_attributes`.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

_CLIENT = None
_INIT_ATTEMPTED = False

# Trace input/output size caps — keep payloads readable and cheap.
_MAX_TEXT_CHARS = 2000
_MAX_BODY_BYTES = 4096


def _client():
    """Lazily return the Langfuse client, or None when disabled/unavailable."""
    global _CLIENT, _INIT_ATTEMPTED
    if _INIT_ATTEMPTED:
        return _CLIENT
    _INIT_ATTEMPTED = True
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import get_client

        _CLIENT = get_client()
    except Exception:  # pragma: no cover - never break the app
        _CLIENT = None
    return _CLIENT


def enabled() -> bool:
    return _client() is not None


class _NullObservation:
    """No-op stand-in so call sites never need `if enabled()` branches."""

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass


@contextlib.contextmanager
def _null_cm() -> Iterator[_NullObservation]:
    yield _NullObservation()


def _truncate(value: Any, limit: int = _MAX_TEXT_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value)} chars total]"
    return value


def root_span_cm(name: str):
    """Context manager for the trace-root span of a request.

    MUST be used as a real `with` block wrapping the code that creates
    child observations, so the OTEL context propagates.
    """
    client = _client()
    if client is None:
        return _null_cm()
    try:
        return client.start_as_current_observation(as_type="span", name=name)
    except Exception:
        return _null_cm()


def span_cm(name: str, input_data: Any = None):
    """Context manager for a child span (guardrails, classification...)."""
    client = _client()
    if client is None:
        return _null_cm()
    try:
        return client.start_as_current_observation(as_type="span", name=name, input=input_data)
    except Exception:
        return _null_cm()


def generation_cm(name: str, model: str | None = None, input_data: Any = None):
    """Context manager for a `generation` observation (one upstream LLM call)."""
    client = _client()
    if client is None:
        return _null_cm()
    try:
        return client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=input_data
        )
    except Exception:
        return _null_cm()


def note_trace_input(input_data: Any) -> None:
    """Set the trace input explicitly (sanitized, relevant data only).

    Must be called while the root span is the active observation (i.e.
    before any child span starts).
    """
    client = _client()
    if client is None:
        return
    try:
        client.update_current_span(input=input_data)
    except Exception:
        pass


def note_route(session_id: str | None, level: str | None, model: str | None, metadata: dict | None = None):
    """Propagate session/route context onto all spans of the current trace.

    Uses `propagate_attributes` (baggage) so downstream spans — including the
    upstream `generation` spans created in the fallback executor — all carry
    session_id and tags, which Langfuse aggregation/filtering relies on.

    IMPORTANT: `propagate_attributes` is a context manager. It only affects
    the active span and spans created *inside* its `with` block, so this
    helper is itself a context manager that must be entered with `with`
    around the code that creates the downstream spans.
    """
    if _client() is None:
        return _null_cm()
    try:
        from langfuse import propagate_attributes

        kwargs: dict[str, Any] = {}
        if session_id:
            kwargs["session_id"] = session_id
        tags = []
        if level:
            tags.append(f"level:{level}")
        if tags:
            kwargs["tags"] = tags
        if metadata:
            kwargs["metadata"] = metadata
        if kwargs:
            return propagate_attributes(**kwargs)
        return _null_cm()
    except Exception:
        return _null_cm()


def shutdown_flush() -> None:
    """Flush pending traces on application shutdown."""
    client = _client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass


class LangfuseTraceMiddleware:
    """Raw ASGI middleware: one Langfuse trace per POST /v1/chat/completions.

    The root span wraps the whole downstream ASGI call in a `with` block, so
    it ends only when the response body has fully been sent (`more_body`
    falsy) — streaming responses get accurate end-to-end latency and all
    child observations (guardrails, classification, upstream generations)
    nest under it.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/chat/completions"
        ):
            await self.app(scope, receive, send)
            return

        state: dict[str, Any] = {"status": None, "chunks": []}

        with root_span_cm("chat-completions") as root:

            async def wrapped_send(message):
                if message["type"] == "http.response.start":
                    state["status"] = message.get("status")
                elif message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    if body and sum(len(c) for c in state["chunks"]) < _MAX_BODY_BYTES:
                        state["chunks"].append(body if isinstance(body, bytes) else bytes(body))
                await send(message)

            try:
                await self.app(scope, receive, wrapped_send)
            except Exception as exc:
                _safe_update(
                    root,
                    output={"status": 500, "error": str(exc)[:500]},
                    level="ERROR",
                    status_message=str(exc)[:500],
                )
                raise
            else:
                _safe_update(root, output=_build_output(state))


def _safe_update(observation, **kwargs: Any) -> None:
    try:
        observation.update(**kwargs)
    except Exception:
        pass


def _build_output(state: dict[str, Any]) -> dict[str, Any]:
    status = state.get("status")
    raw = b"".join(state["chunks"])
    try:
        excerpt = _truncate(raw.decode("utf-8", errors="replace"))
    except Exception:
        excerpt = None
    return {"status": status, "response_excerpt": excerpt}
