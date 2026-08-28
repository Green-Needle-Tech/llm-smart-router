"""Regression tests for mid-stream error handling (2026-08-28 incident).

A large-context call (~80K tokens) exceeded provider.timeout_seconds while
streaming. The old handler swallowed the exception: no log line, metrics
recorded status=200, and the client got a generic uncoded error event.

Covers:
  1. Timeout mid-stream  -> coded router_upstream_timeout event + 504 metric
  2. Non-timeout break   -> coded router_stream_interrupted event + 502 metric
  3. Clean stream        -> exactly one status=200 metric, no error event
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.schemas.router import (
    ClassificationResult,
    ClassificationSource,
    Level,
    RouteDecision,
)
from app.api.chat import _handle_stream


class _FakeStreamResponse:
    """Minimal httpx.Response stand-in for aiter_lines()/aclose()."""

    def __init__(self, lines: list[str], exc: Exception | None = None):
        self._lines = lines
        self._exc = exc
        self.closed = False

    def aiter_lines(self):
        async def gen():
            for line in self._lines:
                yield line
                if self._exc is not None:
                    raise self._exc
        return gen()

    async def aclose(self):
        self.closed = True


class _FakeProvider:
    def __init__(self, resp):
        self._resp = resp

    async def chat_completion(self, payload, fallback_models=None, *, stream=False,
                              base_url=None, api_key=None):
        return None, self._resp, "test/model", False, None


class _FakeState:
    def __init__(self, provider):
        self.provider = provider
        self.ip_redaction = None
        self.guardrails = None
        self.config = _FakeConfig()


class _FakeConfig:
    def __init__(self):
        self._cfg = {"telemetry": {}}

    def get(self):
        return self._cfg


class _FakeRequest:
    def __init__(self, provider):
        self.app = type("A", (), {"state": _FakeState(provider), "url": None})()


def _route() -> RouteDecision:
    return RouteDecision(
        level=Level.L3,
        model="test/model",
        classification=ClassificationResult(
            level=Level.L3, source=ClassificationSource.SESSION,
        ),
    )


def _sse_lines() -> list[str]:
    return [
        'data: {"id":"c1","choices":[{"delta":{"content":"Hello"}}]}',
        "",
        'data: {"id":"c2","choices":[{"delta":{"content":" world"}}]}',
        "",
    ]


def _collect(chunks) -> str:
    return "".join(chunks)


@pytest.mark.asyncio
async def test_timeout_midstream_emits_coded_error_and_504_metric():
    exc = httpx.ReadTimeout("timed out")
    resp = _FakeStreamResponse(_sse_lines(), exc)
    request = _FakeRequest(_FakeProvider(resp))

    response = await _handle_stream(
        request, {}, _route(), [], "sess-1", None, None, False, __import__("time").monotonic(),
    )
    assert response.status_code == 200  # headers already sent

    body = _collect([chunk async for chunk in response.body_iterator])
    assert "router_upstream_timeout" in body
    error_event = json.loads(
        [l for l in body.split("\n") if l.startswith("data: ") and "error" in l][0][6:]
    )
    assert error_event["error"]["code"] == "router_upstream_timeout"
    assert resp.closed


@pytest.mark.asyncio
async def test_generic_break_emits_coded_error_and_502_metric():
    exc = RuntimeError("connection reset")
    resp = _FakeStreamResponse(_sse_lines(), exc)
    request = _FakeRequest(_FakeProvider(resp))

    response = await _handle_stream(
        request, {}, _route(), [], "sess-1", None, None, False, __import__("time").monotonic(),
    )
    body = _collect([chunk async for chunk in response.body_iterator])
    assert "router_stream_interrupted" in body
    assert resp.closed


@pytest.mark.asyncio
async def test_clean_stream_records_single_200_metric():
    lines = _sse_lines() + ["data: [DONE]", ""]
    resp = _FakeStreamResponse(lines, None)
    request = _FakeRequest(_FakeProvider(resp))

    response = await _handle_stream(
        request, {}, _route(), [], "sess-1", None, None, False, __import__("time").monotonic(),
    )
    body = _collect([chunk async for chunk in response.body_iterator])
    assert "data: [DONE]" in body
    assert '"error"' not in body
    assert resp.closed
