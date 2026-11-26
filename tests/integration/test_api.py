"""Integration tests for the API endpoints."""
import os

import httpx
import pytest
import respx
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    os.environ["ROUTER_API_KEY"] = "test-router-key"

    with respx.mock(assert_all_called=False) as rmock:
        rmock.get("https://openrouter.ai/api/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        async with app.router.lifespan_context(app), AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readyz(client):
    resp = await client.get("/readyz")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = [m["id"] for m in data["data"]]
    assert "smart-router" in ids
    assert "smart-router/L1" in ids
    assert "smart-router/L4" in ids


@pytest.mark.asyncio
async def test_auth_required(client):
    # No auth header
    resp = await client.post("/v1/chat/completions", json={
        "model": "smart-router",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_metrics(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "router_requests_total" in resp.text or "# HELP" in resp.text
