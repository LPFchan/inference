"""Common-auth boundary contracts for the public and loopback gateways."""

import httpx
import asyncio
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from grimoire import config
from grimoire import auth
from grimoire import proxy_app


def request(headers=None):
    raw = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw})


def test_bearer_validation_is_scoped_and_does_not_forward_cookie(monkeypatch):
    seen = {}

    async def handler(upstream):
        seen["url"] = str(upstream.url)
        seen["headers"] = dict(upstream.headers)
        return httpx.Response(200, json={"sub": "7", "role": "user", "services": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(auth, "get_proxy_client", lambda: client)
    identity = asyncio.run(auth.authenticate_public(request({
        "Authorization": "Bearer machine-secret",
        "Cookie": "lp_auth=browser-secret",
    })))
    asyncio.run(client.aclose())

    assert identity.sub == "7"
    assert f"service={config.AUTH_TOKEN_SERVICE}" in seen["url"]
    assert seen["headers"]["authorization"] == "Bearer machine-secret"
    assert "cookie" not in seen["headers"]


def test_browser_session_requires_chat_visibility(monkeypatch):
    async def handler(_upstream):
        return httpx.Response(200, json={"sub": "7", "role": "user", "services": ["okdam"]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(auth, "get_proxy_client", lambda: client)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(auth.authenticate_public(request({"Cookie": "lp_auth=browser-secret"})))
    asyncio.run(client.aclose())
    assert caught.value.status_code == 403


def test_auth_outage_fails_closed(monkeypatch):
    async def handler(_upstream):
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(auth, "get_proxy_client", lambda: client)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(auth.authenticate_public(request({"Cookie": "lp_auth=browser-secret"})))
    asyncio.run(client.aclose())
    assert caught.value.status_code == 503


def test_manager_headers_strip_credentials_and_client_identity():
    identity = auth.AuthIdentity("7", "me@example.test", "Me", "user", ("chat",))
    headers = proxy_app._manager_headers({
        "Authorization": "Bearer secret",
        "Cookie": "lp_auth=session",
        "X-Grimoire-Auth-Sub": "attacker",
        "Content-Type": "application/json",
    }, identity)
    lowered = {key.lower(): value for key, value in headers.items()}
    assert "authorization" not in lowered
    assert "cookie" not in lowered
    assert lowered[config.INTERNAL_AUTH_SUB_HEADER] == "7"
    assert lowered[config.INTERNAL_AUTH_ROLE_HEADER] == "user"
