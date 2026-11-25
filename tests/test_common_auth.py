"""Common-auth boundary contracts for the public and loopback gateways."""

import httpx
import asyncio
import pytest
import urllib.parse
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


@pytest.mark.parametrize(
    ("credential_header", "expected_header", "expected_value"),
    [
        ({"Authorization": "Basic abc"}, "authorization", "Basic abc"),
        ({"Authorization": ""}, "authorization", ""),
        ({"X-API-Key": ""}, "x-api-key", ""),
    ],
)
def test_explicit_credential_header_never_falls_back_to_cookie(
    monkeypatch, credential_header, expected_header, expected_value
):
    seen = {}

    async def handler(upstream):
        seen["url"] = str(upstream.url)
        seen["headers"] = dict(upstream.headers)
        return httpx.Response(401)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(auth, "get_proxy_client", lambda: client)
    headers = {**credential_header, "Cookie": "lp_auth=browser-secret"}
    with pytest.raises(HTTPException) as caught:
        asyncio.run(auth.authenticate_public(request(headers)))
    asyncio.run(client.aclose())

    assert caught.value.status_code == 401
    assert f"service={config.AUTH_TOKEN_SERVICE}" in seen["url"]
    assert seen["headers"].get(expected_header, "") == expected_value
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


def test_public_proxy_rejects_stale_expected_account():
    identity = auth.AuthIdentity("account-b", "", "", "user", ("chat",))
    with pytest.raises(HTTPException) as caught:
        proxy_app._enforce_expected_identity(
            request({"X-Grimoire-Expected-Sub": "account-a"}), identity
        )
    assert caught.value.status_code == 409


def test_login_route_returns_to_chat_root():
    response = asyncio.run(proxy_app.login(request()))
    location = response.headers["location"]
    parsed = urllib.parse.urlparse(location)

    assert f"{parsed.scheme}://{parsed.netloc}" == config.AUTH_ORIGIN
    assert parsed.path == "/login"
    assert urllib.parse.parse_qs(parsed.query) == {"to": [f"{config.PUBLIC_ORIGIN}/"]}


@pytest.mark.parametrize(
    "operation",
    [
        lambda req: proxy_app.auth_tokens(req),
        lambda req: proxy_app.auth_token_mode(req),
        lambda req: proxy_app.auth_token_delete(req, "token-id"),
    ],
)
def test_token_management_rejects_explicit_credentials_with_cookie(operation):
    req = request({"Authorization": "Bearer wrong", "Cookie": "lp_auth=browser-secret"})
    with pytest.raises(HTTPException) as caught:
        asyncio.run(operation(req))
    assert caught.value.status_code == 401


def test_token_management_requires_chat_visibility(monkeypatch):
    async def handler(_upstream):
        return httpx.Response(200, json={"sub": "7", "role": "user", "services": ["okdam"]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(auth, "get_proxy_client", lambda: client)
    monkeypatch.setattr(proxy_app, "get_proxy_client", lambda: client)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(proxy_app.auth_tokens(request({"Cookie": "lp_auth=browser-secret"})))
    asyncio.run(client.aclose())
    assert caught.value.status_code == 403


def test_token_management_rejects_stale_expected_account(monkeypatch):
    async def handler(_upstream):
        return httpx.Response(200, json={"sub": "account-b", "role": "user", "services": ["chat"]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(auth, "get_proxy_client", lambda: client)
    monkeypatch.setattr(proxy_app, "get_proxy_client", lambda: client)
    req = request({
        "Cookie": "lp_auth=browser-secret",
        "X-Grimoire-Expected-Sub": "account-a",
    })
    with pytest.raises(HTTPException) as caught:
        asyncio.run(proxy_app.auth_tokens(req))
    asyncio.run(client.aclose())
    assert caught.value.status_code == 409


def test_stateless_proxy_keeps_the_managers_eviction_refusal():
    async def handler(upstream):
        assert upstream.url.path == "/internal/ensure-loaded"
        return httpx.Response(409, json={"detail": "Loading 'emb' would unload chat; only an administrator can do that"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await proxy_app._ensure_loaded(client, "emb", evict=False)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(go())
    assert exc.value.status_code == 409
    assert "only an administrator" in exc.value.detail


def test_public_pwa_manifest_and_icons_bypass_auth(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    (tmp_path / "manifest.webmanifest").write_text('{"name":"Chat","short_name":"Chat"}', encoding="utf-8")
    (tmp_path / "apple-touch-icon.png").write_bytes(b"\x89PNG\r\n\x1a\napple")
    (tmp_path / "favicon.png").write_bytes(b"\x89PNG\r\n\x1a\nfav")
    (tmp_path / "favicon.ico").write_bytes(b"\x00\x00\x01\x00ico")
    (tmp_path / "index.html").write_text("<html>secret</html>", encoding="utf-8")
    icons = tmp_path / "icons"
    icons.mkdir()
    (icons / "icon-192.png").write_bytes(b"\x89PNG\r\n\x1a\n192")
    (icons / "apple-touch-icon.png").write_bytes(b"\x89PNG\r\n\x1a\napple512")

    monkeypatch.setattr(config, "WEBUI_DIR", str(tmp_path))

    with TestClient(proxy_app.app) as client:
        for path, expected_ct in [
            ("/manifest.webmanifest", "application/manifest+json"),
            ("/apple-touch-icon.png", "image/png"),
            ("/apple-touch-icon-precomposed.png", "image/png"),
            ("/apple-touch-icon-240x240.png", "image/png"),
            ("/apple-touch-icon-240x240-precomposed.png", "image/png"),
            ("/favicon.png", "image/png"),
            ("/favicon.ico", "image/x-icon"),
            ("/icons/icon-192.png", "image/png"),
            ("/icons/apple-touch-icon.png", "image/png"),
        ]:
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert expected_ct in resp.headers["content-type"]
            head = client.head(path)
            assert head.status_code == 200, path

        assert client.get("/icons/not-there.png").status_code == 404
        assert client.get("/icons/secret.html").status_code == 404
        assert client.get("/index.html", follow_redirects=False).status_code == 401
        assert client.get("/login", follow_redirects=False).status_code == 307
