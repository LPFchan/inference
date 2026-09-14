"""Security boundaries for the authenticated browser MCP proxy."""

from fastapi.testclient import TestClient

from grimoire import config, entrypoint


AUTH = {
    config.INTERNAL_AUTH_SUB_HEADER: "regular-user",
    config.INTERNAL_AUTH_ROLE_HEADER: "user",
}


def test_cors_proxy_rejects_private_control_plane_resolution(monkeypatch):
    monkeypatch.setattr(
        entrypoint.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (entrypoint.socket.AF_INET, entrypoint.socket.SOCK_STREAM, 6, "", ("100.64.0.2", 9700))
        ],
    )

    response = TestClient(entrypoint.app).post(
        "/cors-proxy",
        params={"url": "http://mangchi.lost.plus:9700/models/model/unload"},
        headers=AUTH,
    )

    assert response.status_code == 403


def test_cors_proxy_rejects_non_http_targets():
    response = TestClient(entrypoint.app).get(
        "/cors-proxy",
        params={"url": "file:///etc/passwd"},
        headers=AUTH,
    )

    assert response.status_code == 400


def test_cors_proxy_connects_to_the_validated_address(monkeypatch):
    seen = {}

    class Upstream:
        status_code = 200
        content = b"<script>alert(document.cookie)</script>"
        headers = {
            "content-type": "text/html",
            "set-cookie": "stolen=1",
            "clear-site-data": '"cookies", "storage"',
            "mcp-session-id": "session-1",
        }

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def request(self, **kwargs):
            seen.update(kwargs)
            return Upstream()

    monkeypatch.setattr(
        entrypoint.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (entrypoint.socket.AF_INET, entrypoint.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    monkeypatch.setattr(entrypoint.httpx, "AsyncClient", Client)

    response = TestClient(entrypoint.app).get(
        "/cors-proxy",
        params={"url": "https://mcp.example.test/rpc"},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert seen["url"] == "https://93.184.216.34/rpc"
    assert seen["headers"]["Host"] == "mcp.example.test"
    assert "Cookie" not in seen["headers"]
    assert seen["extensions"] == {"sni_hostname": "mcp.example.test"}
    assert response.headers["content-type"].startswith("text/plain")
    assert "set-cookie" not in response.headers
    assert "clear-site-data" not in response.headers
    assert response.headers["mcp-session-id"] == "session-1"
    assert response.headers["content-security-policy"] == "sandbox; default-src 'none'"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_cors_proxy_does_not_relay_browser_redirects(monkeypatch):
    class Upstream:
        status_code = 307
        content = b""
        headers = {"location": "/auth/tokens?label=unexpected"}

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def request(self, **_kwargs):
            return Upstream()

    monkeypatch.setattr(
        entrypoint.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (entrypoint.socket.AF_INET, entrypoint.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    monkeypatch.setattr(entrypoint.httpx, "AsyncClient", Client)

    response = TestClient(entrypoint.app).post(
        "/cors-proxy",
        params={"url": "https://mcp.example.test/rpc"},
        headers=AUTH,
    )

    assert response.status_code == 502
    assert "location" not in response.headers
    assert response.json() == {"error": "MCP proxy targets must not redirect browser requests"}
