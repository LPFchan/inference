"""Tests for the authenticated public /web-fetch route."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from grimoire import config, entrypoint
from grimoire.routes import web_fetch


AUTH = {
    config.INTERNAL_AUTH_SUB_HEADER: "regular-user",
    config.INTERNAL_AUTH_ROLE_HEADER: "user",
}


class _FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else []

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


def _patch_dns_and_http(monkeypatch, responses_by_url, dns_map=None):
    seen_calls = []

    def fake_getaddrinfo(host, port, **_kwargs):
        ip = (dns_map or {}).get(host, "93.184.216.34")
        family = web_fetch.socket.AF_INET6 if ":" in ip else web_fetch.socket.AF_INET
        return [(family, web_fetch.socket.SOCK_STREAM, 6, "", (ip, port))]

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        @asynccontextmanager
        async def stream(self, method, url, **kwargs):
            seen_calls.append({"method": method, "url": url, **kwargs})
            resp = responses_by_url[len(seen_calls) - 1]
            yield resp

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(web_fetch.httpx, "AsyncClient", FakeClient)
    return seen_calls


def test_web_fetch_requires_auth():
    response = TestClient(entrypoint.app).post(
        "/web-fetch",
        json={"url": "https://example.com"},
    )
    assert response.status_code == 401


def test_web_fetch_rejects_bad_schemes_credentials_and_local_hosts():
    client = TestClient(entrypoint.app)
    for bad_url in (
        "file:///etc/passwd",
        "ftp://example.com/file",
        "javascript:alert(1)",
        "https://user:pass@example.com/secret",
    ):
        resp = client.post("/web-fetch", json={"url": bad_url}, headers=AUTH)
        assert resp.status_code == 400, f"expected 400 for {bad_url}"

    for blocked_host in (
        "http://localhost/admin",
        "http://sub.localhost:9000/status",
        "http://printer.local/",
        "http://metadata.google.internal/computeMetadata/v1/",
    ):
        resp = client.post("/web-fetch", json={"url": blocked_host}, headers=AUTH)
        assert resp.status_code == 403, f"expected 403 for {blocked_host}"


def test_web_fetch_rejects_private_cgnat_imds_and_ipv4_mapped_ipv6(monkeypatch):
    client = TestClient(entrypoint.app)
    for ip in (
        "127.0.0.1",
        "10.0.0.5",
        "192.168.1.1",
        "100.64.0.2",
        "169.254.169.254",
        "::1",
        "fc00::1",
        "::ffff:127.0.0.1",
    ):
        monkeypatch.setattr(
            web_fetch.socket,
            "getaddrinfo",
            lambda host, port, _ip=ip, **_kwargs: [
                (
                    web_fetch.socket.AF_INET6 if ":" in _ip else web_fetch.socket.AF_INET,
                    web_fetch.socket.SOCK_STREAM,
                    6,
                    "",
                    (_ip, port),
                )
            ],
        )
        resp = client.post("/web-fetch", json={"url": "https://evil.example/"}, headers=AUTH)
        assert resp.status_code == 403, f"expected 403 for resolved IP {ip}"


def test_web_fetch_extracts_html_text_with_pinned_ip_and_truncation(monkeypatch):
    html = (
        b"<html><head><title>Demo</title><style>.x{color:red}</style>"
        b"<script>alert(1)</script></head>"
        b"<body><h1>Hello &amp; Welcome</h1><p>Line 1<br>Line 2</p></body></html>"
    )
    seen = _patch_dns_and_http(
        monkeypatch,
        [_FakeStreamResponse(status_code=200, headers={"content-type": "text/html; charset=utf-8"}, chunks=[html])],
    )

    resp = TestClient(entrypoint.app).post(
        "/web-fetch",
        json={"url": "https://example.com/post?a=1", "max_chars": 256},
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"
    data = resp.json()
    assert data["kind"] == "text"
    assert data["url"] == "https://example.com/post?a=1"
    assert data["truncated"] is False
    assert "alert(1)" not in data["text"]
    assert ".x{color:red}" not in data["text"]
    assert "Hello & Welcome" in data["text"]
    assert "Line 1\nLine 2" in data["text"]

    assert seen[0]["url"] == "https://93.184.216.34/post?a=1"
    assert seen[0]["headers"]["Host"] == "example.com"
    assert seen[0]["extensions"] == {"sni_hostname": "example.com"}


def test_web_fetch_returns_image_data_url_and_sniffs_octet_stream(monkeypatch):
    # Minimal PNG header + payload sent as application/octet-stream across split chunks
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    _patch_dns_and_http(
        monkeypatch,
        [
            _FakeStreamResponse(
                status_code=200,
                headers={"content-type": "application/octet-stream"},
                chunks=[png_bytes[:4], png_bytes[4:]],
            )
        ],
    )

    resp = TestClient(entrypoint.app).post(
        "/web-fetch",
        json={"url": "https://pbs.twimg.com/media/sample?format=png&name=large"},
        headers=AUTH,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "image"
    assert data["content_type"] == "image/png"
    assert data["source_format"] == "PNG"
    assert data["bytes_downloaded"] == len(png_bytes)
    assert data["data_url"] == "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def test_web_fetch_follows_public_redirect_and_blocks_private_redirect(monkeypatch):
    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16
    seen = _patch_dns_and_http(
        monkeypatch,
        [
            _FakeStreamResponse(
                status_code=302,
                headers={"location": "https://cdn.example.net/photo.jpg"},
            ),
            _FakeStreamResponse(
                status_code=200,
                headers={"content-type": "image/jpeg"},
                chunks=[jpeg_bytes],
            ),
        ],
        dns_map={"example.com": "93.184.216.34", "cdn.example.net": "104.18.1.2"},
    )
    resp = TestClient(entrypoint.app).post(
        "/web-fetch",
        json={"url": "https://example.com/pic"},
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert resp.json()["url"] == "https://cdn.example.net/photo.jpg"
    assert resp.json()["kind"] == "image"
    assert len(seen) == 2

    # Redirect to private IP must be rejected with 403
    _patch_dns_and_http(
        monkeypatch,
        [
            _FakeStreamResponse(
                status_code=302,
                headers={"location": "http://internal.example.net/secret"},
            ),
        ],
        dns_map={"example.com": "93.184.216.34", "internal.example.net": "10.0.0.9"},
    )
    blocked = TestClient(entrypoint.app).post(
        "/web-fetch",
        json={"url": "https://example.com/jump"},
        headers=AUTH,
    )
    assert blocked.status_code == 403


def test_web_fetch_enforces_image_size_limit(monkeypatch):
    monkeypatch.setattr(web_fetch, "MAX_IMAGE_BYTES", 64)
    big_jpeg = b"\xff\xd8\xff\xe0" + b"\x11" * 100
    _patch_dns_and_http(
        monkeypatch,
        [
            _FakeStreamResponse(
                status_code=200,
                headers={"content-type": "image/jpeg"},
                chunks=[big_jpeg],
            )
        ],
    )
    resp = TestClient(entrypoint.app).post(
        "/web-fetch",
        json={"url": "https://example.com/huge.jpg"},
        headers=AUTH,
    )
    assert resp.status_code == 413


def test_detect_raster_image_format_covers_isobmff_and_standard_headers():
    assert web_fetch.detect_raster_image_format(b"\x89PNG\r\n\x1a\n1234") == ("image/png", "PNG")
    assert web_fetch.detect_raster_image_format(b"\xff\xd8\xff\xe1") == ("image/jpeg", "JPEG")
    assert web_fetch.detect_raster_image_format(b"RIFF\x00\x00\x00\x00WEBP") == ("image/webp", "WebP")
    assert web_fetch.detect_raster_image_format(b"GIF89a") == ("image/gif", "GIF")
    avif_hdr = (24).to_bytes(4, "big") + b"ftypavif\x00\x00\x00\x00mif1avif"
    assert web_fetch.detect_raster_image_format(avif_hdr) == ("image/avif", "AVIF")
    heic_hdr = (24).to_bytes(4, "big") + b"ftypheic\x00\x00\x00\x00mif1heic"
    assert web_fetch.detect_raster_image_format(heic_hdr) == ("image/heic", "HEIC")
    assert web_fetch.detect_raster_image_format(b"<html>not an image</html>") is None
