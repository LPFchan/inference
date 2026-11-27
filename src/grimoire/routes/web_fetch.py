"""Authenticated public web-fetch route for the chat.lost.plus web PWA.

Fetches public HTTP/HTTPS URLs with SSRF protection (hostname denylist, DNS
resolution check, IP pinning with SNI/Host preservation, per-hop redirect
re-validation, bounded streaming reads, and timeouts). Returns clean extracted
text for web pages/documents or a base64 data URL for raster images so the web
PWA can inspect remote URLs and images with the same capabilities as Chat.app's
`local_web_fetch` tool.
"""

from __future__ import annotations

import asyncio
import base64
from html.parser import HTMLParser
import ipaddress
import socket
import time
import urllib.parse
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from grimoire.auth import require_api

router = APIRouter()

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
DEFAULT_FETCH_MAX_CHARS = 16_000
MIN_FETCH_MAX_CHARS = 256
MAX_FETCH_MAX_CHARS = 64_000
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_REDIRECTS = 5
WEB_TIMEOUT_S = 15.0
HEADER_PROBE_BYTES = 128

_TEXT_MEDIA_TYPES = frozenset({
    "text/plain",
    "application/json",
    "text/markdown",
    "text/html",
    "application/xhtml+xml",
    "image/svg+xml",
})

_RAW_TEXT_MEDIA_TYPES = frozenset({
    "text/plain",
    "application/json",
    "text/markdown",
    "image/svg+xml",
})

_SKIP_HTML_TAGS = frozenset({
    "script",
    "style",
    "noscript",
    "svg",
    "template",
    "iframe",
})

_BLOCK_HTML_TAGS = frozenset({
    "p", "div", "section", "article", "main", "header", "footer", "aside", "nav",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "tr", "blockquote",
    "pre", "hr", "title",
})

_AVIF_BRANDS = frozenset({b"avif", b"avis"})
_HEIC_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx"})
_HEIF_BRANDS = frozenset({b"mif1", b"msf1", b"heif"})


def is_blocked_hostname(host: str) -> bool:
    """Return True if hostname is a known local or internal name."""
    lower = host.strip().rstrip(".").lower()
    return (
        lower == "localhost"
        or lower.endswith(".localhost")
        or lower.endswith(".local")
        or lower.endswith(".internal")
        or lower == "metadata.google.internal"
    )


def is_blocked_ip(raw_ip: str) -> bool:
    """Return True if an IP address is private, loopback, link-local, CGNAT, or non-global."""
    ip = ipaddress.ip_address(raw_ip)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _charset_from_content_type(content_type: str) -> str:
    for part in content_type.split(";")[1:]:
        key, sep, val = part.partition("=")
        if sep and key.strip().lower() == "charset":
            charset = val.strip().strip('"').strip("'")
            if charset:
                return charset
    return "utf-8"


def is_text_content_type(content_type: str) -> bool:
    media = _media_type(content_type)
    return media in _TEXT_MEDIA_TYPES or media.endswith("+json") or media.endswith("+xml")


def is_raster_image_content_type(content_type: str) -> bool:
    media = _media_type(content_type)
    return media.startswith("image/") and media != "image/svg+xml"


def detect_raster_image_format(data: bytes) -> tuple[str, str] | None:
    """Detect raster image (mime_type, format_label) from magic bytes."""
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return ("image/png", "PNG")
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return ("image/jpeg", "JPEG")
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return ("image/gif", "GIF")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ("image/webp", "WebP")
    if len(data) >= 2 and data[:2] == b"BM":
        return ("image/bmp", "BMP")
    if len(data) >= 4 and data[:4] in (b"II*\x00", b"MM\x00*"):
        return ("image/tiff", "TIFF")
    if len(data) >= 4 and data[:4] == b"\x00\x00\x01\x00":
        return ("image/x-icon", "ICO")
    if len(data) >= 4 and data[:4] == b"qoif":
        return ("image/qoi", "QOI")
    if len(data) >= 12 and data[4:8] == b"ftyp":
        box_size = int.from_bytes(data[:4], "big")
        limit = min(len(data), box_size if box_size >= 12 else len(data))
        brands = [data[8:12]]
        offset = 16
        while offset + 4 <= limit:
            brands.append(data[offset : offset + 4])
            offset += 4
        if any(b in _AVIF_BRANDS for b in brands):
            return ("image/avif", "AVIF")
        if any(b in _HEIC_BRANDS for b in brands):
            return ("image/heic", "HEIC")
        if any(b in _HEIF_BRANDS for b in brands):
            return ("image/heif", "HEIF")
    return None


def _format_label_from_mime(mime: str) -> str:
    return {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/webp": "WebP",
        "image/gif": "GIF",
        "image/avif": "AVIF",
        "image/heic": "HEIC",
        "image/heif": "HEIF",
        "image/bmp": "BMP",
        "image/tiff": "TIFF",
        "image/x-icon": "ICO",
        "image/vnd.microsoft.icon": "ICO",
        "image/qoi": "QOI",
    }.get(mime, "image")


class _ReadableTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: list[str] = []

    def _ensure_newline(self) -> None:
        if not self._chunks or not self._chunks[-1].endswith("\n"):
            self._chunks.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low = tag.lower()
        if low in _SKIP_HTML_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if low == "br":
            self._chunks.append("\n")
        elif low in _BLOCK_HTML_TAGS:
            self._ensure_newline()

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low in _SKIP_HTML_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if low in _BLOCK_HTML_TAGS:
            self._ensure_newline()

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        lines: list[str] = []
        prev_blank = True
        for line in raw.splitlines():
            cleaned = " ".join(line.split())
            if not cleaned:
                if not prev_blank:
                    lines.append("")
                    prev_blank = True
            else:
                lines.append(cleaned)
                prev_blank = False
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)


def html_to_text(html: str) -> str:
    """Convert HTML to clean readable text, stripping scripts/styles and collapsing whitespace."""
    parser = _ReadableTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text()


async def validate_and_resolve_url(raw_url: str) -> tuple[urllib.parse.ParseResult, str, str]:
    """Validate URL scheme/host/DNS against SSRF rules and return (parsed, pinned_url, sni_host)."""
    parsed = urllib.parse.urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="Unsupported URL: only public http:// and https:// URLs with a host are allowed",
        )
    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(status_code=400, detail="URL must not contain credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="URL has an invalid port") from exc

    hostname = parsed.hostname
    if is_blocked_hostname(hostname):
        raise HTTPException(
            status_code=403,
            detail=f"Blocked host '{hostname}' (loopback or internal hostname)",
        )

    resolve_port = port if port is not None else (443 if parsed.scheme == "https" else 80)
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            resolve_port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=502,
            detail=f"DNS lookup failed for '{hostname}': {exc}",
        ) from exc

    if not addresses:
        raise HTTPException(
            status_code=502,
            detail=f"DNS lookup returned no addresses for '{hostname}'",
        )

    for item in addresses:
        resolved_ip = item[4][0]
        if is_blocked_ip(resolved_ip):
            raise HTTPException(
                status_code=403,
                detail=f"Blocked resolved IP {resolved_ip} for host '{hostname}'",
            )

    selected_ip = addresses[0][4][0]
    pinned_host = f"[{selected_ip}]" if ":" in selected_ip else selected_ip
    pinned_netloc = f"{pinned_host}:{port}" if port is not None else pinned_host
    pinned_url = urllib.parse.urlunparse(parsed._replace(netloc=pinned_netloc))
    return parsed, pinned_url, hostname


async def _fetch_public_url(raw_url: str) -> tuple[bytes, str, str, tuple[str, str] | None]:
    """Fetch a public HTTP/HTTPS URL following validated redirects with size & timeout caps."""
    current_url = raw_url.strip()
    deadline = time.monotonic() + WEB_TIMEOUT_S

    for _hop in range(MAX_REDIRECTS + 1):
        parsed, pinned_url, sni_host = await validate_and_resolve_url(current_url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HTTPException(status_code=504, detail="Web request timed out")

        req_headers = {
            "Host": parsed.netloc,
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }

        async with httpx.AsyncClient(
            timeout=remaining,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            try:
                async with client.stream(
                    "GET",
                    pinned_url,
                    headers=req_headers,
                    extensions={"sni_hostname": sni_host},
                ) as response:
                    status = response.status_code
                    if 300 <= status < 400:
                        location = response.headers.get("location")
                        if not location:
                            raise HTTPException(
                                status_code=502,
                                detail="Redirect response missing Location header",
                            )
                        current_url = urllib.parse.urljoin(current_url, location.strip())
                        continue

                    if status == 202:
                        raise HTTPException(
                            status_code=502,
                            detail=f"HTTP 202 (rate-limited or bot challenge) for {current_url}",
                        )
                    if not (200 <= status < 300):
                        raise HTTPException(
                            status_code=502,
                            detail=f"HTTP {status} for {current_url}",
                        )

                    content_type = (response.headers.get("content-type") or "text/html").strip()
                    is_text_ct = is_text_content_type(content_type)
                    is_image = is_raster_image_content_type(content_type)
                    sniffed: tuple[str, str] | None = None
                    sniff_done = is_text_ct
                    header_probe = bytearray()
                    buf = bytearray()

                    async for chunk in response.aiter_bytes():
                        if time.monotonic() > deadline:
                            raise HTTPException(status_code=504, detail="Web request timed out")
                        if not chunk:
                            continue
                        if not sniff_done and len(header_probe) < HEADER_PROBE_BYTES:
                            need = HEADER_PROBE_BYTES - len(header_probe)
                            header_probe.extend(chunk[:need])
                            detected = detect_raster_image_format(bytes(header_probe))
                            if detected is not None:
                                sniffed = detected
                                is_image = True
                                sniff_done = True
                            elif len(header_probe) >= HEADER_PROBE_BYTES:
                                sniff_done = True

                        max_bytes = MAX_IMAGE_BYTES if is_image else MAX_TEXT_BYTES
                        if is_image and len(buf) + len(chunk) > max_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Image response exceeds maximum size of "
                                    f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB for {current_url}"
                                ),
                            )
                        room = max_bytes - len(buf)
                        if room <= 0:
                            break
                        if len(chunk) <= room:
                            buf.extend(chunk)
                        else:
                            buf.extend(chunk[:room])
                            break

                    if not sniff_done and len(header_probe) > 0:
                        detected = detect_raster_image_format(bytes(header_probe))
                        if detected is not None:
                            sniffed = detected
                            is_image = True

                    return bytes(buf), content_type, current_url, sniffed if is_image else None
            except httpx.TimeoutException as exc:
                raise HTTPException(status_code=504, detail=f"Web request timed out: {exc}") from exc
            except httpx.RequestError as exc:
                raise HTTPException(status_code=502, detail=f"Web fetch error: {exc}") from exc

    raise HTTPException(
        status_code=502,
        detail=f"Exceeded maximum redirects ({MAX_REDIRECTS})",
    )


def _clamp_max_chars(raw: Any) -> int:
    if isinstance(raw, bool) or raw is None:
        return DEFAULT_FETCH_MAX_CHARS
    if isinstance(raw, (int, float)):
        val = int(raw)
        return max(MIN_FETCH_MAX_CHARS, min(MAX_FETCH_MAX_CHARS, val))
    return DEFAULT_FETCH_MAX_CHARS


@router.post("/web-fetch")
async def web_fetch_endpoint(request: Request):
    """Fetch a public HTTP/HTTPS URL (text/HTML or raster image) on behalf of an authenticated user."""
    require_api(request)

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    raw_url = payload.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise HTTPException(status_code=400, detail="Missing 'url' field")

    max_chars = _clamp_max_chars(payload.get("max_chars"))
    body, content_type, final_url, sniffed = await _fetch_public_url(raw_url)
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }

    if is_raster_image_content_type(content_type) or sniffed is not None:
        if len(body) == 0:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream returned an empty image body for {final_url}",
            )
        if sniffed is not None:
            mime, fmt_label = sniffed
        else:
            mime = _media_type(content_type)
            fmt_label = _format_label_from_mime(mime)
        b64 = base64.b64encode(body).decode("ascii")
        return JSONResponse(
            content={
                "kind": "image",
                "url": final_url,
                "content_type": mime,
                "source_format": fmt_label,
                "bytes_downloaded": len(body),
                "data_url": f"data:{mime};base64,{b64}",
            },
            headers=headers,
        )

    charset = _charset_from_content_type(content_type)
    try:
        raw_text = body.decode(charset, errors="replace")
    except LookupError:
        raw_text = body.decode("utf-8", errors="replace")

    media = _media_type(content_type)
    if media in _RAW_TEXT_MEDIA_TYPES or media.endswith("+json"):
        extracted = raw_text
    else:
        extracted = html_to_text(raw_text)

    truncated = len(extracted) > max_chars
    text = extracted[:max_chars] if truncated else extracted

    return JSONResponse(
        content={
            "kind": "text",
            "url": final_url,
            "content_type": content_type,
            "bytes_downloaded": len(body),
            "truncated": truncated,
            "text": text,
        },
        headers=headers,
    )
