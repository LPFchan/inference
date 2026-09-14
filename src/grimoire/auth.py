"""Shared auth.lost.plus integration and trusted manager identity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import httpx
from fastapi import HTTPException, Request

from grimoire import config
from grimoire.history import identity_hash
from grimoire.proxy.client import get_proxy_client


@dataclass(frozen=True)
class AuthIdentity:
    sub: str
    email: str
    name: str
    role: str
    services: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object) -> "AuthIdentity":
        if not isinstance(payload, dict) or not isinstance(payload.get("sub"), str):
            raise ValueError("auth response did not contain a string sub")
        services = payload.get("services", [])
        if not isinstance(services, list) or not all(isinstance(item, str) for item in services):
            raise ValueError("auth response contained invalid services")
        return cls(
            sub=payload["sub"],
            email=str(payload.get("email") or ""),
            name=str(payload.get("name") or ""),
            role=str(payload.get("role") or "user"),
            services=tuple(services),
        )


def _explicit_credentials(headers: Mapping[str, str]) -> dict[str, str] | None:
    authorization = headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return {"Authorization": authorization}
    api_key = headers.get("x-api-key", "")
    if api_key:
        return {"X-API-Key": api_key}
    return None


async def authenticate_public(request: Request) -> AuthIdentity:
    """Validate a public request with common auth, failing closed."""
    explicit = _explicit_credentials(request.headers)
    if explicit is not None:
        headers = explicit
        params = {"service": config.AUTH_TOKEN_SERVICE}
        visibility = None
    else:
        session = request.cookies.get(config.AUTH_COOKIE_NAME, "")
        if not session:
            raise HTTPException(status_code=401, detail="Authentication required")
        headers = {"Cookie": f"{config.AUTH_COOKIE_NAME}={session}"}
        params = None
        visibility = config.AUTH_VISIBILITY_KEY

    try:
        response = await get_proxy_client().get(
            f"{config.AUTH_ORIGIN}/api/whoami",
            headers=headers,
            params=params,
            timeout=config.AUTH_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc

    if response.status_code in {401, 403}:
        raise HTTPException(status_code=response.status_code, detail="Authentication failed")
    if not response.is_success:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    try:
        identity = AuthIdentity.from_payload(response.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Invalid authentication response") from exc

    if visibility and identity.services and visibility not in identity.services:
        raise HTTPException(status_code=403, detail="Chat access is not enabled for this account")
    return identity


def account_hash(sub: str) -> str:
    """Stable Grimoire storage owner derived from common auth's immutable sub."""
    return identity_hash(f"auth.lost.plus:{sub}")


def require_api(request):
    """Trust identity injected by the loopback-only public proxy."""
    sub = request.headers.get(config.INTERNAL_AUTH_SUB_HEADER, "")
    if not sub:
        raise HTTPException(status_code=401, detail="Missing trusted identity")
    return sub, account_hash(sub)


def require_admin(request):
    """Require the common-auth administrator role on manager routes."""
    sub, user_hash = require_api(request)
    if request.headers.get(config.INTERNAL_AUTH_ROLE_HEADER, "") != "administrator":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return sub, user_hash
