"""Stateless data-plane proxy for the multi-process gateway.

Run under `uvicorn --workers N`. Owns the public port. For the stateless encoder
endpoints (/v1/embeddings, /v1/rerank) it resolves the target model from the
shared route table and round-robins across its replica backends (data
parallelism across GPUs). Everything else — chat (stateful: KV slots,
history), the Responses API, and all admin/management routes — is forwarded to
the single manager process, which owns model lifecycle.

Each worker is an independent process with its own event loop and pooled client,
so throughput scales with --workers past the single-process ceiling.
"""

import copy
import json
import logging
import os
import urllib.parse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response, StreamingResponse

from grimoire import config
from grimoire.auth import AuthIdentity, authenticate_public
from grimoire.proxy.client import init_proxy_client, close_proxy_client, get_proxy_client
from grimoire.proxy.llama import _backend_request_headers, _backend_response_headers
from grimoire.proxy.routes_table import RouteTableReader
from grimoire.registry import registry

logger = logging.getLogger(__name__)

MANAGER_URL = os.environ.get("GRIMOIRE_MANAGER_URL", "http://127.0.0.1:9000").rstrip("/")
# Stateless /v1/* suffixes the proxy workers serve directly; everything else
# (chat/completions, responses, models, props, ...) forwards to the manager.
STATELESS_SUFFIXES = {"embeddings", "rerank", "reranking"}

app = FastAPI(title="Grimoire Proxy", version="0.1.0")

# The proxy workers own the public port, so cross-origin permission belongs here
# rather than on the manager app. Only the read-only stats verbs the standalone
# dashboard needs are opened up. Credentials stay off: dash.lost.plus
# authenticates with a bearer token, never the gw_session cookie, so the browser
# must not be told to attach cookies to these requests.
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "PUT", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        max_age=86400,
    )
    logger.info("CORS enabled for origins: %s", ", ".join(config.CORS_ORIGINS))

_routes = RouteTableReader()
_rr: dict[str, int] = {}


@app.on_event("startup")
async def _startup():
    init_proxy_client()


@app.on_event("shutdown")
async def _shutdown():
    await close_proxy_client()


def _next_replica(model: str, replicas: list[dict]) -> dict:
    """Round-robin pick across a model's replica backends (per-worker counter)."""
    i = _rr.get(model, 0)
    _rr[model] = i + 1
    return replicas[i % len(replicas)]


async def _ensure_loaded(client: httpx.AsyncClient, model: str) -> list[dict]:
    """Ask the manager to load a cold model, then re-read the route table."""
    try:
        await client.post(f"{MANAGER_URL}/internal/ensure-loaded", json={"model": model}, timeout=600.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"manager unavailable: {e}")
    return _routes.replicas(model)


def _manager_headers(headers, identity: AuthIdentity):
    """Forward safe headers plus identity asserted by this loopback proxy."""
    drop = config.HOP_BY_HOP_HEADERS | config.SENSITIVE_PROXY_HEADERS | {"host", "content-length"}
    forwarded = {k: v for k, v in headers.items() if k.lower() not in drop}
    forwarded[config.INTERNAL_AUTH_SUB_HEADER] = identity.sub
    forwarded[config.INTERNAL_AUTH_ROLE_HEADER] = identity.role
    return forwarded


async def _forward_to_manager(
    request: Request, path: str, body: bytes, identity: AuthIdentity
) -> StreamingResponse:
    """Proxy a request verbatim to the manager (chat, responses, admin, ...)."""
    client = get_proxy_client()
    req = client.build_request(
        request.method,
        f"{MANAGER_URL}/{path}",
        headers=_manager_headers(request.headers, identity),
        params=request.query_params,
        content=body,
    )
    upstream = await client.send(req, stream=True)

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body_iter(), status_code=upstream.status_code,
                             headers=_backend_response_headers(upstream.headers))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_v1(request: Request, path: str):
    identity = await authenticate_public(request)
    body = await request.body()

    suffix = path.split("/")[0]
    if suffix not in STATELESS_SUFFIXES:
        # chat/completions, responses, models, ... -> manager (single authority)
        return await _forward_to_manager(request, f"v1/{path}", body, identity)

    # Stateless encoder path: round-robin across replica backends.
    payload = None
    if body and request.headers.get("content-type", "").split(";")[0] == "application/json":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None

    model = registry.resolve(payload.get("model")) if isinstance(payload, dict) else None
    if not model:
        raise HTTPException(status_code=404, detail="No target model resolved")

    client = get_proxy_client()
    replicas = _routes.replicas(model)
    if not replicas:
        replicas = await _ensure_loaded(client, model)
    if not replicas:
        raise HTTPException(status_code=503, detail=f"Model '{model}' not available")

    backend = _next_replica(model, replicas)
    backend_base_url = backend.get("base_url") or f"http://127.0.0.1:{backend['port']}"
    headers = _backend_request_headers(request.headers)
    if isinstance(payload, dict):
        payload = copy.deepcopy(payload)
        payload["model"] = backend.get("backend_model_id") or model
        req = client.build_request("POST", f"{backend_base_url}/v1/{path}",
                                   headers=headers, params=request.query_params, json=payload)
    else:
        req = client.build_request("POST", f"{backend_base_url}/v1/{path}",
                                   headers=headers, params=request.query_params, content=body)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        logger.error(f"backend {backend['port']} unavailable: {e}")
        raise HTTPException(status_code=502, detail="Model server unavailable")

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body_iter(), status_code=upstream.status_code,
                             headers=_backend_response_headers(upstream.headers))


@app.get("/health")
async def health():
    return {"status": "healthy", "pid": os.getpid()}


def _return_to(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        path += f"?{request.url.query}"
    return f"{config.PUBLIC_ORIGIN}{path}"


def _login_url(request: Request) -> str:
    return f"{config.AUTH_ORIGIN}/login?to={urllib.parse.quote(_return_to(request), safe='')}"


async def _browser_identity(request: Request):
    try:
        return await authenticate_public(request)
    except HTTPException as exc:
        accepts_html = "text/html" in request.headers.get("accept", "")
        if exc.status_code == 401 and request.method in {"GET", "HEAD"} and accepts_html:
            return RedirectResponse(_login_url(request), status_code=307)
        raise


def _session_headers(request: Request) -> dict[str, str]:
    session = request.cookies.get(config.AUTH_COOKIE_NAME, "")
    if not session:
        raise HTTPException(status_code=401, detail="Browser session required")
    return {"Cookie": f"{config.AUTH_COOKIE_NAME}={session}"}


async def _auth_management(request: Request, method: str, path: str) -> Response:
    headers = _session_headers(request)
    kwargs: dict = {"headers": headers, "timeout": config.AUTH_TIMEOUT_S}
    if method != "GET":
        kwargs["content"] = await request.body()
        if request.headers.get("content-type"):
            headers["Content-Type"] = request.headers["content-type"]
    try:
        upstream = await get_proxy_client().request(
            method,
            f"{config.AUTH_ORIGIN}{path}",
            params=request.query_params,
            **kwargs,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc
    response_headers = {}
    if upstream.headers.get("content-type"):
        response_headers["content-type"] = upstream.headers["content-type"]
    return Response(upstream.content, status_code=upstream.status_code, headers=response_headers)


@app.get("/login")
async def login(request: Request):
    return RedirectResponse(_login_url(request), status_code=307)


@app.get("/auth/me")
async def auth_me(request: Request):
    identity = await authenticate_public(request)
    return {
        "sub": identity.sub,
        "email": identity.email,
        "name": identity.name,
        "role": identity.role,
        "services": identity.services,
        "manage_url": config.AUTH_ORIGIN,
        "logout_url": f"{config.AUTH_ORIGIN}/logout",
    }


@app.api_route("/auth/tokens", methods=["GET", "POST"])
async def auth_tokens(request: Request):
    return await _auth_management(request, request.method, "/api/tokens")


@app.post("/auth/token-mode")
async def auth_token_mode(request: Request):
    return await _auth_management(request, "POST", "/api/token-mode")


@app.delete("/auth/tokens/{token_id}")
async def auth_token_delete(request: Request, token_id: str):
    # Common auth revokes by immutable token ID in a JSON body.
    body = json.dumps({"id": token_id}).encode()
    headers = _session_headers(request)
    headers["Content-Type"] = "application/json"
    try:
        upstream = await get_proxy_client().request(
            "DELETE",
            f"{config.AUTH_ORIGIN}/api/tokens",
            headers=headers,
            content=body,
            timeout=config.AUTH_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc
    return Response(
        upstream.content,
        status_code=upstream.status_code,
        headers={"content-type": upstream.headers.get("content-type", "application/json")},
    )


# Catch-all for non-/v1 routes (props, models management UI, dashboard, ...) ->
# manager. Registered last so /v1 and /health take precedence.
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_rest(request: Request, path: str):
    if path == "internal" or path.startswith("internal/"):
        raise HTTPException(status_code=404, detail="Not found")
    identity = await _browser_identity(request)
    if isinstance(identity, Response):
        return identity
    body = await request.body()
    return await _forward_to_manager(request, path, body, identity)
