"""Dumpling's transcription gateway using the inference system's Common Auth."""
from contextlib import asynccontextmanager
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from grimoire import config
from grimoire.auth import authenticate_public, _explicit_credentials
from grimoire.proxy.client import init_proxy_client, close_proxy_client
from grimoire.stt_server import MAX_UPLOAD

BACKEND = os.environ.get("GRIMOIRE_STT_BACKEND", "http://127.0.0.1:8791").rstrip("/")


@asynccontextmanager
async def lifespan(app):
    init_proxy_client()
    yield
    await close_proxy_client()


app = FastAPI(title="Inference transcription gateway", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
async def health():
    return {"status": "ok"}


async def forward(request, path):
    await authenticate_public(request)
    if request.method == "POST" and _explicit_credentials(request.headers) is None:
        if request.headers.get("origin") != config.PUBLIC_ORIGIN:
            raise HTTPException(403, "Session-authenticated uploads require the service origin")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_UPLOAD:
            raise HTTPException(413, "Upload exceeds 100 MiB")
    headers = {}
    if request.headers.get("content-type"):
        headers["content-type"] = request.headers["content-type"]
    try:
        async with httpx.AsyncClient(timeout=1800) as client:
            upstream = await client.request(request.method, f"{BACKEND}/{path}", content=bytes(body), headers=headers)
    except httpx.HTTPError as error:
        raise HTTPException(503, "Transcription backend unavailable") from error
    return Response(upstream.content, status_code=upstream.status_code,
                    headers={key: value for key, value in upstream.headers.items()
                             if key.lower() in {"content-type", "x-stt-model", "x-stt-seconds"}})


@app.get("/v1/models")
async def models(request: Request):
    return await forward(request, "v1/models")


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    return await forward(request, "v1/audio/transcriptions")
