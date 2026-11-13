"""Shared, connection-pooled httpx client for backend proxying.

Creating a fresh ``httpx.AsyncClient`` per request opens and tears down a TCP
connection to the local llama-server every time (no keepalive), which caps
high-RPS endpoints (rerank, embeddings) at ~30 req/s. A single pooled client
reused across requests is ~3x faster for those workloads. Initialized in the
app lifespan; both the entrypoint proxies and proxy.llama use it.
"""

import asyncio

import httpx
from fastapi import HTTPException

from grimoire import config

_client: httpx.AsyncClient | None = None


def init_proxy_client() -> httpx.AsyncClient:
    """Create the shared client (idempotent). Call once at app startup."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            # No read limit (prefill and generation run long), but give up
            # connecting quickly so an offline remote host fails over.
            timeout=httpx.Timeout(None, connect=config.DEFAULT_REMOTE_CONNECT_TIMEOUT),
            limits=httpx.Limits(
                max_connections=256,
                max_keepalive_connections=128,
                keepalive_expiry=30.0,
            ),
        )
    return _client


async def close_proxy_client() -> None:
    """Close the shared client at app shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def get_proxy_client() -> httpx.AsyncClient:
    """Return the shared client. Raises if not initialized."""
    if _client is None:
        raise RuntimeError("proxy client not initialized")
    return _client


async def send_while_connected(client: httpx.AsyncClient, req: httpx.Request, request=None) -> httpx.Response:
    """Send ``req`` streamed, cancelling it if ``request``'s client leaves first.

    A queued backend can hold its response headers for many minutes. Without
    this the handler keeps waiting after its client gives up, the backend runs
    the abandoned request anyway, and every client retry queues another copy.
    Once headers arrive, StreamingResponse watches for the disconnect itself.
    """
    if request is None:
        return await client.send(req, stream=True)
    sending = asyncio.ensure_future(client.send(req, stream=True))
    leaving = asyncio.ensure_future(_client_left(request))
    try:
        await asyncio.wait({sending, leaving}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        leaving.cancel()
        if not sending.done():
            # Cancelling closes the backend connection, which is the backend's
            # signal to drop the request.
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
    if sending.cancelled():
        raise HTTPException(status_code=499, detail="Client closed request")
    return sending.result()


async def _client_left(request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass
