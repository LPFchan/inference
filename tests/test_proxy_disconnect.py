"""A client that leaves before the backend answers must not leave the backend request running."""

import asyncio
import socket
import unittest

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from grimoire.proxy.client import send_while_connected
from grimoire.proxy_app import SSE_COMMENT_PING, _with_sse_keepalive


class _Request:
    def __init__(self, leave_after=None):
        self._leave_after = leave_after

    async def receive(self):
        if self._leave_after is None:
            await asyncio.Event().wait()
        await asyncio.sleep(self._leave_after)
        return {"type": "http.disconnect"}


class _Client:
    def __init__(self, answer_after=None):
        self._answer_after = answer_after
        self.cancelled = False

    async def send(self, req, stream=False):
        assert stream
        try:
            if self._answer_after is None:
                await asyncio.Event().wait()
            await asyncio.sleep(self._answer_after)
            return "response"
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class SendWhileConnectedTest(unittest.IsolatedAsyncioTestCase):
    async def test_client_leaving_cancels_backend_request(self):
        client = _Client(answer_after=None)
        with self.assertRaises(HTTPException) as caught:
            await asyncio.wait_for(send_while_connected(client, object(), _Request(leave_after=0.01)), 5)
        self.assertEqual(caught.exception.status_code, 499)
        self.assertTrue(client.cancelled)

    async def test_backend_answer_is_returned_while_client_stays(self):
        client = _Client(answer_after=0.01)
        self.assertEqual(await send_while_connected(client, object(), _Request()), "response")
        self.assertFalse(client.cancelled)

    async def test_without_request_it_only_sends(self):
        self.assertEqual(await send_while_connected(_Client(answer_after=0), object()), "response")


class RealServerDisconnectTest(unittest.IsolatedAsyncioTestCase):
    """Through a real uvicorn: the backend sees its connection close once the client leaves."""

    async def test_backend_connection_closes_when_client_leaves(self):
        import uvicorn

        backend_closed = asyncio.Event()

        async def silent_backend(reader, writer):
            # Accept the request and never answer, like a queued model server.
            await reader.read()  # returns when the proxy closes the connection
            backend_closed.set()
            writer.close()

        backend = await asyncio.start_server(silent_backend, "127.0.0.1", 0)
        backend_port = backend.sockets[0].getsockname()[1]

        app = FastAPI()
        upstream = httpx.AsyncClient(timeout=None)

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            await request.body()
            req = upstream.build_request("POST", f"http://127.0.0.1:{backend_port}/v1/chat/completions", content=b"{}")
            response = await send_while_connected(upstream, req, request)
            await response.aclose()

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off", timeout_graceful_shutdown=1))
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            proxy_port = sock.getsockname()[1]

            async with httpx.AsyncClient(timeout=0.3) as caller:
                with self.assertRaises(httpx.ReadTimeout):
                    await caller.post(f"http://127.0.0.1:{proxy_port}/v1/chat/completions", content=b"{}")

            await asyncio.wait_for(backend_closed.wait(), timeout=5)
        finally:
            server.should_exit = True
            await serving
            await upstream.aclose()
            backend.close()
            await backend.wait_closed()

    async def test_sse_keepalive_disconnect_closes_manager_and_backend(self):
        """A client leaving mid-keepalive must close the manager and backend sockets."""
        import uvicorn

        backend_closed = asyncio.Event()
        mgr_closed = asyncio.Event()

        async def headers_then_silent_backend(reader, writer):
            # Send SSE headers immediately, then stay silent like a queued TensorFold server.
            await reader.read(4096)
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            while await reader.read(1024):
                pass
            backend_closed.set()
            writer.close()

        backend = await asyncio.start_server(headers_then_silent_backend, "127.0.0.1", 0)
        backend_port = backend.sockets[0].getsockname()[1]

        mgr_app = FastAPI()
        mgr_client = httpx.AsyncClient(timeout=None)

        @mgr_app.post("/v1/chat/completions")
        async def mgr_chat(request: Request):
            await request.body()
            req = mgr_client.build_request(
                "POST", f"http://127.0.0.1:{backend_port}/v1/chat/completions", content=b"{}"
            )
            upstream = await send_while_connected(mgr_client, req, request)

            async def body_iter():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    mgr_closed.set()
                    with anyio.CancelScope(shield=True):
                        await upstream.aclose()

            return StreamingResponse(
                body_iter(),
                status_code=upstream.status_code,
                headers={"content-type": "text/event-stream"},
            )

        proxy_app = FastAPI()
        proxy_client = httpx.AsyncClient(timeout=None)

        @proxy_app.post("/v1/chat/completions")
        async def proxy_chat(request: Request):
            body = await request.body()
            req = proxy_client.build_request(
                "POST", f"http://127.0.0.1:{mgr_port}/v1/chat/completions", content=body
            )
            upstream = await send_while_connected(proxy_client, req, request)
            chunks = _with_sse_keepalive(upstream.aiter_raw(), SSE_COMMENT_PING, interval=0.1)

            async def body_iter():
                try:
                    async for chunk in chunks:
                        yield chunk
                finally:
                    with anyio.CancelScope(shield=True):
                        try:
                            await chunks.aclose()
                        finally:
                            await upstream.aclose()

            return StreamingResponse(
                body_iter(),
                status_code=upstream.status_code,
                headers={"content-type": "text/event-stream"},
            )

        mgr_sock = socket.socket()
        mgr_sock.bind(("127.0.0.1", 0))
        mgr_port = mgr_sock.getsockname()[1]
        proxy_sock = socket.socket()
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        mgr_srv = uvicorn.Server(
            uvicorn.Config(mgr_app, log_level="error", lifespan="off", timeout_graceful_shutdown=1)
        )
        proxy_srv = uvicorn.Server(
            uvicorn.Config(proxy_app, log_level="error", lifespan="off", timeout_graceful_shutdown=1)
        )
        mgr_task = asyncio.create_task(mgr_srv.serve(sockets=[mgr_sock]))
        proxy_task = asyncio.create_task(proxy_srv.serve(sockets=[proxy_sock]))
        try:
            while not (mgr_srv.started and proxy_srv.started):
                await asyncio.sleep(0.01)

            async with httpx.AsyncClient(timeout=None) as caller:
                async with caller.stream(
                    "POST", f"http://127.0.0.1:{proxy_port}/v1/chat/completions", content=b"{}"
                ) as resp:
                    self.assertEqual(resp.status_code, 200)
                    async for line in resp.aiter_lines():
                        if line:
                            self.assertEqual(line, ": ping")
                            break

            await asyncio.wait_for(mgr_closed.wait(), timeout=5)
            await asyncio.wait_for(backend_closed.wait(), timeout=5)
            self.assertEqual(proxy_client._transport._pool._connections, [])
            self.assertEqual(mgr_client._transport._pool._connections, [])
        finally:
            mgr_srv.should_exit = True
            proxy_srv.should_exit = True
            await asyncio.gather(mgr_task, proxy_task, return_exceptions=True)
            await proxy_client.aclose()
            await mgr_client.aclose()
            backend.close()
            await backend.wait_closed()


if __name__ == "__main__":
    unittest.main()
