"""A client that leaves before the backend answers must not leave the backend request running."""

import asyncio
import socket
import unittest

import httpx
from fastapi import FastAPI, HTTPException, Request

from grimoire.proxy.client import send_while_connected


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


if __name__ == "__main__":
    unittest.main()
