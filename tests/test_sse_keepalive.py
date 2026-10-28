"""Tests for the public proxy's SSE keepalive during silent upstream prefill."""

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grimoire.proxy_app import ANTHROPIC_PING, _with_sse_keepalive


async def _upstream(*steps):
    """Yield bytes, sleeping for each float step."""
    for step in steps:
        if isinstance(step, float):
            await asyncio.sleep(step)
        else:
            yield step


async def _collect(stream):
    return [chunk async for chunk in stream]


class SseKeepaliveTests(unittest.TestCase):
    def run_keepalive(self, *steps, interval=0.05):
        return asyncio.run(_collect(_with_sse_keepalive(_upstream(*steps), ANTHROPIC_PING, interval)))

    def test_pings_while_upstream_is_silent(self):
        start = b'event: message_start\ndata: {"type": "message_start"}\n\n'
        delta = b'event: content_block_delta\ndata: {}\n\n'
        out = self.run_keepalive(start, 0.18, delta)
        self.assertEqual(out[0], start)
        self.assertEqual(out[-1], delta)
        self.assertGreaterEqual(out.count(ANTHROPIC_PING), 2)

    def test_no_ping_inside_a_split_event(self):
        out = self.run_keepalive(b"data: {\"half\":", 0.18, b" 1}\n\n")
        self.assertEqual(out, [b"data: {\"half\":", b" 1}\n\n"])

    def test_busy_stream_passes_through_unchanged(self):
        chunks = [b"data: %d\n\n" % i for i in range(5)]
        self.assertEqual(self.run_keepalive(*chunks), chunks)


if __name__ == "__main__":
    unittest.main()
