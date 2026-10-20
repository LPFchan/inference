"""Guard against paid reruns, reference leakage and lost empty transcripts."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location("stt_openrouter", Path(__file__).parents[1] / "scripts/stt-benchmark-openrouter.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def fixture(tmp_path, **changes):
    audio = b"fake audio for mock transport"
    (tmp_path / "sample.wav").write_bytes(audio)
    row = {"path": "sample.wav", "language": "ko", "reference": "never send this reference",
           "seconds": 8.0, "sha256": hashlib.sha256(audio).hexdigest()}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    return argparse.Namespace(data=tmp_path, output=tmp_path / "results", workers=2,
                              limit=0, max_cost_usd=4.0, **changes), row


def test_resume_skips_paid_requests_and_keeps_empty_transcripts(tmp_path):
    args, row = fixture(tmp_path)
    calls = []

    def handler(request):
        body = json.loads(request.content)
        assert row["reference"] not in request.content.decode()
        assert body["language"] == "ko"
        assert request.headers["Authorization"] == "Bearer fake-key"
        calls.append(body["model"])
        return httpx.Response(200, json={"text": "", "usage": {"cost": 0.001}},
                              headers={"X-Generation-Id": "mock-generation"})

    transport = httpx.MockTransport(handler)
    asyncio.run(bench.run(args, "fake-key", transport))
    asyncio.run(bench.run(args, "fake-key", transport))
    assert len(calls) == 2
    assert len((args.output / "openrouter-attempts.ndjson").read_text().splitlines()) == 2
    for model in bench.MODELS:
        rows = [json.loads(line) for line in (args.output / f"{model}.jsonl").read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["hypothesis"] == ""
        assert rows[0]["generation_id"] == "mock-generation"


def test_resume_rejects_changed_manifest_before_network(tmp_path):
    args, row = fixture(tmp_path)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"text": "ok"}))
    asyncio.run(bench.run(args, "fake-key", transport))
    row["reference"] = "changed reference"
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="differs from manifest"):
        asyncio.run(bench.run(args, "fake-key", httpx.MockTransport(lambda _: pytest.fail("paid rerun"))))


def test_cost_limit_prevents_requests(tmp_path):
    args, _ = fixture(tmp_path)
    args.max_cost_usd = 0.000001
    with pytest.raises(RuntimeError, match="Cost limit"):
        asyncio.run(bench.run(args, "fake-key", httpx.MockTransport(lambda _: pytest.fail("cost limit bypassed"))))


def test_retry_is_recorded_without_duplicate_prediction(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = {}

    async def no_wait(_):
        pass

    monkeypatch.setattr(bench.asyncio, "sleep", no_wait)

    def handler(request):
        model = json.loads(request.content)["model"]
        calls[model] = calls.get(model, 0) + 1
        if calls[model] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"text": "ok", "usage": {"cost": 0.001}})

    asyncio.run(bench.run(args, "fake-key", httpx.MockTransport(handler)))
    assert set(calls.values()) == {2}
    ledger = [json.loads(line) for line in (args.output / "openrouter-attempts.ndjson").read_text().splitlines()]
    assert len(ledger) == 4
    assert sum(item["cost_reported"] for item in ledger) == 2
    assert all(len((args.output / f"{model}.jsonl").read_text().splitlines()) == 1 for model in bench.MODELS)
