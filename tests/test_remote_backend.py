"""Remote vLLM backend lifecycle, validation, and routing tests."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import grimoire.model_manager as mm
from grimoire.model_manager import ActiveModel, ModelManager
from grimoire.registry import ModelRegistry


REMOTE = {
    "backend": "vllm-remote",
    "remote-agent-url": "http://mangchi.lost.plus:9700",
    "remote-model-id": "remote/model",
    "remote-url": "http://mangchi.lost.plus:8001",
    "backend-model-id": "/models/remote",
    "capabilities": ["completion"],
    "ctx-size": 32768,
}


class _Response:
    def __init__(self, payload=None):
        self.payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Client:
    calls = []

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url):
        self.calls.append(url)
        return _Response({"status": "ok"})


def test_remote_active_model_controls_agent_and_exposes_backend_url():
    _Client.calls = []
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    with patch.object(mm.httpx, "Client", _Client):
        active.start()
        assert active.is_running()
        assert active.backend_url("v1/chat/completions") == (
            "http://mangchi.lost.plus:8001/v1/chat/completions"
        )
        assert asyncio.run(active.get_backend_model_id()) == "/models/remote"
        active.stop()
    assert _Client.calls == [
        "http://mangchi.lost.plus:9700/models/remote%2Fmodel/load",
        "http://mangchi.lost.plus:9700/models/remote%2Fmodel/unload",
    ]
    assert not active.is_running()


def test_registry_accepts_remote_backend_and_rejects_local_gpu_fields(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": {"remote": REMOTE}, "fixed": {}}))
    registry = ModelRegistry(path=str(path), seed_path=str(tmp_path / "missing.json"))
    assert registry.validate("remote") == (True, "OK")

    invalid = dict(REMOTE, **{"gpu-ids": [0]})
    path.write_text(json.dumps({"models": {"remote": invalid}, "fixed": {}}))
    registry.reload()
    valid, reason = registry.validate("remote")
    assert not valid
    assert "gpu-ids" in reason


class _Registry:
    def resolve(self, name):
        return name if name == "remote" else None

    def get(self, name):
        return dict(REMOTE) if name == "remote" else None

    def validate(self, name, gpu_count=None):
        return True, "Valid"

    def get_fixed_gpu(self, name):
        return None

    def get_family_defaults(self, family):
        return {}


def test_manager_remote_start_bypasses_local_gpu_allocator():
    manager = ModelManager(gpu_count=1)

    async def start_active(active):
        active.remote_running = True
        active.status = "loaded"

    with (
        patch.object(mm, "registry", _Registry()),
        patch.object(manager, "_allocate_gpu", AsyncMock(side_effect=AssertionError("allocator called"))),
        patch.object(manager, "_start_active_model", side_effect=start_active),
        patch.object(manager, "_publish_routes"),
    ):
        active = asyncio.run(manager.start_model("remote"))

    assert active.backend_type == "vllm-remote"
    assert active.gpu is None
    assert active.port is None
    assert manager.list_active() == ["remote"]
