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


class _AsyncResponse(_Response):
    def __init__(self, payload=None, status_code=200):
        super().__init__(payload)
        self.status_code = status_code


class _AsyncClient:
    responses = {}
    calls = []

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        self.calls.append(url)
        response = self.responses[url]
        if isinstance(response, BaseException):
            raise response
        return response


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
    def list_all(self):
        return ["remote"]

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


def test_remote_health_probe_requires_agent_residency_and_backend_health():
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    agent_url = "http://mangchi.lost.plus:9700/models/remote%2Fmodel/status"
    backend_url = "http://mangchi.lost.plus:8001/health"
    _AsyncClient.calls = []
    _AsyncClient.responses = {
        agent_url: _AsyncResponse({"resident": True, "alive": True}),
        backend_url: _AsyncResponse(status_code=200),
    }

    with patch.object(mm.httpx, "AsyncClient", _AsyncClient):
        assert asyncio.run(active.probe_remote_health()) is True

    assert _AsyncClient.calls == [agent_url, backend_url]


def test_remote_health_probe_treats_confirmed_absence_as_offline():
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    agent_url = "http://mangchi.lost.plus:9700/models/remote%2Fmodel/status"
    _AsyncClient.calls = []
    _AsyncClient.responses = {
        agent_url: _AsyncResponse({"resident": False, "registered": True}),
    }

    with patch.object(mm.httpx, "AsyncClient", _AsyncClient):
        assert asyncio.run(active.probe_remote_health()) is False

    assert _AsyncClient.calls == [agent_url]


def test_remote_health_probe_reports_agent_loading_without_backend_probe():
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    active.status = "loaded"
    agent_url = "http://mangchi.lost.plus:9700/models/remote%2Fmodel/status"
    _AsyncClient.calls = []
    _AsyncClient.responses = {
        agent_url: _AsyncResponse({"resident": True, "alive": True, "status": "loading"}),
    }

    with patch.object(mm.httpx, "AsyncClient", _AsyncClient):
        assert asyncio.run(active.probe_remote_health()) == "loading"

    assert _AsyncClient.calls == [agent_url]


def test_remote_health_probe_keeps_previous_value_when_agent_times_out():
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    agent_url = "http://mangchi.lost.plus:9700/models/remote%2Fmodel/status"
    _AsyncClient.calls = []
    _AsyncClient.responses = {
        agent_url: mm.httpx.ReadTimeout("agent timed out"),
    }

    with patch.object(mm.httpx, "AsyncClient", _AsyncClient):
        assert asyncio.run(active.probe_remote_health()) is None

    assert _AsyncClient.calls == [agent_url]


def test_remote_status_refresh_keeps_previous_value_when_probe_is_ambiguous():
    manager = ModelManager(gpu_count=1)
    active = ActiveModel("remote", dict(REMOTE), port=None, gpu=None)
    active.remote_running = True
    active.status = "loaded"
    manager.active["remote"] = active

    with (
        patch.object(mm, "registry", _Registry()),
        patch.object(active, "probe_remote_health", AsyncMock(return_value=None)),
        patch.object(manager, "_publish_routes") as publish,
    ):
        refreshed = asyncio.run(manager.refresh_remote_statuses())

    assert refreshed == {}
    assert active.remote_running is True
    assert active.status == "loaded"
    publish.assert_not_called()


def test_remote_status_refresh_applies_confirmed_offline_value():
    manager = ModelManager(gpu_count=1)
    active = ActiveModel("remote", dict(REMOTE), port=None, gpu=None)
    active.remote_running = True
    active.status = "loaded"
    manager.active["remote"] = active

    with (
        patch.object(mm, "registry", _Registry()),
        patch.object(active, "probe_remote_health", AsyncMock(return_value=False)),
        patch.object(manager, "_publish_routes") as publish,
    ):
        refreshed = asyncio.run(manager.refresh_remote_statuses())

    assert refreshed == {"remote": "unloaded"}
    assert active.remote_running is False
    assert active.status == "unloaded"
    publish.assert_called_once_with()


def test_remote_status_refresh_recovers_untracked_healthy_resident():
    manager = ModelManager(gpu_count=1)

    with (
        patch.object(mm, "registry", _Registry()),
        patch.object(
            ActiveModel,
            "probe_remote_health",
            AsyncMock(return_value=True),
        ),
        patch.object(manager, "_publish_routes") as publish,
    ):
        refreshed = asyncio.run(manager.refresh_remote_statuses())

    assert refreshed == {"remote": "loaded"}
    assert manager.active["remote"].remote_running is True
    assert manager.active["remote"].status == "loaded"
    publish.assert_called_once_with()


def test_remote_status_refresh_tracks_loading_then_loaded():
    manager = ModelManager(gpu_count=1)
    active = ActiveModel("remote", dict(REMOTE), port=None, gpu=None)
    active.remote_running = True
    active.status = "loaded"
    manager.active["remote"] = active

    with (
        patch.object(mm, "registry", _Registry()),
        patch.object(active, "probe_remote_health", AsyncMock(side_effect=["loading", True])),
        patch.object(manager, "_publish_routes") as publish,
    ):
        first = asyncio.run(manager.refresh_remote_statuses())
        second = asyncio.run(manager.refresh_remote_statuses())

    assert first == {"remote": "loading"}
    assert second == {"remote": "loaded"}
    assert active.status == "loaded"
    assert publish.call_count == 2
