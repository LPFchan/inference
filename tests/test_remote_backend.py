"""Remote vLLM backend lifecycle, validation, and routing tests."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import grimoire.model_manager as mm
from grimoire.model_manager import ActiveModel, ModelManager
from grimoire.registry import ModelRegistry


REMOTE = {
    "backend": "vllm-remote",
    "remote-agent-url": "http://mangchi.tail.lost.plus:9700",
    "remote-model-id": "remote/model",
    "remote-url": "http://mangchi.tail.lost.plus:8001",
    "backend-model-id": "/models/remote",
    "capabilities": ["completion"],
    "ctx-size": 32768,
}


class _Response:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Client:
    calls = []
    params = []
    response = None

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, params=None):
        self.calls.append(url)
        self.params.append(params)
        return self.response or _Response({"status": "ok"})


class _AsyncResponse(_Response):
    pass


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
            "http://mangchi.tail.lost.plus:8001/v1/chat/completions"
        )
        assert asyncio.run(active.get_backend_model_id()) == "/models/remote"
        active.stop()
    assert _Client.calls == [
        "http://mangchi.tail.lost.plus:9700/models/remote%2Fmodel/load",
        "http://mangchi.tail.lost.plus:9700/models/remote%2Fmodel/unload",
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


def test_manager_failed_remote_unload_keeps_claim_for_retry():
    manager = ModelManager(gpu_count=1)
    active = ActiveModel("remote", dict(REMOTE), port=None, gpu=None)
    active.remote_running = True
    active.status = "loaded"
    manager.active["remote"] = active
    with patch.object(mm, "registry", _Registry()), \
            patch.object(mm.httpx, "Client", _Client), \
            patch.object(mm, "_publish_route_table") as publish:
        with patch.object(_Client, "post", side_effect=mm.httpx.ConnectError("agent unavailable")):
            with pytest.raises(RuntimeError, match="failed to stop remote"):
                asyncio.run(manager.stop_model("remote"))
        assert active.remote_running
        assert manager.get_active("remote") is active
        assert manager.get_status("remote") == "failed"
        assert publish.call_args.args[0] == {}

        assert asyncio.run(manager.stop_model("remote"))
        assert not active.remote_running
        assert manager.active == {}
        assert publish.call_args.args[0] == {}


def test_manager_external_unload_only_releases_claim():
    manager = ModelManager(gpu_count=1)
    cfg = {"backend": "vllm-remote", "remote-url": REMOTE["remote-url"]}
    active = ActiveModel("remote", cfg, port=None, gpu=None)
    active.remote_running = True
    active.status = "loaded"
    manager.active["remote"] = active
    with patch.object(mm, "registry", _Registry()), \
            patch.object(mm.httpx, "Client") as client, \
            patch.object(mm, "_publish_route_table") as publish:
        assert asyncio.run(manager.stop_model("remote"))
        client.assert_not_called()
        assert not active.remote_running
        assert manager.active == {}
        assert publish.call_args.args[0] == {}


def test_remote_health_probe_requires_agent_residency_and_backend_health():
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    agent_url = "http://mangchi.tail.lost.plus:9700/models/remote%2Fmodel/status"
    backend_url = "http://mangchi.tail.lost.plus:8001/health"
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
    agent_url = "http://mangchi.tail.lost.plus:9700/models/remote%2Fmodel/status"
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
    agent_url = "http://mangchi.tail.lost.plus:9700/models/remote%2Fmodel/status"
    _AsyncClient.calls = []
    _AsyncClient.responses = {
        agent_url: _AsyncResponse({"resident": True, "alive": True, "status": "loading"}),
    }

    with patch.object(mm.httpx, "AsyncClient", _AsyncClient):
        assert asyncio.run(active.probe_remote_health()) == "loading"

    assert _AsyncClient.calls == [agent_url]


def test_remote_health_probe_keeps_previous_value_when_agent_times_out():
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    agent_url = "http://mangchi.tail.lost.plus:9700/models/remote%2Fmodel/status"
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


EXTERNAL = {
    key: value for key, value in REMOTE.items()
    if key not in ("remote-agent-url", "remote-model-id")
}


def test_external_remote_has_no_agent_lifecycle(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": {"remote": EXTERNAL}, "fixed": {}}))
    registry = ModelRegistry(path=str(path), seed_path=str(tmp_path / "missing.json"))
    assert registry.validate("remote") == (True, "OK")

    _Client.calls = []
    active = ActiveModel("alias", dict(EXTERNAL), port=None, gpu=None)
    with patch.object(mm.httpx, "Client", _Client):
        active.start()
        assert active.is_running()
        active.stop()
    assert _Client.calls == []
    assert not active.is_running()


def test_external_remote_health_probe_reads_backend_health():
    active = ActiveModel("alias", dict(EXTERNAL), port=None, gpu=None)
    backend_url = "http://mangchi.tail.lost.plus:8001/health"
    _AsyncClient.calls = []
    with patch.object(mm.httpx, "AsyncClient", _AsyncClient):
        _AsyncClient.responses = {backend_url: _AsyncResponse(status_code=200)}
        assert asyncio.run(active.probe_remote_health()) is True
        _AsyncClient.responses = {backend_url: mm.httpx.ConnectError("refused")}
        assert asyncio.run(active.probe_remote_health()) is False
        _AsyncClient.responses = {backend_url: mm.httpx.ReadTimeout("slow")}
        assert asyncio.run(active.probe_remote_health()) is None
    assert _AsyncClient.calls == [backend_url] * 3


def test_remote_load_without_eviction_asks_agent_not_to_evict():
    _Client.calls, _Client.params = [], []
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    active.allow_evict = False
    with patch.object(mm.httpx, "Client", _Client):
        active.start()
    assert _Client.params == [{"evict": "false"}]


def test_remote_agent_refusal_surfaces_its_reason():
    _Client.calls, _Client.params = [], []
    _Client.response = _Response({"detail": "would unload another model"}, status_code=409)
    active = ActiveModel("alias", dict(REMOTE), port=None, gpu=None)
    try:
        with patch.object(mm.httpx, "Client", _Client):
            try:
                active.start()
            except RuntimeError as exc:
                assert str(exc) == "would unload another model"
            else:
                raise AssertionError("a refused load must raise")
    finally:
        _Client.response = None
    assert not active.is_running()


HOSTED = {
    "backend": "vllm-remote",
    "remote-url": "https://opencode.ai/zen/go",
    "remote-api-key-env": "TEST_HOSTED_KEY",
    "remote-session-header": "x-opencode-session",
    "backend-model-id": "deepseek-v4.1-flash",
    "capabilities": ["completion"],
    "ctx-size": 32768,
}


def test_registry_accepts_hosted_remote_with_base_path(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": {"hosted": HOSTED}, "fixed": {}}))
    registry = ModelRegistry(path=str(path), seed_path=str(tmp_path / "missing.json"))
    assert registry.validate("hosted") == (True, "OK")

    path.write_text(json.dumps({"models": {"hosted": dict(HOSTED, **{"remote-api-key-env": ""})}, "fixed": {}}))
    registry.reload()
    valid, reason = registry.validate("hosted")
    assert not valid
    assert "remote-api-key-env" in reason


def test_hosted_remote_headers_carry_key_and_stable_session(monkeypatch):
    monkeypatch.setenv("TEST_HOSTED_KEY", "secret")
    active = ActiveModel("alias", dict(HOSTED), port=None, gpu=None)
    assert active.backend_url("v1/chat/completions") == "https://opencode.ai/zen/go/v1/chat/completions"

    headers = active.remote_headers({"user-agent": "hermes/1.0"}, "caller-a")
    assert headers["authorization"] == "Bearer secret"
    assert headers["user-agent"] == "hermes/1.0"
    assert headers["accept-encoding"] == "identity"
    assert headers["x-opencode-session"] == active.remote_headers({}, "caller-a")["x-opencode-session"]
    assert headers["x-opencode-session"] != active.remote_headers({}, "caller-b")["x-opencode-session"]
    assert active.remote_headers({"x-opencode-session": "own"}, "caller-a")["x-opencode-session"] == "own"
    assert active.remote_headers()["user-agent"] == "grimoire/1.0"

    monkeypatch.delenv("TEST_HOSTED_KEY")
    with pytest.raises(RuntimeError, match="TEST_HOSTED_KEY"):
        active.remote_headers()
    assert ActiveModel("alias", dict(EXTERNAL), port=None, gpu=None).remote_headers() == {}


def test_hosted_remote_health_probe_lists_models_with_key(monkeypatch):
    monkeypatch.setenv("TEST_HOSTED_KEY", "secret")
    active = ActiveModel("alias", dict(HOSTED), port=None, gpu=None)
    models_url = "https://opencode.ai/zen/go/v1/models"
    seen = []

    class _KeyedClient(_AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(kwargs.get("headers"))

    _AsyncClient.calls = []
    _AsyncClient.responses = {models_url: _AsyncResponse(status_code=200)}
    with patch.object(mm.httpx, "AsyncClient", _KeyedClient):
        assert asyncio.run(active.probe_remote_health()) is True
    assert _AsyncClient.calls == [models_url]
    assert seen[0]["authorization"] == "Bearer secret"


OPENCODE = {key: HOSTED[key] for key in (
    "remote-url", "remote-api-key-env", "remote-session-header", "backend-model-id",
)}


def test_registry_fallback_overlays_an_agentless_remote(tmp_path):
    path = tmp_path / "models.json"
    entry = dict(EXTERNAL, fallback=OPENCODE)
    path.write_text(json.dumps({"models": {"primary": entry}, "fixed": {}}))
    registry = ModelRegistry(path=str(path), seed_path=str(tmp_path / "missing.json"))
    assert registry.validate("primary") == (True, "OK")

    for bad in ("other-model", {"remote-url": "not a url"}, dict(OPENCODE, **{"remote-agent-url": "http://a:9700"})):
        path.write_text(json.dumps({"models": {"primary": dict(entry, fallback=bad)}, "fixed": {}}))
        registry.reload()
        valid, reason = registry.validate("primary")
        assert not valid and "'fallback'" in reason


class _FallbackManager:
    def __init__(self):
        self.started = []

    async def start_model(self, name, evict=True):
        self.started.append(name)
        return ActiveModel(name, dict(EXTERNAL), port=None, gpu=None)


class _FallbackRegistry:
    def __init__(self, models):
        self.models = models

    def get(self, name):
        return self.models.get(name)


@pytest.mark.parametrize("error", [
    mm.httpx.ConnectError("refused"),
    mm.httpx.ConnectTimeout("offline"),
    TimeoutError("not ready"),
])
def test_unreachable_model_runs_on_its_fallback(error):
    import grimoire.entrypoint as ep

    manager = _FallbackManager()
    registry = _FallbackRegistry({"primary": dict(EXTERNAL, fallback=OPENCODE)})

    async def attempt(active):
        if active.cfg["remote-url"] == EXTERNAL["remote-url"]:
            raise error
        assert active.name == "primary" and "fallback" not in active.cfg
        return active.backend_url("v1/chat/completions")

    hosted_url = "https://opencode.ai/zen/go/v1/chat/completions"
    request = type("Req", (), {"headers": {}})()
    with patch.object(ep, "manager", manager), patch.object(ep, "registry", registry), \
         patch.object(ep, "_fallback_until", {}), patch.object(ep.time, "monotonic", return_value=1000.0):
        assert asyncio.run(ep._with_fallback("primary", request, attempt)) == hosted_url
        # Inside the hold the fallback serves directly; after it the model is tried again.
        assert asyncio.run(ep._with_fallback("primary", request, attempt)) == hosted_url
        assert manager.started == ["primary"]
        ep.time.monotonic.return_value = 1000.0 + ep.FALLBACK_HOLD_SECONDS
        assert asyncio.run(ep._with_fallback("primary", request, attempt)) == hosted_url
    assert manager.started == ["primary", "primary"]


def test_unreachable_model_without_fallback_and_other_errors_raise():
    import grimoire.entrypoint as ep

    manager = _FallbackManager()
    registry = _FallbackRegistry({"lonely": dict(EXTERNAL), "primary": dict(EXTERNAL, fallback=OPENCODE)})
    request = type("Req", (), {"headers": {}})()

    async def unreachable(active):
        raise mm.httpx.ConnectError("refused")

    async def broken(active):
        raise mm.httpx.ReadError("dropped mid-reply")

    with patch.object(ep, "manager", manager), patch.object(ep, "registry", registry), \
         patch.object(ep, "_fallback_until", {}):
        with pytest.raises(mm.httpx.ConnectError):
            asyncio.run(ep._with_fallback("lonely", request, unreachable))
        with pytest.raises(mm.httpx.ReadError):
            asyncio.run(ep._with_fallback("primary", request, broken))
    assert manager.started == ["lonely", "primary"]


def test_refused_model_with_fallback_does_not_wait_for_startup():
    active = ActiveModel("alias", dict(EXTERNAL, fallback=OPENCODE), port=None, gpu=None)
    active.remote_running = True
    health_url = "http://mangchi.tail.lost.plus:8001/health"
    _AsyncClient.calls = []
    _AsyncClient.responses = {health_url: mm.httpx.ConnectError("refused")}
    with patch.object(mm.httpx, "AsyncClient", _AsyncClient):
        with pytest.raises(mm.httpx.ConnectError):
            asyncio.run(active.wait_ready(timeout=30))
    assert _AsyncClient.calls == [health_url]
