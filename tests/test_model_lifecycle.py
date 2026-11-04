"""Readiness and cleanup ownership regressions without backend operations."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

import grimoire.model_manager as mm
import grimoire.routes.models as routes
from grimoire.presets import PresetManager


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.touch()
    configs = {
        name: {
            "file": str(model_file),
            "chat-template-kwargs": {"reasoning_effort": effort},
        }
        for name, effort in (("owner", "high"), ("alias", "low"))
    }
    registry = Mock()
    registry.resolve.side_effect = lambda name: name if name in configs else None
    registry.get.side_effect = configs.get
    registry.get_fixed_gpu.return_value = None
    registry.get_family_defaults.return_value = {}
    registry.validate.return_value = (True, "OK")
    registry.swap_fixed.return_value = {}
    monkeypatch.setattr(mm, "registry", registry)
    manager = mm.ModelManager(gpu_count=1)
    allocate = AsyncMock(return_value=(mm.GpuPlacement((0,)), []))
    monkeypatch.setattr(manager, "_allocate_gpu", allocate)
    monkeypatch.setattr(manager, "_find_available_port_excluding", lambda *args: 8001)
    publish = Mock()
    monkeypatch.setattr(mm, "_publish_route_table", publish)
    monkeypatch.setattr(routes, "_get_manager", lambda: manager)
    monkeypatch.setattr(routes, "require_admin", lambda request: None)
    return SimpleNamespace(
        manager=manager, registry=registry, configs=configs,
        allocate=allocate, publish=publish, tmp_path=tmp_path,
    )


@pytest.mark.parametrize("backend,requested", [
    ("gpu", "owner"), ("gpu", "alias"),
    ("cpu", "owner"), ("cpu", "alias"), ("remote", "owner"),
])
@pytest.mark.parametrize("readiness_fails", [False, True])
def test_concurrent_start_waits_for_readiness(
    lifecycle, monkeypatch, requested, backend, readiness_fails
):
    manager = lifecycle.manager
    for cfg in lifecycle.configs.values():
        if backend == "cpu":
            cfg["cpu-only"] = True
        elif backend == "remote":
            cfg.pop("file")
            cfg.update({"backend": "vllm-remote", "remote-url": "http://backend.invalid"})

    async def scenario():
        waiting = asyncio.Event()
        release = asyncio.Event()
        launches = []

        def start(active):
            launches.append(active)
            # A real Popen/remote claim is alive before health readiness.
            active.remote_running = True
            active.process = Mock()
            active.process.poll.return_value = None

        def stop(active):
            active.remote_running = False
            active.process = None

        async def wait_ready(active, timeout):
            assert active.is_running()
            waiting.set()
            await release.wait()
            if readiness_fails:
                raise RuntimeError("readiness failed")

        monkeypatch.setattr(mm.ActiveModel, "start", start)
        monkeypatch.setattr(mm.ActiveModel, "stop", stop)
        monkeypatch.setattr(mm.ActiveModel, "wait_ready", wait_ready)

        first = asyncio.create_task(manager.start_model("owner"))
        await asyncio.wait_for(waiting.wait(), timeout=2)
        owner = manager.active["owner"]
        assert owner.is_running()
        assert owner.status == "loading"
        assert manager.get_status("owner") == "loading"
        assert manager._build_route_table() == {}
        second = asyncio.create_task(manager.start_model(requested))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)

        if readiness_fails:
            # The serialized second caller may retry, but cannot reuse a
            # process whose readiness failed.
            assert all(isinstance(result, RuntimeError) for result in results)
            assert all("readiness failed" in str(result) for result in results)
            assert manager._build_route_table() == {}
        else:
            assert results[0] is results[1] is owner
            assert len(launches) == 1
            assert owner.status == "loaded"
            assert list(manager.active) == ["owner"]
            # Healthy direct and compatible aliases must bypass a busy lock.
            async with manager._lock:
                for name in (("owner",) if backend == "remote" else ("owner", "alias")):
                    assert await asyncio.wait_for(manager.start_model(name), 0.5) is owner
            assert lifecycle.publish.call_args.args[0] == manager._build_route_table()
        if backend != "gpu":
            lifecycle.allocate.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("requested", ["owner", "alias"])
@pytest.mark.parametrize("webui", [False, True])
@pytest.mark.parametrize("backend_alive", [False, True])
def test_failed_public_unload_retains_owner_and_allows_retry(
    lifecycle, requested, webui, backend_alive
):
    manager = lifecycle.manager
    active = mm.ActiveModel("owner", lifecycle.configs["owner"], 8001, mm.GpuPlacement((0,)))
    active.status = "loaded"
    active.process = Mock()
    active.process.poll.return_value = None
    manager.active["owner"] = active
    manager._publish_routes()
    assert "owner" in lifecycle.publish.call_args.args[0]

    def cleaned_up():
        active.process = None

    def failed_cleanup():
        if not backend_alive:
            active.process = None
        raise RuntimeError("cleanup failed")

    active.stop = Mock(side_effect=failed_cleanup)

    async def unload():
        if webui:
            request = SimpleNamespace(json=AsyncMock(return_value={"model": requested}))
            return await routes.models_unload(request)
        return await routes.stop_model_endpoint(requested, None)

    with pytest.raises(HTTPException) as error:
        asyncio.run(unload())
    assert error.value.status_code == 409
    assert "cleanup failed" in error.value.detail
    assert active.is_running() == backend_alive
    assert manager.active["owner"] is manager.get_active(requested) is active
    assert manager.get_status(requested) == "failed"
    assert manager.list_active() == (["owner"] if backend_alive else [])
    assert lifecycle.publish.call_args.args[0] == {}

    active.stop.side_effect = cleaned_up
    assert asyncio.run(unload()) == {"status": "stopped", "model": requested}
    assert active.stop.call_count == 2
    assert manager.active == {}
    assert manager.get_active(requested) is None
    assert lifecycle.publish.call_args.args[0] == {}


def test_healthy_unload_removes_proxy_route(lifecycle):
    manager = lifecycle.manager
    active = mm.ActiveModel("owner", lifecycle.configs["owner"], 8001, mm.GpuPlacement((0,)))
    active.status = "loaded"
    active.process = Mock()
    active.process.poll.return_value = None
    active.stop = Mock()
    manager.active["owner"] = active
    manager._publish_routes()
    assert "owner" in lifecycle.publish.call_args.args[0]
    assert asyncio.run(manager.stop_model("owner"))
    active.stop.assert_called_once()
    assert manager.active == {}
    assert lifecycle.publish.call_args.args[0] == {}


def test_preset_reports_partial_failure_and_retains_failed_stop(lifecycle):
    manager = lifecycle.manager
    active = mm.ActiveModel("owner", lifecycle.configs["owner"], 8001, mm.GpuPlacement((0,)))
    active.status = "loaded"
    active.process = Mock()
    active.process.poll.return_value = None
    active.stop = Mock(side_effect=RuntimeError("backend still running"))
    manager.active["owner"] = active
    presets = PresetManager(state_dir=str(lifecycle.tmp_path))
    presets.upsert("training", "", [], {}, gpus=[])

    result = asyncio.run(presets.activate("training", manager, lifecycle.registry))

    assert result["active"] == "training"
    assert result["stopped"] == result["started"] == result["failed"] == []
    assert len(result["warnings"]) == 1
    assert "Failed to stop owner" in result["warnings"][0]
    assert manager.get_active("owner") is active
    assert manager.get_status("owner") == "failed"
    assert manager.gpu_mask == set()
