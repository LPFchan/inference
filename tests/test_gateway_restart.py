"""POST /restart keeps models on other machines loaded; a plain stop does not."""

import asyncio

from fastapi.testclient import TestClient

from grimoire import config, entrypoint
from grimoire.model_manager import ModelManager


class _Active:
    def __init__(self, backend):
        self.cfg = {"backend": backend}
        self.stopped = False

    def stop(self):
        self.stopped = True


def _manager():
    mgr = ModelManager(gpu_count=1)
    mgr._publish_routes = lambda: None
    mgr.active = {"local": _Active("llama"), "remote": _Active("vllm-remote")}
    return mgr, dict(mgr.active)


def test_shutdown_unloads_remote_models():
    mgr, models = _manager()
    asyncio.run(mgr.shutdown())
    assert models["local"].stopped and models["remote"].stopped


def test_restart_keeps_remote_models_loaded():
    mgr, models = _manager()
    asyncio.run(mgr.shutdown(keep_remote=True))
    assert models["local"].stopped
    assert not models["remote"].stopped


def test_restart_route_is_admin_only(monkeypatch):
    exits = []
    monkeypatch.setattr(entrypoint, "_exit_soon", lambda: exits.append(True))
    monkeypatch.setattr(entrypoint.app.state, "restarting", False, raising=False)
    client = TestClient(entrypoint.app)
    headers = {config.INTERNAL_AUTH_SUB_HEADER: "someone"}

    user = client.post("/restart", headers={**headers, config.INTERNAL_AUTH_ROLE_HEADER: "user"})
    assert user.status_code == 403 and exits == []

    admin = client.post("/restart", headers={**headers, config.INTERNAL_AUTH_ROLE_HEADER: "administrator"})
    assert admin.status_code == 200 and exits == [True]
    assert entrypoint.app.state.restarting is True


def test_lifespan_probes_remote_backends_on_boot(monkeypatch):
    calls = []

    async def _fake_refresh():
        calls.append("refresh")
        return {"Inkling-2-Small": "loaded"}

    async def _fake_shutdown(keep_remote=False):
        calls.append(("shutdown", keep_remote))

    async def _noop_sampler():
        await asyncio.sleep(60)

    monkeypatch.setattr(entrypoint, "init_proxy_client", lambda: None)
    monkeypatch.setattr(entrypoint, "close_proxy_client", lambda: asyncio.sleep(0))
    monkeypatch.setattr(entrypoint, "telemetry_sampler", _noop_sampler)
    monkeypatch.setattr(entrypoint.usage_store, "import_legacy_token_stats", lambda *a, **kw: 0)
    monkeypatch.setattr(entrypoint.presets, "get_active_name", lambda: None)
    monkeypatch.setattr(entrypoint.registry, "snapshot", lambda: {"models": {}})
    monkeypatch.setattr(entrypoint.manager, "refresh_remote_statuses", _fake_refresh)
    monkeypatch.setattr(entrypoint.manager, "shutdown", _fake_shutdown)
    monkeypatch.setattr(entrypoint.app.state, "initial_model", None, raising=False)
    monkeypatch.setattr(entrypoint.app.state, "restarting", False, raising=False)

    async def _run():
        async with entrypoint.lifespan(entrypoint.app):
            assert calls == ["refresh"]

    asyncio.run(_run())
    assert calls == ["refresh", ("shutdown", False)]


def test_lifespan_boot_survives_remote_probe_error(monkeypatch):
    async def _failing_refresh():
        raise RuntimeError("remote host unreachable")

    async def _fake_shutdown(keep_remote=False):
        return None

    async def _noop_sampler():
        await asyncio.sleep(60)

    monkeypatch.setattr(entrypoint, "init_proxy_client", lambda: None)
    monkeypatch.setattr(entrypoint, "close_proxy_client", lambda: asyncio.sleep(0))
    monkeypatch.setattr(entrypoint, "telemetry_sampler", _noop_sampler)
    monkeypatch.setattr(entrypoint.usage_store, "import_legacy_token_stats", lambda *a, **kw: 0)
    monkeypatch.setattr(entrypoint.presets, "get_active_name", lambda: None)
    monkeypatch.setattr(entrypoint.registry, "snapshot", lambda: {"models": {}})
    monkeypatch.setattr(entrypoint.manager, "refresh_remote_statuses", _failing_refresh)
    monkeypatch.setattr(entrypoint.manager, "shutdown", _fake_shutdown)
    monkeypatch.setattr(entrypoint.app.state, "initial_model", None, raising=False)
    monkeypatch.setattr(entrypoint.app.state, "restarting", False, raising=False)

    entered = []

    async def _run():
        async with entrypoint.lifespan(entrypoint.app):
            entered.append(True)

    asyncio.run(_run())
    assert entered == [True]
