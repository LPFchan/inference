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
