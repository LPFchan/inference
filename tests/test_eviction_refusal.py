"""A non-administrator's load that would push another model out is a 409."""

import pytest
from fastapi.testclient import TestClient

from grimoire import config, entrypoint
from grimoire.model_manager import LoadRefused


USER = {
    config.INTERNAL_AUTH_SUB_HEADER: "regular-user",
    config.INTERNAL_AUTH_ROLE_HEADER: "user",
}


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/completions"])
def test_public_routes_answer_a_refused_load_with_409(monkeypatch, path):
    seen = []

    async def refuse(model, _preset_bypass=False, *, evict=True):
        seen.append(evict)
        raise LoadRefused("Loading 'cold' would unload warm; only an administrator can do that")

    monkeypatch.setattr(entrypoint.registry, "resolve", lambda name: name)
    monkeypatch.setattr(entrypoint.manager, "start_model", refuse)

    response = TestClient(entrypoint.app).post(path, json={"model": "cold", "messages": []}, headers=USER)

    assert response.status_code == 409
    assert "only an administrator" in response.json()["detail"]
    assert seen == [False]
