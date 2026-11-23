"""Regression tests for settings and downloads routes."""

import asyncio
import json

from starlette.requests import Request

from grimoire.routes import downloads as downloads_route
from grimoire.routes import settings as settings_route
from grimoire.settings import SettingsStore


def _json_request(method: str, path: str, payload: dict, headers: dict[str, str] | None = None) -> Request:
    body = json.dumps(payload).encode("utf-8")
    raw_headers = [(b"content-type", b"application/json")]
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("utf-8"), value.encode("utf-8")))

    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": method, "path": path, "headers": raw_headers}, receive)


def test_put_settings_accepts_null_and_non_string_values(tmp_path, monkeypatch):
    store = SettingsStore(path=str(tmp_path / "settings.sqlite3"))
    monkeypatch.setattr(settings_route, "settings_store", store)
    monkeypatch.setattr(settings_route, "require_api", lambda _req: ("user-1", "hash-1"))

    payload = {
        "config": json.dumps({"systemMessage": "new prompt"}),
        "theme": "dark",
        "defaultModel": None,
        "flag": True,
    }
    req = _json_request("PUT", "/settings", payload, {"X-Grimoire-Expected-Sub": "user-1"})
    result = asyncio.run(settings_route.put_settings(req))

    assert result == {"status": "ok"}
    saved = store.get_all("hash-1")
    assert json.loads(saved["config"])["systemMessage"] == "new prompt"
    assert saved["theme"] == "dark"
    assert saved["defaultModel"] == "null"
    assert saved["flag"] == "true"


def test_migrate_settings_accepts_non_string_values(tmp_path, monkeypatch):
    store = SettingsStore(path=str(tmp_path / "settings.sqlite3"))
    monkeypatch.setattr(settings_route, "settings_store", store)
    monkeypatch.setattr(settings_route, "require_api", lambda _req: ("user-2", "hash-2"))

    payload = {"config": {"systemMessage": "migrated"}, "count": 3}
    req = _json_request("POST", "/settings/migrate", payload, {"X-Grimoire-Expected-Sub": "user-2"})
    result = asyncio.run(settings_route.migrate_settings(req))

    assert result == {"status": "migrated", "count": 2}
    saved = store.get_all("hash-2")
    assert json.loads(saved["config"]) == {"systemMessage": "migrated"}
    assert saved["count"] == "3"


def test_downloads_manifest_and_file_serving(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads_route, "DOWNLOADS_DIR", tmp_path)
    monkeypatch.setattr(downloads_route, "require_api", lambda _req: ("user-1", "hash-1"))

    manifest = {"version": "1.0.0", "mac": {"filename": "Chat_1.0.0_aarch64.dmg"}}
    (tmp_path / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "Chat_1.0.0_aarch64.dmg").write_bytes(b"fake-dmg")

    req = Request({"type": "http", "method": "GET", "path": "/downloads/latest.json", "headers": []})
    assert asyncio.run(downloads_route.downloads_latest(req)) == manifest

    file_req = Request({"type": "http", "method": "GET", "path": "/downloads/Chat_1.0.0_aarch64.dmg", "headers": []})
    response = asyncio.run(downloads_route.downloads_file(file_req, "Chat_1.0.0_aarch64.dmg"))
    assert response.path == tmp_path / "Chat_1.0.0_aarch64.dmg"
