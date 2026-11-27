"""Tests for POST /history/{conversation_id}/title and GET /auth/services."""

import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from grimoire import auth, config, entrypoint, proxy_app
from grimoire.history import HistoryStore
from grimoire.model_manager import ActiveModel
from grimoire.registry import ModelRegistry
from grimoire.routes import history as history_route


def _raw_request(headers=None, method="GET", path="/"):
    raw = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    return Request({"type": "http", "method": method, "path": path, "query_string": b"", "headers": raw})


class _FakeManager:
    def __init__(self, active_models=None):
        self._active = dict(active_models or {})
        self.start_calls = []

    def get_active(self, model_name):
        return self._active.get(model_name)

    def list_active(self):
        return list(self._active.keys())

    def pick_replica(self, model_name, conversation_key=None, caller_key=None):
        return None

    async def start_replica(self, model_name, replica=None, *, evict=True):
        self.start_calls.append((model_name, replica, evict))
        active = self._active.get(model_name)
        if active is not None:
            return active
        raise httpx.ConnectError(f"{model_name} unavailable")


def _make_registry(tmp_path, models, fixed=None, family_defaults=None):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": models,
                "fixed": fixed or {},
                "family_defaults": family_defaults or {},
            }
        ),
        encoding="utf-8",
    )
    return ModelRegistry(path=str(path), seed_path="")


@pytest.fixture
def title_env(tmp_path, monkeypatch):
    store = HistoryStore(str(tmp_path / "history.sqlite3"))
    reg = _make_registry(
        tmp_path,
        models={
            "Inkling-2-Small": {
                "backend": "vllm-remote",
                "remote-url": "http://spark2.test:8889",
                "backend-model-id": "Inkling-2-Small",
                "capabilities": ["completion", "multimodal"],
                "family": "inkling",
            },
            "GLM-5.3": {
                "backend": "vllm-remote",
                "remote-url": "https://hosted.test/api",
                "remote-api-key-env": "TEST_HOSTED_KEY",
                "backend-model-id": "glm-5.3",
                "capabilities": ["completion"],
                "family": "glm",
            },
            "gemma-4-31B": {
                "backend": "llama",
                "file": "gemma4.gguf",
                "capabilities": ["completion"],
                "family": "gemma4",
            },
            "eastself-embedder-0.6B": {
                "backend": "llama",
                "file": "embed.gguf",
                "capabilities": ["embedding"],
            },
            "eastself-reranker-0.6B": {
                "backend": "llama",
                "file": "rerank.gguf",
                "capabilities": ["reranking"],
            },
            "image-only": {
                "backend": "vllm-remote",
                "remote-url": "http://img.test:9710",
                "capabilities": ["image-generation"],
            },
        },
        fixed={"eastself-embedder-0.6B": 0, "eastself-reranker-0.6B": 0},
        family_defaults={
            "inkling": {"vllm-reasoning-efforts": ["low", "medium", "high", "max"]},
            "glm": {"vllm-reasoning-efforts": ["low", "high", "max"]},
        },
    )
    inkling_active = ActiveModel("Inkling-2-Small", reg.get("Inkling-2-Small"), port=None, gpu=None)
    inkling_active.backend_model_id = "Inkling-2-Small"
    inkling_active.remote_running = True

    embed_active = ActiveModel("eastself-embedder-0.6B", reg.get("eastself-embedder-0.6B"), port=8100, gpu=0)
    rerank_active = ActiveModel("eastself-reranker-0.6B", reg.get("eastself-reranker-0.6B"), port=8101, gpu=0)

    mgr = _FakeManager(
        {
            "eastself-embedder-0.6B": embed_active,
            "eastself-reranker-0.6B": rerank_active,
            "Inkling-2-Small": inkling_active,
        }
    )

    monkeypatch.setattr(history_route, "_history_store", lambda: store)
    monkeypatch.setattr(history_route, "_get_registry", lambda: reg)
    monkeypatch.setattr(history_route, "_get_manager", lambda: mgr)

    client = TestClient(entrypoint.app)
    headers = {
        config.INTERNAL_AUTH_SUB_HEADER: "user-1",
        config.INTERNAL_AUTH_ROLE_HEADER: "administrator",
    }
    return store, reg, mgr, client, headers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"SQLite Lock Debugging"', "SQLite Lock Debugging"),
        ("Title: 'Async Queue Fix.'", "Async Queue Fix"),
        ("Subject: **Deploy Script Cleanup**", "Deploy Script Cleanup"),
        ("Topic: `Docker Cache Tuning`", "Docker Cache Tuning"),
        ("### FastAPI Route Testing.", "FastAPI Route Testing"),
        ("<think>Thinking about a good title...</think>\n\nPython Packaging Guide\nSecond line ignored", "Python Packaging Guide"),
        ("<think>unclosed thinking that ran out of tokens", None),
        ("파이썬 비동기 디버깅 가이드", "파이썬 비동기 디버깅 가이드"),
        ("A", None),
        ("   \n  ", None),
        ("", None),
        ("x" * 120, "x" * 80),
        ("  Multiple   spaces\tand   tabs.  ", "Multiple spaces and tabs"),
    ],
)
def test_clean_generated_title_edge_cases(raw, expected):
    assert history_route._clean_generated_title(raw) == expected


def test_title_generation_tree_extracts_clean_text_and_preserves_last_modified(title_env, monkeypatch):
    store, _, _, client, headers = title_env
    user_hash = auth.account_hash("user-1")

    store.upsert_conversation_tree(
        user_hash,
        {"id": "conv-1", "name": "New chat", "lastModified": 123456789, "model": "Inkling-2-Small"},
    )
    store.create_message_branch(
        user_hash,
        "conv-1",
        {"id": "m-root", "parent": None, "role": "system", "type": "root", "content": ""},
    )
    multimodal_user = json.dumps(
        [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}},
            {"type": "text", "text": "Why does pytest hang on TestClient?"},
        ]
    )
    store.create_message_branch(
        user_hash,
        "conv-1",
        {"id": "m-u1", "parent": "m-root", "role": "user", "content": multimodal_user},
    )
    # Tool-call assistant row with empty content, followed by tool result, then real answer
    store.create_message_branch(
        user_hash,
        "conv-1",
        {"id": "m-a1-tool", "parent": "m-u1", "role": "assistant", "content": "", "toolCalls": "[]"},
    )
    store.create_message_branch(
        user_hash,
        "conv-1",
        {"id": "m-t1", "parent": "m-a1-tool", "role": "tool", "content": "tool raw output"},
    )
    store.create_message_branch(
        user_hash,
        "conv-1",
        {
            "id": "m-a2",
            "parent": "m-t1",
            "role": "assistant",
            "content": "Because the background task was not cancelled.",
            "model": "Inkling-2-Small",
        },
    )
    store.patch_conversation_tree(user_hash, "conv-1", {"lastModified": 123456789})

    seen = {}

    async def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["json"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": 'Title: "Pytest TestClient Hang Fix."'}}]},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(history_route, "get_proxy_client", lambda: http_client)

    resp = client.post("/history/conv-1/title", headers=headers)
    asyncio.run(http_client.aclose())

    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": "conv-1", "title": "Pytest TestClient Hang Fix"}
    assert seen["url"] == "http://spark2.test:8889/v1/chat/completions"
    assert "x-grimoire-conversation-id" not in seen["headers"]
    assert "x-grimoire-kv-conversation-id" not in seen["headers"]
    assert seen["json"]["stream"] is False
    assert seen["json"]["reasoning_effort"] == "none"
    assert seen["json"]["max_tokens"] == 48
    assert "tools" not in seen["json"]
    prompt = seen["json"]["messages"][0]["content"]
    assert "Why does pytest hang on TestClient?" in prompt
    assert "Because the background task was not cancelled." in prompt
    assert "iVBORw0KGgo" not in prompt
    assert "tool raw output" not in prompt

    saved = store.get_conversation_tree(user_hash, "conv-1")
    assert saved["name"] == "Pytest TestClient Hang Fix"
    assert saved["lastModified"] == 123456789


def test_title_generation_flat_legacy_conversation(title_env, monkeypatch):
    store, _, _, client, headers = title_env
    user_hash = auth.account_hash("user-1")

    conv = store.create_conversation(
        user_hash,
        title="New chat",
        model="Inkling-2-Small",
        messages=[
            {"role": "user", "content": "How do I configure nginx reverse proxy?"},
            {"role": "assistant", "content": "Use proxy_pass inside a location block."},
        ],
    )
    assert store.get_conversation_tree(user_hash, conv["id"])["currNode"] is None

    async def fake_completion(model_name, prompt_text, caller_hash):
        assert model_name == "Inkling-2-Small"
        assert "nginx reverse proxy" in prompt_text
        assert "proxy_pass" in prompt_text
        return "Nginx Reverse Proxy Setup"

    monkeypatch.setattr(history_route, "_call_title_completion", fake_completion)

    resp = client.post(f"/history/{conv['id']}/title", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": "Nginx Reverse Proxy Setup"}


def test_title_guards_skip_custom_and_in_flight_rename_and_allow_force(title_env, monkeypatch):
    store, _, _, client, headers = title_env
    user_hash = auth.account_hash("user-1")

    conv = store.create_conversation(
        user_hash,
        title="Already Named",
        model="Inkling-2-Small",
        messages=[{"role": "user", "content": "Hello world"}],
    )

    called = []

    async def fake_completion(model_name, prompt_text, caller_hash):
        called.append(model_name)
        return "Generated Title"

    monkeypatch.setattr(history_route, "_call_title_completion", fake_completion)

    # 1. Existing custom title with force=False skips LLM call completely
    resp = client.post(f"/history/{conv['id']}/title", json={}, headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": None, "skipped": True}
    assert called == []

    # 2. force=True overwrites custom title
    resp = client.post(f"/history/{conv['id']}/title", json={"force": True}, headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": "Generated Title"}
    assert store.get_conversation_tree(user_hash, conv["id"])["name"] == "Generated Title"

    # 3. expected_title mismatch before LLM call skips
    resp = client.post(
        f"/history/{conv['id']}/title",
        json={"expected_title": "New chat"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": None, "skipped": True}

    # 4. In-flight manual rename during LLM call is NOT clobbered
    conv2 = store.create_conversation(
        user_hash,
        title="New chat",
        model="Inkling-2-Small",
        messages=[{"role": "user", "content": "Build a parser"}],
    )

    async def racing_completion(model_name, prompt_text, caller_hash):
        store.patch_conversation_tree(user_hash, conv2["id"], {"name": "User Renamed Midflight"})
        return "Stale Auto Title"

    monkeypatch.setattr(history_route, "_call_title_completion", racing_completion)
    resp = client.post(
        f"/history/{conv2['id']}/title",
        json={"expected_title": "New chat"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv2["id"], "title": None, "skipped": True}
    assert store.get_conversation_tree(user_hash, conv2["id"])["name"] == "User Renamed Midflight"


def test_model_resolution_filters_non_completion_and_never_evicts_local_gpu(title_env, monkeypatch):
    store, reg, mgr, client, headers = title_env
    user_hash = auth.account_hash("user-1")

    conv = store.create_conversation(
        user_hash,
        title="New chat",
        model=None,
        messages=[{"role": "user", "content": "Explain turboquant"}],
    )

    # Rejects explicit embedder, reranker, image-gen, unknown model with 400
    for bad_model in ("eastself-embedder-0.6B", "eastself-reranker-0.6B", "image-only", "no-such-model"):
        resp = client.post(f"/history/{conv['id']}/title", json={"model": bad_model}, headers=headers)
        assert resp.status_code == 400

    # Unloaded local GPU model (gemma-4-31B) returns title=None and NEVER calls start_replica
    resp = client.post(f"/history/{conv['id']}/title", json={"model": "gemma-4-31B"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": None}
    assert mgr.start_calls == []

    # Fallback when conv has no model skips fixed embedder/reranker and selects Inkling-2-Small
    seen_model = []

    async def capture_completion(model_name, prompt_text, caller_hash):
        seen_model.append(model_name)
        return "TurboQuant Overview"

    monkeypatch.setattr(history_route, "_call_title_completion", capture_completion)
    resp = client.post(f"/history/{conv['id']}/title", headers=headers)
    assert resp.status_code == 200
    assert seen_model == ["Inkling-2-Small"]
    assert resp.json() == {"conversation_id": conv["id"], "title": "TurboQuant Overview"}


def test_hosted_model_without_none_uses_low_effort_and_higher_max_tokens(title_env, monkeypatch):
    store, reg, mgr, client, headers = title_env
    user_hash = auth.account_hash("user-1")
    monkeypatch.setenv("TEST_HOSTED_KEY", "secret-key")

    glm_active = ActiveModel("GLM-5.3", reg.get("GLM-5.3"), port=None, gpu=None)
    glm_active.backend_model_id = "glm-5.3"
    mgr._active["GLM-5.3"] = glm_active

    conv = store.create_conversation(
        user_hash,
        title="New chat",
        model="GLM-5.3",
        messages=[{"role": "user", "content": "Compare Rust and Go"}],
    )

    seen = {}

    async def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["json"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Rust vs Go Comparison"}}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(history_route, "get_proxy_client", lambda: http_client)

    resp = client.post(f"/history/{conv['id']}/title", headers=headers)
    asyncio.run(http_client.aclose())

    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": "Rust vs Go Comparison"}
    assert seen["url"] == "https://hosted.test/api/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer secret-key"
    assert seen["json"]["reasoning_effort"] == "low"
    assert seen["json"]["max_tokens"] == 256
    assert "chat_template_kwargs" not in seen["json"]


def test_local_llama_backend_sets_enable_thinking_false(title_env, monkeypatch):
    store, reg, mgr, client, headers = title_env
    user_hash = auth.account_hash("user-1")

    gemma_active = ActiveModel("gemma-4-31B", reg.get("gemma-4-31B"), port=8123, gpu=0)
    gemma_active.backend_model_id = "gemma-4-31B"
    mgr._active["gemma-4-31B"] = gemma_active

    conv = store.create_conversation(
        user_hash,
        title="New chat",
        model="gemma-4-31B",
        messages=[{"role": "user", "content": "Explain Raft consensus"}],
    )

    seen = {}

    async def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["json"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Raft Consensus Basics"}}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(history_route, "get_proxy_client", lambda: http_client)

    resp = client.post(f"/history/{conv['id']}/title", headers=headers)
    asyncio.run(http_client.aclose())

    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": "Raft Consensus Basics"}
    assert seen["url"] == "http://127.0.0.1:8123/v1/chat/completions"
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["json"]["max_tokens"] == 64


def test_title_error_cases_and_non_fatal_backend_failure(title_env, monkeypatch):
    store, _, _, client, headers = title_env
    user_hash = auth.account_hash("user-1")
    other_hash = auth.account_hash("user-2")

    # 404 unknown conversation
    assert client.post("/history/does-not-exist/title", headers=headers).status_code == 404

    # 404 another user's conversation
    other_conv = store.create_conversation(
        other_hash, title="New chat", messages=[{"role": "user", "content": "secret"}]
    )
    assert client.post(f"/history/{other_conv['id']}/title", headers=headers).status_code == 404

    # 400 malformed JSON and non-object JSON
    conv = store.create_conversation(
        user_hash,
        title="New chat",
        model="Inkling-2-Small",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert client.post(
        f"/history/{conv['id']}/title",
        content=b"{bad json",
        headers={**headers, "content-type": "application/json"},
    ).status_code == 400
    assert client.post(f"/history/{conv['id']}/title", json=["not", "a", "dict"], headers=headers).status_code == 400

    # 400 when conversation has no text messages
    empty_conv = store.create_conversation(user_hash, title="New chat", model="Inkling-2-Small", messages=[])
    assert client.post(f"/history/{empty_conv['id']}/title", headers=headers).status_code == 400

    # Non-fatal {"title": None} on backend HTTP 500 or network error
    async def fail_handler(_req: httpx.Request):
        return httpx.Response(500, text="boom")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(fail_handler))
    monkeypatch.setattr(history_route, "get_proxy_client", lambda: http_client)

    resp = client.post(f"/history/{conv['id']}/title", headers=headers)
    asyncio.run(http_client.aclose())
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": None}


def test_auth_services_proxies_with_browser_cookie_and_rejects_unauth_or_bearer(monkeypatch):
    seen = []

    async def handler(upstream: httpx.Request):
        seen.append((upstream.method, str(upstream.url), dict(upstream.headers)))
        if upstream.url.path == "/api/whoami":
            return httpx.Response(200, json={"sub": "7", "role": "user", "services": ["chat"]})
        if upstream.url.path == "/api/services":
            return httpx.Response(200, json={"services": [{"slug": "shares", "mcp": "https://shares.lost.plus/mcp"}]})
        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(auth, "get_proxy_client", lambda: http_client)
    monkeypatch.setattr(proxy_app, "get_proxy_client", lambda: http_client)

    # Authenticated browser session proxies to auth.lost.plus/api/services
    resp = asyncio.run(
        proxy_app.auth_services(_raw_request({"Cookie": "lp_auth=browser-secret"}, path="/auth/services"))
    )
    assert resp.status_code == 200
    assert json.loads(resp.body) == {"services": [{"slug": "shares", "mcp": "https://shares.lost.plus/mcp"}]}
    assert seen[-1][0] == "GET"
    assert seen[-1][1] == f"{config.AUTH_ORIGIN}/api/services"
    assert seen[-1][2]["cookie"] == "lp_auth=browser-secret"

    # Rejects unauthenticated request (no cookie)
    with pytest.raises(HTTPException) as unauth_exc:
        asyncio.run(proxy_app.auth_services(_raw_request(path="/auth/services")))
    assert unauth_exc.value.status_code == 401

    # Rejects Bearer credential even when cookie is present
    with pytest.raises(HTTPException) as bearer_exc:
        asyncio.run(
            proxy_app.auth_services(
                _raw_request(
                    {"Authorization": "Bearer machine-token", "Cookie": "lp_auth=browser-secret"},
                    path="/auth/services",
                )
            )
        )
    assert bearer_exc.value.status_code == 401

    asyncio.run(http_client.aclose())


def test_attachment_only_user_message_falls_back_to_assistant_reply(title_env, monkeypatch):
    store, _, _, client, headers = title_env
    user_hash = auth.account_hash("user-1")

    image_only = json.dumps(
        [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}}]
    )
    conv = store.create_conversation(
        user_hash,
        title="New chat",
        model="Inkling-2-Small",
        messages=[
            {"role": "user", "content": image_only},
            {"role": "assistant", "content": "That is a red spider lily (Lycoris radiata)."},
        ],
    )

    seen_prompts = []

    async def fake_completion(model_name, prompt_text, caller_hash):
        seen_prompts.append(prompt_text)
        return "Red Spider Lily Identification"

    monkeypatch.setattr(history_route, "_call_title_completion", fake_completion)

    resp = client.post(f"/history/{conv['id']}/title", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": "Red Spider Lily Identification"}
    assert "red spider lily" in seen_prompts[0]
    assert "/9j/4AAQ" not in seen_prompts[0]


def test_first_exchange_skips_tool_preamble_and_stops_before_turn_two(title_env, monkeypatch):
    store, _, _, client, headers = title_env
    user_hash = auth.account_hash("user-1")

    store.upsert_conversation_tree(
        user_hash,
        {"id": "conv-url", "name": "New chat", "lastModified": 999, "model": "Inkling-2-Small"},
    )
    store.create_message_branch(
        user_hash,
        "conv-url",
        {"id": "u1", "parent": None, "role": "user", "content": "https://x.com/example/status/123 what's this??"},
    )
    store.create_message_branch(
        user_hash,
        "conv-url",
        {
            "id": "a1-preamble",
            "parent": "u1",
            "role": "assistant",
            "content": "let me pull up that link first, one sec.",
            "toolCalls": json.dumps([{"id": "tc1", "type": "function", "function": {"name": "tweet_fetch", "arguments": "{}"}}]),
        },
    )
    store.create_message_branch(
        user_hash,
        "conv-url",
        {"id": "t1", "parent": "a1-preamble", "role": "tool", "content": "raw tweet json"},
    )
    store.create_message_branch(
        user_hash,
        "conv-url",
        {
            "id": "a1-final",
            "parent": "t1",
            "role": "assistant",
            "content": "That thread is a breakdown of the 2014 MH370 satellite video hoax.",
            "model": "Inkling-2-Small",
        },
    )
    store.create_message_branch(
        user_hash,
        "conv-url",
        {"id": "u2", "parent": "a1-final", "role": "user", "content": "unrelated follow-up"},
    )
    store.create_message_branch(
        user_hash,
        "conv-url",
        {"id": "a2", "parent": "u2", "role": "assistant", "content": "Turn two answer should not leak."},
    )

    seen_prompts = []

    async def fake_completion(model_name, prompt_text, caller_hash):
        seen_prompts.append(prompt_text)
        return "MH370 Video Hoax Breakdown"

    monkeypatch.setattr(history_route, "_call_title_completion", fake_completion)

    resp = client.post("/history/conv-url/title", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": "conv-url", "title": "MH370 Video Hoax Breakdown"}
    assert "MH370 satellite video hoax" in seen_prompts[0]
    assert "let me pull up that link" not in seen_prompts[0]
    assert "Turn two answer" not in seen_prompts[0]


@pytest.mark.parametrize(
    ("title", "user_text", "expected"),
    [
        ("", "Hello world", True),
        ("New chat", "Hello world", True),
        ("New Chat", "Hello world", True),
        ("Apple의 쿼런틴 제거 커맨드좀", "Apple의 쿼런틴 제거 커맨드좀", True),
        (
            "woaahhhh no way!! https://x.com/viviencheng__/st",
            "woaahhhh no way!! https://x.com/viviencheng__/status/21256310503198314",
            True,
        ),
        (
            "https://x.com/theo/status/2125177503526588877\nco",
            "https://x.com/theo/status/2125177503526588877\ncontext: extra text",
            True,
        ),
        (
            "Explain how KV cache prefix matching works in ll…",
            "Explain how KV cache prefix matching works in llama.cpp when tools change",
            True,
        ),
        (
            "Explain how KV cache prefix matching works in ll...",
            "Explain how KV cache prefix matching works in llama.cpp when tools change",
            True,
        ),
        ("MIDI의 표준 정의와 역할", "midi가 무엇의 표준이라고 말할수 있을까?", False),
        ("Custom User Title", "Explain how KV cache works", False),
    ],
)
def test_is_ungenerated_title_detection(title, user_text, expected):
    assert history_route._is_ungenerated_title(title, user_text) is expected


def test_auto_model_resolution_falls_back_from_unloaded_local_model(title_env, monkeypatch):
    store, _, mgr, client, headers = title_env
    user_hash = auth.account_hash("user-1")

    conv = store.create_conversation(
        user_hash,
        title="Old Title",
        model="gemma-4-31B",  # local model not currently active
        messages=[
            {"role": "user", "content": "Help me debug systemd timers"},
            {"role": "assistant", "content": "Check systemctl list-timers first."},
        ],
    )

    seen_models = []

    async def fake_completion(model_name, prompt_text, caller_hash):
        seen_models.append(model_name)
        return "Systemd Timer Debugging"

    monkeypatch.setattr(history_route, "_call_title_completion", fake_completion)

    # Without explicit body model, regenerate falls back to active completion model (Inkling-2-Small)
    resp = client.post(f"/history/{conv['id']}/title", json={"force": True}, headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": conv["id"], "title": "Systemd Timer Debugging"}
    assert seen_models == ["Inkling-2-Small"]
    assert mgr.start_calls == []


def test_run_title_backfill_once_backfills_ungenerated_only_and_preserves_order(title_env, monkeypatch):
    store, _, _, _, _ = title_env
    user_a = auth.account_hash("user-a")
    user_b = auth.account_hash("user-b")
    now_ms = 2_000_000_000_000
    old_ms = now_ms - 600_000  # 10 minutes ago
    fresh_ms = now_ms - 10_000  # 10 seconds ago (still active Turn 1)

    # 1. Empty conversation (no messages) -> ignored by SQL filter
    store.upsert_conversation_tree(
        user_a, {"id": "c-empty", "name": "New chat", "lastModified": old_ms, "model": "Inkling-2-Small"}
    )

    # 2. Old conversation with 48-char raw prompt placeholder and unloaded local model -> backfilled
    long_prompt = "do you have any idea why they decided to train with 384 experts instead of 128?"
    store.upsert_conversation_tree(
        user_a, {"id": "c-prefix", "name": long_prompt[:48], "lastModified": old_ms, "model": "gemma-4-31B"}
    )
    store.create_message_branch(
        user_a, "c-prefix", {"id": "m-p-u", "parent": None, "role": "user", "content": long_prompt}
    )
    store.create_message_branch(
        user_a, "c-prefix", {"id": "m-p-a", "parent": "m-p-u", "role": "assistant", "content": "Fine-grained MoE improves routing efficiency."}
    )
    store.patch_conversation_tree(user_a, "c-prefix", {"lastModified": old_ms})

    # 3. Old conversation under another user with exact short prompt title -> backfilled
    short_prompt = "Apple의 쿼런틴 제거 커맨드좀"
    store.upsert_conversation_tree(
        user_b, {"id": "c-exact", "name": short_prompt, "lastModified": old_ms - 1000, "model": "Inkling-2-Small"}
    )
    store.create_message_branch(
        user_b, "c-exact", {"id": "m-e-u", "parent": None, "role": "user", "content": short_prompt}
    )
    store.create_message_branch(
        user_b, "c-exact", {"id": "m-e-a", "parent": "m-e-u", "role": "assistant", "content": "xattr -dr com.apple.quarantine ..."}
    )
    store.patch_conversation_tree(user_b, "c-exact", {"lastModified": old_ms - 1000})

    # 4. Old conversation already titled -> untouched
    store.upsert_conversation_tree(
        user_a, {"id": "c-titled", "name": "Custom Titled Chat", "lastModified": old_ms - 2000, "model": "Inkling-2-Small"}
    )
    store.create_message_branch(
        user_a, "c-titled", {"id": "m-t-u", "parent": None, "role": "user", "content": "Completely different prompt"}
    )
    store.patch_conversation_tree(user_a, "c-titled", {"lastModified": old_ms - 2000})

    # 5. Fresh conversation (< min_age_s) -> skipped so active Turn 1 is not raced
    store.upsert_conversation_tree(
        user_a, {"id": "c-fresh", "name": "New chat", "lastModified": fresh_ms, "model": "Inkling-2-Small"}
    )
    store.create_message_branch(
        user_a, "c-fresh", {"id": "m-f-u", "parent": None, "role": "user", "content": "Streaming right now"}
    )
    store.patch_conversation_tree(user_a, "c-fresh", {"lastModified": fresh_ms})

    # 6. Conversation renamed mid-flight during LLM call -> NOT overwritten
    store.upsert_conversation_tree(
        user_a, {"id": "c-race", "name": "New chat", "lastModified": old_ms - 3000, "model": "Inkling-2-Small"}
    )
    store.create_message_branch(
        user_a, "c-race", {"id": "m-r-u", "parent": None, "role": "user", "content": "Race condition check"}
    )
    store.patch_conversation_tree(user_a, "c-race", {"lastModified": old_ms - 3000})

    calls = []

    async def fake_completion(model_name, prompt_text, caller_hash):
        calls.append((model_name, prompt_text))
        if "384 experts" in prompt_text:
            return "MoE 384 Experts Architecture"
        if "쿼런틴" in prompt_text:
            return "macOS 쿼런틴 제거 명령어"
        if "Race condition check" in prompt_text:
            store.patch_conversation_tree(user_a, "c-race", {"name": "Manual Override"})
            return "Should Not Overwrite"
        return "Fallback Title"

    monkeypatch.setattr(history_route, "_call_title_completion", fake_completion)

    stats = asyncio.run(
        history_route.run_title_backfill_once(
            min_age_s=120,
            max_per_run=10,
            step_pause_s=0,
            now_ms_fn=lambda: now_ms,
        )
    )

    assert stats["updated"] == 2
    assert stats["candidates"] == 3
    assert stats["skipped"] == 1
    assert stats["failed"] == 0
    assert all(model == "Inkling-2-Small" for model, _ in calls)

    saved_prefix = store.get_conversation_tree(user_a, "c-prefix")
    assert saved_prefix["name"] == "MoE 384 Experts Architecture"
    assert saved_prefix["lastModified"] == old_ms

    saved_exact = store.get_conversation_tree(user_b, "c-exact")
    assert saved_exact["name"] == "macOS 쿼런틴 제거 명령어"
    assert saved_exact["lastModified"] == old_ms - 1000

    assert store.get_conversation_tree(user_a, "c-titled")["name"] == "Custom Titled Chat"
    assert store.get_conversation_tree(user_a, "c-fresh")["name"] == "New chat"
    assert store.get_conversation_tree(user_a, "c-race")["name"] == "Manual Override"


def test_run_title_backfill_once_caps_retries_on_repeated_failures(title_env, monkeypatch):
    store, _, _, _, _ = title_env
    user_a = auth.account_hash("user-a")
    now_ms = 2_000_000_000_000
    old_ms = now_ms - 600_000

    store.upsert_conversation_tree(
        user_a, {"id": "c-fail", "name": "New chat", "lastModified": old_ms, "model": "Inkling-2-Small"}
    )
    store.create_message_branch(
        user_a, "c-fail", {"id": "m-fail", "parent": None, "role": "user", "content": "Unparseable title case"}
    )
    store.patch_conversation_tree(user_a, "c-fail", {"lastModified": old_ms})

    attempts = 0

    async def failing_completion(model_name, prompt_text, caller_hash):
        nonlocal attempts
        attempts += 1
        return None

    monkeypatch.setattr(history_route, "_call_title_completion", failing_completion)
    failed_counts: dict[tuple[str, str], int] = {}

    for _ in range(5):
        asyncio.run(
            history_route.run_title_backfill_once(
                min_age_s=120,
                max_per_run=10,
                step_pause_s=0,
                failed_counts=failed_counts,
                now_ms_fn=lambda: now_ms,
            )
        )

    assert attempts == history_route._TITLE_BACKFILL_MAX_RETRIES
