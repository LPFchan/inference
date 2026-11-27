"""History route handlers (/history/*)."""

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import re
import time

import httpx
from fastapi import APIRouter, HTTPException, Request

from grimoire.auth import require_api
from grimoire.chat_template import apply_chat_template_kwargs
from grimoire.model_manager import ActiveModel, LoadRefused
from grimoire.proxy.client import get_proxy_client
from grimoire.registry import BACKEND_LLAMA, BACKEND_VLLM_REMOTE

router = APIRouter()
logger = logging.getLogger(__name__)

_DEFAULT_TITLE_PLACEHOLDERS = {"", "New chat", "New Chat"}
_TITLE_PLACEHOLDER_PREFIX_CHARS = 48
_TITLE_TEXT_CLAMP = 1200
_TITLE_MAX_CHARS = 80
_TITLE_TIMEOUT_S = 30.0
_TITLE_BACKFILL_INTERVAL_S = int(os.environ.get("GRIMOIRE_TITLE_BACKFILL_INTERVAL_S", "3600"))
_TITLE_BACKFILL_INITIAL_DELAY_S = int(os.environ.get("GRIMOIRE_TITLE_BACKFILL_INITIAL_DELAY_S", "60"))
_TITLE_BACKFILL_MIN_AGE_S = int(os.environ.get("GRIMOIRE_TITLE_BACKFILL_MIN_AGE_S", "120"))
_TITLE_BACKFILL_MAX_PER_RUN = int(os.environ.get("GRIMOIRE_TITLE_BACKFILL_MAX_PER_RUN", "20"))
_TITLE_BACKFILL_MAX_RETRIES = 3
_TITLE_BACKFILL_STEP_PAUSE_S = 0.25
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_TITLE_PREFIX_RE = re.compile(r"^(?:title|subject|topic)\s*:\s*", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")
_WRAP_PAIRS = (
    ("**", "**"),
    ("__", "__"),
    ("`", "`"),
    ('"', '"'),
    ("'", "'"),
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
)


def _history_store():
    from grimoire.entrypoint import history_store
    return history_store


def _get_manager():
    from grimoire.entrypoint import manager
    return manager


def _get_registry():
    from grimoire.registry import registry
    return registry


@router.get("/history")
async def list_history(request: Request):
    """List conversations for the authenticated account (tree-aware shape)."""
    _, user_hash = require_api(request)
    conversations = _history_store().list_conversations_tree(user_hash)
    return {"conversations": conversations}


@router.post("/history")
async def create_history(request: Request):
    """Create a conversation for the authenticated account.

    Webui upsert path: pass {id, name, lastModified, currNode, ...}.
    Legacy gateway path: pass {title, model, messages: [...]}.
    """
    _, user_hash = require_api(request)
    data = await request.json()
    if data.get("id") or data.get("name") is not None or data.get("lastModified") is not None:
        try:
            return _history_store().upsert_conversation_tree(user_hash, data)
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return _history_store().create_conversation(
        user_hash,
        title=data.get("title") or "New chat",
        model=data.get("model"),
        messages=data.get("messages") or [],
    )


@router.get("/history/{conversation_id}")
async def get_history(conversation_id: str, request: Request):
    """Return one server-side conversation with tree-shaped messages."""
    _, user_hash = require_api(request)
    try:
        return _history_store().get_conversation_tree(user_hash, conversation_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/history/{conversation_id}")
async def update_history(conversation_id: str, request: Request):
    """Replace metadata/messages for one server-side conversation."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return _history_store().replace_conversation(
            user_hash,
            conversation_id,
            title=data.get("title"),
            model=data.get("model"),
            messages=data.get("messages"),
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/history/{conversation_id}")
async def patch_history(conversation_id: str, request: Request):
    """Partial-update conversation metadata (webui's updateConversation)."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return _history_store().patch_conversation_tree(user_hash, conversation_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/history/{conversation_id}")
async def delete_history(conversation_id: str, request: Request):
    """Delete one conversation; pass ?with_forks=true to cascade through forks."""
    _, user_hash = require_api(request)
    with_forks = request.query_params.get("with_forks", "").lower() in {"1", "true", "yes", "on"}
    try:
        _history_store().delete_conversation_with_options(user_hash, conversation_id, delete_with_forks=with_forks)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": conversation_id}


@router.patch("/history/messages/{message_id}")
async def patch_history_message_by_id(message_id: str, request: Request):
    """Webui's updateMessage doesn't carry convId; resolve it from the message row."""
    _, user_hash = require_api(request)
    conv_id = _history_store().find_message_conversation(user_hash, message_id)
    if not conv_id:
        raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found")
    data = await request.json()
    try:
        _history_store().update_message_tree(user_hash, conv_id, message_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"updated": message_id}


@router.delete("/history/messages/{message_id}")
async def delete_history_message_by_id(message_id: str, request: Request):
    """Webui's deleteMessage doesn't carry convId; resolve it from the message row."""
    _, user_hash = require_api(request)
    conv_id = _history_store().find_message_conversation(user_hash, message_id)
    if not conv_id:
        raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found")
    cascade = request.query_params.get("cascade", "").lower() in {"1", "true", "yes", "on"}
    deleted = _history_store().delete_message_tree(user_hash, conv_id, message_id, cascade=cascade)
    return {"deleted": deleted}


@router.post("/history/{conversation_id}/messages")
async def create_history_message(conversation_id: str, request: Request):
    """Create a message branch under parent_id and update the conversation's currNode."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return _history_store().create_message_branch(user_hash, conversation_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/history/{conversation_id}/messages/{message_id}")
async def patch_history_message(conversation_id: str, message_id: str, request: Request):
    """Partial-update a message (webui's updateMessage)."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        _history_store().update_message_tree(user_hash, conversation_id, message_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"updated": message_id}


@router.delete("/history/{conversation_id}/messages/{message_id}")
async def delete_history_message(conversation_id: str, message_id: str, request: Request):
    """Delete a message; pass ?cascade=true to delete the whole subtree."""
    _, user_hash = require_api(request)
    cascade = request.query_params.get("cascade", "").lower() in {"1", "true", "yes", "on"}
    try:
        deleted = _history_store().delete_message_tree(user_hash, conversation_id, message_id, cascade=cascade)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": deleted}


@router.post("/history/{conversation_id}/fork")
async def fork_history(conversation_id: str, request: Request):
    """Fork a conversation at a specific message into a new conversation."""
    _, user_hash = require_api(request)
    data = await request.json()
    at_message_id = data.get("at_message_id") or data.get("atMessageId")
    name = data.get("name") or "Forked chat"
    include_attachments = data.get("include_attachments", data.get("includeAttachments", True))
    if not at_message_id:
        raise HTTPException(status_code=400, detail="Missing 'at_message_id'")
    try:
        return _history_store().fork_conversation(
            user_hash, conversation_id, at_message_id, name, include_attachments
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/history/import")
async def import_history(request: Request):
    """Bulk-import conversations in the webui's exported shape."""
    _, user_hash = require_api(request)
    data = await request.json()
    return _history_store().import_conversations_tree(user_hash, data)


def _extract_text_content(content) -> str:
    """Extract plain text from string or OpenAI multimodal content without leaking base64."""
    if content is None:
        return ""
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except ValueError:
                return content
            if isinstance(parsed, list):
                return _extract_text_content(parsed)
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
            elif isinstance(item, dict):
                part_type = item.get("type")
                if part_type in (None, "text", "input_text") and isinstance(item.get("text"), str):
                    if item["text"].strip():
                        parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("type") in (None, "text", "input_text") and isinstance(content.get("text"), str):
            return content["text"]
    return ""


def _active_message_path(conv: dict) -> list[dict]:
    """Walk root -> currNode when valid; fall back to chronological linear order."""
    messages = [m for m in (conv.get("messages") or []) if isinstance(m, dict)]
    if not messages:
        return []
    curr_node = conv.get("currNode")
    by_id = {m["id"]: m for m in messages if isinstance(m.get("id"), str)}
    if isinstance(curr_node, str) and curr_node in by_id:
        path: list[dict] = []
        visited: set[str] = set()
        cursor: str | None = curr_node
        while cursor is not None and cursor in by_id and cursor not in visited:
            visited.add(cursor)
            node = by_id[cursor]
            path.append(node)
            parent = node.get("parent")
            cursor = parent if isinstance(parent, str) else None
        path.reverse()
        return path
    return messages


def _has_tool_calls(msg: dict) -> bool:
    """Return True when a stored message carries one or more tool calls."""
    tc = msg.get("toolCalls") if "toolCalls" in msg else msg.get("tool_calls")
    if not tc:
        return False
    if isinstance(tc, str):
        stripped = tc.strip()
        if not stripped or stripped == "[]":
            return False
        try:
            return bool(json.loads(stripped))
        except ValueError:
            return True
    return bool(tc)


def _extract_first_exchange(path: list[dict]) -> tuple[str, str]:
    """Extract clamped first user text and the final assistant reply of Turn 1."""
    filtered = [
        m for m in path
        if m.get("type") != "root" and m.get("role") != "tool"
    ]
    user_idx: int | None = None
    user_text = ""
    for idx, msg in enumerate(filtered):
        if msg.get("role") == "user":
            user_idx = idx
            user_text = _extract_text_content(msg.get("content")).strip()
            break

    assistant_text = ""
    fallback_assistant_text = ""
    if user_idx is not None:
        for msg in filtered[user_idx + 1:]:
            role = msg.get("role")
            if role == "user":
                break
            if role == "assistant":
                candidate = _extract_text_content(msg.get("content")).strip()
                if not candidate:
                    continue
                fallback_assistant_text = candidate
                if not _has_tool_calls(msg):
                    assistant_text = candidate

    chosen_assistant = assistant_text or fallback_assistant_text
    return user_text[:_TITLE_TEXT_CLAMP], chosen_assistant[:_TITLE_TEXT_CLAMP]


def _build_title_prompt(user_text: str, assistant_text: str) -> str:
    """Build a tiny standalone prompt requesting only a plain 3-6 word title."""
    lines = [
        "Write a concise 3-6 word title for this conversation in the same language as the user's message.",
        'Reply with ONLY the title in plain text: no quotes, no markdown, no prefixes like "Title:", no trailing period.',
        "",
    ]
    if user_text:
        lines.append(f"User: {user_text}")
    if assistant_text:
        lines.append(f"Assistant: {assistant_text}")
    return "\n".join(lines)


def _clean_generated_title(raw: str | None) -> str | None:
    """Normalize raw model output into a clean single-line conversation title."""
    if not isinstance(raw, str):
        return None
    text = _THINK_BLOCK_RE.sub("", raw)
    text = _UNCLOSED_THINK_RE.sub("", text)
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    if not line:
        return None

    for _ in range(5):
        prev = line
        line = _MD_HEADING_RE.sub("", line).strip()
        line = _TITLE_PREFIX_RE.sub("", line).strip()
        for left, right in _WRAP_PAIRS:
            if len(line) >= len(left) + len(right) and line.startswith(left) and line.endswith(right):
                line = line[len(left):len(line) - len(right)].strip()
        if line == prev:
            break

    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"\.+$", "", line).strip()
    line = line[:_TITLE_MAX_CHARS].rstrip()
    if len(line) < 2:
        return None
    return line


def _is_completion_model(model_name: str, reg=None) -> bool:
    reg = reg or _get_registry()
    cfg = reg.get(model_name)
    if not isinstance(cfg, dict):
        return False
    caps = cfg.get("capabilities") or []
    return "completion" in caps


def _is_title_model_runnable(model_name: str, reg=None, manager=None) -> bool:
    """Return True when title completion can run without cold-loading a local GPU."""
    reg = reg or _get_registry()
    manager = manager or _get_manager()
    if manager.get_active(model_name) is not None:
        return True
    cfg = reg.get(model_name) or {}
    if cfg.get("backend", BACKEND_LLAMA) == BACKEND_VLLM_REMOTE:
        return True
    return bool(cfg.get("fallback"))


def _resolve_title_model(body_model, conv: dict, path: list[dict]) -> str:
    """Resolve a completion-capable model for title generation."""
    reg = _get_registry()
    manager = _get_manager()

    if body_model is not None:
        if not isinstance(body_model, str) or not body_model.strip():
            raise HTTPException(status_code=400, detail="Invalid 'model'")
        resolved = reg.resolve(body_model.strip())
        if not resolved or not _is_completion_model(resolved, reg):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{body_model}' is not a completion model",
            )
        return resolved

    candidates: list[str] = []
    if isinstance(conv.get("model"), str) and conv["model"].strip():
        candidates.append(conv["model"].strip())
    for msg in reversed(path):
        if msg.get("role") == "assistant" and isinstance(msg.get("model"), str) and msg["model"].strip():
            candidates.append(msg["model"].strip())
            break

    resolved_candidates: list[str] = []
    for candidate in candidates:
        resolved = reg.resolve(candidate)
        if resolved and _is_completion_model(resolved, reg):
            if _is_title_model_runnable(resolved, reg, manager):
                return resolved
            if resolved not in resolved_candidates:
                resolved_candidates.append(resolved)

    active_completion = [
        name for name in manager.list_active()
        if _is_completion_model(name, reg)
    ]
    fixed = set(reg.list_fixed().keys()) if hasattr(reg, "list_fixed") else set()
    non_fixed = [name for name in active_completion if name not in fixed]
    fallback_candidates = non_fixed or active_completion
    if fallback_candidates:
        return fallback_candidates[0]

    if resolved_candidates:
        return resolved_candidates[0]

    raise HTTPException(status_code=400, detail="No completion model available for title generation")


def _active_backend_url(active, path: str) -> str:
    builder = getattr(active, "backend_url", None)
    if builder is not None:
        return builder(path)
    return f"http://127.0.0.1:{active.port}/{path.lstrip('/')}"


def _build_title_payload(
    backend_model_id: str,
    prompt_text: str,
    cfg: dict,
    family_defaults: dict | None,
    is_hosted: bool,
) -> dict:
    """Build a standalone, thinking-suppressed completion payload tailored to the backend."""
    payload: dict = {
        "model": backend_model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "temperature": 0.2,
    }
    backend_type = cfg.get("backend", BACKEND_LLAMA)
    family = cfg.get("family")
    efforts = list((family_defaults or {}).get("vllm-reasoning-efforts") or [])

    if backend_type == BACKEND_LLAMA:
        payload["max_tokens"] = 64
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload = apply_chat_template_kwargs(payload, cfg, family_defaults)
    elif "none" in efforts or family == "inkling":
        payload["reasoning_effort"] = "none"
        payload["max_tokens"] = 48
    elif "low" in efforts:
        payload["reasoning_effort"] = "low"
        payload["max_tokens"] = 256
    else:
        payload["max_tokens"] = 64
        if not is_hosted:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload = apply_chat_template_kwargs(payload, cfg, family_defaults)
    return payload


async def _resolve_title_backend(model_name: str, user_hash: str):
    """Return (ActiveModel, effective_cfg) without ever evicting or cold-loading local GPUs."""
    reg = _get_registry()
    manager = _get_manager()
    cfg = reg.get(model_name) or {}

    active = manager.get_active(model_name)
    if active is not None:
        return active, cfg

    backend_type = cfg.get("backend", BACKEND_LLAMA)
    fallback = cfg.get("fallback")

    def _make_fallback():
        fb_cfg = {k: v for k, v in cfg.items() if k != "fallback"}
        fb_cfg.update(fallback)
        return ActiveModel(model_name, fb_cfg, port=None, gpu=None), fb_cfg

    if backend_type != BACKEND_VLLM_REMOTE:
        if fallback:
            return _make_fallback()
        logger.info("Skipping title generation for unloaded local model '%s'", model_name)
        return None, cfg

    try:
        replica = manager.pick_replica(model_name, caller_key=user_hash)
        active = await manager.start_replica(model_name, replica, evict=False)
        return active, cfg
    except (httpx.HTTPError, TimeoutError, LoadRefused, RuntimeError) as exc:
        if fallback:
            logger.warning("Title backend '%s' unreachable (%s); using fallback", model_name, exc)
            return _make_fallback()
        logger.warning("Title backend '%s' unavailable: %s", model_name, exc)
        return None, cfg


async def _call_title_completion(model_name: str, prompt_text: str, user_hash: str) -> str | None:
    """Run an in-process, standalone non-streaming completion for title generation."""
    active, effective_cfg = await _resolve_title_backend(model_name, user_hash)
    if active is None:
        return None

    reg = _get_registry()
    family_defaults = reg.get_family_defaults(effective_cfg.get("family"))
    try:
        get_id = getattr(active, "get_backend_model_id", None)
        backend_model_id = await get_id() if callable(get_id) else (active.backend_model_id or model_name)
        remote_headers_fn = getattr(active, "remote_headers", None)
        headers = remote_headers_fn(None, user_hash) if callable(remote_headers_fn) else {}
    except Exception as exc:
        logger.warning("Failed preparing title request for '%s': %s", model_name, exc)
        return None

    payload = _build_title_payload(
        backend_model_id,
        prompt_text,
        effective_cfg,
        family_defaults,
        bool(getattr(active, "hosted_remote", False)),
    )
    url = _active_backend_url(active, "v1/chat/completions")

    try:
        response = await get_proxy_client().post(
            url,
            headers=headers,
            json=payload,
            timeout=_TITLE_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        logger.warning("Title completion request failed for '%s': %s", model_name, exc)
        return None

    if response.status_code != 200:
        logger.warning("Title completion for '%s' returned HTTP %s", model_name, response.status_code)
        return None

    try:
        body = response.json()
        choices = body.get("choices") or []
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
    except (ValueError, AttributeError, IndexError, TypeError) as exc:
        logger.warning("Malformed title completion response from '%s': %s", model_name, exc)
        return None

    return content if isinstance(content, str) else None


def _should_skip_title(current_name: str, expected_title: str | None, force: bool) -> bool:
    if force:
        return False
    if expected_title is not None:
        return current_name != expected_title
    return current_name.strip() not in _DEFAULT_TITLE_PLACEHOLDERS


@router.post("/history/{conversation_id}/title")
async def generate_history_title(conversation_id: str, request: Request):
    """Generate and persist a short LLM title for a conversation."""
    _, user_hash = require_api(request)
    store = _history_store()
    try:
        conv = store.get_conversation_tree(user_hash, conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    raw_body = await request.body()
    if raw_body.strip():
        try:
            data = json.loads(raw_body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")
    else:
        data = {}

    body_model = data.get("model")
    expected_title = data.get("expected_title")
    if expected_title is not None and not isinstance(expected_title, str):
        raise HTTPException(status_code=400, detail="'expected_title' must be a string")
    force = bool(data.get("force", False))

    current_name = conv.get("name") or ""
    if _should_skip_title(current_name, expected_title, force):
        return {"conversation_id": conversation_id, "title": None, "skipped": True}

    path = _active_message_path(conv)
    user_text, assistant_text = _extract_first_exchange(path)
    if not user_text and not assistant_text:
        raise HTTPException(status_code=400, detail="Conversation has no text messages to title")

    model_name = _resolve_title_model(body_model, conv, path)
    prompt_text = _build_title_prompt(user_text, assistant_text)
    raw_title = await _call_title_completion(model_name, prompt_text, user_hash)
    title = _clean_generated_title(raw_title)
    if not title:
        return {"conversation_id": conversation_id, "title": None}

    preserved_last_modified = conv.get("lastModified", 0)
    if not force:
        try:
            latest = store.get_conversation_tree(user_hash, conversation_id)
        except KeyError:
            return {"conversation_id": conversation_id, "title": None, "skipped": True}
        latest_name = latest.get("name") or ""
        if _should_skip_title(latest_name, expected_title, force=False):
            return {"conversation_id": conversation_id, "title": None, "skipped": True}
        preserved_last_modified = latest.get("lastModified", preserved_last_modified)

    try:
        store.patch_conversation_tree(
            user_hash,
            conversation_id,
            {"name": title, "lastModified": preserved_last_modified},
        )
    except KeyError:
        return {"conversation_id": conversation_id, "title": None, "skipped": True}

    return {"conversation_id": conversation_id, "title": title}


def _is_ungenerated_title(current_name: str, user_text: str) -> bool:
    """Return True if current_name is still a placeholder or raw first-prompt snippet."""
    stripped = (current_name or "").strip()
    if stripped in _DEFAULT_TITLE_PLACEHOLDERS:
        return True
    prompt = (user_text or "").strip()
    if not prompt:
        return False
    if stripped == prompt:
        return True
    if stripped.endswith("…"):
        prefix = stripped[:-1]
        if len(prefix) >= 20 and prompt.startswith(prefix):
            return True
    elif stripped.endswith("..."):
        prefix = stripped[:-3]
        if len(prefix) >= 20 and prompt.startswith(prefix):
            return True
    cp_prefix = prompt[:_TITLE_PLACEHOLDER_PREFIX_CHARS].strip()
    if cp_prefix and stripped == cp_prefix:
        return True
    u16_prefix = (
        prompt.encode("utf-16-le")[: _TITLE_PLACEHOLDER_PREFIX_CHARS * 2]
        .decode("utf-16-le", errors="ignore")
        .strip()
    )
    if u16_prefix and stripped == u16_prefix:
        return True
    return False


def _header_last_modified_ms(header: dict) -> int:
    raw_ms = header.get("last_modified_ms")
    if isinstance(raw_ms, (int, float)) and raw_ms > 0:
        return int(raw_ms)
    updated_at = header.get("updated_at")
    if isinstance(updated_at, str) and updated_at.strip():
        try:
            dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return 0
    return 0


async def run_title_backfill_once(
    *,
    min_age_s: int | None = None,
    max_per_run: int | None = None,
    step_pause_s: float | None = None,
    failed_counts: dict[tuple[str, str], int] | None = None,
    now_ms_fn=None,
) -> dict[str, int]:
    """Scan conversations with ungenerated placeholder titles and backfill them."""
    min_age = _TITLE_BACKFILL_MIN_AGE_S if min_age_s is None else min_age_s
    limit = _TITLE_BACKFILL_MAX_PER_RUN if max_per_run is None else max_per_run
    pause_s = _TITLE_BACKFILL_STEP_PAUSE_S if step_pause_s is None else step_pause_s
    now_ms = int(now_ms_fn()) if now_ms_fn is not None else int(time.time() * 1000)
    cutoff_ms = now_ms - max(0, min_age) * 1000
    failures = failed_counts if failed_counts is not None else {}

    store = _history_store()
    headers = store.list_conversations_with_messages()
    stats = {"scanned": len(headers), "candidates": 0, "updated": 0, "skipped": 0, "failed": 0}
    if limit <= 0 or not headers:
        return stats

    attempted = 0
    for header in headers:
        if attempted >= limit:
            break
        if min_age > 0 and _header_last_modified_ms(header) > cutoff_ms:
            continue

        conv_id = header["id"]
        user_hash = header["user_hash"]
        header_title = header.get("title") or ""
        retry_key = (conv_id, header_title)
        if failures.get(retry_key, 0) >= _TITLE_BACKFILL_MAX_RETRIES:
            stats["skipped"] += 1
            continue

        try:
            conv = store.get_conversation_tree(user_hash, conv_id)
        except KeyError:
            continue

        current_name = conv.get("name") or ""
        path = _active_message_path(conv)
        user_text, assistant_text = _extract_first_exchange(path)
        if not user_text and not assistant_text:
            continue
        if not _is_ungenerated_title(current_name, user_text):
            continue

        stats["candidates"] += 1
        try:
            model_name = _resolve_title_model(None, conv, path)
        except HTTPException:
            stats["skipped"] += 1
            continue

        if not _is_title_model_runnable(model_name):
            stats["skipped"] += 1
            continue

        attempted += 1
        prompt_text = _build_title_prompt(user_text, assistant_text)
        raw_title = await _call_title_completion(model_name, prompt_text, user_hash)
        title = _clean_generated_title(raw_title)
        if not title:
            failures[(conv_id, current_name)] = failures.get((conv_id, current_name), 0) + 1
            stats["failed"] += 1
            continue

        try:
            latest = store.get_conversation_tree(user_hash, conv_id)
        except KeyError:
            stats["skipped"] += 1
            continue
        latest_name = latest.get("name") or ""
        if latest_name != current_name:
            stats["skipped"] += 1
            continue

        preserved_last_modified = latest.get("lastModified", conv.get("lastModified", 0))
        try:
            store.patch_conversation_tree(
                user_hash,
                conv_id,
                {"name": title, "lastModified": preserved_last_modified},
            )
            failures.pop((conv_id, current_name), None)
            stats["updated"] += 1
        except KeyError:
            stats["skipped"] += 1

        if pause_s > 0 and attempted < limit:
            await asyncio.sleep(pause_s)

    return stats


async def title_backfill_loop(
    *,
    interval_s: int | None = None,
    initial_delay_s: int | None = None,
) -> None:
    """Periodic background task that generates titles for conversations missed at Turn 1."""
    interval = _TITLE_BACKFILL_INTERVAL_S if interval_s is None else interval_s
    delay = _TITLE_BACKFILL_INITIAL_DELAY_S if initial_delay_s is None else initial_delay_s
    if interval <= 0:
        logger.info("Periodic title backfill disabled (interval=%ss)", interval)
        return

    logger.info("Periodic title backfill enabled (every %ss, initial delay %ss)", interval, delay)
    failed_counts: dict[tuple[str, str], int] = {}
    try:
        if delay > 0:
            await asyncio.sleep(delay)
        while True:
            try:
                stats = await run_title_backfill_once(failed_counts=failed_counts)
                if stats["updated"] or stats["failed"]:
                    logger.info(
                        "Title backfill pass finished: updated=%d failed=%d candidates=%d scanned=%d",
                        stats["updated"],
                        stats["failed"],
                        stats["candidates"],
                        stats["scanned"],
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Title backfill pass failed: %s", exc)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.debug("Periodic title backfill stopped")
        raise
