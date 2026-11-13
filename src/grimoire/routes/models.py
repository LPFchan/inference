"""Model management route handlers (/v1/models, /models, /switch, /stop, /props, /ingest)."""

import asyncio
import logging
import math
import os

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from grimoire.auth import is_admin, require_api, require_admin
from grimoire.config import (
    DEFAULT_CTX_SIZE,
    DEFAULT_GENERATION_PARAMS,
    MODEL_STATUS_LOADED,
    MODEL_STATUS_UNLOADED,
)
from grimoire.ingest import download_model_file, model_filename_from_url
from grimoire.registry import (
    MODELS_DIR,
    registry,
)


router = APIRouter()
logger = logging.getLogger(__name__)


def _backend_url(active, path):
    builder = getattr(active, "backend_url", None)
    if builder is not None:
        return builder(path)
    return f"http://127.0.0.1:{active.port}/{path.lstrip('/')}"


def _get_manager():
    from grimoire.entrypoint import manager
    return manager


def _synthetic_props(model_name=None):
    cfg = registry.get(model_name) if model_name else None
    capabilities = (cfg or {}).get("capabilities", []) or []
    has_vision = "multimodal" in capabilities or "vision" in capabilities
    return {
        "default_generation_settings": {
            "id": 0,
            "id_task": -1,
            "n_ctx": (cfg or {}).get("ctx-size", DEFAULT_CTX_SIZE),
            "speculative": False,
            "is_processing": False,
            "params": dict(DEFAULT_GENERATION_PARAMS),
            "prompt": "",
            "next_token": {
                "has_next_token": False,
                "has_new_line": False,
                "n_remain": 0,
                "n_decoded": 0,
                "stopping_word": "",
            },
        },
        "total_slots": (cfg or {}).get("parallel", 1),
        "model_path": (cfg or {}).get("file", ""),
        "role": "router",
        "modalities": {"vision": bool(has_vision), "audio": False},
        "chat_template": "",
        "bos_token": "",
        "eos_token": "",
        "build_info": "grimoire",
    }


def _model_payload_name(payload):
    if not isinstance(payload, dict):
        return None
    name = payload.get("model")
    return name if isinstance(name, str) and name else None


def _alpha_payload_value(payload, body: bytes):
    if isinstance(payload, bool):
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        value = payload.get("alpha", payload.get("scale"))
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
    if isinstance(payload, str):
        try:
            return float(payload)
        except ValueError:
            return None
    if body:
        try:
            return float(body.decode("utf-8").strip())
        except (UnicodeDecodeError, ValueError):
            return None
    return None


async def _active_alpha_target(request: Request):
    manager = _get_manager()
    requested = request.query_params.get("model")
    if requested:
        resolved = registry.resolve(requested)
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Model '{requested}' was not found in the registry")
        active = manager.get_active(resolved)
        if not active:
            raise HTTPException(status_code=404, detail=f"Model '{resolved}' is not active")
        return resolved, active

    active_names = manager.list_active()
    if len(active_names) != 1:
        raise HTTPException(
            status_code=409,
            detail=f"Expected exactly one active model or ?model=..., found {len(active_names)}",
        )
    model_name = active_names[0]
    active = manager.get_active(model_name)
    if not active:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not active")
    return model_name, active


async def _backend_lora_adapters(active) -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(_backend_url(active, "lora-adapters"))
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Backend rejected /lora-adapters GET: {resp.text}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Backend returned invalid /lora-adapters JSON") from exc
    return data if isinstance(data, list) else []


@router.get("/v1/models")
async def get_v1_models(request: Request, refresh_remote: bool = False):
    """Return all registry models in OpenAI-compatible + llama.cpp router shape."""
    require_api(request)
    manager = _get_manager()
    if refresh_remote:
        await manager.refresh_remote_statuses()
    data = registry.list_metadata()
    for item in data:
        name = item["id"]
        cfg = registry.get(name) or {}
        item["active"] = manager.get_active(name) is not None
        status = manager.get_status(name)
        # A fallback is always up, so the model answers even while its own
        # backend is down and nothing is loaded for it.
        if status == MODEL_STATUS_UNLOADED and cfg.get("fallback"):
            status = MODEL_STATUS_LOADED
        item["status"] = {"value": status}
        item["in_cache"] = True
        # llama backends carry `file`.
        item["path"] = cfg.get("file") or cfg.get("target") or ""
        item["context_window"] = cfg.get("ctx-size", DEFAULT_CTX_SIZE)
    return {"object": "list", "data": data}


@router.get("/models")
async def get_models(request: Request):
    """Return registry and active model info."""
    require_api(request)
    manager = _get_manager()
    return {
        "models": registry.list_all(),
        "metadata": registry.list_metadata(),
        "fixed": registry.list_fixed(),
        "active": manager.list_active(),
        "gpu_count": manager.gpu_count
    }


@router.get("/status")
async def status(request: Request):
    """Return system status."""
    require_api(request)
    manager = _get_manager()
    active_info = {}
    for name in manager.list_active():
        active = manager.get_active(name)
        if not active:
            continue
        active_info[name] = {
            "port": active.port,
            "started": active.started.isoformat(),
            "running": active.is_running(),
            **manager.override_metadata(name),
        }
    return {
        "models": registry.list_all(),
        "fixed": registry.list_fixed(),
        "active": active_info,
        "gpu_count": manager.gpu_count,
        "runtime_overrides": {
            name: manager.override_metadata(name)
            for name in manager.runtime_override_names()
        },
    }


def _runtime_control_response(model_name, operation, manager):
    metadata = manager.override_metadata(model_name)
    return {
        "status": operation,
        "operation": operation,
        "model": model_name,
        "active": manager.get_active(model_name) is not None,
        **metadata,
    }


async def _runtime_control_call(model_name, operation, call):
    manager = _get_manager()
    try:
        await call(manager)
        resolved = registry.resolve(model_name) or model_name
        return _runtime_control_response(resolved, operation, manager)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/models/{model_name}/clone")
async def clone_model_endpoint(model_name: str, request: Request):
    require_admin(request)
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    gpu_ids = data.get("gpu_ids")
    if (not isinstance(gpu_ids, list) or len(gpu_ids) < 2
            or any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids)):
        raise HTTPException(status_code=400, detail="gpu_ids must contain at least two unique non-negative integers")
    tensor_split = data.get("tensor_split")
    if tensor_split is not None:
        if (not isinstance(tensor_split, list)
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in tensor_split)):
            raise HTTPException(status_code=400, detail="tensor_split must be an array of numbers")
    return await _runtime_control_call(
        model_name, "cloned",
        lambda manager: manager.clone_model(model_name, gpu_ids, tensor_split),
    )


@router.post("/models/{model_name}/declone")
async def declone_model_endpoint(model_name: str, request: Request):
    require_admin(request)
    return await _runtime_control_call(
        model_name, "decloned", lambda manager: manager.declone_model(model_name),
    )


@router.post("/models/{model_name}/pin")
async def pin_model_endpoint(model_name: str, request: Request):
    require_admin(request)
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    if not isinstance(data, dict) or "gpu" not in data:
        raise HTTPException(status_code=400, detail="Body must contain gpu")
    return await _runtime_control_call(
        model_name, "pinned", lambda manager: manager.pin_model(model_name, data["gpu"]),
    )


@router.post("/models/{model_name}/unpin")
async def unpin_model_endpoint(model_name: str, request: Request):
    require_admin(request)
    return await _runtime_control_call(
        model_name, "unpinned", lambda manager: manager.unpin_model(model_name),
    )


@router.post("/switch/{model_name}")
async def switch_model(model_name: str, request: Request):
    """Start a model with GPU allocation."""
    require_admin(request)
    manager = _get_manager()
    try:
        active = await manager.start_model(model_name)
        return {
            "status": "started",
            "model": model_name,
            "gpu": active.gpu,
            "gpus": getattr(active, "gpus", [] if active.gpu is None else [active.gpu]),
            "port": active.port
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{model_name}")
async def stop_model_endpoint(model_name: str, request: Request):
    """Stop an active model."""
    require_admin(request)
    manager = _get_manager()
    if not manager.get_active(model_name):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not active")
    try:
        await manager.stop_model(model_name)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "stopped", "model": model_name}


@router.post("/models/load")
async def models_load(request: Request):
    """Router-mode alias of /switch/{name}, called by stock llama.cpp webui."""
    require_admin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    name = _model_payload_name(payload)
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'model' in body")
    return await switch_model(name, request)


@router.post("/models/unload")
async def models_unload(request: Request):
    """Router-mode alias of /stop/{name}, called by stock llama.cpp webui."""
    require_admin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    name = _model_payload_name(payload)
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'model' in body")
    return await stop_model_endpoint(name, request)


@router.get("/alpha")
async def get_alpha(request: Request):
    """Return runtime LoRA adapter scales for the active backend."""
    require_api(request)
    model_name, active = await _active_alpha_target(request)
    return {
        "model": model_name,
        "port": active.port,
        "adapters": await _backend_lora_adapters(active),
    }


@router.post("/alpha")
async def set_alpha(request: Request):
    """Set runtime LoRA alpha for the active backend without rebaking GGUFs."""
    require_admin(request)
    body = await request.body()
    try:
        payload = await request.json()
    except Exception:
        payload = None
    alpha = _alpha_payload_value(payload, body)
    if alpha is None or not math.isfinite(alpha) or alpha < 0:
        raise HTTPException(status_code=400, detail="Body must be a non-negative finite alpha float")

    adapter_id_raw = request.query_params.get("id")
    if adapter_id_raw is None and isinstance(payload, dict):
        body_id = payload.get("id", payload.get("adapter_id"))
        if body_id is not None:
            adapter_id_raw = str(body_id)
    if adapter_id_raw is None:
        adapter_id_raw = "0"
    try:
        adapter_id = int(adapter_id_raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="'id' must be an integer")
    if adapter_id < 0:
        raise HTTPException(status_code=400, detail="'id' must be non-negative")

    model_name, active = await _active_alpha_target(request)
    before = await _backend_lora_adapters(active)
    # The backend replaces its entire adapter-scale list on every call, so a
    # single-entry payload silently zeroes every other adapter's scale. Preserve
    # the current scale for every adapter not targeted by this request.
    scales_by_id = {
        int(entry["id"]): entry.get("scale", 1.0)
        for entry in before
        if isinstance(entry, dict) and "id" in entry
    }
    scales_by_id[adapter_id] = alpha
    updated_payload = [
        {"id": entry_id, "scale": scale} for entry_id, scale in sorted(scales_by_id.items())
    ]
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            _backend_url(active, "lora-adapters"),
            json=updated_payload,
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Backend rejected alpha update: {resp.text}")
    after = await _backend_lora_adapters(active)
    return {
        "status": "updated",
        "model": model_name,
        "port": active.port,
        "adapter_id": adapter_id,
        "alpha": alpha,
        "before": before,
        "adapters": after,
    }


@router.post("/ingest")
async def ingest_model(request: Request):
    """Download and register a new model."""
    require_admin(request)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    model_alias = data.get("alias")
    model_url = data.get("url")
    ctx_size = data.get("ctx-size", DEFAULT_CTX_SIZE)

    if not model_alias or not model_url:
        raise HTTPException(status_code=400, detail="Missing 'alias' or 'url'")

    try:
        model_filename = model_filename_from_url(model_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_dir = os.path.join(MODELS_DIR, "gguf")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        raise HTTPException(status_code=409, detail=f"Model file already exists at {model_path}")

    try:
        logger.info(f"Downloading model from {model_url} to {model_path}")
        await asyncio.to_thread(download_model_file, model_url, model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")

    try:
        registry.add(model_alias, {
            "file": f"gguf/{model_filename}",
            "mmproj": None,
            "ctx-size": ctx_size,
        })
        logger.info(f"Added model {model_alias} to registry")
        return {"status": "added", "model": model_alias}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/props")
async def props(request: Request):
    """Router-mode /props for the stock llama.cpp webui.

    Without ?model=<id> returns server-wide router props.
    With ?model=<id>&autoload=false returns synthetic per-model props from registry.
    With ?model=<id> (autoload not false) starts the model and proxies its real /props.
    """
    require_api(request)
    model_name = request.query_params.get("model")
    autoload = request.query_params.get("autoload", "true").lower() not in {"false", "0", "no", "off"}

    if not model_name:
        return _synthetic_props()

    resolved = registry.resolve(model_name)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not in registry")

    manager = _get_manager()
    if not autoload and not manager.get_active(resolved):
        return _synthetic_props(resolved)

    try:
        active = await manager.start_model(resolved, evict=is_admin(request))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start {resolved} for /props: {e}")
        raise HTTPException(status_code=502, detail="Model server unavailable")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_backend_url(active, "props"))
        if resp.status_code == 200:
            data = resp.json()
            data["role"] = "router"
            return data
    except Exception as e:
        logger.info(f"Falling back to synthetic /props for {resolved}: {e}")
    return _synthetic_props(resolved)
