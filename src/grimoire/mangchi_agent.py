"""Mangchi residency agent.

Runs on the Jetson AGX Thor (mangchi) and owns the lifecycle of vLLM processes
on its single Blackwell GPU. Grimoire (on another host) drives it over HTTP to
make remote vLLM models load/unload the way grimoire's local models do.

See DEC-20260909-002. The design is "many registered, few resident": an
unbounded registry of launch specs, and a resident set bounded by the Thor's
unified-memory budget, with LRU eviction and pinning.

The agent is the single authority on what is running on the Thor. It does not
track grimoire's state and grimoire does not track mangchi's memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

logger = logging.getLogger("mangchi_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

SPECS_PATH = os.environ.get(
    "MANGCHI_AGENT_SPECS", str(Path(__file__).resolve().parents[2] / "etc" / "mangchi-agent.json")
)
# Total unified memory the residency set may use, in GiB. Thor has 128; leave
# headroom for the OS, CUDA runtime, and KV cache growth beyond estimate.
MEMORY_BUDGET_GIB = float(os.environ.get("MANGCHI_AGENT_BUDGET_GIB", "110"))
HEALTH_TIMEOUT_S = float(os.environ.get("MANGCHI_AGENT_HEALTH_TIMEOUT_S", "600"))
STOP_TIMEOUT_S = float(os.environ.get("MANGCHI_AGENT_STOP_TIMEOUT_S", "60"))


@dataclass
class LaunchSpec:
    """How to serve one registered model. Adding a model = adding a spec."""

    model_path: str
    resident_gb: float
    serve_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    port: int = 8000
    pinned: bool = False
    vllm_bin: str = "vllm"


@dataclass
class Resident:
    name: str
    spec: LaunchSpec
    process: subprocess.Popen
    port: int
    started_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


def load_specs(path: str = SPECS_PATH) -> dict[str, LaunchSpec]:
    raw = json.loads(Path(path).read_text())
    specs: dict[str, LaunchSpec] = {}
    for name, cfg in raw.get("models", {}).items():
        specs[name] = LaunchSpec(
            model_path=cfg["model_path"],
            resident_gb=float(cfg["resident_gb"]),
            serve_args=list(cfg.get("serve_args", [])),
            env=dict(cfg.get("env", {})),
            port=int(cfg.get("port", 8000)),
            pinned=bool(cfg.get("pinned", False)),
            vllm_bin=cfg.get("vllm_bin", "vllm"),
        )
    return specs


class ResidencyManager:
    """Owns vLLM processes on the Thor. Single-device budget + LRU + pinning."""

    def __init__(self, specs: dict[str, LaunchSpec], budget_gib: float = MEMORY_BUDGET_GIB):
        self.specs = specs
        self.budget_gib = budget_gib
        self.resident: dict[str, Resident] = {}
        self._lock = asyncio.Lock()

    def _used_gib(self) -> float:
        return sum(r.spec.resident_gb for r in self.resident.values())

    def _free_gib(self) -> float:
        return self.budget_gib - self._used_gib()

    async def _stop(self, name: str) -> None:
        r = self.resident.get(name)
        if not r:
            return
        logger.info("stopping %s (pid %s)", name, r.process.pid)
        # SIGTERM and wait — SIGKILL mid-CUDA-op can wedge the Thor's GPU.
        r.process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, r.process.wait),
                timeout=STOP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("%s did not exit in %ss; SIGKILL as last resort", name, STOP_TIMEOUT_S)
            r.process.kill()
            await asyncio.get_event_loop().run_in_executor(None, r.process.wait)
        self.resident.pop(name, None)

    async def _evict_until_fits(self, need_gib: float, exclude: str) -> None:
        """Evict least-recently-used unpinned residents until need_gib fits."""
        while self._free_gib() < need_gib:
            candidates = [r for r in self.resident.values() if not r.spec.pinned and r.name != exclude]
            if not candidates:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"cannot free {need_gib:.1f} GiB for '{exclude}': "
                        "all residents are pinned"
                    ),
                )
            victim = min(candidates, key=lambda r: r.last_used)
            logger.info("evicting LRU resident %s to make room", victim.name)
            await self._stop(victim.name)

    async def _wait_healthy(self, r: Resident) -> None:
        url = f"http://127.0.0.1:{r.port}/health"
        deadline = time.time() + HEALTH_TIMEOUT_S
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() < deadline:
                if r.process.poll() is not None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"vLLM for '{r.name}' exited during startup (rc={r.process.returncode})",
                    )
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(3)
        raise HTTPException(status_code=504, detail=f"'{r.name}' did not become healthy in {HEALTH_TIMEOUT_S}s")

    async def load(self, name: str) -> dict:
        async with self._lock:
            if name not in self.specs:
                raise HTTPException(status_code=404, detail=f"unknown model '{name}'")
            if name in self.resident:
                r = self.resident[name]
                r.last_used = time.time()
                return {"name": name, "status": "already-resident", "port": r.port}
            spec = self.specs[name]
            await self._evict_until_fits(spec.resident_gb, exclude=name)
            cmd = [spec.vllm_bin, "serve", spec.model_path, "--port", str(spec.port), *spec.serve_args]
            env = {**os.environ, **spec.env}
            logger.info("launching %s: %s", name, " ".join(cmd))
            proc = subprocess.Popen(cmd, env=env, start_new_session=True)
            r = Resident(name=name, spec=spec, process=proc, port=spec.port)
            self.resident[name] = r
            try:
                await self._wait_healthy(r)
            except Exception:
                self.resident.pop(name, None)
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                raise
            return {"name": name, "status": "loaded", "port": r.port, "resident_gb": spec.resident_gb}

    async def unload(self, name: str) -> dict:
        async with self._lock:
            if name not in self.resident:
                return {"name": name, "status": "not-resident"}
            await self._stop(name)
            return {"name": name, "status": "unloaded"}

    def status(self, name: Optional[str] = None) -> dict:
        def one(r: Resident) -> dict:
            return {
                "name": r.name,
                "port": r.port,
                "pid": r.process.pid,
                "alive": r.process.poll() is None,
                "pinned": r.spec.pinned,
                "resident_gb": r.spec.resident_gb,
                "uptime_s": round(time.time() - r.started_at, 1),
                "url": f"http://127.0.0.1:{r.port}/v1",
            }

        if name is not None:
            r = self.resident.get(name)
            if not r:
                return {"name": name, "resident": False, "registered": name in self.specs}
            d = one(r)
            d["resident"] = True
            return d
        return {
            "budget_gib": self.budget_gib,
            "used_gib": round(self._used_gib(), 2),
            "free_gib": round(self._free_gib(), 2),
            "registered": sorted(self.specs.keys()),
            "resident": [one(r) for r in self.resident.values()],
        }

    async def shutdown(self) -> None:
        async with self._lock:
            for name in list(self.resident.keys()):
                await self._stop(name)


def create_app(manager: Optional[ResidencyManager] = None) -> FastAPI:
    mgr = manager or ResidencyManager(load_specs())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await mgr.shutdown()

    app = FastAPI(title="mangchi-residency-agent", lifespan=lifespan)
    app.state.manager = mgr

    @app.post("/models/{name}/load")
    async def load_model(name: str):
        return await mgr.load(name)

    @app.post("/models/{name}/unload")
    async def unload_model(name: str):
        return await mgr.unload(name)

    @app.get("/models/{name}/status")
    async def model_status(name: str):
        return mgr.status(name)

    @app.get("/status")
    async def global_status():
        return mgr.status()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "grimoire.mangchi_agent:app",
        host=os.environ.get("MANGCHI_AGENT_HOST", "0.0.0.0"),
        port=int(os.environ.get("MANGCHI_AGENT_PORT", "9700")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
