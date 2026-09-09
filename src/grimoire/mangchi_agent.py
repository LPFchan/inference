"""Mangchi residency agent.

Runs on the Jetson AGX Thor (mangchi) and owns the lifecycle of vLLM processes
on its single Blackwell GPU. Grimoire (on another host) drives it over HTTP to
make remote vLLM models load/unload the way grimoire's local models do.

See DEC-20260909-002. The design is "many registered, few resident": an
unbounded registry of launch specs, and a resident set bounded by the Thor's
unified-memory budget, with LRU eviction and pinning.

The agent is the single authority on what is running on the Thor. It does not
track grimoire's state and grimoire does not track mangchi's memory.

Known limitation: residency is tracked in this process's memory. If shutdown
cannot confirm a model's process group is dead, the reservation is retained for
the running manager but is lost when the agent exits — a fresh agent has no
record of surviving workers. Cross-restart reconciliation (scanning for and
reaping orphaned vLLM processes at startup before admitting models) is a
deliberate follow-up; see DEC-20260909-002.
"""

from __future__ import annotations

import asyncio
import ipaddress
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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

# Trusted source networks: loopback, the LAN, and the tailnet. The agent exposes
# process lifecycle (load/unload), so it must not answer arbitrary internet
# clients even though it binds all interfaces (it has to serve both grimoire,
# which resolves mangchi.lost.plus -> the LAN IP, and tailnet peers). Fails
# closed if the env override is empty.
_DEFAULT_TRUSTED = "127.0.0.0/8,10.0.0.0/24,100.64.0.0/10,::1/128"
TRUSTED_NETWORKS = [
    ipaddress.ip_network(x.strip())
    for x in os.environ.get("MANGCHI_AGENT_TRUSTED_CIDRS", _DEFAULT_TRUSTED).split(",")
    if x.strip()
]


@dataclass
class LaunchSpec:
    """How to serve one registered model. Adding a model = adding a spec."""

    model_path: str
    resident_gb: float
    gpu_mem_util: Optional[float] = None
    model_path_host: Optional[str] = None  # host path, for docker -v mount
    serve_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    port: int = 8000
    pinned: bool = False
    vllm_bin: str = "vllm"
    # When vllm_docker_image is set, the model runs as a sibling docker container
    # (the agent itself runs in a container with the docker socket mounted).
    vllm_docker_image: Optional[str] = None
    docker_runtime: str = "nvidia"
    models_dir_container: str = "/models"
    models_dir_host: Optional[str] = None


@dataclass
class Resident:
    name: str
    spec: LaunchSpec
    process: subprocess.Popen
    port: int
    pgid: Optional[int] = None
    container: Optional[str] = None  # set when launched as a docker container
    started_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


def load_specs(path: str = SPECS_PATH) -> dict[str, LaunchSpec]:
    raw = json.loads(Path(path).read_text())
    specs: dict[str, LaunchSpec] = {}
    for name, cfg in raw.get("models", {}).items():
        specs[name] = LaunchSpec(
            model_path=cfg["model_path"],
            resident_gb=float(cfg["resident_gb"]),
            gpu_mem_util=(float(cfg["gpu_mem_util"]) if "gpu_mem_util" in cfg else None),
            model_path_host=cfg.get("model_path_host"),
            serve_args=list(cfg.get("serve_args", [])),
            env=dict(cfg.get("env", {})),
            port=int(cfg.get("port", 8000)),
            pinned=bool(cfg.get("pinned", False)),
            vllm_bin=cfg.get("vllm_bin", "vllm"),
            vllm_docker_image=cfg.get("vllm_docker_image"),
            docker_runtime=cfg.get("docker_runtime", "nvidia"),
            models_dir_container=cfg.get("models_dir_container", "/models"),
            models_dir_host=cfg.get("models_dir_host"),
        )
    return specs


def build_launch_command(name: str, spec: LaunchSpec) -> list[str]:
    """The argv used to start one model's vLLM process (host binary or docker)."""
    serve = ["serve", spec.model_path, "--port", str(spec.port)]
    if spec.gpu_mem_util is not None:
        serve += ["--gpu-memory-utilization", str(spec.gpu_mem_util)]
    serve += list(spec.serve_args)
    if not spec.vllm_docker_image:
        return [spec.vllm_bin, *serve]
    # Sibling-container launch: name it so status/cleanup can find it, map the
    # port, and mount the model dir read-only. --init gives a reaping PID 1.
    cmd = [
        "docker", "run", "-d", "--init",
        "--name", f"mangchi-vllm-{name}",
        "--runtime", spec.docker_runtime,
        "-p", f"{spec.port}:{spec.port}",
    ]
    if spec.models_dir_host:
        cmd += ["-v", f"{spec.models_dir_host}:{spec.models_dir_container}:ro"]
    for k, v in spec.env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [spec.vllm_docker_image, "vllm", *serve]
    return cmd


class ResidencyManager:
    """Owns vLLM processes on the Thor. Single-device budget + LRU + pinning."""

    def __init__(self, specs: dict[str, LaunchSpec], budget_gib: float = MEMORY_BUDGET_GIB):
        self.specs = specs
        self.budget_gib = budget_gib
        self.resident: dict[str, Resident] = {}
        self._lock = asyncio.Lock()
        self._stop_tasks: dict[str, "asyncio.Task"] = {}
        self._load_tasks: dict[str, "asyncio.Task"] = {}
        self._closing = False

    def _used_gib(self) -> float:
        return sum(r.spec.resident_gb for r in self.resident.values())

    def _free_gib(self) -> float:
        return self.budget_gib - self._used_gib()

    def _group_alive(self, r: Resident) -> bool:
        """True if any member of the resident's process group still exists."""
        if r.container:
            return self._container_running(r.container)
        if r.pgid is None:
            return r.process.poll() is None
        try:
            os.killpg(r.pgid, 0)
            return True
        except ProcessLookupError:
            # No such process/group: confirmed gone.
            return False
        except PermissionError:
            # Group exists but we can't signal it — it is NOT confirmed gone.
            # Treating this as absent would release the reservation over live
            # workers. Surface it as alive so shutdown fails honestly.
            return True

    @staticmethod
    def _docker(args: list[str]) -> tuple[int, str]:
        try:
            out = subprocess.run(
                ["docker", *args], capture_output=True, text=True, timeout=30
            )
            return out.returncode, (out.stdout or "").strip()
        except Exception as exc:
            return 1, str(exc)

    def _container_running(self, name: str) -> bool:
        rc, out = self._docker(["inspect", "-f", "{{.State.Running}}", name])
        return rc == 0 and out == "true"

    def _stop_container(self, name: str, timeout_s: int) -> None:
        # docker stop sends SIGTERM then SIGKILL after the timeout — graceful
        # first, matching the Thor's SIGKILL-avoidance need.
        self._docker(["stop", "-t", str(timeout_s), name])

    def _remove_container(self, name: str) -> None:
        self._docker(["rm", "-f", name])

    def _signal_group(self, r: Resident, sig: int) -> None:
        # Prefer the pgid captured at launch — reliable even after the launcher
        # has been reaped, and avoids os.getpgid() on a stale/reused PID.
        if r.pgid is not None:
            try:
                os.killpg(r.pgid, sig)
                return
            except (ProcessLookupError, PermissionError):
                pass
        try:
            r.process.send_signal(sig)
        except ProcessLookupError:
            pass

    def _reap_launcher(self, r: Resident) -> None:
        """Reap the launcher so an exited one doesn't linger as a zombie and make
        the group look alive. Non-blocking."""
        try:
            r.process.wait(timeout=0)
        except Exception:
            pass

    async def _wait_group_gone(self, r: Resident, timeout: float) -> bool:
        """Poll until no group member survives (reaping the launcher meanwhile).

        Returns True on confirmed group death, False on timeout. Cannot rely on
        signal-0 alone to distinguish a zombie from a live worker, so the
        launcher is reaped each poll.
        """
        deadline = time.time() + timeout
        while True:
            self._reap_launcher(r)
            if not self._group_alive(r):
                return True
            if time.time() >= deadline:
                return False
            await asyncio.sleep(0.5)

    async def _reap_group(self, r: Resident) -> None:
        """Stop the whole process group and confirm it is gone.

        Raises RuntimeError if the group cannot be confirmed dead — the caller
        must NOT release the reservation in that case, or a replacement could be
        admitted over live workers holding CUDA memory.
        """
        if r.container:
            await self._reap_container(r)
            return
        self._signal_group(r, signal.SIGTERM)
        if await self._wait_group_gone(r, STOP_TIMEOUT_S):
            return
        logger.warning("group for %s survived SIGTERM; SIGKILL as last resort", r.name)
        self._signal_group(r, signal.SIGKILL)
        if await self._wait_group_gone(r, 10):
            return
        raise RuntimeError(
            f"process group for '{r.name}' could not be confirmed dead after SIGKILL"
        )

    async def _reap_container(self, r: Resident) -> None:
        """Stop+remove the model's container, confirming it is gone."""
        name = r.container
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._stop_container, name, int(STOP_TIMEOUT_S))
        deadline = time.time() + STOP_TIMEOUT_S + 15
        while self._container_running(name):
            if time.time() >= deadline:
                raise RuntimeError(f"container '{name}' would not stop")
            await asyncio.sleep(0.5)
        await loop.run_in_executor(None, self._remove_container, name)

    def _ensure_stop_task(self, name: str) -> "asyncio.Task":
        """Return the single in-flight stop task for name, creating it if needed.

        The task OWNS the manager lock and runs the full stop to completion,
        detached from any caller. Callers wait on it via asyncio.shield, so a
        cancelled caller detaches without cancelling the stop.
        """
        task = self._stop_tasks.get(name)
        if task is None or task.done():
            task = asyncio.ensure_future(self._stop_locked(name))
            self._stop_tasks[name] = task
            task.add_done_callback(lambda t, n=name: self._drop_task(self._stop_tasks, n, t))
        return task

    async def _stop_locked(self, name: str) -> bool:
        """Full stop under the manager lock. Runs to completion inside its own
        task regardless of caller cancellation. Raises RuntimeError if the group
        cannot be confirmed dead — the reservation is retained in that case.
        """
        async with self._lock:
            r = self.resident.get(name)
            if not r:
                return False
            logger.info("stopping %s (pid %s, pgid %s)", name, r.process.pid, r.pgid)
            await self._reap_group(r)  # raises if not confirmed dead
            self.resident.pop(name, None)
            return True

    async def _stop(self, name: str) -> bool:
        """Wait for the model's stop to finish. Caller cancellation only detaches
        the caller (via shield); the stop task itself runs to completion.
        Returns True if a resident was stopped, False if it wasn't resident."""
        task = self._ensure_stop_task(name)
        return await asyncio.shield(task)

    def _plan_evictions(self, need_gib: float, exclude: str) -> list[str]:
        """Compute the LRU victim list to fit need_gib WITHOUT stopping anything.

        Returns the names to evict (oldest-first). Raises 409 if it cannot be
        done — feasibility is decided before any resident is harmed.
        """
        if need_gib > self.budget_gib:
            raise HTTPException(
                status_code=409,
                detail=f"'{exclude}' needs {need_gib:.1f} GiB, above budget {self.budget_gib:.1f} GiB",
            )
        free = self._free_gib()
        if free >= need_gib:
            return []
        candidates = sorted(
            (r for r in self.resident.values() if not r.spec.pinned and r.name != exclude),
            key=lambda r: r.last_used,
        )
        victims: list[str] = []
        for r in candidates:
            if free >= need_gib:
                break
            victims.append(r.name)
            free += r.spec.resident_gb
        if free < need_gib:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cannot free {need_gib:.1f} GiB for '{exclude}': "
                    "remaining residents are pinned"
                ),
            )
        return victims

    async def _evict_until_fits(self, need_gib: float, exclude: str) -> None:
        """Evict the planned LRU victims to fit need_gib (feasibility pre-checked).

        Runs INLINE — the caller (the load task) already holds the lock, so it
        reaps victims directly rather than going through the detached stop task
        (which would deadlock on the same lock).
        """
        for victim in self._plan_evictions(need_gib, exclude):
            logger.info("evicting LRU resident %s to make room", victim)
            await self._stop_inline(victim)

    async def _stop_inline(self, name: str) -> None:
        """Stop a resident while the caller ALREADY holds the lock. Retains the
        reservation if the group can't be confirmed dead."""
        r = self.resident.get(name)
        if not r:
            return
        logger.info("stopping %s (pid %s, pgid %s)", name, r.process.pid, r.pgid)
        await self._reap_group(r)  # raises if not confirmed dead
        self.resident.pop(name, None)

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
        task = self._ensure_load_task(name)
        return await asyncio.shield(task)

    def _ensure_load_task(self, name: str) -> "asyncio.Task":
        task = self._load_tasks.get(name)
        if task is None or task.done():
            task = asyncio.ensure_future(self._load_locked(name))
            self._load_tasks[name] = task
            task.add_done_callback(lambda t, n=name: self._drop_task(self._load_tasks, n, t))
        return task

    def _drop_task(self, table: dict, name: str, task: "asyncio.Task") -> None:
        # Remove only if the entry still points at THIS task, and surface any
        # non-cancellation exception so a detached failure isn't silently lost.
        if table.get(name) is task:
            table.pop(name, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("lifecycle task for %s failed: %r", name, exc)

    async def _load_locked(self, name: str) -> dict:
        async with self._lock:
            if self._closing:
                raise HTTPException(status_code=503, detail="agent is shutting down")
            if name not in self.specs:
                raise HTTPException(status_code=404, detail=f"unknown model '{name}'")
            if name in self.resident:
                r = self.resident[name]
                r.last_used = time.time()
                return {"name": name, "status": "already-resident", "port": r.port}
            spec = self.specs[name]
            await self._evict_until_fits(spec.resident_gb, exclude=name)
            cmd = build_launch_command(name, spec)
            env = {**os.environ, **spec.env}
            logger.info("launching %s: %s", name, " ".join(cmd))
            if spec.vllm_docker_image:
                # Clear any stale container with our name from a prior run, then
                # launch. docker run -d returns immediately; the tracked entity
                # is the named container, not the (already-exited) CLI process.
                self._remove_container(f"mangchi-vllm-{name}")
            proc = subprocess.Popen(cmd, env=env, start_new_session=True)
            if spec.vllm_docker_image:
                pgid = None
                container = f"mangchi-vllm-{name}"
                rc = proc.wait()
                if rc != 0:
                    self._remove_container(container)
                    raise HTTPException(
                        status_code=500,
                        detail=f"docker run for '{name}' failed (rc={rc})",
                    )
            else:
                container = None
                try:
                    pgid = os.getpgid(proc.pid)
                except (ProcessLookupError, PermissionError):
                    pgid = None
            r = Resident(name=name, spec=spec, process=proc, port=spec.port, pgid=pgid, container=container)
            self.resident[name] = r
            try:
                await self._wait_healthy(r)
            except BaseException:
                # Startup failed: shut the process group down and WAIT, so the
                # lock isn't released while a live process holds GPU/port. The
                # caller may have been cancelled, but THIS task continues.
                await self._stop_inline(name)
                raise
            return {"name": name, "status": "loaded", "port": r.port, "resident_gb": spec.resident_gb}

    async def unload(self, name: str) -> dict:
        # Always go through the serialized stop task: it waits for any in-flight
        # load of this model to finish (lock ordering) before deciding residency,
        # so an unload can't miss a model that's mid-launch.
        try:
            stopped = await self._stop(name)
        except RuntimeError as exc:
            # Group not confirmed dead: reservation retained, report honestly.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if not stopped:
            return {"name": name, "status": "not-resident"}
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
        # Stop accepting new loads, let in-flight lifecycle work settle, then
        # stop whatever became resident — so a detached load finishing mid-
        # shutdown isn't left running.
        self._closing = True
        pending = [t for t in list(self._load_tasks.values()) + list(self._stop_tasks.values()) if not t.done()]
        if pending:
            # Shield the drain: cancelling shutdown must not cancel the in-flight
            # lifecycle tasks (bare gather would propagate cancellation to them).
            drain = asyncio.gather(*pending, return_exceptions=True)
            await asyncio.shield(drain)
        for name in list(self.resident.keys()):
            try:
                await self._stop(name)
            except RuntimeError:
                logger.error("shutdown: could not confirm '%s' dead; leaving reservation", name)


def create_app(manager: Optional[ResidencyManager] = None) -> FastAPI:
    mgr = manager or ResidencyManager(load_specs())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await mgr.shutdown()

    app = FastAPI(title="mangchi-residency-agent", lifespan=lifespan)
    app.state.manager = mgr

    @app.middleware("http")
    async def _trusted_sources_only(request: Request, call_next):
        host = request.client.host if request.client else None
        try:
            addr = ipaddress.ip_address(host) if host else None
        except ValueError:
            addr = None
        allowed = bool(TRUSTED_NETWORKS) and addr is not None and any(
            addr in net for net in TRUSTED_NETWORKS
        )
        if not allowed:
            return JSONResponse(status_code=403, content={"detail": "forbidden source"})
        return await call_next(request)

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
