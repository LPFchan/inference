"""Mangchi residency agent logic: budget, LRU eviction, pinning, unknown models.

These tests stub out subprocess launch and health polling so no real vLLM,
process, or network is needed. They pin the residency rules from
DEC-20260909-002: registered-vs-resident, memory budget, LRU eviction, pinning.
"""

import asyncio

import pytest

from grimoire import mangchi_agent as agent
from grimoire.mangchi_agent import LaunchSpec, ResidencyManager
from fastapi import HTTPException


def run(coro):
    return asyncio.run(coro)


class FakeProc:
    _next_pid = 1000

    def __init__(self, *a, **k):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.returncode = None
        self._signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self._signals.append(sig)
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if timeout is not None and self.returncode is None:
            import subprocess as _sp
            raise _sp.TimeoutExpired(cmd="fake", timeout=timeout)
        return self.returncode


def _specs():
    return {
        "small": LaunchSpec(model_path="/m/small", resident_gb=30, port=8001),
        "large": LaunchSpec(model_path="/m/large", resident_gb=80, port=8002),
        "pinned": LaunchSpec(model_path="/m/pinned", resident_gb=40, port=8003, pinned=True),
    }


@pytest.fixture
def mgr(monkeypatch):
    monkeypatch.setattr(agent.subprocess, "Popen", FakeProc)
    # Neutralize every OS process/signal boundary so FakeProc's invented PIDs
    # can never reach a real process or process group. Individual tests can
    # re-stub these to observe calls.
    monkeypatch.setattr(agent.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(agent.os, "killpg", lambda pgid, sig: None)
    m = ResidencyManager(_specs(), budget_gib=110)

    async def _no_healthy(r):
        return None

    monkeypatch.setattr(m, "_wait_healthy", _no_healthy)
    # Group reaping must not actually block on fake processes.
    monkeypatch.setattr(m, "_group_alive", lambda r: False)
    return m


def test_load_unknown_model_404(mgr):
    with pytest.raises(HTTPException) as e:
        run(mgr.load("nope"))
    assert e.value.status_code == 404


def test_load_within_budget(mgr):
    out = run(mgr.load("small"))
    assert out["status"] == "loaded"
    st = mgr.status()
    assert st["used_gib"] == 30
    assert "small" in [r["name"] for r in st["resident"]]


def test_reload_is_idempotent(mgr):
    run(mgr.load("small"))
    out = run(mgr.load("small"))
    assert out["status"] == "already-resident"
    assert len(mgr.resident) == 1


def test_lru_eviction_makes_room(mgr):
    mgr.budget_gib = 150
    run(mgr.load("small"))   # 30 used, 120 free
    run(mgr.load("pinned"))  # 70 used, 80 free
    # large needs 80, exactly the free headroom; loading it fills the device.
    # Lowering the budget to 110 forces eviction: 70 used, 40 free, need 80,
    # so the LRU unpinned resident (small, 30) is evicted, leaving 70 free...
    # still short, but pinned cannot be evicted, so this must 409.
    mgr.budget_gib = 110
    mgr.resident["small"].last_used -= 100
    mgr.resident["pinned"].last_used -= 10
    with pytest.raises(HTTPException) as e:
        run(mgr.load("large"))
    assert e.value.status_code == 409


def test_lru_eviction_succeeds_when_it_frees_enough(mgr):
    # small + another unpinned small-ish resident; evicting one frees enough.
    mgr.specs["small2"] = LaunchSpec(model_path="/m/small2", resident_gb=30, port=8004)
    mgr.budget_gib = 100
    run(mgr.load("small"))    # 30 used, 70 free
    run(mgr.load("small2"))   # 60 used, 40 free
    # large needs 80; only 40 free. Evicting both small residents frees 60,
    # leaving large (80) as the sole resident within the 100 budget.
    mgr.resident["small"].last_used -= 100   # small is oldest
    mgr.resident["small2"].last_used -= 10
    out = run(mgr.load("large"))
    assert out["status"] == "loaded"
    names = set(mgr.resident.keys())
    assert "small" not in names
    assert "small2" not in names
    assert "large" in names
    assert mgr.status()["used_gib"] == 80


def test_eviction_blocked_when_all_pinned(mgr):
    run(mgr.load("pinned"))  # 40 pinned, 70 free
    # large needs 80 but only 70 free and the only resident is pinned.
    with pytest.raises(HTTPException) as e:
        run(mgr.load("large"))
    assert e.value.status_code == 409
    assert "pinned" in mgr.resident  # untouched


def test_unload(mgr):
    run(mgr.load("small"))
    out = run(mgr.unload("small"))
    assert out["status"] == "unloaded"
    assert mgr.status()["used_gib"] == 0


def test_unload_not_resident(mgr):
    out = run(mgr.unload("small"))
    assert out["status"] == "not-resident"


def test_status_reports_registered_and_resident(mgr):
    run(mgr.load("small"))
    st = mgr.status("small")
    assert st["resident"] is True
    assert st["url"].endswith("/v1")
    missing = mgr.status("large")
    assert missing["resident"] is False
    assert missing["registered"] is True
    unreg = mgr.status("ghost")
    assert unreg["registered"] is False


def test_specs_file_parses():
    specs = agent.load_specs()
    assert "qwen3.8-27b-uncensored-nvfp4" in specs
    assert "qwen3.8-flash-next-uncensored-nvfp4" in specs
    flash = specs["qwen3.8-flash-next-uncensored-nvfp4"]
    assert flash.vllm_docker_image == "mangchi-vllm:thor-v0.29-ple-mmap"
    assert flash.env.get("VLLM_PLE_MMAP") == "1"
    assert "VLLM_PLE_CPU_OFFLOAD" not in flash.env
    assert "--enforce-eager" in flash.serve_args
    assert "--no-enable-flashinfer-autotune" in flash.serve_args
    assert flash.resident_gb > specs["qwen3.8-27b-uncensored-nvfp4"].resident_gb


def test_docker_launch_command_shape():
    specs = agent.load_specs()
    name = "qwen3.8-27b-uncensored-nvfp4"
    cmd = agent.build_launch_command(name, specs[name])
    assert cmd[0] == "docker" and "run" in cmd and "-d" in cmd
    assert f"mangchi-vllm-{name}" in cmd
    assert "--runtime" in cmd and "nvidia" in cmd
    # port published, model dir mounted ro, image + vllm serve present
    joined = " ".join(cmd)
    assert "-p 8001:8001" in joined
    assert ":ro" in joined
    assert "vllm serve /models/qwen3.8-27b-uncensored" in joined
    assert "--gpu-memory-utilization 0.22" in joined


def test_container_resident_alive_and_reap(mgr, monkeypatch):
    # A container-backed resident is tracked by docker state, not host pgid.
    mgr.specs["cont"] = LaunchSpec(
        model_path="/models/x", resident_gb=10, port=9001,
        vllm_docker_image="img", models_dir_host="/host/models",
    )
    states = {"running": True}
    removed = []
    monkeypatch.setattr(mgr, "_container_running", lambda c: states["running"])
    monkeypatch.setattr(mgr, "_stop_container", lambda c, t: states.__setitem__("running", False))
    monkeypatch.setattr(mgr, "_remove_container", lambda c: removed.append(c))
    # docker run -d exits 0 immediately on success
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *a, **k: type("P", (), {"wait": lambda s: 0, "pid": 1})())
    run(mgr.load("cont"))
    r = mgr.resident["cont"]
    assert r.container == "mangchi-vllm-cont"
    out = run(mgr.unload("cont"))
    assert out["status"] == "unloaded"
    assert "mangchi-vllm-cont" in removed


def test_group_alive_container_branch(monkeypatch):
    # _group_alive on a container resident consults docker state, not host pgid.
    m = ResidencyManager(_specs(), budget_gib=110)
    r = agent.Resident(
        name="c", spec=_specs()["small"], process=None, port=1, container="mangchi-vllm-c"
    )
    monkeypatch.setattr(m, "_container_running", lambda c: True)
    assert m._group_alive(r) is True
    monkeypatch.setattr(m, "_container_running", lambda c: False)
    assert m._group_alive(r) is False


def test_specs_have_per_model_gpu_mem_util():
    # Per-instance --gpu-memory-utilization must reflect each model's share, not
    # a blanket near-1.0 (which would let two models both claim ~the whole device).
    specs = agent.load_specs()
    for name, spec in specs.items():
        assert spec.gpu_mem_util is not None, name
        assert 0.0 < spec.gpu_mem_util < 0.9, name


def test_infeasible_load_does_not_harm_residents(mgr):
    # 40 pinned + 30 unpinned under 110 budget; 80 cannot fit even after evicting
    # the unpinned resident, so the request must 409 WITHOUT touching anyone.
    run(mgr.load("small"))    # 30 unpinned
    run(mgr.load("pinned"))   # 40 pinned
    mgr.resident["small"].last_used -= 100
    with pytest.raises(HTTPException) as e:
        run(mgr.load("large"))
    assert e.value.status_code == 409
    # pre-check must have rejected before evicting: both residents intact.
    assert "small" in mgr.resident
    assert "pinned" in mgr.resident


def test_oversized_spec_rejected_before_eviction(mgr):
    mgr.specs["huge"] = LaunchSpec(model_path="/m/huge", resident_gb=200, port=8009)
    run(mgr.load("small"))
    with pytest.raises(HTTPException) as e:
        run(mgr.load("huge"))
    assert e.value.status_code == 409
    assert "small" in mgr.resident  # not evicted by an impossible request


def test_failed_start_cleans_up_and_waits(mgr, monkeypatch):
    # Health check raises: the failed process must be stopped (and waited on)
    # and removed from residency, not abandoned holding memory/port.
    async def _boom(r):
        raise RuntimeError("never healthy")

    monkeypatch.setattr(mgr, "_wait_healthy", _boom)
    with pytest.raises(RuntimeError):
        run(mgr.load("small"))
    assert "small" not in mgr.resident
    assert mgr.status()["used_gib"] == 0


def test_stop_signals_process_group(mgr, monkeypatch):
    run(mgr.load("small"))
    r = mgr.resident["small"]
    calls = []
    monkeypatch.setattr(agent.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    run(mgr.unload("small"))
    # SIGTERM (15) must go to the captured process group, not only the PID.
    assert (r.pgid, 15) in calls


def test_cleanup_waits_for_slow_group(mgr, monkeypatch):
    # A group that stays alive for a few polls must delay _stop until it's gone;
    # this proves cleanup actually waits (a non-waiting _stop would pass earlier
    # tests but fail here).
    run(mgr.load("small"))
    r = mgr.resident["small"]
    polls = {"n": 0}

    def _alive(res):
        polls["n"] += 1
        return polls["n"] <= 3  # alive for first 3 checks, then gone

    monkeypatch.setattr(mgr, "_group_alive", _alive)
    monkeypatch.setattr(agent.asyncio, "sleep", _instant_sleep)
    run(mgr.unload("small"))
    assert polls["n"] > 3  # reaped only after the group stopped reporting alive
    assert "small" not in mgr.resident


async def _instant_sleep(_):
    return None


def test_unload_waits_for_inflight_load(mgr, monkeypatch):
    # A load in progress must not be missed by unload: the serialized stop task
    # waits for the load (lock ordering) and then stops the now-resident model.
    started = agent.asyncio.Event()
    release = agent.asyncio.Event()

    async def _slow_healthy(r):
        started.set()
        await release.wait()

    async def _scenario():
        monkeypatch.setattr(mgr, "_wait_healthy", _slow_healthy)
        load_task = agent.asyncio.ensure_future(mgr.load("small"))
        await started.wait()
        # load is mid-flight; unload must not return not-resident and miss it.
        unload_task = agent.asyncio.ensure_future(mgr.unload("small"))
        await agent.asyncio.sleep(0)  # let unload queue its stop
        release.set()
        loaded = await load_task
        unloaded = await unload_task
        return loaded, unloaded

    loaded, unloaded = run(_scenario())
    assert loaded["status"] == "loaded"
    assert unloaded["status"] == "unloaded"
    assert "small" not in mgr.resident


def test_unconfirmed_group_death_retains_reservation(mgr, monkeypatch):
    # If the group never reports gone, unload must NOT claim success or release
    # the reservation — a replacement must not be admitted over live workers.
    run(mgr.load("small"))
    monkeypatch.setattr(mgr, "_group_alive", lambda r: True)  # never dies
    monkeypatch.setattr(agent.asyncio, "sleep", _instant_sleep)
    # Drive the poll loop to timeout instantly by advancing time.
    clock = {"t": 0.0}
    monkeypatch.setattr(agent.time, "time", lambda: clock["t"])

    async def _advance(_):
        clock["t"] += 1000  # jump past every deadline on each poll

    monkeypatch.setattr(agent.asyncio, "sleep", _advance)
    with pytest.raises(HTTPException) as e:
        run(mgr.unload("small"))
    assert e.value.status_code == 500
    assert "small" in mgr.resident          # reservation retained
    assert mgr.status()["used_gib"] == 30   # accounting NOT dropped


def test_zombie_launcher_does_not_block_reap(mgr, monkeypatch):
    # An exited-but-unreaped launcher is a zombie in the group; reaping it during
    # the wait lets group polling see the group as gone without SIGKILL timeout.
    # The group's disappearance is made to DEPEND on the launcher having been
    # reaped — so if reaping is removed, the group never "dies" and this fails.
    run(mgr.load("small"))
    r = mgr.resident["small"]
    state = {"launcher_reaped": False}
    r.process.returncode = 0  # launcher has exited (but stays a zombie until reaped)

    def _reap(res):
        state["launcher_reaped"] = True

    def _alive(res):
        # group looks alive until the launcher zombie is reaped
        return not state["launcher_reaped"]

    kills = []
    monkeypatch.setattr(mgr, "_reap_launcher", _reap)
    monkeypatch.setattr(mgr, "_group_alive", _alive)
    monkeypatch.setattr(mgr, "_signal_group", lambda res, sig: kills.append(sig))
    monkeypatch.setattr(agent.asyncio, "sleep", _instant_sleep)
    out = run(mgr.unload("small"))
    assert out["status"] == "unloaded"
    assert "small" not in mgr.resident
    assert state["launcher_reaped"] is True
    assert 9 not in kills  # SIGKILL was not needed once the zombie was reaped
