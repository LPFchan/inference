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

    def wait(self):
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
    m = ResidencyManager(_specs(), budget_gib=110)

    async def _no_healthy(r):
        return None

    monkeypatch.setattr(m, "_wait_healthy", _no_healthy)
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
    assert flash.env.get("VLLM_PLE_CPU_OFFLOAD") == "1"
    assert flash.resident_gb > specs["qwen3.8-27b-uncensored-nvfp4"].resident_gb
