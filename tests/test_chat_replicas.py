import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-usage.sqlite3"))

import grimoire.config as config
import grimoire.model_manager as mm_module


CONFIGS = {
    "chat": {"file": "chat.gguf", "replica_peers": ["chat-g1"]},
    "chat-g1": {"file": "chat.gguf"},
    "solo": {"file": "solo.gguf"},
}


class FakeRegistry:
    def get(self, name):
        return CONFIGS.get(name)


class FakeActive:
    def __init__(self, name, inflight=0, loaded=True):
        self.name = name
        self.cfg = CONFIGS[name]
        self.inflight = inflight
        self.status = config.MODEL_STATUS_LOADED if loaded else config.MODEL_STATUS_LOADING

    def is_running(self):
        return True


class PickReplicaTests(unittest.TestCase):
    def setUp(self):
        self._old_registry = mm_module.registry
        mm_module.registry = FakeRegistry()
        self.manager = mm_module.ModelManager(gpu_count=2)
        self.manager.active = {"chat": FakeActive("chat"), "chat-g1": FakeActive("chat-g1")}

    def tearDown(self):
        mm_module.registry = self._old_registry

    def test_model_without_peers_serves_itself(self):
        self.manager.active["solo"] = FakeActive("solo")
        self.assertEqual(self.manager.pick_replica("solo", "conv", "user"), "solo")

    def test_unloaded_peer_is_skipped(self):
        self.manager.active["chat-g1"] = FakeActive("chat-g1", loaded=False)
        self.assertEqual(self.manager.pick_replica("chat", "conv", "user"), "chat")

    def test_only_peer_loaded_serves_without_cold_loading_primary(self):
        del self.manager.active["chat"]
        self.assertEqual(self.manager.pick_replica("chat", None, "user"), "chat-g1")

    def test_least_busy_replica_wins(self):
        self.manager.active["chat"].inflight = 1
        self.assertEqual(self.manager.pick_replica("chat", None, "user"), "chat-g1")
        self.manager.active["chat"].inflight = 0
        self.manager.active["chat-g1"].inflight = 2
        self.assertEqual(self.manager.pick_replica("chat", None, "user"), "chat")

    def test_idle_caller_keeps_the_same_replica(self):
        picks = {self.manager.pick_replica("chat", None, "user") for _ in range(5)}
        self.assertEqual(len(picks), 1)

    def test_conversation_stays_on_its_replica_even_when_busier(self):
        first = self.manager.pick_replica("chat", "conv", "user")
        self.manager.active[first].inflight = 3
        self.assertEqual(self.manager.pick_replica("chat", "conv", "user"), first)

    def test_conversation_moves_when_its_replica_stops(self):
        first = self.manager.pick_replica("chat", "conv", "user")
        del self.manager.active[first]
        other = "chat-g1" if first == "chat" else "chat"
        self.assertEqual(self.manager.pick_replica("chat", "conv", "user"), other)

    def test_conversation_started_on_one_copy_stays_when_peer_loads(self):
        del self.manager.active["chat-g1"]
        self.assertEqual(self.manager.pick_replica("chat", "conv", "user"), "chat")
        self.manager.active["chat"].inflight = 3
        self.manager.active["chat-g1"] = FakeActive("chat-g1")
        self.assertEqual(self.manager.pick_replica("chat", "conv", "user"), "chat")

    def test_concurrent_conversations_spread_across_replicas(self):
        seen = set()
        for i in range(8):
            name = self.manager.pick_replica("chat", f"conv-{i}", "user")
            self.manager.active[name].inflight += 1
            seen.add(name)
        self.assertEqual(seen, {"chat", "chat-g1"})
        self.assertEqual(self.manager.active["chat"].inflight, self.manager.active["chat-g1"].inflight)


MIRROR_CONFIGS = {
    "chat": {"file": "chat.gguf"},
    "other": {"file": "other.gguf"},
    "encoder": {"file": "enc.gguf", "vram-budget-mib": 1000},
}


class MirrorRegistry:
    def resolve(self, name):
        return name if name in MIRROR_CONFIGS else None

    def get(self, name):
        return MIRROR_CONFIGS.get(name)

    def validate(self, name, gpu_count=None):
        return True, None

    def get_fixed_gpu(self, name):
        return None

    def is_fixed(self, name):
        return False

    def get_family_defaults(self, family):
        return {}


class _Process:
    def poll(self):
        return None


def _fake_start(active):
    active.process = _Process()


def _fake_stop(active):
    active.process = None


async def _ready(active, timeout=None):
    return None


class MirrorModeTests(unittest.TestCase):
    def setUp(self):
        self.manager = mm_module.ModelManager(gpu_count=2)
        self.manager.mirror = True
        self.patches = [
            patch.object(mm_module, "registry", MirrorRegistry()),
            patch.object(mm_module.ActiveModel, "start", _fake_start),
            patch.object(mm_module.ActiveModel, "stop", _fake_stop),
            patch.object(mm_module.ActiveModel, "wait_ready", _ready),
            patch.object(self.manager, "_publish_routes", lambda: None),
            patch.object(self.manager, "_get_gpu_free_vram_mib", lambda gpu: 20000),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def run_async(self, coro):
        return asyncio.run(coro)

    def gpus(self):
        return {name: active.gpu for name, active in self.manager.active.items()}

    def test_loading_a_model_starts_a_hidden_twin_on_the_other_gpu(self):
        self.run_async(self.manager.start_model("chat"))
        self.assertEqual(self.gpus(), {"chat": 0, "chat@mirror": 1})
        self.assertEqual(self.manager.list_active(), ["chat"])
        self.assertEqual(self.manager.mirror_gpu("chat"), 1)
        self.assertIsNone(self.manager.get_active("chat").twin_of)

    def test_chat_spreads_across_model_and_twin(self):
        self.run_async(self.manager.start_model("chat"))
        seen = set()
        for i in range(4):
            name = self.manager.pick_replica("chat", f"conv-{i}", "user")
            self.manager.active[name].inflight += 1
            seen.add(name)
        self.assertEqual(seen, {"chat", "chat@mirror"})

    def test_picked_twin_is_served_without_a_registry_entry(self):
        self.run_async(self.manager.start_model("chat"))
        twin = self.run_async(self.manager.start_replica("chat", "chat@mirror"))
        self.assertEqual(twin.name, "chat@mirror")
        self.run_async(self.manager._stop_twin("chat"))
        self.assertEqual(self.run_async(self.manager.start_replica("chat", "chat@mirror")).name, "chat")

    def test_chat_route_sends_a_picked_twin_its_request(self):
        import grimoire.entrypoint as entrypoint

        self.run_async(self.manager.start_model("chat"))
        served = []

        async def attempt(active):
            served.append(active.name)

        with patch.object(entrypoint, "manager", self.manager), \
             patch.object(entrypoint, "registry", MirrorRegistry()), \
             patch.object(entrypoint, "is_admin", lambda request: True):
            self.run_async(entrypoint._with_fallback("chat", None, attempt, replica="chat@mirror"))
        self.assertEqual(served, ["chat@mirror"])

    def test_route_table_lists_twin_as_a_replica_only(self):
        self.run_async(self.manager.start_model("chat"))
        routes = self.manager._build_route_table()
        self.assertEqual(list(routes), ["chat"])
        self.assertEqual(len(routes["chat"]["replicas"]), 2)

    def test_stopping_a_model_stops_its_twin(self):
        self.run_async(self.manager.start_model("chat"))
        self.assertTrue(self.run_async(self.manager.stop_model("chat")))
        self.assertEqual(self.manager.active, {})

    def test_loading_another_model_replaces_both_copies(self):
        self.run_async(self.manager.start_model("chat"))
        self.run_async(self.manager.start_model("other"))
        self.assertEqual(sorted(self.gpus().items()), [("other", 0), ("other@mirror", 1)])

    def test_budgeted_twin_only_where_it_fits(self):
        self.run_async(self.manager.start_model("chat"))
        with patch.object(self.manager, "_get_gpu_free_vram_mib", lambda gpu: 20000 if gpu == 0 else 500):
            self.run_async(self.manager.start_model("encoder"))
        self.assertNotIn("encoder@mirror", self.manager.active)
        self.assertIn("chat@mirror", self.manager.active)

    def test_failed_twin_leaves_the_model_serving(self):
        async def fail_twin(active, timeout=None):
            if active.name.endswith("@mirror"):
                raise RuntimeError("out of memory")

        with patch.object(mm_module.ActiveModel, "wait_ready", fail_twin):
            active = self.run_async(self.manager.start_model("chat"))
        self.assertEqual(active.name, "chat")
        self.assertEqual(self.gpus(), {"chat": 0})

    def test_no_twin_outside_mirror_mode(self):
        self.manager.mirror = False
        self.run_async(self.manager.start_model("chat"))
        self.assertEqual(self.gpus(), {"chat": 0})

    def test_leaving_mirror_mode_stops_twins(self):
        self.run_async(self.manager.start_model("chat"))
        self.run_async(self.manager.prepare_preset_activation(
            "free", manual_control=True, gpu_mask=None, mirror=False))
        self.assertFalse(self.manager.mirror)
        self.assertEqual(self.gpus(), {"chat": 0})


if __name__ == "__main__":
    unittest.main()
