import os
import sys
import tempfile
import unittest
from pathlib import Path


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

    def test_concurrent_conversations_spread_across_replicas(self):
        seen = set()
        for i in range(8):
            name = self.manager.pick_replica("chat", f"conv-{i}", "user")
            self.manager.active[name].inflight += 1
            seen.add(name)
        self.assertEqual(seen, {"chat", "chat-g1"})
        self.assertEqual(self.manager.active["chat"].inflight, self.manager.active["chat-g1"].inflight)


if __name__ == "__main__":
    unittest.main()
