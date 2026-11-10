import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Qwen38RegistryTests(unittest.TestCase):
    def test_flash_next_remote_alias_matches_the_served_snapshot(self):
        models = json.loads((ROOT / "etc" / "models.grimoire.json").read_text())["models"]
        config = models["qwen3.8-flash-next"]

        self.assertEqual(
            config["backend-model-id"],
            "qwen3.8-flash-next",
        )
        self.assertEqual(config["ctx-size"], 393216)


if __name__ == "__main__":
    unittest.main()
