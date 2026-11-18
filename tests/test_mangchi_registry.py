import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MangchiRegistryTests(unittest.TestCase):
    def _models(self):
        return json.loads((ROOT / "etc" / "models.mangchi.json").read_text())["models"]

    def test_no_bare_valued_flags_in_extra_args(self):
        # A bare --flash-attn in extra-args would be appended after the
        # gateway's own `--flash-attn on` and greedily consume the next token
        # (e.g. --chat-template-kwargs) as its value, breaking startup.
        for name, cfg in self._models().items():
            extra = cfg.get("extra-args", [])
            self.assertNotIn("--flash-attn", extra, f"{name} re-adds bare --flash-attn")


if __name__ == "__main__":
    unittest.main()
