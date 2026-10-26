"""CPU checks for the CSF scale decoder ported from vllm#60646."""
from pathlib import Path
import sys
from types import ModuleType
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/mangchi-vllm"))
logger_stub = ModuleType("vllm.logger")
logger_stub.init_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("vllm", ModuleType("vllm"))
sys.modules.setdefault("vllm.logger", logger_stub)
from vllm_csf_scales import decode_csf_scale_streams, decode_nvfp4_csf_scale


def encode(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference encoder: row base = row min (capped at 240), outliers as exceptions."""
    raw = scales.view(torch.uint8).to(torch.int64)
    rows, _ = raw.shape
    bases = raw.min(dim=1, keepdim=True).values.clamp(max=240)
    codes = raw - bases
    outlier = (codes < 0) | (codes > 15)
    codes = torch.where(outlier, 0, codes)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    fixed = torch.cat(
        (bases.to(torch.uint8).reshape(rows // 16, 16), packed.reshape(rows // 16, -1)),
        dim=1,
    )
    positions = outlier.reshape(-1).nonzero().reshape(-1)
    words = (raw.reshape(-1)[positions] << 24) | positions
    return fixed, words.to(torch.int32).view(torch.uint32)


class CsfScales(unittest.TestCase):
    def test_decode_matches_source_scales(self):
        torch.manual_seed(0)
        # Expert gate/up and down scale shapes in Qwen3.8 Flash-Next.
        for shape in ((640, 160), (2560, 40)):
            raw = torch.randint(0x30, 0x40, shape, dtype=torch.uint8)
            raw.view(-1)[torch.randperm(raw.numel())[:64]] = 0x7E
            fixed, exceptions = encode(raw.view(torch.float8_e4m3fn))
            self.assertGreaterEqual(exceptions.numel(), 64)
            decoded = decode_nvfp4_csf_scale(fixed, exceptions)
            self.assertEqual(decoded.dtype, torch.float8_e4m3fn)
            self.assertTrue(torch.equal(decoded.view(torch.uint8), raw))

    def test_stream_pairs_are_replaced_by_the_scale(self):
        raw = torch.full((16, 4), 0x38, dtype=torch.uint8)
        fixed, exceptions = encode(raw.view(torch.float8_e4m3fn))
        out = dict(decode_csf_scale_streams([
            ("e.weight", torch.zeros(1)),
            ("e.weight_scale.nvfp4_csf_exceptions", exceptions),
            ("e.weight_scale.nvfp4_csf_fixed", fixed),
        ]))
        self.assertEqual(list(out), ["e.weight", "e.weight_scale"])
        self.assertTrue(torch.equal(out["e.weight_scale"].view(torch.uint8), raw))

    def test_unpaired_stream_is_rejected(self):
        fixed, _ = encode(torch.full((16, 4), 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn))
        with self.assertRaisesRegex(ValueError, "without a partner"):
            list(decode_csf_scale_streams([("e.weight_scale.nvfp4_csf_fixed", fixed)]))


if __name__ == "__main__":
    unittest.main()
