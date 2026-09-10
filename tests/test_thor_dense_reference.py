"""Torch CPU scale/layout tests; no CUDA or vLLM import needed."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/mangchi-vllm"))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from thor_nvfp4.dense_adapter import has_uniform_alpha, prepare_tensors
    from thor_nvfp4.diagnose import unswizzle_scales


@unittest.skipIf(torch is None, "Requires Torch CPU")
class DenseScaleTests(unittest.TestCase):
    def test_hybrid_scalar_requires_exact_uniformity(self):
        alpha = torch.full((5120,), .125)
        self.assertTrue(has_uniform_alpha(alpha, 5120))
        alpha[-1] = torch.nextafter(alpha[-1], torch.tensor(float("inf")))
        self.assertFalse(has_uniform_alpha(alpha, 5120))
        self.assertFalse(has_uniform_alpha(torch.ones(128), 96))
        with self.assertRaises(ValueError):
            has_uniform_alpha(torch.full((5120,), float("nan")), 5120)
    def setUp(self):
        self.w = torch.arange(96 * 2560).remainder(256).to(torch.uint8).reshape(96, 2560)
        self.sf = torch.full((96, 320), .125, dtype=torch.float8_e4m3fn)

    def test_distinct_fused_divisors_and_lossless_padding(self):
        wg, ag = torch.tensor([45056., 16384.]), torch.tensor([53.5, 53.5])
        w, sf, alpha, scale = prepare_tensors(self.w, self.sf, wg, ag, [48, 48], divisors=True)
        self.assertTrue(torch.equal(w[:96], self.w))
        self.assertEqual(w.shape, (128, 2560))
        self.assertFalse(w[96:].any())
        linear_sf = unswizzle_scales(sf, 128, 320)
        self.assertTrue(torch.equal(linear_sf[:96], self.sf.view(torch.uint8)))
        self.assertFalse(linear_sf[96:].any())
        torch.testing.assert_close(alpha[:48], (wg[0].reciprocal() * ag[0].reciprocal()).expand(48), rtol=0, atol=0)
        torch.testing.assert_close(alpha[48:96], (wg[1].reciprocal() * ag[1].reciprocal()).expand(48), rtol=0, atol=0)
        self.assertFalse(alpha[96:].any())
        torch.testing.assert_close(scale, ag[:1].reciprocal(), rtol=0, atol=0)

    def test_modelopt_multipliers_match_ct_divisors(self):
        wg, ag = torch.tensor([2048., 4096.]), torch.tensor([128., 128.])
        ct = prepare_tensors(self.w, self.sf, wg, ag, [48, 48], divisors=True)
        mo = prepare_tensors(self.w, self.sf, wg.reciprocal(), ag.reciprocal(), [48, 48], divisors=False)
        for actual, expected in zip(ct, mo):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_aligned_weight_reuses_checkpoint_storage(self):
        w = torch.zeros((5120, 3072), dtype=torch.uint8)
        sf = torch.ones((5120, 384), dtype=torch.float8_e4m3fn)
        packed, _, _, _ = prepare_tensors(
            w, sf, torch.ones(1), torch.ones(1), [5120], divisors=True
        )
        self.assertEqual(packed.data_ptr(), w.data_ptr())

    def test_distinct_fused_alpha_is_exact_with_each_tile_broadcast(self):
        from thor_nvfp4.dense_contract import LOGICAL_WIDTHS, validate_alpha_layout
        for (n, k), widths in LOGICAL_WIDTHS.items():
            if not validate_alpha_layout(n, k, widths):
                continue
            products = torch.arange(1, len(widths) + 1, dtype=torch.float32) / 8192
            alpha = torch.repeat_interleave(products, torch.tensor(widths))
            for tile in (128, 256):
                broadcast = alpha[::tile].repeat_interleave(tile)
                torch.testing.assert_close(broadcast, alpha, rtol=0, atol=0)

    def test_rejects_unrepresentable_or_invalid_scales(self):
        wg = torch.tensor([1., 2.])
        for ag in (torch.tensor([1., 2.]), torch.tensor([0., 0.]), torch.tensor([float("nan"), 1.])):
            with self.assertRaises(ValueError):
                prepare_tensors(self.w, self.sf, wg, ag, [48, 48], divisors=True)
        with self.assertRaises(ValueError):
            prepare_tensors(self.w, self.sf[:, :160], wg, torch.ones(2), [48, 48], divisors=True)
        with self.assertRaisesRegex(ValueError, "layout"):
            prepare_tensors(self.w, self.sf, wg, torch.ones(2), [32, 64], divisors=True)


if __name__ == "__main__":
    unittest.main()
