"""CPU numerical tests; run with the candidate image's Torch installation."""
from pathlib import Path
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/mangchi-vllm"))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from thor_nvfp4.adapter import swizzle_scales
    from thor_nvfp4.check import quant_dequant, unpack
    from thor_nvfp4 import check
    from thor_nvfp4.diagnose import GateCollector, require_quality, require_repeat_stable, unswizzle_scales, validate_input_rounding, validate_layout


@unittest.skipIf(torch is None, "Requires Torch; runnable on CPU inside the candidate image")
class ThorReferenceTests(unittest.TestCase):
    def test_diagnostic_collector_keeps_later_failures_and_then_raises(self):
        gates = GateCollector(collect_failures=True)
        def fail(message):
            raise AssertionError(message)
        with contextlib.redirect_stdout(io.StringIO()):
            gates.check(fail, "FC2 failure")
            self.assertEqual(gates.check(lambda: "later slot executed"), "later slot executed")
            gates.check(fail, "repeat failure")
        with self.assertRaisesRegex(AssertionError, "(?s)FC2 failure.*repeat failure"):
            gates.finish()
        with self.assertRaisesRegex(AssertionError, "immediate"):
            GateCollector(collect_failures=False).check(fail, "immediate")

    def test_repeat_gate_reports_but_accepts_small_atomic_order_changes(self):
        first = torch.tensor([1., 2.], dtype=torch.bfloat16)
        changed = torch.tensor([1.0078125, 2.], dtype=torch.bfloat16)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            require_repeat_stable("same", first, first.clone())
            require_quality("small error", changed, first)
            require_repeat_stable("different", changed, first)
            require_repeat_stable("reverse", first, changed)
        self.assertIn("bitwise_equal=False", output.getvalue())

    def test_repeat_stability_is_stricter_than_stage_quality(self):
        first = torch.ones(16, dtype=torch.bfloat16)
        with contextlib.redirect_stdout(io.StringIO()):
            # Two BF16 ULPs pass the 2% stage gate but exceed 1% repeat drift.
            changed = first + .015625
            require_quality("stage", changed, first)
            with self.assertRaisesRegex(AssertionError, "1% pairwise repeat-stability"):
                require_repeat_stable("repeat", changed, first)
            for delta in (.03, .05):
                with self.assertRaises(AssertionError):
                    require_repeat_stable("race-sized drift", first + delta, first)

    def test_repeat_stability_rejects_nonfinite_and_handles_zero(self):
        zeros = torch.zeros(16)
        with contextlib.redirect_stdout(io.StringIO()):
            require_repeat_stable("zeros", zeros, zeros.clone())
            for value in (float("nan"), float("inf"), 1.):
                with self.assertRaises(AssertionError):
                    require_repeat_stable("invalid", torch.full_like(zeros, value), zeros)

    def test_input_boundary_exception_is_adjacent_local_and_scale_exact(self):
        values = torch.zeros(1, 16)
        values[0, :2] = torch.tensor([.25, 6])
        _, expected_codes, sf = quant_dequant(values, 1.0, return_fields=True)
        adjacent = expected_codes.clone()
        adjacent[0, 0] = 1
        validate_input_rounding(values, 1.0, adjacent, sf, expected_codes, sf)
        distant = expected_codes.clone()
        distant[0, 0] = 2
        with self.assertRaisesRegex(AssertionError, "adjacent"):
            validate_input_rounding(values, 1.0, distant, sf, expected_codes, sf)
        wrong_sign = adjacent.clone()
        wrong_sign[0, 0] |= 8
        with self.assertRaisesRegex(AssertionError, "sign"):
            validate_input_rounding(values, 1.0, wrong_sign, sf, expected_codes, sf)
        with self.assertRaisesRegex(AssertionError, "scales"):
            validate_input_rounding(values, 1.0, adjacent, sf + 1, expected_codes, sf)
        for value, accepted in ((.25000005, True), (.251, False)):
            values[0, 0] = value
            _, expected_codes, sf = quant_dequant(values, 1.0, return_fields=True)
            changed = expected_codes.clone()
            changed[0, 0] = 0
            if accepted:
                validate_input_rounding(values, 1.0, changed, sf, expected_codes, sf)
            else:
                with self.assertRaisesRegex(AssertionError, "rounding-boundary"):
                    validate_input_rounding(values, 1.0, changed, sf, expected_codes, sf)

    def test_quality_gate_rejects_drift_and_nonfinite_values(self):
        expected = torch.ones(16)
        with contextlib.redirect_stdout(io.StringIO()):
            require_quality("unchanged", expected, expected)
            require_quality("zero", torch.zeros(16), torch.zeros(16))
            for actual in (expected * 1.03, torch.full((16,), float("nan")),
                           torch.full((16,), float("inf"))):
                with self.assertRaisesRegex(AssertionError, "2%"):
                    require_quality("bad", actual, expected)

    def test_fp4_midpoints_round_to_even(self):
        values = torch.tensor([[-6, -5, -3.5, -2.5, -1.75, -1.25, -.75, -.25,
                                .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6]])
        expected = torch.tensor([[-6, -4, -4, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 4, 4, 6]])
        result, codes, scales = quant_dequant(values, 1.0, return_fields=True)
        torch.testing.assert_close(result, expected.float(), rtol=0, atol=0)
        self.assertEqual(codes[0, 7].item(), 8)  # negative zero
        packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        torch.testing.assert_close(unpack(packed, scales.view(torch.float8_e4m3fn), 1.0), result, rtol=0, atol=0)

    def test_per_row_global_scale_matches_scalar_calls(self):
        values = torch.linspace(-3, 3, 64).reshape(2, 32)
        scales = torch.tensor([[.01], [.07]])
        actual = quant_dequant(values, scales)
        expected = torch.cat([quant_dequant(values[i:i + 1], scales[i, 0]) for i in range(2)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(quant_dequant(torch.zeros(2, 32), scales), torch.zeros(2, 32))

    def test_scale_swizzle_inverse_preserves_every_byte(self):
        raw = torch.arange(256 * 8).remainder(256).to(torch.uint8).reshape(1, 256, 8)
        swizzled = swizzle_scales(raw.view(torch.float8_e4m3fn))
        restored = unswizzle_scales(swizzled[0], 256, 8)
        self.assertTrue(torch.equal(restored, raw[0]))

    def test_reference_stage_overrides_preserve_the_same_calculation(self):
        original = [torch.full((2, 32, 8), 0x22, dtype=torch.uint8),
                    torch.full((2, 16, 8), 0x22, dtype=torch.uint8),
                    torch.full((2, 32, 1), .125).to(torch.float8_e4m3fn),
                    torch.full((2, 16, 1), .125).to(torch.float8_e4m3fn),
                    torch.full((2, 2), .1), torch.full((2, 2), .01),
                    torch.full((2,), .1), torch.full((2,), .01)]
        x = torch.linspace(.1, 1, 16).reshape(1, 16)
        ids = torch.tensor([[1, 0]], dtype=torch.int32)
        routes = torch.tensor([[.3, .7]])
        with patch.object(check, "INTERMEDIATE", 16), patch.object(check, "TOP_K", 2):
            trace = {}
            expected = check.reference(x, ids, routes, original, trace=trace)
            actual_input = quant_dequant(x.repeat_interleave(2, 0), .01)
            from_input = check.reference(x, ids, routes, original, input_override=actual_input)
            from_intermediate = check.reference(x, ids, routes, original, intermediate_override=trace["intermediate"])
        torch.testing.assert_close(from_input, expected, rtol=0, atol=0)
        torch.testing.assert_close(from_intermediate, expected, rtol=0, atol=0)

    def test_routing_validation_rejects_duplicate_and_wrong_expert_rows(self):
        ids = torch.arange(10, dtype=torch.int32).reshape(1, 10)
        mapping = torch.full((1280,), -1, dtype=torch.int32)
        mapping[::128] = torch.arange(10, dtype=torch.int32)
        trace = dict(mapping=mapping, groups=torch.arange(10, dtype=torch.int32),
                     limits=torch.arange(10, dtype=torch.int32) * 128 + 1,
                     tile_count=torch.tensor([10], dtype=torch.int32))
        rows, expanded = validate_layout(trace, ids)
        self.assertTrue(torch.equal(rows, torch.arange(10) * 128))
        self.assertTrue(torch.equal(expanded, torch.arange(10)))
        mapping[128] = 0
        with self.assertRaisesRegex(AssertionError, "exactly once"):
            validate_layout(trace, ids)
        mapping[128] = 1
        trace["groups"][1] = 0
        with self.assertRaisesRegex(AssertionError, "does not match"):
            validate_layout(trace, ids)


if __name__ == "__main__":
    unittest.main()
