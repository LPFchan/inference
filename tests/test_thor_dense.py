"""Dependency-free contract and patch guards for the dense opt-in."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/mangchi-vllm"))
from thor_nvfp4.dense_contract import LOGICAL_WIDTHS, SHAPES, select_tile_n, use_cutlass, validate_alpha_layout, validate_shape
from thor_nvfp4.dense_source import replace_once
from thor_nvfp4.install import SOURCES, patch_dense_vllm


class DenseContractTests(unittest.TestCase):
    def test_hybrid_requires_uniform_alpha_and_large_m(self):
        for m in (1, 157, 511, 512, 608, 1567, 1568, 2048, 262144):
            self.assertEqual(use_cutlass(m, 34816, 5120, True), m >= 1568)
            self.assertFalse(use_cutlass(m, 34816, 5120, False))
        with self.assertRaises(ValueError):
            use_cutlass(2048, 96, 5120, True)
        with self.assertRaises(ValueError):
            use_cutlass(2048, 34816, 5120, 1)

    def test_alpha_boundaries_cover_both_reachable_tiles(self):
        for (n, k), widths in LOGICAL_WIDTHS.items():
            self.assertEqual(validate_alpha_layout(n, k, widths), n != 96)
        with self.assertRaisesRegex(ValueError, "layout"):
            validate_alpha_layout(34816, 5120, (17344, 17472))
        # Even a future allowlisted layout must satisfy the alignment invariant.
        from unittest.mock import patch
        with patch.dict(LOGICAL_WIDTHS, {(34816, 5120): (17344, 17472)}):
            with self.assertRaisesRegex(ValueError, "boundary"):
                validate_alpha_layout(34816, 5120, (17344, 17472))

    def test_exact_shapes_and_padding(self):
        for n, k in SHAPES:
            self.assertEqual(validate_shape(n, k, [n]), max(n, 128))
        for n, k, widths in ((128, 5120, [128]), (96, 2560, [96]), (96, 5120, [48])):
            with self.assertRaises(ValueError):
                validate_shape(n, k, widths)

    def test_dense_source_is_pinned(self):
        self.assertEqual(SOURCES["kernelSrcs/gemm_cutedsl/gemm_blackwell_nvfp4_ws.py"],
                         "a74e79be79a0b2e67f6432370ed8742a5c6cfaaabeee123f453a8ee3799b2cbd")
        with self.assertRaises(ValueError):
            replace_once("bad bad", "bad", "ok")

    def test_vllm_patches_fail_closed_and_idempotent(self):
        fixtures = {
            "linear": "def init(use_a16=False):\n    config = NvFp4LinearLayerConfig()\n",
            "ct": "class C:\n    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n        # Rename CT checkpoint names\n        pass\n",
            "modelopt": "class C:\n    def process_weights_after_loading(self, layer) -> None:\n        self.fmt.pre_process(layer)\n",
        }
        for name, text in fixtures.items():
            patched = patch_dense_vllm(name, text)
            ast.parse(patched)
            self.assertEqual(patched, patch_dense_vllm(name, patched))
            with self.assertRaises(ValueError):
                patch_dense_vllm(name, "changed")
            with self.assertRaises(ValueError):
                patch_dense_vllm(name, patched.replace("return", "raise"))

    def test_standalone_gate_covers_all_shapes_and_sizes(self):
        tree = ast.parse((ROOT / "docker/mangchi-vllm/thor_nvfp4/dense_check.py").read_text())
        cases = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and n.targets[0].id == "CASES")
        self.assertEqual({(n, k) for n, k, _ in cases}, SHAPES)
        self.assertTrue(any(isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                            and n.target.id == "m" and ast.literal_eval(n.iter) == (1, 7, 129, 512, 608, 1568, 2048, 2205)
                            for n in ast.walk(tree)))

    def test_two_variant_dispatch_boundaries_and_small_output(self):
        for n, k in SHAPES:
            for m in (1, 7, 129, 511, 512, 608, 1567, 1568, 2048, 262144):
                expected = 256 if (n, k) == (5120, 6144) or (m >= 512 and n != 96) else 128
                self.assertEqual(select_tile_n(m, n, k), expected)
        with self.assertRaises(ValueError):
            select_tile_n(262145, 5120, 6144)

    def test_serving_precompiles_both_tiles_and_reports_ready_backend(self):
        runtime = (ROOT / "docker/mangchi-vllm/thor_nvfp4/dense_runtime.py").read_text()
        tree = ast.parse(runtime)
        warmup = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "warmup_dense")
        compiled = [n for n in ast.walk(warmup) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "_compile_dense"]
        self.assertEqual({n.args[1].value for n in compiled if isinstance(n.args[1], ast.Constant)}, {128, 256})
        self.assertIn("if (0, tile_n, tile_uniform_alpha) not in _READY_TILES:", runtime)
        self.assertIn("_compile_dense(device_index, 128, False)", runtime)
        self.assertIn("_compile_dense(device_index, 128, True)", runtime)
        self.assertIn("_compile_dense(device_index, 256, True)", runtime)
        self.assertIn("tile_uniform_alpha = output_size != 96", runtime)
        adapter = (ROOT / "docker/mangchi-vllm/thor_nvfp4/dense_adapter.py").read_text()
        self.assertLess(adapter.index("warmup_dense(weight.device.index)"), adapter.index("THOR_DENSE_BACKEND ready"))
        self.assertIn("flush=True", adapter)

    def test_hybrid_shares_one_quantization_and_fixes_cache_keys(self):
        runtime = (ROOT / "docker/mangchi-vllm/thor_nvfp4/dense_runtime.py").read_text()
        self.assertEqual(runtime.count("helper.quantize("), 1)
        self.assertIn('backend="cutlass"', runtime)
        tree = ast.parse(runtime)
        warm = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "warmup_dense")
        calls = [n for n in ast.walk(warm) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_compile_dense"]
        self.assertTrue(all(len(call.args) == 3 for call in calls))

    def test_runtime_rejects_unaccepted_tiles_before_cuda_access(self):
        tree = ast.parse((ROOT / "docker/mangchi-vllm/thor_nvfp4/dense_runtime.py").read_text())
        compile_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_compile_dense")
        # Execute the first guard alone: rejected tactics must not need Torch,
        # CUDA, cache state, or a supported GPU to fail closed.
        guard = ast.Module(body=[compile_fn.body[0]], type_ignores=[])
        code = compile(ast.fix_missing_locations(guard), "tile_guard", "exec")
        for tile in (0, 64, 192, 512):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                exec(code, {"tile_n": tile})
        for tile in (128, 256):
            exec(code, {"tile_n": tile})
        launch_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "launch_dense")
        tile_checks = [n for n in ast.walk(launch_fn) if isinstance(n, ast.Compare)
                       and isinstance(n.left, ast.Name) and n.left.id == "tile_n"
                       and isinstance(n.ops[0], ast.NotIn)]
        self.assertEqual([ast.literal_eval(n.comparators[0]) for n in tile_checks], [(128, 256)])
        bench = ast.parse((ROOT / "docker/mangchi-vllm/thor_nvfp4/dense_bench.py").read_text())
        tiles = next(ast.literal_eval(n.value) for n in bench.body if isinstance(n, ast.Assign)
                     and n.targets[0].id == "TILES")
        self.assertEqual(tiles, (128, 256))

    def test_compiled_launch_omits_constexpr_parameters(self):
        tree = ast.parse((ROOT / "docker/mangchi-vllm/thor_nvfp4/dense_runtime.py").read_text())
        launch = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "gemm")
        self.assertEqual(len(launch.args), 9)  # six pointers, M/N/K
        self.assertEqual({k.arg for k in launch.keywords}, {"max_active_clusters", "stream"})


if __name__ == "__main__":
    unittest.main()
