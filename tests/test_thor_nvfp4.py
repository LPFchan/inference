"""CPU checks for the narrow native Thor adapter; hardware accuracy is separate."""
from pathlib import Path
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/mangchi-vllm"))
from thor_nvfp4.contract import fc1_row_order, permuted_rows, scale_offset, supported
from thor_nvfp4.install import HOOK, PATCHED_HOOK, adapt_python, device_body, patch_vllm


class ThorNvfp4Contracts(unittest.TestCase):
    def setUp(self):
        self.config = dict(sm=(11, 0), quant_method="NVFP4", group_size=16,
                           hidden=2560, intermediate=640, experts=512, top_k=10,
                           activation="silu", dtype="torch.bfloat16",
                           parallel_sizes=(1, 1, 1, 1, 1))

    def test_selection_is_narrow(self):
        self.assertTrue(supported(**self.config))
        for key, value in [("sm", (10, 0)), ("sm", (12, 1)),
                           ("quant_method", "W4A16_NVFP4"), ("group_size", 32),
                           ("hidden", 2048), ("intermediate", 1280), ("experts", 256),
                           ("top_k", 8), ("activation", "relu2"), ("dtype", "torch.float16"),
                           ("parallel_sizes", (2, 1, 1, 1, 1)), ("has_bias", True),
                           ("lora", True), ("swiglu_parameters", (10.0, None, None))]:
            with self.subTest(key=key, value=value):
                self.assertFalse(supported(**{**self.config, key: value}))

    def test_fc1_lossless_interleave(self):
        order = fc1_row_order()
        self.assertEqual(sorted(order), list(range(1280)))
        self.assertEqual(order[:64], list(range(640, 704)))
        self.assertEqual(order[64:128], list(range(64)))
        self.assertEqual(order[-64:], list(range(576, 640)))
        with self.assertRaises(ValueError):
            fc1_row_order(65)

    def test_scale_swizzle_is_bijective(self):
        offsets = {scale_offset(row, col, 256, 8) for row in range(256) for col in range(8)}
        self.assertEqual(offsets, set(range(256 * 8)))
        self.assertEqual(scale_offset(32, 0, 256, 8), 4)
        self.assertEqual(scale_offset(1, 0, 256, 8), 16)
        self.assertEqual(scale_offset(0, 4, 256, 8), 512)
        with self.assertRaises(ValueError):
            scale_offset(256, 0, 256, 8)

    def test_workspace_bound_covers_worst_case_padding(self):
        for tokens in (0, 1, 10, 128, 512, 8192, 262144):
            size = permuted_rows(tokens)
            self.assertEqual(size % 128, 0)
            self.assertGreaterEqual(size, tokens * 10 + 512 * 127)
        for tokens in (-1, 262145, 1.5):
            with self.assertRaises(ValueError):
                permuted_rows(tokens)

    def test_registration_patch_is_idempotent_and_fail_closed(self):
        patched = patch_vllm("before\n" + HOOK + "\nafter\n")
        self.assertIn(PATCHED_HOOK, patched)
        self.assertEqual(patch_vllm(patched), patched)
        for invalid in ("", HOOK + "\n" + HOOK, "_thor_nvfp4_make_method\n" + HOOK):
            with self.assertRaises(ValueError):
                patch_vllm(invalid)

    def test_export_adaptation_preserves_kernel_call(self):
        source = ('from export_common import make_ptr\n'
                  'def export(args):\n    compiled = kernel()\n'
                  '    os.makedirs(args.output_dir, exist_ok=True)\n'
                  '    compiled.export_to_c(args.output_dir, args.file_name, args.function_prefix)\n'
                  '    return verify_export(args.output_dir, args.file_name)\n')
        adapted = adapt_python("export_fc1_kernel.py", source)
        self.assertIn("from .export_common import make_ptr", adapted)
        self.assertIn("compiled = kernel()", adapted)
        self.assertIn("return compiled", adapted)
        self.assertNotIn("export_to_c", adapted)
        compile(adapted, "fixture", "exec")
        with self.assertRaises(ValueError):
            adapt_python("export_fc1_kernel.py", "changed upstream")

    def test_device_extraction_omits_tensor_rt_host_code(self):
        fixture = '/* NVIDIA license */\nnamespace\n{\n__global__ void device() {}\n} // namespace\nvoid host() {}'
        result = device_body(fixture)
        self.assertIn("NVIDIA license", result)
        self.assertIn("__global__ void device()", result)
        self.assertNotIn("void host", result)
        with self.assertRaises(ValueError):
            device_body("changed upstream")

    def test_all_adapter_python_parses(self):
        for path in (ROOT / "docker/mangchi-vllm/thor_nvfp4").glob("*.py"):
            compile(path.read_text(), str(path), "exec")

    def test_adapter_preserves_fallback_and_shared_expert_contract(self):
        class Base:
            def __init__(self, quant_config, moe_config):
                self.original = True

        class MethodBase:
            def __init__(self, moe_config):
                self.moe = moe_config
                self.moe_kernel = None

        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(get_device_capability=lambda device: (11, 0))
        fake_torch.int32, fake_torch.float32 = "int32", "float32"
        fake_base = ModuleType("vllm.model_executor.layers.fused_moe.fused_moe_method_base")
        fake_base.FusedMoEMethodBase = MethodBase
        fake_logger = ModuleType("vllm.logger")
        fake_logger.init_logger = lambda name: SimpleNamespace(info=lambda message: None)
        fake_runtime = ModuleType("thor_nvfp4.runtime")
        fake_runtime.run = lambda *args: args
        modules = {"torch": fake_torch, fake_base.__name__: fake_base,
                   fake_logger.__name__: fake_logger, fake_runtime.__name__: fake_runtime}
        path = ROOT / "docker/mangchi-vllm/thor_nvfp4/adapter.py"
        spec = importlib.util.spec_from_file_location("thor_nvfp4._adapter_test", path)
        adapter = importlib.util.module_from_spec(spec)
        config = SimpleNamespace(quant_method="NVFP4", group_size=16)
        moe = SimpleNamespace(device="cuda:0", hidden_dim=2560,
                              intermediate_size_per_partition=640, num_local_experts=512,
                              num_experts=512, experts_per_token=10,
                              activation=SimpleNamespace(value="silu"), in_dtype="torch.bfloat16",
                              tp_size=1, dp_size=1, ep_size=1, pcp_size=1, sp_size=1,
                              has_bias=False, is_lora_enabled=False, swiglu_limit=None,
                              swiglu_alpha=None, swiglu_beta=None, moe_backend="auto")
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
            spec.loader.exec_module(adapter)
            method = adapter.make_method(Base)
            os.environ.pop("VLLM_THOR_CUTEDSL_MOE", None)
            self.assertIs(type(method(config, moe)), Base)
            os.environ["VLLM_THOR_CUTEDSL_MOE"] = "1"
            native = method(config, moe)
            self.assertIs(type(native), method)
            self.assertFalse(native.supports_eplb)
            self.assertIsNone(native.get_fused_moe_quant_config(None))
            config.quant_method = "W4A16_NVFP4"
            self.assertIs(type(method(config, moe)), Base)
            config.quant_method = "NVFP4"
            moe.moe_backend = "marlin"
            self.assertIs(type(method(config, moe)), Base)
            native._weights = ["prepared"]
            value = SimpleNamespace()
            value.contiguous = lambda: value
            value.to = lambda dtype: value
            output = native.apply(None, value, value, value, object(), value)
            self.assertEqual(output[-1], ["prepared"])


if __name__ == "__main__":
    unittest.main()
