"""CPU checks for the narrow native Thor adapter; hardware accuracy is separate."""
from pathlib import Path
import ast
import importlib.util
import os
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/mangchi-vllm"))
from thor_nvfp4.contract import fc1_row_order, permuted_rows, scale_offset, supported
from thor_nvfp4.install import HOOK, PATCHED_HOOK, adapt_python, device_body, patch_vllm


class ThorNvfp4Contracts(unittest.TestCase):
    def test_all_cases_gate_stages_and_replay_graphs(self):
        source = ast.parse((ROOT / "docker/mangchi-vllm/thor_nvfp4/check.py").read_text())
        case_loop = next(node for node in ast.walk(source) if isinstance(node, ast.For)
                         and isinstance(node.target, ast.Tuple)
                         and all(isinstance(item, ast.Name) for item in node.target.elts)
                         and [item.id for item in node.target.elts] == ["tokens", "pattern"])
        self.assertEqual(ast.literal_eval(case_loop.iter),
                         ((1, "shared"), (10, "shared"), (129, "shared"), (32, "scattered")))
        calls = [node for node in ast.walk(case_loop) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id == "validate_case" for node in calls))
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == "replay" for node in calls))
        gates = ast.parse((ROOT / "docker/mangchi-vllm/thor_nvfp4/diagnose.py").read_text())
        calls = [node for node in ast.walk(gates) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.args and isinstance(node.args[0], ast.Constant)]
        collected = [node for node in ast.walk(gates) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "check"
                     and len(node.args) > 1 and isinstance(node.args[0], ast.Name)]
        enforced = {node.args[1].value for node in collected if node.args[0].id == "require_quality"
                    and isinstance(node.args[1], ast.Constant)}
        self.assertTrue({"fc1_swiglu_requant_using_native_input",
                         "fc2_finalize_using_native_intermediate", "scatter_vs_sum_of_isolated_native_slots"} <= enforced)
        self.assertNotIn("input_quantization", enforced)
        self.assertTrue(any(node.args[0].id == "validate_input_rounding" for node in collected))
        input_metrics = [node for node in calls if node.args[0].value == "informational_input_quantization"]
        self.assertEqual(len(input_metrics), 1)
        self.assertEqual(input_metrics[0].func.id, "metrics")
        informational = [node for node in calls if node.args[0].value == "informational_end_to_end_quantization_sensitivity"]
        self.assertEqual(len(informational), 1)
        self.assertEqual(informational[0].func.id, "metrics")
        self.assertTrue(any(node.args[0].id == "require_repeat_stable" for node in collected))

    def test_fc2_uses_register_atomic_epilogue(self):
        source = ('def export(args):\n    kernel = Kernel(\n        use_blkred=True,\n    )\n'
                  '    compiled = compile_kernel(kernel)\n'
                  '    compiled.export_to_c(args.output_dir, args.file_name, args.function_prefix)\n'
                  '    return verify_export(args.output_dir, args.file_name)\n')
        adapted = adapt_python("export_fc2_kernel.py", source)
        self.assertIn("use_blkred=False", adapted)
        self.assertNotIn("use_blkred=True", adapted)
        compile(adapted, "fc2-fixture", "exec")
        with self.assertRaisesRegex(ValueError, "epilogue selection"):
            adapt_python("export_fc2_kernel.py", source.replace("use_blkred=True", "use_blkred=unknown"))

    def test_compiled_launches_pass_only_runtime_arguments(self):
        source = (ROOT / "docker/mangchi-vllm/thor_nvfp4/runtime.py").read_text()
        calls = {node.func.id: node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id in ("fc1", "fc2")}
        self.assertEqual(set(calls), {"fc1", "fc2"})
        # Pinned FC1: 13 pointers + 5 Int64 dimensions. FC2: 12 pointers +
        # 6 Int64 dimensions. Both retain dynamic cluster count and stream.
        # Constexpr tile_size/scaling_vector_size/activation_type are pruned.
        for name, call in calls.items():
            with self.subTest(kernel=name):
                self.assertEqual(len(call.args), 18)
                self.assertEqual({kw.arg for kw in call.keywords}, {"max_active_clusters", "stream"})
                self.assertEqual(len(call.keywords), 2)
        self.assertEqual([ast.unparse(arg) for arg in calls["fc1"].args[-5:]],
                         ["tokens * TOP_K", "padded", "2 * INTERMEDIATE", "HIDDEN", "EXPERTS"])
        self.assertEqual([ast.unparse(arg) for arg in calls["fc2"].args[-6:]],
                         ["padded", "HIDDEN", "INTERMEDIATE", "EXPERTS", "tokens", "TOP_K"])

    def test_cuda_dependency_pins_have_available_index_and_compatible_versions(self):
        # Published cuda-python 13.3.1 requires cuda-bindings~=13.3.1;
        # Torch 2.11's CUDA wheel requires cuda-bindings>=13.0.3,<14.
        # The inherited Jetson index only exposes 13.0.1 for these packages.
        dockerfile = (ROOT / "docker/mangchi-vllm/Dockerfile").read_text()
        install = dockerfile.split('RUN if [ "${THOR_CUTEDSL_MOE}" = "1" ]; then', 1)[1].split("; fi", 1)[0]
        self.assertIn("--extra-index-url https://pypi.org/simple", install)
        pins = dict(re.findall(r'"(cuda-python|cuda-bindings)==([0-9.]+)"', install))
        self.assertEqual(set(pins), {"cuda-python", "cuda-bindings"})
        umbrella = tuple(map(int, pins["cuda-python"].split(".")))
        bindings = tuple(map(int, pins["cuda-bindings"].split(".")))
        self.assertEqual(umbrella[:2], bindings[:2])
        self.assertGreaterEqual(bindings, umbrella)
        self.assertGreaterEqual(bindings, (13, 0, 3))
        self.assertLess(bindings, (14,))

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
