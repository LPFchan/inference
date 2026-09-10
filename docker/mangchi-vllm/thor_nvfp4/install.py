"""Fetch and adapt the minimum pinned NVIDIA source; patch one pinned vLLM hook."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
from urllib.request import urlopen

from thor_nvfp4.contract import SOURCE_SHA

KERNEL_DIR = "kernelSrcs/nvfp4_moe_cutedsl/"
SOURCES = {
    "kernelSrcs/gemm_cutedsl/gemm_blackwell_nvfp4_ws.py": "a74e79be79a0b2e67f6432370ed8742a5c6cfaaabeee123f453a8ee3799b2cbd",
    "kernelSrcs/gemm_cutedsl/common.py": "95983027908e8049216f947167817ec75a55659f729c7dbbd52582c9b7c43c51",
    KERNEL_DIR + "blockscaled_contiguous_gather_grouped_gemm_act_fusion.py": "9a1c08088b1614870c6e411eca2107875cdf35b1e54663e5011cff3ccc8f815d",
    KERNEL_DIR + "blockscaled_contiguous_grouped_gemm_finalize_fusion.py": "d441aff45d23eb043f993fb7f1e839473bdba38665d3a69732a69785021c95ba",
    KERNEL_DIR + "custom_pipeline.py": "6c15e7f4473a3e33c5b93f55e1a185f214ab15602b08f18f071e9d8bd1d46b39",
    KERNEL_DIR + "cute_utils.py": "4d9909d3ad2ea160515ca0448b3ec67e08b904cb9f63e7f69669e9f3d2e776c1",
    KERNEL_DIR + "moe_compat.py": "71c4319cfd7c68aa4c332127bec42b923548864449b55b8242c0813d30b0e6c0",
    KERNEL_DIR + "export_common.py": "5d71e0a87ec64341127e7f2ccd372a79de7cfe7f7923529e1feff57caf665f1e",
    KERNEL_DIR + "export_fc1_kernel.py": "e6f19a30223967d87031a274ad4058eaaebaf7da05129659ef7a11ca98ca4cd2",
    KERNEL_DIR + "export_fc2_kernel.py": "81539ff82ec4c3f4cb84e6e6eede20755baf2118726a1431dfbf6782226dfbe7",
    "cpp/kernels/moe/fp4SupportKernels/buildLayout.cu": "91edb32f350115feac820676156ffe3f5c31722accd31ea1fda73b11da91c813",
    "cpp/kernels/moe/fp4SupportKernels/fp4Quantize.cu": "1fdd7fc1e420a06ee04d24f4256ee7fb6ae3b650adf5092643070d002b3abcda",
    "LICENSE": "267b087adacda7c301467b1cbfc396299cb439bb8a606d938d02e6183884cf8f",
}
HOOK = "ModelOptNvFp4Config.FusedMoEMethodCls = ModelOptNvFp4FusedMoE"
PATCHED_HOOK = (
    "from thor_nvfp4.adapter import make_method as _thor_nvfp4_make_method\n"
    "ModelOptNvFp4Config.FusedMoEMethodCls = _thor_nvfp4_make_method(ModelOptNvFp4FusedMoE)"
)


def patch_vllm(text):
    if text.count(PATCHED_HOOK) == 1 and HOOK not in text:
        return text
    if text.count(HOOK) != 1 or "_thor_nvfp4_make_method" in text:
        raise ValueError("Pinned vLLM ModelOpt registration hook changed")
    return text.replace(HOOK, PATCHED_HOOK)


def adapt_python(name, text):
    if name == "gemm_blackwell_nvfp4_ws.py":
        from .dense_source import adapt_dense_source
        return adapt_dense_source(text)
    # Keep helpers private to this package; no generic names added to sys.path.
    for module in ("moe_compat", "custom_pipeline", "cute_utils", "export_common",
                   "blockscaled_contiguous_gather_grouped_gemm_act_fusion",
                   "blockscaled_contiguous_grouped_gemm_finalize_fusion"):
        text = re.sub(rf"\bfrom {module} import", f"from .{module} import", text)
    if name == "export_fc2_kernel.py":
        # The pinned bulk-reduce path reuses shared output storage without a
        # bulk commit/wait. Select its existing register-atomic epilogue, which
        # has no asynchronous shared-memory source lifetime to manage.
        if text.count("        use_blkred=True,") != 1:
            raise ValueError("Pinned FC2 epilogue selection changed")
        text = text.replace("        use_blkred=True,", "        use_blkred=False,")
    if name in ("export_fc1_kernel.py", "export_fc2_kernel.py"):
        target = "    compiled.export_to_c(args.output_dir, args.file_name, args.function_prefix)\n    return verify_export(args.output_dir, args.file_name)"
        if text.count(target) != 1:
            raise ValueError(f"Pinned export contract changed: {name}")
        # The same NVIDIA tracing wrapper now returns its compiled Python callable.
        # Export dummy buffers are released after compilation, before model loading.
        text = text.replace("    os.makedirs(args.output_dir, exist_ok=True)\n", "")
        text = text.replace(target, "    return compiled")
    return text


def patch_dense_vllm(name, text):
    from .dense_source import replace_once
    marker = "# Thor dense NVFP4 integration"
    if name == "linear":
        old = "    config = NvFp4LinearLayerConfig()"
        new = """    # Thor dense NVFP4 integration
    from thor_nvfp4.dense_adapter import select_dense_kernel
    thor_kernel = select_dense_kernel(use_a16)
    if thor_kernel is not None:
        return thor_kernel
    config = NvFp4LinearLayerConfig()"""
    elif name == "ct":
        old = "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n        # Rename CT checkpoint names"
        new = """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Thor dense NVFP4 integration
        from thor_nvfp4.dense_adapter import prepare_dense_method
        if prepare_dense_method(self, layer, divisors=True):
            return
        # Rename CT checkpoint names"""
    elif name == "modelopt":
        old = "    def process_weights_after_loading(self, layer) -> None:\n        self.fmt.pre_process(layer)"
        new = """    def process_weights_after_loading(self, layer) -> None:
        # Thor dense NVFP4 integration
        from thor_nvfp4.dense_adapter import prepare_dense_method
        if prepare_dense_method(self, layer, divisors=False):
            return
        self.fmt.pre_process(layer)"""
    else:
        raise ValueError(f"Unknown dense patch target: {name}")
    if marker in text:
        if text.count(marker) != 1 or text.count(new) != 1:
            raise ValueError("Malformed or duplicate Thor dense integration")
        return text
    return replace_once(text, old, new)


def device_body(text):
    start = "namespace\n{\n"
    end = "\n} // namespace\n"
    if text.count(start) != 1 or end not in text:
        raise ValueError("Pinned anonymous device namespace changed")
    license_header = text[:text.index("*/") + 2]
    body = text.split(start, 1)[1].split(end, 1)[0]
    # The original host TensorRT dispatcher is replaced by prepare.cu's Torch binding.
    # vLLM uses -1 for padded or dropped routes. Match build_layout_kernel by
    # skipping invalid expert ids before the quantizer indexes global scales.
    marker = "        int const expert = topkIds[routedRowIdx];"
    body = body.replace(marker, marker + "\n        if (expert < 0 || expert >= 512) { continue; }")
    return license_header + "\n// Derived by thor_nvfp4/install.py from " + SOURCE_SHA + "\n" + body + "\n"


def fetch_sources():
    result = {}
    for path, expected in SOURCES.items():
        url = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-Edge-LLM/{SOURCE_SHA}/{path}"
        with urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"NVIDIA source hash mismatch: {path}")
        result[path] = data.decode()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args()
    sources = fetch_sources()
    for path, source in sources.items():
        if path.endswith(".py"):
            compile(adapt_python(Path(path).name, source), path, "exec")
        elif path.endswith(".cu"):
            device_body(source)
    if args.check_sources:
        print(f"Verified {len(sources)} pinned NVIDIA sources and adaptations")
        return
    if args.vllm_root is None:
        parser.error("--vllm-root is required for installation")
    target = args.vllm_root / "model_executor/layers/quantization/modelopt.py"
    patched = patch_dense_vllm("modelopt", patch_vllm(target.read_text()))
    linear_target = args.vllm_root / "model_executor/kernels/linear/__init__.py"
    ct_target = args.vllm_root / "model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py"
    linear_patched = patch_dense_vllm("linear", linear_target.read_text())
    ct_patched = patch_dense_vllm("ct", ct_target.read_text())
    root = Path(__file__).parent / "nvidia"
    original = root / "original"
    original.mkdir(parents=True, exist_ok=True)
    for path, source in sources.items():
        name = Path(path).name
        (original / name).write_text(source)
        if name.endswith(".py"):
            installed_name = "dense_common.py" if path == "kernelSrcs/gemm_cutedsl/common.py" else name
            (root / installed_name).write_text(adapt_python(name, source))
        elif name.endswith(".cu"):
            (root / (name + ".inc")).write_text(device_body(source))
        else:
            (root / name).write_text(source)
    (root / "__init__.py").write_text('"""NVIDIA source adapted at the pinned revision."""\n')
    target.write_text(patched)
    linear_target.write_text(linear_patched)
    ct_target.write_text(ct_patched)


if __name__ == "__main__":
    main()
