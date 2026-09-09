"""Disable QSA's unsupported cooperative top-k kernel on Jetson AGX Thor."""

import sysconfig
from pathlib import Path


site_packages = Path(sysconfig.get_paths()["purelib"])
qsa_path = site_packages / "vllm/models/qwen4_exp/nvidia/ops/qsa.py"
source = qsa_path.read_text()

old = """            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
"""
new = """            and current_platform.has_device_capability(90)
            # Thor (SM110) rejects this thread-block-cluster launch. Use the
            # existing persistent_topk fallback, as vLLM already does on SM120.
            and not current_platform.is_device_capability_family(110)
            and not current_platform.is_device_capability_family(120)
"""

if old not in source:
    raise RuntimeError(f"QSA cooperative top-k compatibility gate not found in {qsa_path}")

qsa_path.write_text(source.replace(old, new, 1))
print(f"Disabled QSA cooperative top-k on SM110 in {qsa_path}")
