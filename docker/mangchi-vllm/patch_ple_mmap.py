"""Install the optional Qwen4Exp PLE mmap hook into vLLM."""

import sysconfig
from pathlib import Path

target = (
    Path(sysconfig.get_paths()["purelib"])
    / "vllm"
    / "models"
    / "qwen4_exp"
    / "nvidia"
    / "ple_layer.py"
)
hook = (
    "\n# Optional SSD-backed PLE table (adapted from blazux/qwen3.8-Flash-DGX).\n"
    "from vllm_ple_mmap import apply as _apply_ple_mmap\n"
    "_apply_ple_mmap(Qwen4ExpNGramEmbedding)\n"
)
text = target.read_text()
if "_apply_ple_mmap(Qwen4ExpNGramEmbedding)" not in text:
    target.write_text(text + hook)
print(f"installed PLE mmap hook in {target}")
