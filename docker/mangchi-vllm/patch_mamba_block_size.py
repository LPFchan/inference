"""Fix hybrid-model prefix caching: use the mamba block size for state seeding/splitting.

Ported from blazux/qwen3.8-Flash-DGX src/patch_mamba_block_size.py (Apache-2.0),
diagnosed 2026-08-29 on GB10. EngineCore._initialize_kv_caches sets
cache_config.block_size to the MIN group block size; for Qwen3.8-Flash-Next one
KV group is the QSA raw-key ring (block 8 with MTP, 4 without), while the GDN
Mamba block is 1600. Two consumers used cache_config.block_size as the Mamba
block size: the align-mode state-slot seed in mamba_hybrid.py (a prefix hit
seeds a slot past the request's block-table row, restoring the all-zero null
block) and the scheduler's block-aligned prefill split (states almost never
captured at a real Mamba boundary, so cold requests rarely cache anything).
"""

import sysconfig
from pathlib import Path

site_packages = Path(sysconfig.get_paths()["purelib"])
mamba_hybrid = site_packages / "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
scheduler = site_packages / "vllm/v1/core/sched/scheduler.py"

source = mamba_hybrid.read_text()
old = (
    "                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size\n"
)
new = (
    "                (new_req_data.num_computed_tokens - 1)\n"
    "                // (self.cache_config.mamba_block_size or self.cache_config.block_size)\n"
)
if old not in source:
    raise RuntimeError(f"mamba_hybrid state-seed line not found in {mamba_hybrid}")
mamba_hybrid.write_text(source.replace(old, new, 1))

source = scheduler.read_text()
old = (
    "        block_size = self.cache_config.block_size\n"
    "        # The last block-aligned position whose state can be cached."
)
new = (
    "        block_size = self.block_size  # scheduler block size (LCM of groups) == mamba block size\n"
    "        # The last block-aligned position whose state can be cached."
)
if old not in source:
    raise RuntimeError(f"scheduler block-aligned split line not found in {scheduler}")
scheduler.write_text(source.replace(old, new, 1))

print("Applied mamba block_size fix in mamba_hybrid.py and scheduler.py")
