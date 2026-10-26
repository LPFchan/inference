"""NVFP4 Qwen4Exp PLE row decode for the SSD-backed table (vllm_ple_mmap).

Decode from vllm-project/vllm#56273 (Apache-2.0). FP4-Local-format checkpoints
store the n-gram table as packed E2M1 rows (two values per byte), one E4M3
block scale per 16 values, and one FP32 global scale. Only gathered rows are
decoded to the activation dtype.
"""

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
)
from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 16


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_rows(
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """CPU reference decode: low nibble is the even element."""
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    values = table[codes & 0x7] * torch.where(codes >= 8, -1.0, 1.0)
    block_scales = scales.float().repeat_interleave(_BLOCK_SIZE, dim=-1)
    return (values * block_scales * global_scale.float().reshape(())).to(dtype)


@triton.jit
def _e4m3_inline(byte):
    """Decode an FP8 E4M3FN byte to float32 without an FP8 pointer type."""
    bits = byte.to(tl.int32)
    exponent = (bits >> 3) & 0xF
    mantissa = (bits & 0x7).to(tl.float32)
    normal = (1.0 + mantissa * 0.125) * tl.exp2((exponent - 7).to(tl.float32))
    val = tl.where(exponent == 0, mantissa * 0.001953125, normal)
    val = tl.where((bits & 0x7F) == 0x7F, float("nan"), val)
    return tl.where((bits & 0x80) != 0, -val, val)


@triton.jit
def _lookup_nvfp4_ple_embedding_kernel(
    weight_ptr,
    scale_ptr,
    global_scale_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    BLOCK_D: tl.constexpr,
):
    """Decode one NVFP4 PLE row per program into the activation dtype."""
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    packed = tl.load(
        weight_ptr + idx * (embedding_dim // 2) + offsets // 2, mask=mask, other=0
    )
    # The low nibble holds the even element and the high nibble the odd one.
    codes = (packed >> ((offsets % 2) * 4)) & 0xF
    scales = _e4m3_inline(
        tl.load(scale_ptr + idx * (embedding_dim // 16) + offsets // 16, mask=mask, other=0)
    )
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    values = _e2m1_inline(codes) * scales * global_scale
    tl.store(output_ptr + row * embedding_dim + offsets, values, mask=mask)
