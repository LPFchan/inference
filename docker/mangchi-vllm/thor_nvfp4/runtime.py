"""NVIDIA CuTe DSL split FC1/FC2 launch and Torch-owned scratch memory."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch

from .contract import EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, permuted_rows


@lru_cache(maxsize=1)
def warmup(device_index):
    """Compile before vLLM starts CUDA graph capture; do not JIT inside forward."""
    if device_index != 0:
        raise ValueError("Thor NVFP4 requires visible CUDA device 0")
    if torch.cuda.get_device_capability(device_index) != (11, 0):
        raise RuntimeError("Thor NVFP4 requires SM110")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Thor NVFP4 must be warmed before CUDA graph capture")
    from torch.utils.cpp_extension import load
    from .nvidia.export_fc1_kernel import export_fc1
    from .nvidia.export_fc2_kernel import export_fc2

    root = Path(__file__).parent
    with torch.cuda.device(device_index):
        prepare = load(
            name="thor_nvfp4_prepare_e8b2952",
            sources=[str(root / "prepare.cu")],
            extra_include_paths=[str(root)],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_110a,code=sm_110a"],
            verbose=False,
        )
        args = SimpleNamespace(
            activation="swiglu", mma_tiler_n=128, output_dtype="bf16",
            dummy_tokens=128, dummy_experts=EXPERTS, dummy_top_k=TOP_K,
            dummy_hidden_size=256, dummy_intermediate_size=128,
        )
        fc1 = export_fc1(args)
        fc2 = export_fc2(args)
    return prepare, fc1, fc2


def _pointer(tensor, dtype, align=16):
    from .nvidia.export_common import make_ptr
    return make_ptr(dtype, tensor.data_ptr(), assumed_align=align)


@torch.library.custom_op("thor_nvfp4::moe", mutates_args=())
def run(x: torch.Tensor, ids: torch.Tensor, routes: torch.Tensor,
        weights: list[torch.Tensor]) -> torch.Tensor:
    import cuda.bindings.driver as cuda
    import cutlass as c

    if x.ndim != 2 or x.shape[1] != HIDDEN or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("Thor NVFP4 expects CUDA BF16 [T,2560]")
    tokens = x.shape[0]
    padded = permuted_rows(tokens)
    if ids.shape != (tokens, TOP_K) or routes.shape != ids.shape:
        raise ValueError("Thor NVFP4 expects routing [T,10]")
    if ids.dtype != torch.int32 or routes.dtype != torch.float32:
        raise ValueError("Thor NVFP4 expects INT32 ids and FP32 router weights")
    if any(t.device != x.device or not t.is_contiguous() for t in [x, ids, routes, *weights]):
        raise ValueError("Thor NVFP4 tensors must be contiguous on the same device")
    if len(weights) != 8:
        raise ValueError("Thor NVFP4 requires eight prepared weight/scale tensors")
    if tokens == 0:
        return torch.empty_like(x)
    shapes = [(512, 1280, 1280), (512, 2560, 320),
              (512, 10, 40, 32, 4, 4), (512, 20, 10, 32, 4, 4),
              (512,), (512,), (512,), (512,)]
    for index, (value, shape) in enumerate(zip(weights, shapes)):
        dtype = torch.uint8 if index < 4 else torch.float32
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Unsupported prepared Thor NVFP4 tensor {index}")
    for value in [x, ids, routes, *weights]:
        if value.data_ptr() % 16:
            raise ValueError("Thor NVFP4 pointers require 16-byte alignment")
    if weights[0].data_ptr() % 32 or weights[1].data_ptr() % 32:
        raise ValueError("Thor NVFP4 packed weights require 32-byte alignment")
    if not warmup.cache_info().currsize:
        raise RuntimeError("Call warmup before using the Thor NVFP4 operator")
    if x.device.index != 0:
        raise ValueError("The first Thor NVFP4 implementation supports visible CUDA device 0 only")
    prepare, fc1, fc2 = warmup(x.device.index)
    w1, w2, s1, s2, alpha1, input_scale, alpha2, down_scale = weights
    alloc = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=x.device)
    a = alloc((tokens * TOP_K, HIDDEN // 2), torch.uint8)
    a_sf = alloc((tokens * TOP_K, HIDDEN // 16), torch.uint8)
    intermediate = alloc((padded, INTERMEDIATE // 2), torch.uint8)
    intermediate_sf = alloc((padded, INTERMEDIATE // 16), torch.uint8)
    mapping = alloc((padded,), torch.int32)
    groups = alloc((padded // 128,), torch.int32)
    limits = alloc((padded // 128,), torch.int32)
    tile_count = alloc((1,), torch.int32)
    result = torch.zeros_like(x)
    prepare.prepare(x, ids, input_scale, a, a_sf, mapping, groups, limits, tile_count)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    clusters = torch.cuda.get_device_properties(x.device).multi_processor_count
    p4 = lambda t: _pointer(t, c.Float4E2M1FN, 32)
    p8 = lambda t: _pointer(t, c.Float8E4M3FN)
    pf = lambda t: _pointer(t, c.Float32)
    pi = lambda t: _pointer(t, c.Int32)
    # CuTe DSL removes Constexpr parameters from the compiled callable. Tile
    # size, scale-vector size, and activation were fixed by the export wrappers.
    fc1(
        p4(a), p4(w1), p8(a_sf), p8(s1), p4(intermediate), p8(intermediate_sf),
        pf(alpha1), pf(input_scale), pf(down_scale), pi(groups), pi(limits),
        pi(mapping), pi(tile_count), tokens * TOP_K, padded, 2 * INTERMEDIATE,
        HIDDEN, EXPERTS,
        max_active_clusters=clusters, stream=stream,
    )
    fc2(
        p4(intermediate), p4(w2), p8(intermediate_sf), p8(s2),
        _pointer(result, c.BFloat16, 32), pf(alpha2), pf(down_scale),
        pi(groups), pi(limits), pi(mapping), pi(tile_count), pf(routes),
        padded, HIDDEN, INTERMEDIATE, EXPERTS, tokens, TOP_K,
        max_active_clusters=clusters, stream=stream,
    )
    return result


@run.register_fake
def _fake(x, ids, routes, weights):
    return torch.empty_like(x)
