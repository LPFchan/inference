"""Torch-owned memory and stream ABI for the NVIDIA dense SM110 kernel."""
from functools import lru_cache
from pathlib import Path

import torch

from .dense_contract import select_tile_n, use_cutlass, validate_shape

_READY_TILES = set()
_CUTLASS_READY = False


def warmup_cutlass():
    global _CUTLASS_READY
    if _CUTLASS_READY:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm CUTLASS before graph capture")
    if torch.cuda.get_device_capability(0) != (11, 0):
        raise ValueError("Dense hybrid requires SM110")
    from flashinfer.gemm.gemm_base import get_cutlass_fp4_gemm_module
    get_cutlass_fp4_gemm_module(11, 0)
    _CUTLASS_READY = True


def warmup_dense(device_index=0, tile_n=None, tile_uniform_alpha=True):
    if tile_n is not None:
        return _compile_dense(device_index, tile_n, tile_uniform_alpha)
    baseline = _compile_dense(device_index, 128, True)
    _compile_dense(device_index, 256, True)
    _compile_dense(device_index, 128, False)
    return baseline


@lru_cache(maxsize=3)
def _compile_dense(device_index, tile_n, tile_uniform_alpha=True):
    if tile_n not in (128, 256):
        raise ValueError("Unsupported dense CuTeDSL N tile")
    if type(tile_uniform_alpha) is not bool or (not tile_uniform_alpha and tile_n != 128):
        raise ValueError("Per-column alpha is supported only for the B/A 128 tile")
    if device_index != 0 or torch.cuda.get_device_capability(device_index) != (11, 0):
        raise ValueError("Dense CuTeDSL requires visible SM110 device 0")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm dense CuTeDSL before CUDA graph capture")
    import cutlass as c
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import make_ptr
    from torch.utils.cpp_extension import load
    from .nvidia.gemm_blackwell_nvfp4_ws import GemmBlackwellNvFp4WS
    root = Path(__file__).parent
    helper = load(name="thor_dense_prepare_e8b2952", sources=[str(root / "dense_prepare.cu")],
                  extra_include_paths=[str(root)],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_110a,code=sm_110a"], verbose=False)
    kernel = GemmBlackwellNvFp4WS(acc_dtype=c.Float32, mma_tiler_mn=(128, tile_n),
                                 cluster_shape_mn=(1, 1), sf_vec_size=16)
    kernel.tile_uniform_alpha = tile_uniform_alpha
    # Raw pointers suffice for tracing: no dummy GPU model tensors are needed.
    ptr = lambda dtype: make_ptr(dtype, 0, assumed_align=32)
    compiled = cute.compile(kernel.wrapper, ptr(c.Float4E2M1FN), ptr(c.Float4E2M1FN),
                            ptr(c.Float8E4M3FN), ptr(c.Float8E4M3FN), ptr(c.BFloat16),
                            ptr(c.Float32), 128, 128, 5120, 16, c.Int32(1),
                            cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    _READY_TILES.add((device_index, tile_n, tile_uniform_alpha))
    return helper, compiled


def launch_dense(x, weight, sf, alpha, input_scale, output_size, trace=None, *, tile_n=None,
                 uniform_alpha=False):
    n, k = weight.shape[0], x.shape[-1]
    expected_n = validate_shape(output_size, k, [output_size])
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or x.device.index != 0:
        raise ValueError("Dense CuTeDSL requires CUDA BF16 [M,K] on visible device 0")
    if n != expected_n or weight.shape != (n, k // 2) or weight.dtype != torch.uint8:
        raise ValueError("Invalid dense packed weight shape/dtype")
    if sf.dtype != torch.uint8 or sf.numel() != n * k // 16:
        raise ValueError("Invalid dense swizzled scale shape/dtype")
    if alpha.dtype != torch.float32 or alpha.shape != (n,):
        raise ValueError("Invalid dense per-column alpha")
    if input_scale.dtype != torch.float32 or input_scale.numel() != 1:
        raise ValueError("Invalid dense input scale")
    if any(t.device != x.device or not t.is_contiguous() or t.data_ptr() % 32
           for t in (x, weight, sf, alpha, input_scale)):
        raise ValueError("Dense tensors must be contiguous, aligned, and on one CUDA device")
    m = x.shape[0]
    if not 0 <= m <= 262144:
        raise ValueError("Dense M must be in [0,262144]")
    cutlass = use_cutlass(m, output_size, k, uniform_alpha)
    if cutlass and not _CUTLASS_READY:
        raise RuntimeError("Call warmup_cutlass before hybrid launch")
    if not m:
        return torch.empty((0, output_size), dtype=x.dtype, device=x.device)
    if tile_n is None:
        tile_n = select_tile_n(m, output_size, k)
    if tile_n not in (128, 256) or n % tile_n:
        raise ValueError("Dense N must cover complete epilogue tiles (per-column alpha bounds)")
    tile_uniform_alpha = output_size != 96
    if (0, tile_n, tile_uniform_alpha) not in _READY_TILES:
        raise RuntimeError("Call warmup_dense for the selected tile before dense launch")
    import cutlass as c
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import make_ptr
    helper, gemm = warmup_dense(0, tile_n, tile_uniform_alpha)
    padded_m = (m + 127) // 128 * 128
    a = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    a_sf = torch.empty((padded_m, k // 16), dtype=torch.uint8, device=x.device)
    helper.quantize(x, input_scale, a, a_sf)
    if trace is not None:
        trace.update(packed=a, sf=a_sf, backend="cutlass" if cutlass else "cutedsl")
    if cutlass:
        from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
        output = flashinfer_scaled_fp4_mm(
            a, weight, a_sf, sf.reshape(n, k // 16), alpha[:1],
            torch.bfloat16, backend="cutlass")
        if output.shape != (m, n) or output.dtype != torch.bfloat16 or output.device != x.device:
            raise RuntimeError("CUTLASS dense output contract changed")
        return output
    # Conservatively retain padded M backing despite upstream wrapper predication.
    output = torch.empty((padded_m, n), dtype=torch.bfloat16, device=x.device)
    ptr = lambda t, dtype: make_ptr(dtype, t.data_ptr(), assumed_align=32)
    gemm(ptr(a, c.Float4E2M1FN), ptr(weight, c.Float4E2M1FN),
         ptr(a_sf, c.Float8E4M3FN), ptr(sf, c.Float8E4M3FN), ptr(output, c.BFloat16),
         ptr(alpha, c.Float32), m, n, k,
         max_active_clusters=torch.cuda.get_device_properties(0).multi_processor_count,
         stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    return output[:m, :output_size].contiguous()


@torch.library.custom_op("thor_nvfp4::dense", mutates_args=())
def dense_run(x: torch.Tensor, weight: torch.Tensor, sf: torch.Tensor,
              alpha: torch.Tensor, input_scale: torch.Tensor, output_size: int,
              uniform_alpha: bool = False) -> torch.Tensor:
    return launch_dense(x, weight, sf, alpha, input_scale, output_size,
                        uniform_alpha=uniform_alpha)


@dense_run.register_fake
def _fake(x, weight, sf, alpha, input_scale, output_size, uniform_alpha=False):
    return torch.empty((x.shape[0], output_size), dtype=x.dtype, device=x.device)
