"""Opt-in linear integration; preserve fused globals before vLLM collapses them."""
import os

import torch

from .dense_contract import validate_alpha_layout, validate_shape


def has_uniform_alpha(alpha, output_size):
    """Preparation-only proof; never inspect GPU scalar values during serving."""
    if alpha.dtype != torch.float32 or alpha.shape != (max(output_size, 128),):
        raise ValueError("Invalid prepared hybrid alpha")
    if not torch.isfinite(alpha).all() or not (alpha[:output_size] > 0).all():
        raise ValueError("Nonfinite prepared hybrid alpha")
    return output_size != 96 and bool(torch.equal(alpha, alpha[:1].expand_as(alpha)))


def select_dense_kernel(use_a16=False):
    if os.environ.get("VLLM_THOR_CUTEDSL_DENSE") != "1" or use_a16:
        return None
    from vllm.platforms import current_platform
    if not current_platform.is_cuda() or not current_platform.is_device_capability(110):
        return None
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    if config.parallel_config.tensor_parallel_size != 1:
        raise ValueError("Thor CuTeDSL dense requires TP=1")
    if config.model_config.dtype != torch.bfloat16:
        raise ValueError("Thor CuTeDSL dense requires BF16 model dtype")
    from vllm.model_executor.kernels.linear import _get_linear_backend
    from vllm import envs
    if _get_linear_backend() != "auto" or envs.VLLM_BATCH_INVARIANT:
        raise ValueError("Use the Thor dense opt-in with the default linear backend")
    from vllm.logger import init_logger
    init_logger(__name__).info_once("Using NVIDIA ThorDenseNvFp4Kernel (SM110 CuTeDSL W4A4)")
    return ThorDenseNvFp4Kernel()


def prepare_tensors(weight, block_scales, weight_globals, input_globals, widths, *, divisors):
    """CPU-testable preparation; byte-preserving weights/scales and column alpha."""
    from .adapter import swizzle_scales
    weight, block_scales, weight_globals, input_globals = (
        t.as_subclass(torch.Tensor) for t in (weight, block_scales, weight_globals, input_globals))
    n, packed_k = weight.shape
    padded_n = validate_shape(n, packed_k * 2, widths)
    validate_alpha_layout(n, packed_k * 2, widths)
    if weight.dtype != torch.uint8 or block_scales.dtype != torch.float8_e4m3fn:
        raise ValueError("Dense NVFP4 requires uint8 weights and E4M3 block scales")
    if tuple(block_scales.shape) != (n, packed_k // 8):
        raise ValueError("Dense NVFP4 requires group-16 block scales")
    if any(t.device != weight.device for t in (block_scales, weight_globals, input_globals)):
        raise ValueError("Dense tensors must share one device")
    wg, ag = weight_globals.float().flatten(), input_globals.float().flatten()
    if wg.numel() != len(widths) or ag.numel() != len(widths):
        raise ValueError("Each logical projection needs its own global scales")
    if any(not torch.isfinite(t).all() or not (t > 0).all() for t in (wg, ag)):
        raise ValueError("Global scales must be finite and positive")
    if not torch.equal(ag, ag[0].expand_as(ag)):
        raise ValueError("Fused input global scales differ; one activation quantization cannot represent them")
    if not torch.isfinite(block_scales.float()).all() or not (block_scales.float() >= 0).all():
        raise ValueError("Block scales must be finite and nonnegative")
    if divisors:
        wg, ag = wg.reciprocal(), ag.reciprocal()
    products = wg * ag
    if not torch.isfinite(products).all() or not (products > 0).all():
        raise ValueError("Global scale products overflow or underflow")
    alpha = torch.zeros(padded_n, dtype=torch.float32, device=weight.device)
    alpha[:n] = torch.repeat_interleave(products, torch.tensor(widths, device=weight.device))
    packed = torch.zeros((padded_n, packed_k), dtype=torch.uint8, device=weight.device)
    packed[:n] = weight
    sf = torch.zeros((padded_n, packed_k // 8), dtype=torch.uint8, device=weight.device)
    sf[:n] = block_scales.view(torch.uint8)
    swizzled = swizzle_scales(sf.view(torch.float8_e4m3fn).unsqueeze(0))[0]
    return packed, swizzled, alpha, ag[:1].contiguous()


class ThorDenseNvFp4Kernel:
    _reported = False

    def input_quant_key(self):
        # Keep vLLM's fused activation quantization off: this backend packs once.
        return None

    def prepare(self, layer, *, divisors):
        weight = layer.weight_packed if divisors else layer.weight
        wg = layer.weight_global_scale if divisors else layer.weight_scale_2
        ag = layer.input_global_scale if divisors else layer.input_scale
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (11, 0):
            raise ValueError("Thor dense requires SM110 CUDA weights")
        prepared = prepare_tensors(weight, layer.weight_scale, wg, ag,
                                   layer.logical_widths, divisors=divisors)
        from .adapter import plain_parameter
        for name, tensor in zip(("thor_dense_weight", "thor_dense_sf", "thor_dense_alpha", "thor_dense_input_scale"), prepared):
            layer.register_parameter(name, plain_parameter(tensor))
        layer.thor_dense_uniform_alpha = has_uniform_alpha(prepared[2], weight.shape[0])
        # Retain checkpoint-global vectors for auditable fused-scale provenance.
        for name in (("weight_packed", "weight_scale") if divisors else ("weight", "weight_scale")):
            delattr(layer, name)
        from .dense_runtime import warmup_cutlass, warmup_dense
        warmup_dense(weight.device.index)
        if layer.thor_dense_uniform_alpha:
            warmup_cutlass()
        if not ThorDenseNvFp4Kernel._reported:
            # Emit only after successful preparation/compilation, independently
            # of vLLM's rank-aware init logger (which can suppress early logs).
            print("THOR_DENSE_BACKEND ready: NVIDIA e8b2952 SM110 CuTeDSL W4A4; "
                  "BF16 output; exact tile-broadcast fused scales (B/A per-column); tiles=128,256; "
                  "tile256 for aligned N at M>=512 and out-proj K6144; "
                  "CUTLASS for M>=1568 with preparation-proven uniform alpha; "
                  "one native activation pack; shared packed weights/scales", flush=True)
            ThorDenseNvFp4Kernel._reported = True

    def apply_weights(self, layer, x, bias=None):
        from .dense_runtime import dense_run
        shape = (*x.shape[:-1], layer.output_size_per_partition)
        out = dense_run(x.reshape(-1, x.shape[-1]).contiguous(), layer.thor_dense_weight,
                        layer.thor_dense_sf, layer.thor_dense_alpha, layer.thor_dense_input_scale,
                        layer.output_size_per_partition, layer.thor_dense_uniform_alpha)
        if bias is not None:
            out = out + bias
        return out.reshape(shape)


def prepare_dense_method(method, layer, *, divisors):
    if not isinstance(method.kernel, ThorDenseNvFp4Kernel):
        return False
    group_size = method.group_size if divisors else method.ctx.group_size
    if group_size != 16:
        raise ValueError("Thor dense requires group size 16")
    method.kernel.prepare(layer, divisors=divisors)
    return True
