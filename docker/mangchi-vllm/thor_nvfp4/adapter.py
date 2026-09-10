"""Opt-in ModelOpt integration for the pinned vLLM FusedMoEMethod API."""
import os

import torch

from .contract import EXPERTS, HIDDEN, INTERMEDIATE, fc1_row_order, supported


def swizzle_scales(scales):
    experts, rows, cols = scales.shape
    if rows % 128 or cols % 4:
        raise ValueError("Thor NVFP4 block scales require N%128=0 and (K/16)%4=0")
    # Operate on raw bytes: reshaping FP8 tensors must not round their values.
    return (scales.view(torch.uint8).reshape(experts, rows // 128, 4, 32, cols // 4, 4)
            .permute(0, 1, 4, 3, 2, 5).contiguous())


def plain_parameter(value):
    """Register a tensor without retaining vLLM parameter-subclass behavior."""
    plain = value.as_subclass(torch.Tensor)
    return torch.nn.Parameter(plain, requires_grad=False)


def prepare_weights(layer):
    expected = {
        "w13_weight": ((EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2), torch.uint8),
        "w2_weight": ((EXPERTS, HIDDEN, INTERMEDIATE // 2), torch.uint8),
        "w13_weight_scale": ((EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16), torch.float8_e4m3fn),
        "w2_weight_scale": ((EXPERTS, HIDDEN, INTERMEDIATE // 16), torch.float8_e4m3fn),
        "w13_weight_scale_2": ((EXPERTS, 2), torch.float32),
        "w2_weight_scale_2": ((EXPERTS,), torch.float32),
        "w13_input_scale": ((EXPERTS, 2), torch.float32),
        "w2_input_scale": ((EXPERTS,), torch.float32),
    }
    device = layer.w13_weight.device
    for name, (shape, dtype) in expected.items():
        value = getattr(layer, name)
        if value.shape != shape or value.dtype != dtype or value.device != device or not value.is_contiguous():
            raise ValueError(f"Unsupported Thor NVFP4 checkpoint tensor: {name}")
        if "scale" in name:
            fp32 = value.float()
            invalid = fp32 < 0 if name in ("w13_weight_scale", "w2_weight_scale") else fp32 <= 0
            if not torch.isfinite(fp32).all() or invalid.any():
                raise ValueError(f"Thor NVFP4 requires finite nonnegative block scales and positive global scales: {name}")
    for name in ("w13_weight_scale_2", "w13_input_scale"):
        value = getattr(layer, name)
        if not torch.equal(value[:, 0], value[:, 1]):
            raise ValueError(f"Thor NVFP4 requires exactly equal gate/up {name}; no lossy rescaling is applied")
    order = torch.tensor(fc1_row_order(), dtype=torch.long, device=device)
    w1 = layer.w13_weight.index_select(1, order)
    s1 = swizzle_scales(layer.w13_weight_scale.view(torch.uint8).index_select(1, order).view(torch.float8_e4m3fn))
    s2 = swizzle_scales(layer.w2_weight_scale)
    # NVIDIA's FC1/FC2 epilogues multiply these weight global scales by their
    # activation global scales internally. Multiplying alpha here would apply
    # the activation scale twice. ModelOpt stores dequantization multipliers.
    values = [w1, layer.w2_weight, s1, s2,
              layer.w13_weight_scale_2[:, 0].contiguous(),
              layer.w13_input_scale[:, 0].contiguous(),
              layer.w2_weight_scale_2, layer.w2_input_scale]
    # Replace source storage to avoid retaining a second packed copy per layer.
    names = ["w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
             "w13_weight_scale_2", "w13_input_scale", "w2_weight_scale_2", "w2_input_scale"]
    for name, value in zip(names, values):
        setattr(layer, name, plain_parameter(value))
    return [getattr(layer, name) for name in names]


def make_method(base):
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.logger import init_logger

    class ThorModelOptNvfp4MoEMethod(base):
        def __new__(cls, quant_config, moe_config):
            if os.environ.get("VLLM_THOR_CUTEDSL_MOE") != "1":
                return base(quant_config, moe_config)
            m = moe_config
            use = supported(
                sm=torch.cuda.get_device_capability(m.device),
                quant_method=quant_config.quant_method, group_size=quant_config.group_size,
                hidden=m.hidden_dim, intermediate=m.intermediate_size_per_partition,
                experts=m.num_local_experts, top_k=m.experts_per_token,
                activation=m.activation.value, dtype=str(m.in_dtype),
                parallel_sizes=(m.tp_size, m.dp_size, m.ep_size, m.pcp_size, m.sp_size),
                has_bias=m.has_bias, lora=m.is_lora_enabled,
                swiglu_parameters=(m.swiglu_limit, m.swiglu_alpha, m.swiglu_beta),
            )
            if not use or m.num_experts != EXPERTS or m.moe_backend != "auto":
                return base(quant_config, moe_config)
            return object.__new__(cls)

        def __init__(self, quant_config, moe_config):
            FusedMoEMethodBase.__init__(self, moe_config)
            self.quant_config = quant_config
            self.use_a16 = False
            self.use_global_sf = False
            self._weights = None
            init_logger(__name__).info("Using experimental NVIDIA SM110 CuTe DSL NVFP4 W4A4 MoE")

        @property
        def supports_eplb(self):
            return False

        @property
        def topk_indices_dtype(self):
            return torch.int32

        def get_fused_moe_quant_config(self, layer):
            return None

        def process_weights_after_loading(self, layer):
            from .runtime import warmup
            if layer.apply_router_weight_on_input or layer.expert_map is not None:
                raise ValueError("Thor NVFP4 supports local experts and output router weighting only")
            self._weights = prepare_weights(layer)
            warmup(layer.w13_weight.device.index)

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            from .runtime import run
            # mk_can_overlap_shared_experts is False: MoERunner already executes
            # NO_OVERLAP shared experts and combines their output after this call.
            # It still passes the shared-expert object through this legacy API.
            if self._weights is None:
                raise RuntimeError("Thor NVFP4 weights have not been prepared")
            return run(x.contiguous(), topk_ids.to(torch.int32).contiguous(),
                       topk_weights.to(torch.float32).contiguous(), self._weights)

    return ThorModelOptNvfp4MoEMethod
