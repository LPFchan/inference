"""Standalone exact-geometry GPU correctness gate, run before full-model loading."""
import argparse

import torch

from .adapter import prepare_weights
from .contract import EXPERTS, HIDDEN, INTERMEDIATE, TOP_K
from .runtime import run, warmup


def unpack(weight, block_scale, global_scale):
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=weight.device)
    return table[codes] * block_scale.float().repeat_interleave(16, dim=-1) * global_scale


def quant_dequant(value, global_scale):
    blocks = value.reshape(value.shape[0], -1, 16)
    sf = (blocks.abs().amax(-1) / (6 * global_scale)).to(torch.float8_e4m3fn).float()
    scale = sf[..., None] * global_scale
    normalized = torch.where(scale > 0, blocks / scale, 0)
    mids = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=value.device)
    magnitude = normalized.abs().contiguous()
    code = torch.bucketize(magnitude, mids)
    # E2M1 rounds midpoint ties to the even code, including the nonuniform bins.
    tie = (magnitude == mids[code.clamp(max=6)]) & (code < 7) & ((code % 2) == 1)
    code = code + tie.to(code.dtype)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=value.device)
    return (levels[code] * normalized.sign() * scale).reshape_as(value)


def reference(x, ids, routes, original):
    w1, w2, s1, s2, g1, a1, g2, a2 = original
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in ids.unique().tolist():
        tokens, choices = torch.where(ids == expert)
        values = quant_dequant(x[tokens].float(), a1[expert, 0])
        first = values @ unpack(w1[expert], s1[expert], g1[expert, 0]).T
        gate, up = first.split(INTERMEDIATE, dim=-1)
        intermediate = quant_dequant(torch.nn.functional.silu(gate) * up, a2[expert])
        second = intermediate @ unpack(w2[expert], s2[expert], g2[expert]).T
        result.index_add_(0, tokens, second * routes[tokens, choices, None])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    warmup(0)
    if args.compile_only:
        print("SM110 BF16 input pack and SwiGLU/BF16 CuTe DSL kernels compiled")
        return
    layer = torch.nn.Module()
    specs = {
        "w13_weight": ((EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2), torch.uint8, None),
        "w2_weight": ((EXPERTS, HIDDEN, INTERMEDIATE // 2), torch.uint8, None),
        "w13_weight_scale": ((EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16), torch.float8_e4m3fn, .125),
        "w2_weight_scale": ((EXPERTS, HIDDEN, INTERMEDIATE // 16), torch.float8_e4m3fn, .125),
        "w13_weight_scale_2": ((EXPERTS, 2), torch.float32, .1),
        "w13_input_scale": ((EXPERTS, 2), torch.float32, .01),
        "w2_weight_scale_2": ((EXPERTS,), torch.float32, .1),
        "w2_input_scale": ((EXPERTS,), torch.float32, .01),
    }
    for name, (shape, dtype, fill) in specs.items():
        tensor = (torch.randint(0, 256, shape, dtype=dtype, device="cuda") if fill is None
                  else torch.full(shape, fill, dtype=dtype, device="cuda"))
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    # Nonuniform expert global scales catch accidental collapse to one layer scale.
    with torch.no_grad():
        layer.w2_input_scale.mul_(torch.linspace(.8, 1.2, EXPERTS, device="cuda"))
        layer.w2_weight_scale_2.mul_(torch.linspace(.8, 1.2, EXPERTS, device="cuda"))
    names = list(specs)
    original = [getattr(layer, name) for name in names]
    prepared = prepare_weights(layer)
    for tokens, pattern in ((1, "shared"), (10, "shared"), (129, "shared"), (32, "scattered")):
        x = (torch.randn(tokens, HIDDEN, device="cuda") * .25).to(torch.bfloat16)
        ids = torch.arange(TOP_K, device="cuda").expand(tokens, -1)
        if pattern == "scattered":
            ids = (ids + torch.arange(tokens, device="cuda")[:, None] * TOP_K) % EXPERTS
        ids = ids.to(torch.int32).contiguous()
        routes = torch.softmax(torch.randn(tokens, TOP_K, device="cuda"), dim=-1)
        expected = reference(x, ids, routes, original)
        actual = run(x, ids, routes, prepared).float()
        torch.cuda.synchronize()
        relative_rmse = ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
        print(f"tokens={tokens} routing={pattern} relative_rmse={relative_rmse:.6f} cosine={cosine:.6f}")
        if not torch.isfinite(actual).all() or relative_rmse > .02 or cosine < .999:
            raise RuntimeError("Native kernel failed the NVFP4 reference accuracy gate")
    # Capture/replay exercises pointer lifetime and caller-owned scratch storage.
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        run(x, ids, routes, prepared)
    torch.cuda.current_stream().wait_stream(capture_stream)
    with torch.cuda.graph(graph, stream=capture_stream):
        captured = run(x, ids, routes, prepared)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.float(), actual, rtol=.02, atol=.002)
    print("PASS: exact-geometry numerical comparisons and CUDA graph replay")


if __name__ == "__main__":
    main()
