"""Stage isolation for the standalone correctness test; never used by serving."""
import torch

from .check import quant_dequant, reference, unpack
from .contract import HIDDEN, INTERMEDIATE, TOP_K, fc1_row_order
from .runtime import _run_impl


def unswizzle_scales(value, rows, cols):
    """Inverse of the pinned NVIDIA [M/128,K/64,32,4,4] scale layout."""
    return (value.reshape(rows // 128, cols // 4, 32, 4, 4)
            .permute(0, 3, 2, 1, 4).contiguous().reshape(rows, cols))


def validate_layout(trace, ids):
    mapping = trace["mapping"]
    count = int(trace["tile_count"].item())
    if not 0 < count <= trace["groups"].numel():
        raise AssertionError("Invalid native tile count")
    rows = torch.where(mapping >= 0)[0]
    expanded = mapping[rows].long()
    expected = torch.arange(ids.numel(), device=ids.device)
    if not torch.equal(expanded.sort().values, expected):
        raise AssertionError("Native mapping must contain each routed row exactly once")
    if rows.numel() and rows.max().item() >= count * 128:
        raise AssertionError("Native mapping contains rows beyond the valid tile count")
    for tile in range(count):
        start = tile * 128
        limit = int(trace["limits"][tile].item())
        expert = int(trace["groups"][tile].item())
        if not start < limit <= start + 128:
            raise AssertionError("Invalid native tile limit")
        indices = mapping[start:limit].long()
        if (indices < 0).any() or not (ids.flatten()[indices] == expert).all():
            raise AssertionError("Native tile expert does not match routed ids")
        if (mapping[limit:start + 128] != -1).any():
            raise AssertionError("Native tile padding must be -1")
    return rows, expanded


def metrics(name, actual, expected):
    actual, expected = actual.float().flatten(), expected.float().flatten()
    error = (actual - expected).norm()
    expected_norm = expected.norm()
    rmse = (error / expected_norm.clamp_min(1e-30)).item()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
    max_abs = (actual - expected).abs().max().item()
    print(f"diagnostic {name}: relative_rmse={rmse:.6g} cosine={cosine:.6g} max_abs={max_abs:.6g}")
    return rmse


def codes(packed):
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()


def report_fields(name, packed, sf, reference_codes, reference_sf):
    actual_codes = codes(packed)
    code_error = (actual_codes != reference_codes).float().mean().item()
    scale_error = (sf != reference_sf).float().mean().item()
    print(f"diagnostic {name}: differing_fp4_codes={code_error:.6%} differing_fp8_scales={scale_error:.6%}")


def diagnose(x, ids, routes, original, prepared):
    """Keep the failing gate intact while locating the first divergent stage."""
    trace = {}
    actual = _run_impl(x, ids, routes, prepared, trace=trace)
    torch.cuda.synchronize()
    rows, expanded = validate_layout(trace, ids)
    print("diagnostic routing: every expanded row appears once under the correct expert")
    inverse = torch.argsort(torch.tensor(fc1_row_order(), device=x.device))
    for expert in ids.unique().tolist():
        scale1 = unswizzle_scales(prepared[2][expert], 2 * INTERMEDIATE, HIDDEN // 16)
        scale2 = unswizzle_scales(prepared[3][expert], HIDDEN, INTERMEDIATE // 16)
        checks = [torch.equal(prepared[0][expert][inverse], original[0][expert]),
                  torch.equal(prepared[1][expert], original[1][expert]),
                  torch.equal(scale1[inverse], original[2][expert].view(torch.uint8)),
                  torch.equal(scale2, original[3][expert].view(torch.uint8))]
        if not all(checks):
            raise AssertionError(f"Lossless weight/scale repacking failed for expert {expert}")
    print("diagnostic repacking: active experts' weight and block-scale bytes match the checkpoint")

    expert_ids = ids.flatten().long()
    input_scale = prepared[5][expert_ids, None]
    native_input = unpack(trace["input_packed"], trace["input_sf"].view(torch.float8_e4m3fn), input_scale)
    expanded_input = x.repeat_interleave(TOP_K, dim=0).float()
    expected_input, input_codes, input_sf = quant_dequant(expanded_input, input_scale, return_fields=True)
    metrics("input_quantization", native_input, expected_input)
    report_fields("input_quantization", trace["input_packed"], trace["input_sf"], input_codes, input_sf)

    reference_trace = {}
    reference(x, ids, routes, original, input_override=native_input, trace=reference_trace)
    linear_sf = unswizzle_scales(trace["intermediate_sf"], trace["mapping"].numel(), INTERMEDIATE // 16)
    native_intermediate = torch.empty_like(reference_trace["intermediate"])
    down_scale = prepared[7][expert_ids, None]
    native_intermediate[expanded] = unpack(
        trace["intermediate_packed"][rows], linear_sf[rows].view(torch.float8_e4m3fn), down_scale[expanded])
    metrics("fc1_swiglu_requant_using_native_input", native_intermediate, reference_trace["intermediate"])
    _, intermediate_codes, intermediate_sf = quant_dequant(
        reference_trace["activated"], down_scale, return_fields=True)
    report_fields("fc1_swiglu_requant", trace["intermediate_packed"][rows], linear_sf[rows],
                  intermediate_codes[expanded], intermediate_sf[expanded])
    # Supplying the native intermediate removes both earlier quantization stages
    # from the FC2/finalize comparison.
    expected_fc2 = reference(x, ids, routes, original, intermediate_override=native_intermediate)
    metrics("fc2_finalize_using_native_intermediate", actual, expected_fc2)

    # One nonzero slot per token removes cross-expert accumulation. These are
    # diagnostic launches only; zero-weight slots still run through the kernel.
    isolated_sum = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(TOP_K):
        one_slot = torch.zeros_like(routes)
        one_slot[:, slot] = routes[:, slot]
        contribution = _run_impl(x, ids, one_slot, prepared)
        expected = reference(x, ids, one_slot, original, intermediate_override=native_intermediate)
        metrics(f"fc2_slot_{slot}", contribution, expected)
        isolated_sum.add_(contribution.float())
    metrics("scatter_vs_sum_of_isolated_native_slots", actual, isolated_sum)
    print("diagnostic complete; original acceptance thresholds remain unchanged")
