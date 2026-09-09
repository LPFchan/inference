"""Stage acceptance for the standalone correctness test; never used by serving."""
import math

import torch

from .check import quant_dequant, reference, unpack
from .contract import HIDDEN, INTERMEDIATE, TOP_K, fc1_row_order
from .runtime import _run_impl

MAX_RELATIVE_RMSE = .02
MIN_COSINE = .999
# BF16 unit roundoff is 2**-8. For two top-k=10 sums, an independent,
# uniform-rounding estimate is u*sqrt(2*(TOP_K-1)/3) = 0.00957.
# Round to 1% as a separate repeat-stability screening limit. This is not a
# worst-case bound (cancellation and correlated rounding can exceed it).
MAX_REPEAT_RELATIVE_DRIFT = .01
# The CUDA input pack uses FP32 reciprocal/multiply operations, including
# rcp.approx.ftz, while the independent emulator divides. Eight FP32 epsilons
# allow their few rounding steps near an E2M1 midpoint; they do not allow a
# nonadjacent code or a wrong FP8 block scale.
INPUT_BOUNDARY_RTOL = 8 * 2**-23


class GateCollector:
    """Diagnostic mode records numerical failures until all stages are measured."""
    def __init__(self, collect_failures):
        self.collect_failures = collect_failures
        self.failures = []

    def check(self, function, *args):
        try:
            return function(*args)
        except AssertionError as error:
            if not self.collect_failures:
                raise
            self.failures.append(str(error))
            print(f"FAIL: {error}")

    def finish(self):
        if self.failures:
            raise AssertionError("Collected native MoE gate failures:\n" + "\n".join(self.failures))


def require_repeat_stable(name, actual, previous):
    metrics(name, actual, previous)
    differing = (actual != previous).sum().item()
    # Symmetric normalization makes pair order immaterial. A zero pair has
    # zero drift; a zero/nonzero pair has unit drift and cannot pass.
    actual32, previous32 = actual.float(), previous.float()
    scale = torch.maximum(actual32.norm(), previous32.norm()).clamp_min(1e-30)
    drift = ((actual32 - previous32).norm() / scale).item()
    print(f"diagnostic {name}: bitwise_equal={torch.equal(actual, previous)} "
          f"differing_elements={differing} pairwise_relative_drift={drift:.6g}")
    if (not torch.isfinite(actual).all() or not torch.isfinite(previous).all()
            or not math.isfinite(drift) or drift > MAX_REPEAT_RELATIVE_DRIFT):
        raise AssertionError(f"{name}: same-input output failed the 1% pairwise repeat-stability gate")


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
    cosine = (1.0 if expected_norm.item() == 0 and actual.norm().item() == 0 else
              torch.nn.functional.cosine_similarity(actual, expected, dim=0).item())
    max_abs = (actual - expected).abs().max().item()
    print(f"diagnostic {name}: relative_rmse={rmse:.6g} cosine={cosine:.6g} max_abs={max_abs:.6g}")
    return rmse, cosine


def require_quality(name, actual, expected):
    rmse, cosine = metrics(name, actual, expected)
    if (not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
            or not math.isfinite(rmse) or not math.isfinite(cosine)
            or rmse > MAX_RELATIVE_RMSE or cosine < MIN_COSINE):
        raise AssertionError(f"{name} failed the 2% relative-RMSE / 0.999 cosine gate")


def validate_input_rounding(value, global_scale, actual_codes, actual_sf,
                            expected_codes, expected_sf):
    global_scale = torch.as_tensor(global_scale, device=value.device).reshape(-1, 1)
    sf = actual_sf.view(torch.float8_e4m3fn).float()
    if (not torch.isfinite(value).all() or not torch.isfinite(global_scale).all()
            or not (global_scale > 0).all() or not torch.isfinite(sf).all()
            or not (sf >= 0).all()):
        raise AssertionError("Input values and scales must be finite, with positive global scales")
    if not (((actual_codes >= 0) & (actual_codes <= 15)).all()
            and ((expected_codes >= 0) & (expected_codes <= 15)).all()):
        raise AssertionError("Input FP4 codes must be in [0,15]")
    if not torch.equal(actual_sf, expected_sf):
        raise AssertionError("Input FP8 block scales must match the independent emulator exactly")
    actual_magnitude, expected_magnitude = actual_codes & 7, expected_codes & 7
    signed_zero = (actual_magnitude == 0) & (expected_magnitude == 0)
    changed = (actual_codes != expected_codes) & ~signed_zero
    adjacent = (actual_magnitude - expected_magnitude).abs() == 1
    same_sign = (actual_codes >> 3) == (expected_codes >> 3)
    if not (adjacent[changed] & same_sign[changed]).all():
        raise AssertionError("Input FP4 differences must preserve sign and select adjacent levels")
    mids = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=value.device)
    lower_code = torch.minimum(actual_magnitude, expected_magnitude).clamp(max=6).long()
    midpoint = mids[lower_code]
    scale = (sf * global_scale).repeat_interleave(16, -1)
    normalized = torch.where(scale > 0, value.abs() / scale, 0)
    tolerance = INPUT_BOUNDARY_RTOL * midpoint.clamp_min(1)
    if not ((normalized - midpoint).abs()[changed] <= tolerance[changed]).all():
        raise AssertionError("Input FP4 difference is outside the FP32 rounding-boundary allowance")
    # Independently check error to the original input, including unchanged
    # codes. FP64 evaluates all eight representable magnitudes without borrowing
    # the emulator's selected code. Crossing a midpoint by delta can increase
    # nearest-level error by at most 2*delta in normalized units.
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=value.device)
    decoded = (levels[actual_magnitude] * (1 - 2 * (actual_codes >> 3))
               * sf.repeat_interleave(16, -1) * global_scale)
    if not torch.isfinite(decoded).all():
        raise AssertionError("Decoded input values must be finite")
    scale64 = (sf.double() * global_scale.double()).repeat_interleave(16, -1)
    magnitude64 = value.double().abs()
    nearest_error = magnitude64.clone()
    for level in levels.tolist()[1:]:
        nearest_error = torch.minimum(nearest_error, (magnitude64 - level * scale64).abs())
    decoded64 = (levels.double()[actual_magnitude] * scale64
                 * (1 - 2 * (actual_codes >> 3)))
    allowance = 2 * INPUT_BOUNDARY_RTOL * torch.maximum(scale64, magnitude64)
    arithmetic_slack = 8 * 2**-52 * torch.maximum(magnitude64, decoded64.abs())
    if not ((decoded64 - value.double()).abs()
            <= nearest_error + allowance + arithmetic_slack).all():
        raise AssertionError("Input quantization exceeds the pointwise nearest-level error bound")


def codes(packed):
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()


def report_fields(name, packed, sf, reference_codes, reference_sf):
    actual_codes = codes(packed)
    code_error = (actual_codes != reference_codes).float().mean().item()
    scale_error = (sf != reference_sf).float().mean().item()
    print(f"diagnostic {name}: differing_fp4_codes={code_error:.6%} differing_fp8_scales={scale_error:.6%}")


def validate_case(x, ids, routes, original, prepared, *, verbose=False):
    """Require every native stage to pass against its actual input representation."""
    trace = {}
    actual = _run_impl(x, ids, routes, prepared, trace=trace)
    torch.cuda.synchronize()
    gates = GateCollector(collect_failures=verbose)
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
    gates.check(validate_input_rounding, expanded_input, input_scale, codes(trace["input_packed"]),
                trace["input_sf"], input_codes, input_sf)
    metrics("informational_input_quantization", native_input, expected_input)
    if verbose:
        report_fields("input_quantization", trace["input_packed"], trace["input_sf"], input_codes, input_sf)

    reference_trace = {}
    reference(x, ids, routes, original, input_override=native_input, trace=reference_trace)
    linear_sf = unswizzle_scales(trace["intermediate_sf"], trace["mapping"].numel(), INTERMEDIATE // 16)
    native_intermediate = torch.empty_like(reference_trace["intermediate"])
    down_scale = prepared[7][expert_ids, None]
    native_intermediate[expanded] = unpack(
        trace["intermediate_packed"][rows], linear_sf[rows].view(torch.float8_e4m3fn), down_scale[expanded])
    gates.check(require_quality, "fc1_swiglu_requant_using_native_input", native_intermediate, reference_trace["intermediate"])
    _, intermediate_codes, intermediate_sf = quant_dequant(
        reference_trace["activated"], down_scale, return_fields=True)
    if verbose:
        report_fields("fc1_swiglu_requant", trace["intermediate_packed"][rows], linear_sf[rows],
                      intermediate_codes[expanded], intermediate_sf[expanded])
    # Supplying the native intermediate removes both earlier quantization stages
    # from the FC2/finalize comparison.
    expected_fc2 = reference(x, ids, routes, original, intermediate_override=native_intermediate)
    gates.check(require_quality, "fc2_finalize_using_native_intermediate", actual, expected_fc2)

    # One nonzero slot per token removes cross-expert accumulation. These are
    # diagnostic launches only; zero-weight slots still run through the kernel.
    isolated_sum = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(TOP_K):
        one_slot = torch.zeros_like(routes)
        one_slot[:, slot] = routes[:, slot]
        contribution = _run_impl(x, ids, one_slot, prepared)
        expected = reference(x, ids, one_slot, original, intermediate_override=native_intermediate)
        gates.check(require_quality, f"fc2_slot_{slot}", contribution, expected)
        isolated_sum.add_(contribution.float())
    gates.check(require_quality, "scatter_vs_sum_of_isolated_native_slots", actual, isolated_sum)
    # Keep all outputs alive and compare every pair, so allocator reuse cannot
    # make a previous result silently alias the next one. BF16 atomic reduction
    # does not promise a fixed summation order; gate numerical stability and
    # report bitwise equality as an informational diagnostic.
    repeats = [actual]
    for repeat in range(1, 4):
        current = _run_impl(x, ids, routes, prepared)
        torch.cuda.synchronize()
        gates.check(require_quality, f"repeat_{repeat}_fc2_reference", current, expected_fc2)
        for prior, previous in enumerate(repeats):
            gates.check(require_repeat_stable, f"repeat_{repeat}_vs_{prior}", current, previous)
        repeats.append(current)
    # Independently quantizing the input can choose the opposite side of an
    # E2M1 midpoint; SwiGLU can amplify that difference through the MoE. The
    # stage gates above determine acceptance, while this remains a sensitivity
    # measurement for the complete quantized pipeline.
    metrics("informational_end_to_end_quantization_sensitivity", actual, reference(x, ids, routes, original))
    gates.finish()
    print("PASS: routed input, FC1, FC2, and scatter stage gates")
    return actual
