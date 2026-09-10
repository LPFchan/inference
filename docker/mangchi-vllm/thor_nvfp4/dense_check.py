"""Standalone SM110 dense gate; run before opting the full model into serving."""
import argparse

import torch

from .check import quant_dequant, unpack
from .dense_adapter import has_uniform_alpha, prepare_tensors
from .dense_runtime import dense_run, launch_dense, warmup_cutlass, warmup_dense
from .diagnose import (codes, metrics, require_quality, require_repeat_stable,
                       unswizzle_scales, validate_input_rounding)

CASES = ((34816, 5120, (17408, 17408)), (5120, 17408, (5120,)),
         (14336, 5120, (12288, 1024, 1024)), (5120, 6144, (5120,)),
         (16384, 5120, (2048, 2048, 6144, 6144)), (96, 5120, (48, 48)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    warmup_dense(0)
    warmup_cutlass()
    if args.compile_only:
        print("PASS: dense pack, CuTe variants, and SM110 CUTLASS module compiled")
        return
    profiles = [(n, k, widths, hybrid) for hybrid in (False, True)
                for n, k, widths in CASES if not hybrid or n != 96]
    for n, k, widths, hybrid in profiles:
        w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        sf = torch.full((n, k // 16), .125, dtype=torch.float8_e4m3fn, device="cuda")
        # Distinct fused globals are mandatory, including B/A's 48-row boundary.
        wg = torch.linspace(400., 1600., len(widths), device="cuda")
        if hybrid:
            wg.fill_(400.)
        ag = torch.full_like(wg, 100.)
        prepared = prepare_tensors(w, sf, wg, ag, list(widths), divisors=True)
        eligible = hybrid and has_uniform_alpha(prepared[2], n)
        column_wg = torch.repeat_interleave(wg.reciprocal(), torch.tensor(widths, device="cuda"))
        reference_w = unpack(w, sf, column_wg[:, None]).double()
        for m in (1, 7, 129, 512, 608, 1568, 2048, 2205):
            print(f"dense case M={m} N={n} K={k} hybrid={hybrid}")
            x = (torch.randn(m, k, device="cuda") * .25).to(torch.bfloat16)
            trace = {}
            actual = launch_dense(x, *prepared, n, trace=trace, uniform_alpha=eligible)
            assert trace["backend"] == ("cutlass" if eligible and m >= 1568 else "cutedsl")
            torch.cuda.synchronize()
            linear_sf = unswizzle_scales(trace["sf"], (m + 127) // 128 * 128, k // 16)[:m]
            native_x = unpack(trace["packed"], linear_sf.view(torch.float8_e4m3fn), ag[0].reciprocal())
            expected_x, expected_codes, expected_sf = quant_dequant(x.float(), ag[0].reciprocal(), return_fields=True)
            validate_input_rounding(x.float(), ag[0].reciprocal(), codes(trace["packed"]), linear_sf,
                                    expected_codes, expected_sf)
            metrics("informational_dense_input", native_x, expected_x)
            expected = (native_x.double() @ reference_w.T).float()
            require_quality("dense_gemm_using_native_input", actual, expected)
            outputs = [actual]
            for repeat in range(3):
                output = dense_run(x, *prepared, n, eligible)
                torch.cuda.synchronize()
                require_quality(f"dense_repeat_{repeat}_reference", output, expected)
                for prior in outputs:
                    require_repeat_stable("dense_repeat_pair", output, prior)
                outputs.append(output)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                dense_run(x, *prepared, n, eligible)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph, stream=stream):
                captured = dense_run(x, *prepared, n, eligible)
            for _ in range(3):
                captured.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                require_quality("dense_graph_reference", captured, expected)
                require_repeat_stable("dense_graph_repeat", captured, actual)
        del reference_w, w, sf, prepared
    print("PASS: all dense shapes, fused scales, repeats, and CUDA graphs")


if __name__ == "__main__":
    main()
