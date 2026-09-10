"""Isolated tactic census with reference/repeat gates and CUDA-graph timings."""
import argparse
import contextlib
import io
import json
import random
import statistics
import time

import torch

from .check import unpack
from .dense_adapter import prepare_tensors
from .dense_check import CASES
from .dense_runtime import launch_dense, warmup_dense
from .diagnose import require_quality, require_repeat_stable, unswizzle_scales

TILES = (128, 256)


def quality(*args):
    with contextlib.redirect_stdout(io.StringIO()):
        require_quality(*args)


def stable(*args):
    with contextlib.redirect_stdout(io.StringIO()):
        require_repeat_stable(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(123)
    warmup_dense(0, 128, False)
    supported = []
    for tile in TILES:
        start = time.perf_counter()
        try:
            warmup_dense(0, tile)
        except Exception as error:
            print("COMPILE " + json.dumps(dict(tile=tile, error=str(error))), flush=True)
            continue
        supported.append(tile)
        print("COMPILE " + json.dumps(dict(tile=tile, seconds=time.perf_counter()-start)), flush=True)
    if 128 not in supported:
        raise RuntimeError("Baseline tactic failed compilation")
    for n, k, widths in CASES:
        w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        sf = torch.full((n, k // 16), .125, dtype=torch.float8_e4m3fn, device="cuda")
        wg = torch.linspace(400., 1600., len(widths), device="cuda")
        ag = torch.full_like(wg, 100.)
        prepared = prepare_tensors(w, sf, wg, ag, list(widths), divisors=True)
        eligible = [t for t in supported if prepared[0].shape[0] % t == 0]
        for tile in set(supported) - set(eligible):
            print("SKIP " + json.dumps(dict(n=n, k=k, tile=tile, reason="incomplete N epilogue tile; unsafe alpha read")), flush=True)
        column_wg = torch.repeat_interleave(wg.reciprocal(), torch.tensor(widths, device="cuda"))
        reference_w = unpack(w, sf, column_wg[:, None])
        for m in (1, 7, 129, 2048):
            x = (torch.randn(m, k, device="cuda") * .25).to(torch.bfloat16)
            trace = {}
            baseline = launch_dense(x, *prepared, n, trace=trace, tile_n=128)
            linear_sf = unswizzle_scales(trace["sf"], (m+127)//128*128, k//16)[:m]
            native_x = unpack(trace["packed"], linear_sf.view(torch.float8_e4m3fn), ag[0].reciprocal())
            # FP32 with TF32 disabled; the standalone acceptance gate additionally
            # uses FP64. Reference and quantizer work stay outside timed regions.
            expected = native_x @ reference_w.T
            quality("benchmark_baseline_reference", baseline, expected)
            graphs, outputs = {}, {}
            for tile in eligible:
                print(f"CHECK M={m} N={n} K={k} tile={tile}", flush=True)
                output = launch_dense(x, *prepared, n, tile_n=tile)
                quality("tactic_reference", output, expected)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    launch_dense(x, *prepared, n, tile_n=tile)
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(graph, stream=stream):
                    captured = launch_dense(x, *prepared, n, tile_n=tile)
                for repeat in range(3):
                    graph.replay()
                    torch.cuda.synchronize()
                    quality("tactic_graph_reference", captured, expected)
                    stable("tactic_graph_repeat", captured, output)
                graphs[tile], outputs[tile] = graph, captured
            timings = {tile: [] for tile in eligible}
            for _ in range(args.rounds):
                order = eligible.copy()
                rng.shuffle(order)
                for tile in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        graphs[tile].replay()
                    end.record()
                    end.synchronize()
                    timings[tile].append(start.elapsed_time(end) * 1000 / args.iterations)
            print("BENCH " + json.dumps(dict(m=m, n=n, k=k, median_us={t: statistics.median(v) for t,v in timings.items()},
                                               samples_us=timings)), flush=True)
            del graphs, outputs, baseline, trace, expected, native_x
        del reference_w, w, sf, prepared


if __name__ == "__main__":
    main()
