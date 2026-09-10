# Experimental Thor NVFP4 experts

This package adapts NVIDIA TensorRT-Edge-LLM's native SM110 split CuTe DSL
FC1/FC2 kernels to the ModelOpt MoE method in the pinned vLLM image. It is an
experimental implementation pending CUDA compilation, numerical checks, and
full-model validation on Thor. CPU tests do not establish GPU correctness or
speed. Production acceptance remains in `records/PLANS.md`.

## Source and build contract

- NVIDIA source: `NVIDIA/TensorRT-Edge-LLM` revision
  `e8b29522938901f6df19ebeedd4b69bc8edbcd97`, Apache-2.0.
- vLLM source: `5fd5dd5cf4ac8e9f09b6fae3f3603e9a3cb88aaa`.
- `install.py` pins SHA256 for eight Python kernel/helper files, two CUDA setup
  files, and LICENSE. It retains untouched originals and their license headers
  inside the installed `thor_nvfp4/nvidia/original` directory.
- Python adaptations make imports package-relative and return the compiled
  kernel from NVIDIA's export wrapper. CUDA adaptations retain device code,
  replace TensorRT host dispatch with a Torch binding, and skip vLLM's `-1`
  padded-route sentinel before indexing activation scales. NVIDIA's full
  TensorRT runtime is unnecessary.
- The FC2 exporter selects NVIDIA's register-atomic epilogue (`use_blkred=False`).
  Its pinned bulk-reduction path issues asynchronous shared-to-global reductions
  without a bulk commit/wait before shared-buffer reuse or a producer barrier
  before row reads. Register atomics avoid that shared-memory lifetime hazard;
  this source correction still needs hardware validation.
- Dependencies: CuTe DSL 4.7.0 with CUDA 13, CuPy CUDA 13 13.6.0, cuda-python
  13.3.1, cuda-bindings 13.3.1, and the image's existing Torch/CUDA compiler.
  The opt-in install adds official PyPI alongside the inherited Jetson index
  so these pinned releases are available. The CUDA Python umbrella and binding
  versions satisfy each other's metadata and Torch's `cuda-bindings>=13.0.3,<14`
  requirement. The existing cuda-pathfinder 1.5.1 satisfies their constraints.
  Kernels compile once
  during weight preparation, before graph capture, using the Torch/CuTe caches.

The default Docker build keeps its existing dependency set. Build the separate
candidate from `docker/mangchi-vllm` after scheduling free GPU memory:

```sh
docker build --build-arg THOR_CUTEDSL_MOE=1 \
  -t mangchi-vllm:thor-cutedsl-e8b2952 -f Dockerfile .
```

Enable selection with `VLLM_THOR_CUTEDSL_MOE=1` and `--moe-backend auto` in a
separate canary. Exact supported geometry is H=2560, I=640, E=512, top-k=10,
SwiGLU, BF16 IO, ModelOpt NVFP4 group-16, single-device execution. W4A16,
other shapes/dtypes, explicit alternative backends, LoRA, bias, clamped
SwiGLU, and distributed modes retain vLLM's existing method. Once selected,
unsupported tensor layouts or unequal gate/up scales fail during loading.
The compressed-tensors method is unchanged; this first adapter targets the
downloaded checkpoint's actual `quant_method=modelopt` format.

## Numerical mapping

Packed weights remain FP4. Loading reorders FC1's `[gate, up]` rows into
64-row `[up, gate]` chunks and swizzles FP8 block-scale bytes to NVIDIA's MMA
layout. Gate/up global weight and activation scales must match exactly. Block
scales can be zero; global scales must be positive and finite. No weight
dequantization/requantization occurs during loading or inference.

NVIDIA's routed input pack reads BF16 directly and applies each expert's
activation global scale. FC1 performs FP4 matrix multiplication, SwiGLU, and
FP4 intermediate quantization. FC2 performs FP4 matrix multiplication and
weighted scatter into BF16 output. The CuTe epilogues multiply the weight
global scale by the activation global scale internally. The adapter passes
the weight global scale as `alpha`, avoiding a second multiplication by the
activation scale. Routing ids/weights come from vLLM. The legacy vLLM runner
executes shared experts separately.

The initial CUDA setup uses NVIDIA's general layout and routed-quantization
kernels for both decode and prefill. The fused single-token setup is a later
performance option if measurements justify it. Scratch allocations use Torch
on the current CUDA stream, including during CUDA graph capture.

## Verification gates

CPU/static checks from the repository root:

```sh
python3 tests/test_thor_nvfp4.py
PYTHONPATH=docker/mangchi-vllm python3 -m thor_nvfp4.install --check-sources
git diff --check
```

First run these inside the separate candidate container with the NVIDIA
runtime, a writable compiler cache, and sufficient unused memory:

```sh
python -m thor_nvfp4.check --compile-only
python -m thor_nvfp4.check
```

The numerical test gates each stage separately. Routing and repacking must
match exactly. Input quantization must have exactly matching FP8 block scales;
any differing FP4 codes must preserve sign and choose adjacent E2M1 levels
within `8 * float32 epsilon * max(1, abs(midpoint))` of their rounding midpoint
in normalized FP4 units. This small allowance covers FP32 reciprocal/multiply
rounding in CUDA versus division in the independent emulator. It cannot excuse
wrong scales, nonadjacent levels, or differences away from a rounding boundary.
Inputs, scales, and decoded values must be finite. An independent FP64 check
requires each decoded value's error to the original input to be no greater than
the nearest representable level's error plus
`2 * INPUT_BOUNDARY_RTOL * max(block_scale * global_scale, abs(input))`
(with eight FP64 epsilons of arithmetic slack). The factor two bounds the
extra error from choosing the other side of a midpoint. This check also covers
codes that agree with the emulator. Input emulator-comparison RMSE and cosine
are informational: valid opposite midpoint choices need not agree in aggregate.

FC1/SwiGLU/requant is compared against a reference driven by the actual native
quantized input. FC2/finalize is compared against a reference driven by the
actual native intermediate. Each routing slot is checked separately, and the
combined scatter is compared with the sum of isolated native slot outputs.
These downstream stages require finite values, relative RMSE <= 0.02, and cosine >= 0.999.
All four token/routing cases run these gates and CUDA graph replay. The complete
pipeline comparison against independently emulated input quantization remains
an informational sensitivity metric: SwiGLU can amplify a few valid opposite
choices at FP4 rounding boundaries.

`python -m thor_nvfp4.check --diagnose` additionally reports differing FP4 code
and FP8 scale counts and collects numerical failures through all stages, slots,
and repeated launches before raising. Structural routing/layout failures stop
immediately. Four same-input outputs each pass the strict reference quality
gate and are compared pairwise using a separate 1% repeat-stability limit:
`norm(a-b) / max(norm(a), norm(b)) <= 0.01`. Nonfinite values fail. Bitwise
equality and differing-element counts are informational diagnostics.

BF16 unit roundoff is `u=2^-8`. Ten contributions require nine rounded
additions after the first exact addition to zero. An independent uniform-rounding
estimate for the difference between two sums is `u*sqrt(2*9/3) = 0.00957`;
the screening limit rounds this to 1%. This is a practical numerical-stability
budget, not a worst-case error theorem: correlated rounding and cancellation
can exceed it and require investigation. It is narrower than the stage-quality
floor and rejects 3–5% repeat drift. Hardware validation must confirm that the
register-atomic path meets it. BF16 atomics have unspecified summation order;
bitwise determinism is not an acceptance requirement.
Stage validation retains scratch buffers and synchronizes
only in the standalone test; serving does not run these checks. CPU tests in
`tests/test_thor_nvfp4_reference.py` verify rounding, boundary rejection, and
stage-reference helpers using Torch without a GPU. Do not relax thresholds for
unexplained failures. The synthetic test retains one original layer's weights
and needs several GiB of free memory; it does not prove correctness on the
downloaded checkpoint.

Next load the downloaded W4A4 checkpoint in the canary with the existing
PLE mmap, persistent QSA top-k, FP8 attention KV, and 262144 context settings.
Require the native-backend log, a deterministic short chat, long prefill,
and decode. Verify shared experts and QSA behavior in that full-model run.

Record identical prompt/output token counts, clocks, power mode, CUDA-graph
mode, MTP setting, and warm-up policy for Marlin versus native. Compare both
on the same W4A4 checkpoint to separate kernel effects from checkpoint
changes; also record the existing W4A16 production baseline separately.
Use CUDA events for isolated layer timings and client timing for full-model
prefill/decode. FlashInfer can serve as an additional control when usable.
Preserve the production image until all correctness and performance gates pass.
# Dense Qwen3.8-27B W4A4 opt-in

The dense adapter is separate from the Flash-Next MoE adapter. Build the existing
`THOR_CUTEDSL_MOE=1` image variant to install both pinned kernel families, then
set `VLLM_THOR_CUTEDSL_DENSE=1` only for the dense candidate. It uses NVIDIA
TensorRT-Edge-LLM `e8b29522938901f6df19ebeedd4b69bc8edbcd97`'s
`gemm_cutedsl/gemm_blackwell_nvfp4_ws.py`, with a per-column FP32 global-scale
epilogue and BF16 output. Imported source hashes and Apache-2.0 originals remain
auditable through `install.py`. This is experimental until hardware acceptance.

The initial contract is SM110, visible device 0, TP=1, BF16 input/output,
NVFP4 W4A4 group-16, and these `(N,K)` pairs:
`(34816,5120)`, `(5120,17408)`, `(14336,5120)`, `(5120,6144)`,
`(16384,5120)`, `(96,5120)`. The 96-row B/A projection pads weights, scales, and
alpha to 128 rows with zeros and slices the result back to 96. All other shapes
fail after explicit opt-in. W4A16, non-SM110 devices, and opt-out keep the original
vLLM selection. The switch does not affect MoE dispatch. Existing BF16 state,
convolution, norm, vision, embedding, and lm-head paths remain unchanged.
Use the default `--linear-backend=auto`; conflicting backend selection and
`VLLM_BATCH_INVARIANT` are rejected. The raw GEMM launch has six pointers
`A,B,SFA,SFB,C,alpha`, runtime Int64 `M,N,K`, Int32 `max_active_clusters`, and
the caller's `CUstream`. Scale-vector size 16 is specialized at compilation.

Compressed-tensors divisors and ModelOpt multipliers are intercepted before
vLLM reduces fused globals. Each logical slice retains its own weight global;
activation globals must agree exactly across slices or loading fails. Packed
FP4 weight bytes are retained, FP8 block scales are losslessly swizzled, and
each invocation performs one NVIDIA BF16-to-FP4 activation quantization.
Column alpha is the product of forward activation and weight scales and is
applied in the FP32 accumulator before BF16 conversion. Every non-B/A logical
boundary must align to both 128- and 256-column tiles. Each such tile loads its
exact alpha once per epilogue thread and broadcasts it across its accumulator
values. Distinct fused slices retain distinct products. Padded B/A uses
per-column alpha because its 48-column boundary crosses a tile. Preparation
rejects unaudited or misaligned layouts. No FP16 intermediate or weight
requantization is used.

The dense backend uses CUTLASS through the pinned FlashInfer interface only
when M>=1568 and preparation proves every output column has exactly the same
positive FP32 alpha. CuTeDSL handles smaller M, distinct fused globals, and
padded B/A. Both branches share the packed weights, 128x4-swizzled FP8 scales,
and one NVIDIA activation pack. CUTLASS receives a scalar view of the exact
prepared alpha; no maximum or approximate equality is permitted. Its module
is warmed before serving, and errors propagate without a backend fallback.
The shared CUTLASS workspace is 32 MiB; no second checkpoint copy is retained.

Validation commands inside a candidate image:

```sh
python -m thor_nvfp4.install --check-sources
python -m thor_nvfp4.dense_check --compile-only
python -m thor_nvfp4.dense_check
```

The numerical gate covers all six shape classes at M=1,7,129,512,608,1568,2048,2205 with deliberately
different fused weight globals, plus five uniform-alpha hybrid shape profiles
at those same sizes. It checks backend selection, pointwise-valid activation rounding, an
independent FP64 GEMM reference driven by the native quantized input, repeated
launches, and poisoned-output CUDA graph replay. It retains the 2%/.999 output-quality floor
and separate 1% pairwise stability screen. This standalone gate allocates large
FP64 reference matrices; allow several GiB of free GPU memory. It does not
execute on the serving path.

Full-model acceptance used the exact preetpatel checkpoint at 262144 context,
confirmed the `THOR_DENSE_BACKEND ready` marker, checked health and a coherent
437 answer for `19 * 23`, and benchmarked identical raw completion requests
against the preserved CUTLASS control. Kernel compilation or a healthy HTTP
endpoint alone is not sufficient evidence.

## Dense tactic policy

Serving precompiles three variants before graph capture: tile-broadcast alpha
with N-tile 128 and 256, and per-column alpha with N-tile 128 for B/A.
It selects 256 for `(N,K)=(5120,6144)` at every nonempty M and for
other N-divisible-by-256 projections at M>=512. All remaining cases use 128,
including padded B/A.
This tile policy applies when the hybrid dispatcher selects CuTeDSL.

The pinned scheduler uses 1568-token Mamba-aligned cache blocks with prefix
caching enabled. A fresh 2176-token request therefore executes 1568 and 608
token chunks. These actual call sizes are covered by the dispatch and GPU
gates. Matched component measurements support tile256 for both chunks and
uniform-alpha CUTLASS for the 1568-token chunk. The 608-token output projection
keeps CuTeDSL because its eager full launch path is faster there.
No serving-time tuning or first-launch compilation is permitted. The first
successfully prepared layer prints `THOR_DENSE_BACKEND ready` with source,
precision, scale semantics, and policy, independent of vLLM's early rank logger.

Compare the accepted variants with `python -m thor_nvfp4.dense_bench`. It
compiles N-tiles 128 and 256, excludes incomplete epilogue tiles,
checks numerical output/repeats/graphs, then records seven shuffled rounds of
20 CUDA-graph replays per compatible tactic. The 192 variant failed its first
unrestricted check (17.3% RMSE); complete per-column alpha coverage is now a
launch requirement. Production compilation and launch reject every tile except
128 and 256. Tile 256 is excluded for B/A. Tile 64's small B/A savings did not
justify another tile width.

Two disposable Thor sweeps with the per-column epilogue agreed on the selected
tile-width improvements. Median times
from the repeat sweep, including activation packing and output slicing:

| Projection | M | Tile 128 (µs) | Tile 256 (µs) |
|---|---:|---:|---:|
| gate/up | 2048 | 3003.3 | 2300.6 |
| down | 2048 | 1663.8 | 1366.6 |
| attention QKV | 2048 | 1317.8 | 1026.6 |
| GDN QKV/Z | 2048 | 1473.5 | 1171.4 |
| attention/GDN output | 2048 | 639.7 | 533.5 |
| attention/GDN output | 1 | 51.35 | 39.07 |

Weighted by the model's projection counts, these component timings initially
implied 20.7% less dense-projection time at M=2048 and 1.2% at M=1. Scheduler
tracing later showed that prefix-enabled serving actually splits the matched
prompt into M=1568 and M=608, which produced the final policy above. The final
whole-model medians were 1966.52 prompt tok/s and 10.24 decode tok/s, versus
2021.0 and 8.94 for the old CUTLASS control. No clocks or power settings were
altered. The cold CUDA helper build remains about 41 seconds and should be
baked into the production image rather than paid at every fresh-container
startup.
