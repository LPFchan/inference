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
  replace TensorRT host dispatch with a Torch binding, and guard expert ids
  before indexing activation scales. NVIDIA's full TensorRT runtime is unnecessary.
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
Input dequantized values must also meet relative RMSE <= 0.02 and cosine >= 0.999.

FC1/SwiGLU/requant is compared against a reference driven by the actual native
quantized input. FC2/finalize is compared against a reference driven by the
actual native intermediate. Each routing slot is checked separately, and the
combined scatter is compared with the sum of isolated native slot outputs.
Every stage requires finite values, relative RMSE <= 0.02, and cosine >= 0.999.
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
