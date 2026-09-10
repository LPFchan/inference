# Plans

## Steady-State Architecture

Single Bee binary (`Anbeeld/beellama.cpp`) serves DFlash (`--spec-type dflash`), PFlash (via standalone `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse. Legacy `backend:dflash` daemon retired. See git history for the completed migration (Phases 1-7).

## CUDA Flash-Attention Scratch (deployed baseline)

The May Atomic V2 work was retired when Grimoire migrated to the current TheTom engine fork. The current build ports scoped CUDA flash-attention K/V scratch allocation to the pinned engine and keeps CUDA graphs off. Qwen3.8-27B reasoning aliases use a 237,568-token context, a physical prompt batch of 128, the BF16 vision projector on CPU, the MTP head on GPU, and symmetric turbo4 K/V. This configuration maximizes the repeatedly tested long-context boundary on one RTX 3090 while retaining vision and MTP.

The incident audit and OpenCode soak are recorded in `records/research/RSH-20260828-001-qwen38-long-context-crash.md`. The GPU-projector boundary tests are recorded in `records/research/RSH-20260829-001-qwen38-gpu-mmproj-190k.md`. The CPU-projector maximum boundary is recorded in `records/research/RSH-20260829-002-qwen38-cpu-mmproj-237k.md`.

A graph-safe reusable scratch reservation remains optional future work. Revisit it only if CUDA-graph throughput becomes more valuable than the currently verified long-agent-context headroom.

## Thor-native NVFP4 MoE for Qwen3.8 Flash-Next

Replace the target model's Marlin W4A16 expert fallback with NVIDIA's native
SM110 CuTeDSL W4A4 MoE kernel from TensorRT-Edge-LLM. Keep the current
`mangchi-vllm:thor-qsa-fp8-5fd5dd5` image and its 262,144-token service as the
rollback until the new backend passes correctness, full-model, and performance
gates.

### Implementation path

1. Pin the imported TensorRT-Edge-LLM source revision and preserve its
   Apache-2.0 provenance. Adapt the SM110 runner behind a narrow vLLM expert
   backend rather than copying unrelated TensorRT-Edge runtime code.
2. Support the exact Flash-Next target geometry first: hidden size 2,560,
   intermediate size 640, 512 experts, top-k 10, fused SwiGLU, ModelOpt NVFP4
   weights and static NVFP4 activations. Reject unsupported layouts and
   quantization schemes explicitly.
3. Map compressed-tensors/ModelOpt W4A4 weights, block scales, global scales,
   routing output, activation quantization, and BF16 output to the runner
   without quantize-dequantize-quantize conversions on the expert hot path.
4. Package the adapter and kernel build reproducibly in
   `docker/mangchi-vllm/`, retaining the SSD-backed PLE, persistent QSA top-k,
   FP8 attention KV, and native 262K context configuration.
5. Validate kernel output against a trusted BF16 or Marlin reference across
   representative token counts and routing patterns before loading the whole
   checkpoint. Then run short chat, long-prefill, and decode tests on Mangchi.
6. Benchmark Marlin, FlashInfer CUTLASS if its one-line SM110 dispatch fix is
   usable, and the NVIDIA CuTeDSL backend under identical clocks and requests.
   FlashInfer is a control measurement, not the primary implementation target.

### Acceptance gates

- The backend is selected only on SM110/SM110a for compatible true W4A4
  NVFP4 MoE layers; W4A16 and unsupported layers keep their existing backend.
- Numerical comparison finds no material correctness regression against the
  reference path, and unsupported shapes fail cleanly before kernel launch.
- Qwen3.8 Flash-Next starts from the downloaded W4A4 checkpoint, completes a
  deterministic chat smoke, and retains the 262,144-token configured context.
- Measured prefill and decode throughput are recorded against the current
  Marlin deployment using the same prompt, output length, power mode, clocks,
  CUDA-graph mode, and MTP setting.
- Deployment happens only after the native backend is both correct and faster;
  otherwise the current FP8-QSA/Marlin image remains production.

### 2026-09-10 canary result

The pinned NVIDIA CuTeDSL MoE path now loads the complete Flash-Next checkpoint,
handles vLLM's padded `-1` expert routes, serves a deterministic generation, and
retains FP8 KV with the configured 262,144-token context. The native path reached
1,671 prompt tok/s and 10.56 decode tok/s in the initial canary. Correctness is
accepted; performance is not yet accepted against the 4,500 prompt tok/s Thor
report, so this remains a canary rather than the production default.

## Thor-native NVFP4 dense path for Qwen3.8-27B

The 27B W4A4 checkpoint now has an opt-in SM110 dense backend based on NVIDIA's
TensorRT-Edge-LLM CuTeDSL kernel at revision `e8b2952`. It preserves each fused
projection's checkpoint-global scale, performs one BF16-to-FP4 activation pack,
and returns BF16 without a quantize-dequantize-quantize round trip.

The measured scheduler-aware policy is intentionally narrow:

- CuTeDSL handles decode, inputs below 1,568 rows, every distinct-scale fused
  projection, and B/A. Aligned projections use the 256-column tile from 512
  rows; B/A retains its safe 128-column per-column-scale specialization.
- At 1,568 rows or more, FlashInfer CUTLASS is used only when preparation proves
  the complete fused output has one exact alpha. Packed weights, swizzled block
  scales, and the single native activation pack are shared; no checkpoint copy
  or collapsed maximum scale is introduced.
- Any unreviewed shape, dtype, group size, tensor-parallel layout, scale layout,
  architecture, or tile fails before launch.

Acceptance evidence: 70 static tests and 88 SM110 GPU cases passed, including
FP64-reference comparisons, unaligned M, backend assertions, 792 repeat/graph
comparisons with zero drift, and a maximum relative RMSE of 0.00171549. The
full model answered `19 * 23` as 437 and retained a 262,144-token configured
context with FP8 KV. On the matched raw-completions workload, the final exact
hybrid reached a median 1,966.52 prompt tok/s and 10.24 decode tok/s. The old
CUTLASS control reached 2,021.0 and 8.94 tok/s respectively, but emitted the
fused-scale-collapse warning. The exact hybrid therefore keeps 97.3% of that
prefill rate and improves decode by 14.5% without the accuracy compromise.

## Mangchi vLLM build minimization follow-up

Keep the native vLLM wheel before Python-only patches in the container build so
adapter and telemetry edits reuse the expensive CUDA layer. Persist or bake the
exact SM110 Torch, CuTeDSL, Triton, and FlashInfer JIT artifacts used by the two
accepted models.

After both model paths are stable, audit a minimal source build rather than
removing extensions by file size alone. FA3 is unused on SM110 and can be
removed. FA2 is required by the current vision encoder; either retain only the
BF16 head-size variants used by these checkpoints, or remove FA2 only after an
operator decision to run `--language-model-only`. Remove other quant/kernel
families only after import, symbol, startup, text, vision, 262K-context, and
rollback tests prove that neither target model reaches them.

## Backlog (Future Interest)

| Priority | Item | Why Deferred | Prerequisite |
|----------|------|-------------|--------------|
| 1 | **DDTree tree-mode verify** — enable `--spec-branch-budget > 0` with larger `GGML_DFLASH_MAX_VERIFY_TOKENS` | Benchmarked (2026-05-18): tree-mode (budget=22) = 62.4 tok/s vs flat = 61.6 tok/s (+1%). The draft is already well-matched to the target for greedy decode (85% acceptance), so tree-mode adds negligible benefit. Deferred unless non-greedy sampling or a lower-quality draft changes the trade-off. | The 25-token cap fix is already deployed. Env var auto-derived by model_manager.py. |
| 2 | **GPU tape recording** — tree-mode DDTree verify using `dflash_tape_*` | Only needed if single-spec throughput becomes a bottleneck | Multi-spec batch decode |

## Not Pursuing

- Multi-spec batched decode — single-spec sufficient for current workload
- Monitoring for 413 path — (user decision)
- VMM park/unpark — measured (RSH-20260518-001): adds 2.5 GB VRAM overhead with no TTFT benefit for single-model. Unnecessary unless multi-model GPU sharing is needed later.
