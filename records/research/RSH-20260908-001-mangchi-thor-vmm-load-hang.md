# RSH-20260908-001: Jetson Thor (mangchi) llama.cpp Load Hang — VMM Allocator + --fit Planner

Opened: 2026-09-08 19-30-00 KST
Recorded by agent: codex

## Question

Why does llama-server (TheTom turboquant @ 407f3237, built for sm_110) hang
while loading Qwen3.8-Flash-Next UD-IQ4_XS on the Jetson AGX Thor (mangchi),
and how do we get a model to actually serve on this box?

## Symptoms

- `load_model: loading model ...` prints, then one of two stalls:
  1. With default `--fit on`: stops right after `MoE cache fit kept stock
     placement ...` — 90-100% CPU, ~0 disk I/O (weights never read).
  2. With `--fit off`: gets past the planner, weights partially load (~66 GB
     of 93.7 GB), then the process sleeps and never opens the port.
- The stuck process ignores SIGTERM and sometimes SIGKILL for tens of seconds;
  while it lives, `nvidia-smi` and any new nvidia-container process block in
  D state (GPU effectively wedged until the process finally dies).
- Kernel log (`dmesg`) shows repeated:
  `NVRM: GPU0 nvCheckOkFailedNoLog: Check failed: Out of memory
  [NV_ERR_NO_MEMORY] ... _memdescAllocInternal(pMemDesc) @ mem_desc.c:1336`
  — the CUDA VMM allocator failing, not a true capacity OOM (122 GB total,
  weights are 93.7 GB).

## Findings

- This is a **known llama.cpp-on-Jetson unified-memory issue**, not a defect in
  our build or the model. Two independent upstream threads cover it:
  - ggml-org/llama.cpp#21039 (Jetson Thor unified memory / vLLM/SGLang/llama.cpp)
  - the `--fit` memory planner hanging on Thor; one reporter had qwen4exp
    specifically misbehave on SM110.
- `--fit off` is a real (if partial) lever: it removes the first (planner)
  hang. It does not fix the VMM allocator OOM during weight load.
- Root cause of the load stall is the **CUDA VMM pool**
  (`ggml_cuda_pool_vmm::alloc` → `cuMemCreate`) on unified memory. Same
  allocator as RSH-20260523-001 (VMM pool OOM during FA scratch on the x86 box),
  but here it fires during model load rather than mid-inference.

## What we tried (in order)

1. Full gateway-style flags (mmproj, `--cache-type-* turbo4`, `-ngl 999`, FA on)
   → hang #1 (fit planner).
2. Minimal flags, f16 cache → same hang #1. Not config-specific.
3. `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` → no change; still hang #1.
4. `--fit off` → past planner, but hang #2 (VMM OOM at ~66 GB loaded).

## Fix being applied

- Rebuild the mangchi image with `GGML_CUDA_NO_VMM=ON` so the CUDA backend uses
  the legacy non-VMM pool instead of the VMM allocator. Done per-target in the
  Dockerfile (`GRIMOIRE_CUDA_NO_VMM_MANGCHI=ON`, grimoire keeps VMM); part of
  the build_config cache key so it forces a clean rebuild. Commit 426f86c.
- The `GGML_USE_VMM` macro is gated on `!defined(GGML_CUDA_NO_VMM)` in
  `common.cuh:254`, so the CMake option is the correct lever.

## Open questions / follow-ups

- Whether NO_VMM alone clears the load hang, or we also need `--fit off` baked
  into the mangchi model config (extra-args) once the planner path is reachable.
- If VMM-less load works, re-test `--fit on` to see if the planner hang was
  downstream of the same allocator or is a separate Thor planner bug.
- Long-context (256k) behavior on 128 GB unified memory is unverified; KV cache
  sizing vs. weights is the next constraint after the load succeeds.
