# RSH-20260915-001: Mangchi vLLM Load-Speed Bottleneck
Opened: 2026-09-15 03-13-02 KST
Recorded by agent: codex

## Question

Is Flash-Next startup limited by Mangchi's NVMe throughput, and can model-load
work be cached without reducing usable KV below three native 262,144-token
contexts?

## Findings

- Direct NVMe reads reached 3.32 GiB/s at queue depth 1 and 5.22 GiB/s at
  queue depth 16. The original target-weight loader needed 387.91 seconds for
  about 73 GiB of resident weights, only about 0.19 GiB/s. Storage was not the
  bottleneck.
- The source checkpoint contains roughly 300,000 tensors. vLLM spent most of
  target loading converting ModelOpt NVFP4 expert tensors into the prepared
  Thor CuTe DSL layout, not reading bytes from disk.
- Persisting `/root/.cache`, including Triton's compiled kernels, reduced a
  warm source-checkpoint restart from about 753 to 539 seconds. Target weights
  still took 378.28 seconds, but engine profiling and warmup fell from 231.97
  to 54.27 seconds.
- Saving vLLM's post-load state once produced 20 sequential safetensors shards
  totaling 73.36 GiB. The accepted load read those prepared target weights in
  65.81 seconds, loaded the original MTP draft in 43.50 seconds, and reported
  118.36 seconds for total model loading. A warm-compiler restart became
  healthy about 154 seconds after container launch, 4.9 times faster than the
  old 753-second baseline.
- A production restart after rebuilding the Thor image exposed a separate
  cold CuTe DSL compilation cost. Prepared target loading still took 66.22
  seconds, but total model loading rose to 271.52 seconds and the API became
  healthy in about 379 seconds. A following restart returned to 68.11 seconds
  for the target, 117.43 seconds for all model loading, and about 154 seconds
  to health. NVIDIA's `CUTE_DSL_CACHE_DIR` created no files for these exported
  kernels, so it was not retained as a placebo setting. Both observed startup
  times remain below the gateway's 600-second deadline.
- The explicit 13.49 GiB FP8 KV reservation produced 844,883 cache tokens:
  3.22 native contexts or 2.15 extended 393,216-token contexts. Four short
  concurrent requests ran together with zero waiting. Host available memory
  remained about 17 GiB after load and inference.
- `--skip-mm-profiling` only skips the synthetic maximum-image sizing pass;
  vision remains enabled. A real PNG request succeeded after the optimized
  load.

## Required Compatibility Work

Stock vLLM sharded-state loading assumes saved and fresh parameter shapes are
identical. The Thor backend's prepared expert scales are six-dimensional, and
the mmap-backed PLE table is intentionally absent from a fresh model state.
The accepted path therefore:

1. restores only the eight known prepared Thor expert tensors by replacing
   their fresh parameter storage;
2. validates and reuses that prepared layout instead of swizzling it twice;
3. ignores PLE state because the original checkpoint remains its mmap source;
4. passes `draft_load_config` through the MTP/EAGLE loader so the small draft
   continues using the normal checkpoint loader.

## Rejected Paths

- vLLM's generic multithreaded loader can queue enough converted tensors to
  exceed Thor's unified memory, so it was not enabled.
- RunAI's streaming sharded loader pushed host available memory below the
  agent's 6 GiB safety floor during startup, so the lower-memory standard
  sharded loader remains selected.
- Startup-plan caching does not remove ModelOpt weight conversion and does not
  apply when KV bytes are explicit.
- Pointing `CUTE_DSL_CACHE_DIR` at the persistent cache mount produced no
  compiler artifacts for the exported Thor kernels and did not provide a
  durable cache to retain.
- Two sharded checkpoints made with mismatched serving geometry were removed.
  They were derived artifacts; the source checkpoint was untouched.

## Validation

- Deterministic text: `19 * 23` returned `437` with thinking disabled.
- Vision: the model correctly described the local robot-logo PNG.
- Concurrency: four simultaneous 80-word requests all returned HTTP 200; vLLM
  logged four running and zero waiting.
- Unit tests: 34 Mangchi agent and registry tests passed.
