# RSH-20260912-004: Thor vs DGX Spark on Flash-Next W4A4, matched engine

Opened: 2026-09-12 15-40-00 KST
Recorded by agent: claude

## Question

Mangchi (Jetson AGX Thor, SM110) or a DGX Spark (GB10, SM121) for serving
Qwen3.8 Flash-Next NVFP4? Spark has more SMs, Thor the higher paper FP4 rate,
and both share 128 GB of unified memory at ~273 GB/s. Does the Spark serve this
checkpoint faster, with everything except the silicon held equal?

## Method

Borrowed spark01 (iso's, `ahri` account). Both hosts served
`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4` at revision `be794b99`, verified
file-for-file identical: 422 files, no size mismatches.

`scripts/spark/bench.py` ran unchanged on both: four fixed prompts at
temperature 0, three passes with the first discarded, plus cold prefill probes
whose length is calibrated against the server's own `prompt_tokens` and whose
leading nonce defeats prefix caching. Headline metrics are end-to-end tok/s and
spec-decode steps/s; both survive the reasoning-flush artifact described below.

The Spark was measured twice. The first run used vLLM's public day-0 image and
is **not** a like-for-like result: it had BF16 KV (that build's QSA rejects FP8)
and no reduced-vocabulary drafting. The second used `scripts/spark/Dockerfile`,
which rebuilds vLLM at Mangchi's exact pin `5fd5dd5cf4ac` for sm_121 and applies
the same `docker/mangchi-vllm/` patches, giving both hosts one engine build,
FP8 KV, draft vocab 98304, SSD-backed PLE, eager execution, MTP k=4,
`--max-num-seqs 1`, 8192 batched tokens and 0.72 memory utilisation.

## Findings

Matched engine, warm, medians and means over passes 2-3:

| measure | Thor | Spark | delta |
| --- | ---: | ---: | ---: |
| prefill 7.9k tok | 2,722 | 2,801 | +2.9% |
| prefill 35k tok | 2,727 | 2,911 | +6.7% |
| decode code | 37.56 | 42.84 | +14.1% |
| decode prose | 21.23 | 25.20 | +18.7% |
| decode reasoning | 34.74 | 41.95 | +20.8% |
| decode arith | 36.16 | 44.95 | +24.3% |
| MTP steps/s | 9.24-9.50 | 10.74-11.09 | +15-17% |

Acceptance length agrees within 3.2% on code, prose and reasoning, while
steps/s is uniformly 15-17% higher on the Spark. Equal draft quality with
faster steps is a hardware result, not a workload or measurement difference.

- **The Spark serves this checkpoint faster on both axes.** Decode margins
  exceed the ~5% noise floor others report for this model; the 7.9k prefill
  margin does not and should be read as a tie.
- **Both hosts use native FP4 for the routed experts**, by different kernels.
  The Spark logs `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`, declining
  `MARLIN` and `EMULATION`. Thor logs no such line because `thor_nvfp4`
  replaces vLLM's ModelOpt method with NVIDIA CuTeDSL kernels and is silent;
  the v8 image was confirmed built with `THOR_CUTEDSL_MOE=1`.
- **Thor's MTP draft head runs on untuned Triton.** The checkpoint excludes
  `mtp.*` from quantisation, so that MoE takes the unquantized path: the Spark
  selects FlashInfer CUTLASS, Thor selects Triton and warns
  `Using default MoE config. Performance might be sub-optimal!` for
  `E=512,N=640,device_name=NVIDIA_Thor.json`. At k=4 that head runs four of
  every five forwards, so the gap above partly reflects a missing tuned config
  rather than the chip. Tuning is tracked separately.
- **The first Spark result was wrong and is retracted.** It read as a tie on
  decode and a 13-14% Thor win on prefill. Both artifacts came from the day-0
  image's BF16 KV and missing draft vocab; FP8 KV alone moved 7.9k prefill from
  2,336 to 2,801 tok/s.

## Corrections made to the measurement

- Decode rate was initially computed as tokens after the first streamed delta.
  With thinking on, this build flushes the whole reasoning block at once, which
  produced 475,008 tok/s on the arithmetic prompt. The script now refuses to
  report a rate over a flush.
- Two benchmark clients once ran concurrently against a `--max-num-seqs 1`
  server, halving throughput and disagreeing internally (steps/s 2.84 against
  10.89 in one run). Internal disagreement between prompts is the contention
  signature; absolute values alone do not reveal it.

## Build notes

vLLM's official runtime image lacks six things a source build needs: `git`,
`cmake`, a Rust toolchain with `setuptools-rust`, the bare `libnvrtc.so`
development symlink, `cusparse.h` (present only in the pip CUDA tree), and
tolerance for external CUDA projects that SM121 does not use. Adding the whole
pip CUDA include directory to `CPATH` is the wrong fix: it supplies a second
copy of CCCL and the crt headers and the trees miscompile against each other.

Upstream has since pushed to PR #55557, so `refs/pull/55557/head` no longer
resolves to the pinned commit. Both Dockerfiles now fetch the SHA directly.
PR #55557 is still unmerged as of main `e7edf17c`, so FP8 QSA is not available
from a nightly image.

## Open questions

- Whether a tuned `E=512,N=640,device_name=NVIDIA_Thor.json` closes part of the
  decode gap. A full 1,920-config sweep is running on Mangchi.
- The Spark's own best MTP depth was not swept; k=4 was inherited from Thor.
- Neither host got CUDA graphs, because a single unit must keep the PLE table on
  NVMe and a host gather cannot be captured. `patch_ple_graph_split.py` is the
  untested path to lifting that on both.
