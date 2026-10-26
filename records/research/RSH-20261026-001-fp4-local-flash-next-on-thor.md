# RSH-20261026-001: FP4-Local Flash-Next on Thor
Opened: 2026-10-26 20-33-52 KST
Recorded by agent: claude-code

## Question

Does an FP4-Local-format (NVIDIA / Local Inference Lab QAD) uncensored
Flash-Next beat the production W4A4 NVFP4 Flash-Next on Mangchi in speed,
memory, or quality, and what does the pinned vLLM need to serve it on SM110?

## Candidate

`qwen3.8-flash-next-fp4local-abliterated-v2`, 96 GiB on disk. It uses ModelOpt
`MIXED_PRECISION`: NVFP4 W4A4 routed experts with CSF-compressed block scales
(`byte-window4-fixed-stream-u24-exceptions/1`), MXFP8 attention and dense
linears, W4A16 NVFP4 MTP-head experts, a BF16 vision tower, and an NVFP4 n-gram
PLE table (320,001,536 rows, 26.8 GiB).

## What Serving It Needed

B12X kernels target SM120/SM121 only. The literal port was not needed:
production's Thor CuTe DSL MoE kernels already run the NVFP4 experts.

- CSF scale decode at load: a port of vllm#60646.
- NVFP4 PLE rows read from SSD through `vllm_ple_mmap` and decoded on the GPU
  (row decode from vllm#56273). Keeping the table resident added 26.8 GiB.
  Host memory then fell to 3 GiB on the first 8k-token prefill, and the canary
  watchdog stopped it.
- `ModelOptMixedPrecisionConfig` builds its NVFP4 expert method directly, which
  bypassed the Thor registration hook. Without the second hook in
  `thor_nvfp4/install.py`, the experts silently ran on `VLLM_CUTLASS`.
- `VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel`. FlashInfer's MXFP8
  kernel asserts `N >= 128` at runtime but accepts every layer in
  `can_implement`. The GDN `in_proj_ba` (N=96) crashes warmup. Upstream
  vllm#54223 selects per layer; this pin does not have it. Marlin MXFP8 matched
  BF16 exactly (cosine 1.0) in a kernel probe.
- vLLM can deliver the PLE shards and the table's `weight_scale_2` in separate
  `load_weights` calls. The NVFP4 table is therefore recognised from the
  checkpoint's shard dtype (`U8`), not from whether the scale has loaded yet.

## Findings

Same flags as production v10, including 14,485,411,127 KV bytes, MTP=4, and
`QWEN4EXP_DRAFT_VOCAB=98304`. Measured with `scripts/spark/bench.py`; warm
medians.

| | Production v10 | FP4-Local (Marlin MXFP8) |
|---|---|---|
| Weights in memory | 79.41 GiB | 73.48 GiB |
| KV cache | 844,883 tokens | 844,883 tokens |
| Prefill 7.9k / 35.4k | 2,697 / 2,696 tok/s | 1,940 / 1,936 tok/s |
| Code e2e (accept len) | 41.3 tok/s (4.06) | 33.7 tok/s (4.02) |
| Prose e2e (accept len) | 24.2 tok/s (2.35) | 20.4 tok/s (2.32) |
| Reasoning e2e (accept len) | 39.7 tok/s (3.91) | 32.8 tok/s (3.86) |
| Arithmetic e2e (accept len) | 38.2 tok/s (4.40) | 26.4 tok/s (3.33) |
| Steps/s, code / prose | 10.17 / 10.36 | 8.35 / 8.75 |
| Exact answers | 10/10 | 10/10 |
| Refusals (thinking off and medium) | 0/8 | 0/8 |
| Han characters in Korean prose | 0 | 0 |

- FP4-Local saves 5.9 GiB of weights and loses 16-28% of throughput.
- Draft acceptance is unchanged, so the loss is per step: about 17% fewer
  steps per second. The MXFP8 attention and dense linears read twice the bytes
  of production's NVFP4 dense layers. Decode on Thor is memory-bound.
- Quality was equal on this small probe. It cannot show the QAD accuracy gains
  the format claims. Those would need a longer evaluation.

## Conclusion

FP4-Local stays off production on Mangchi. The supporting code is kept on the
`fp4local-thor` branch so a future FP4-Local-format checkpoint can load on the
pinned vLLM.

## Follow-Up Routes

- Porting vllm#54223 would let FlashInfer serve the large MXFP8 layers. That
  can recover prefill: a kernel probe measured 475 µs against BF16's 826 µs at
  M=2048. It cannot recover decode, where FlashInfer was slower than Marlin at
  M=1 (152 µs against 43 µs on 2560x4096).
- Copying the 96 GiB checkpoint from Grimoire to Mangchi ran at about 12 MB/s
  on both the LAN address and Tailscale, though both NICs negotiate 1-2.5 Gb.
  The limit is likely on the home network.
