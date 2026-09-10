# RSH-20260910-001: Thor W4A4 Prefill Batch Sweep

Opened: 2026-09-10 17-41-08 KST
Recorded by agent: codex

## Question

Can larger `--max-num-batched-tokens` settings raise peak prefill throughput
above 3,000 tokens/s for the native SM110 W4A4 Qwen3.8 models while retaining
their 262,144-token context limit?

## Method

Both models used `mangchi-vllm:thor-dense-candidate-v6-vision-minfa`, eager
execution, FP8 KV, one sequence, disabled FlashInfer autotuning, and the native
CuTeDSL backend selected for that checkpoint. Each successful setting received
three streaming chat requests containing 36,885 or 36,886 prompt tokens and one
output token. A nonce near the start prevented vLLM prefix-cache reuse.

End-to-end rates use vLLM's first-token latency. Peak rates use the largest
token delta divided by its elapsed-time delta from the live chunked-prefill
events. The prompt intentionally repeats a short token sequence, so the warm
Flash-Next results measure favorable PLE locality and are peak-performance
evidence rather than a claim about arbitrary prompts.

## Qwen3.8-27B W4A4

All settings started with the full 262,144-token limit. Medians across three
runs:

| Token budget | Peak chunk tok/s | End-to-end tok/s | KV capacity | Chunks |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2,120 | 779 | 1,279,081 | 23 |
| 4,096 | 1,826 | 757 | 1,268,535 | 12 |
| 8,192 | 1,443 | 762 | 1,245,937 | 5 |
| 16,384 | 1,025 | 749 | 1,206,766 | 3 |

The 2,048 setting is the clear peak-throughput winner. Larger chunks make the
attention-heavy work slower and do not materially improve full-prompt time.
Changing this scheduler limit cannot take the dense model to 3,000 prompt
tokens/s; further work must profile the model components and kernel tactics.

## Qwen3.8 Flash-Next W4A4

Flash-Next showed a first-request penalty after each restart. No runtime JIT
warning appeared. SSD-backed PLE page faults or another shape-specific cold
path are plausible causes, but the sweep does not isolate them. The table keeps
the first run separate and reports the median of runs two and three as warm
steady state.

| Token budget | Cold end-to-end / peak tok/s | Warm end-to-end tok/s | Warm peak tok/s | KV capacity |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2,515 / 5,358* | 2,591 | 3,507 | 332,612 |
| 4,096 | 791 / 2,855 | 2,814 | 3,747 | 304,425 |
| 8,192 | 2,704 / 3,026 | 2,962 | 3,967 | 270,600 |
| 16,384 | did not start | — | — | estimated 175,616 |

`*` The 5,358 tok/s chunk was not reproduced; later 2,048-token runs peaked at
3,534 and 3,480 tok/s. It is not a sustained result.

At a 0.68 memory allocation, the 16,384 setting left 2.49 GiB for KV while
3.55 GiB was required and vLLM rejected the 262,144-token configuration. At
8,192, only 8,456 KV tokens of margin remain above the configured maximum. It
is the fastest tested 0.68 canary, but 4,096 retains 42,281 tokens of KV margin
and gives up only about 5% warm end-to-end throughput.

Raising `--gpu-memory-utilization` to 0.70 made the 16,384 server start with
4.95 GiB of KV capacity: 363,619 tokens, or 1.39 concurrent 262,144-token
requests. Memory was no longer the blocker. Its first 36,886-token request then
spent more than 600 seconds in the engine worker before the client timed out.
The worker remained CPU-bound, vLLM reported no scheduled prompt tokens, and
there was no CUDA or JIT-monitor error. The native MoE runtime accepts up to
262,144 input tokens, but its hardware correctness gate covers only 1, 10, 32,
and 129 tokens. A roughly 15,680-token scheduler chunk therefore crosses an
untested native launch boundary. This sweep does not prove whether the stall is
inside MoE preparation, the CuTeDSL launch, or another CPU-side QSA/PLE step.

## Conclusion

- Keep 2,048 for the dense 27B model.
- Flash-Next demonstrably exceeds 3,000 tok/s at both 4,096 and 8,192.
- Prefer 4,096 for an operational Flash-Next default at 0.68 unless the
  operator explicitly accepts 8,192's narrow full-context memory margin.
- A 0.70 allocation gives 16,384 enough KV memory, but that setting is not
  usable until the first-request stall is isolated. Extend the native MoE gate
  to served chunk sizes and test splitting large MoE inputs into proven smaller
  launches before another full-model attempt.
- Profile the 27B projection, attention, GDN, and non-FP4 time separately;
  scheduler batch width is not its missing optimization.
- Measure Flash-Next with varied natural/token-random prompts before treating
  these favorable warm PLE-locality rates as typical workload throughput.
