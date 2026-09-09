# RSH-20260909-001: Qwen3.8 Flash-Next on Jetson Thor
Opened: 2026-09-09 20-10-50 KST
Recorded by agent: codex

## Question

Can `blazux/qwen3.8-Flash-DGX` run on the Jetson AGX Thor, and can its PLE
table remain on SSD instead of exhausting Thor's 128 GB unified memory?

## Findings

- The published Blazux image is specific to DGX Spark/GB10 (`sm_121`). Its
  deterministic top-k extension and several kernel patches are built for that
  architecture, so the whole image is not a safe Thor (`sm_110a`) replacement.
- The useful portable part is its Apache-2.0 PLE mmap loader. Flash-Next only
  looks up a small number of hashed PLE rows per token, so the table can remain
  in its safetensors file and use Linux's SSD-backed page cache.
- This checkpoint stores 128 BF16 PLE tensors in one 102.4 GB file (95.4 GiB),
  with 320,001,536 rows of width 160. Stock vLLM first allocates the full table
  in unified memory, which is why CPU offload did not help.
- The local Thor image uses vLLM v0.29's newer `Qwen4ExpNGramEmbedding`, not
  Blazux's older class. The mmap loader therefore needs a small port rather
  than a direct copy.

## Implementation and validation

- Kept the existing Thor-native CUDA, PyTorch, and vLLM build.
- Added an opt-in `Qwen4ExpNGramEmbedding` patch that replaces only the PLE
  allocation with a tiny placeholder and gathers requested rows from mmap.
- Wrapped the CPU gather as a vLLM custom op so it stays outside compiled CUDA
  graphs. Initial deployment uses `--enforce-eager` until compiled mode is
  separately validated.
- Verified mmap output byte-for-byte against safetensors across shard
  boundaries.
- Built `mangchi-vllm:thor-v0.29-ple-mmap`. A real checkpoint load mapped all
  128 PLE shards from SSD and loaded the remaining model in 71.11 GiB, passing
  the previous allocation failure.
- `gpu-memory-utilization=0.62` left only 0.81 GiB for KV, below the 3.44 GiB
  required for one 131,072-token request. The deployed setting is 0.65, which
  adds about 3.7 GiB to the memory budget while retaining system headroom.
- FlashInfer 0.6.8 is the newest known-working aarch64 release in this image,
  but it predates vLLM v0.29's `set_autotune_process_group` API. Startup uses
  vLLM's supported `--no-enable-flashinfer-autotune` switch and retains
  FlashInfer's normal heuristic kernel selection.
- vLLM also selects a CUDA thread-block-cluster cooperative top-k kernel for
  QSA on SM110. Thor rejects its launch configuration. The image excludes
  SM110 from that gate so it uses vLLM's existing `persistent_topk` fallback,
  matching the upstream workaround already used for SM120.
- The final Thor run became healthy in 9 minutes 44 seconds, created a 4.02
  GiB KV cache for 152,212 tokens, and completed two OpenAI-compatible chat
  requests. The warm request generated 34 tokens in 5.5 seconds. The first
  request also JIT-compiled four Triton QSA kernels, as expected.
- Raised the residency agent's default health timeout from 10 to 15 minutes;
  the measured startup left only 16 seconds of margin under the old limit.

## Rejected paths

- Running the complete Blazux GB10 image on Thor: architecture-specific and
  unnecessary.
- `VLLM_PLE_CPU_OFFLOAD=1`: CPU and GPU use the same physical memory on Thor,
  so this cannot solve the capacity problem.
- PLE prewarming: disabled because reading the entire 95.4 GiB table into page
  cache would compete with model memory.

## Sources

- https://github.com/blazux/qwen3.8-Flash-DGX (commit
  `bd60fcb1b492ca920f74df7462f05da7b6d98f73` inspected on 2026-09-09)
- https://github.com/vllm-project/vllm (v0.29.0 commit
  `98dff2a81d747d1dba01a47f939f48c3526d4206`)
- https://github.com/vllm-project/vllm/issues/47266 (same cooperative top-k
  failure and fallback on Blackwell SM120)
