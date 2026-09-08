# DEC-20260909-001: Move mangchi from llama.cpp/GGUF to vLLM/NVFP4

Opened: 2026-09-09 03-10-00 KST
Recorded by agent: codex

## Metadata

- Status: accepted
- Deciders: operator, codex
- Area: mangchi inference runtime, model storage, `etc/models.mangchi.json`
- Related: DEC-20260908-001 (one inference repo, two targets),
  RSH-20260908-001 (Thor VMM load hang)
- Greenlit by operator 2026-09-09: "pull both nvfp4 and purge all gguf";
  "transition away from llama.cpp to vLLM on mangchi"

## Decision

Mangchi stops serving models with llama.cpp/GGUF and moves to **vLLM running
NVFP4-quantized safetensors**. The two GGUF model sets previously staged for
mangchi (Qwen3.8-Flash-Next UD-IQ4_XS and the Qwen3.8-27B family, ~124 GB) are
purged; they are replaced by two gated NVFP4 safetensors repos from
`orcarouter`:

- `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` (~24.7 GB, 5 shards + extra)
- `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` (~183.5 GB, 17 shards + MTP)

The runtime is the purpose-built NVIDIA Jetson Thor image
`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` (SM110 / JetPack 7), not a
from-source build. Flash-Next's giant n-gram PLE table is kept off the GPU via
vLLM's PLE host offload (`VLLM_PLE_CPU_OFFLOAD`).

## Context

Mangchi is a Jetson AGX Thor (aarch64, single Blackwell `sm_110`, CUDA 13,
128 GB unified memory). Blackwell has native NVFP4 hardware support, so NVFP4
is the quant this chip is designed to run. The original plan was llama.cpp
GGUF; that hit a wall on three fronts:

- **NVFP4 GGUF for these models is too large or unavailable.** The target
  uncensored variants ship as NVFP4 *safetensors*, which llama.cpp cannot load.
- **llama.cpp on Thor underperforms and is fragile here.** Measured on
  Qwen3.8-Flash-Next (UD-IQ4_XS): ~293 tok/s prefill, ~12 tok/s decode, and
  the numbers were flat across both a 6B-active hybrid and a dense 27B,
  pointing at a hardware ceiling llama.cpp could not use better. Loading also
  required `GGML_CUDA_NO_VMM=ON` + `--fit off` to avoid an
  `NV_ERR_NO_MEMORY` hang (RSH-20260908-001), and CUDA graphs had to be
  enabled per-target while dropping a conflicting FA patch (0011).
- **vLLM has first-class NVFP4-on-Blackwell support and a PLE offload path**
  purpose-built for this exact model.

The decisive technical finding (operator's insight, confirmed by reading the
safetensors shard headers): Flash-Next-NVFP4 is 183.5 GB on disk, but
**~102 GB of that is a single shard that is entirely the n-gram PLE embedding
table** (`layers.1.ple.ple_embedding.ngram_embedding.*`, BF16 `[2500012,160]`
shards). The MoE experts total ~68 GB; everything else is ~8 GB. The PLE table
is a lookup table, not on the GEMM hot path, so it can stream from host
RAM/SSD. With PLE offloaded, the resident footprint drops to roughly
**~76 GB + KV cache**, which fits the Thor's 128 GB with headroom for long
context. Without offload the model does not fit at all.

## Options Considered

### Stay on llama.cpp, convert NVFP4 safetensors to GGUF

- Upside: keeps the existing grimoire/mangchi llama.cpp stack and gateway.
- Downside: NVFP4 safetensors -> GGUF NVFP4 conversion is not a supported,
  validated path; llama.cpp's measured Thor performance is poor; and the GGUF
  builds for these uncensored variants do not exist. High risk, low reward.

### vLLM from the NVIDIA Thor image (chosen)

- Upside: native NVFP4 on Blackwell; official PLE-CPU-offload support; a
  maintained SM110 image; no fragile VMM/patch workarounds.
- Downside: abandons the shared llama.cpp gateway on mangchi; mangchi's
  serving path diverges from grimoire's.

## Rationale

The hardware favors NVFP4, the models only exist as NVFP4 safetensors, and
vLLM is the only runtime that loads them natively with a working offload story
for the PLE table. The llama.cpp path was measurably slow and required
accumulating hacks (NO_VMM, fit-off, per-target graphs, dropped patches) that
added maintenance burden for no performance gain.

## Consequences

- GGUF model files are purged from mangchi's SSD (~124 GB freed).
- `etc/models.mangchi.json` (GGUF/turbo4/llama.cpp-specific) is obsolete for
  serving and must be replaced with a vLLM-oriented model config.
- The mangchi Dockerfile / llama.cpp build target is no longer the serving
  path. It can remain for reference or be removed in a follow-up; grimoire's
  llama.cpp target is unaffected.
- KV cache persistence and speculative decoding, previously via the llama.cpp
  gateway, need re-implementation under vLLM on mangchi (MTP draft is present
  in the Flash-Next repo; vLLM supports MTP/EAGLE spec decode).
- A follow-up DEC should decide whether the mangchi gateway/entrypoint is
  reworked to front vLLM or replaced by vLLM's own OpenAI-compatible server.
