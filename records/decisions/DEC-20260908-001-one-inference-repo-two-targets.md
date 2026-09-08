# DEC-20260908-001: One inference repo targeting grimoire and mangchi

Opened: 2026-09-08 18-05-00 KST
Recorded by agent: codex

## Metadata

- Status: accepted
- Deciders: operator, codex
- Area: repo layout, `Dockerfile`, `docker-compose.yml`, `registry.py`/`model_manager.py` GPU logic, `etc/models.json`
- Related: DEC-20260517-004 (TheTom turboquant as canonical base), DEC-20260622-001 (GPU co-location allocator)
- Greenlit by operator 2026-09-08: rename + pin bump + ARM64 support

## Decision

Rename `LPFchan/grimoire` to `LPFchan/inference` and turn it into a
single-source-of-truth inference repo that builds for **two target machines** —
`grimoire` (the existing multi-GPU x86 box) and `mangchi` (an NVIDIA Jetson
AGX Thor: ARM64, single Blackwell GPU `sm_110`, 128 GB unified memory) — via a
per-target build/config switch rather than a fork.

The machine target is selected by one build/runtime knob (working name
`INFERENCE_TARGET`, defaulting to `grimoire` so existing behavior is
unchanged). The target drives:

- **CUDA base image and arch.** `grimoire` keeps `nvidia/cuda:12.8.1` x86 with
  `GRIMOIRE_CMAKE_CUDA_ARCHITECTURES=86;89`. `mangchi` uses an ARM64 CUDA 13
  base with arch `110` (Thor is compute capability 11.0).
- **GPU model.** `grimoire` keeps the multi-GPU pool (pin/evict/clone, the
  DEC-20260622-001 allocator). `mangchi` is single-GPU: the allocator is still
  exercised but always resolves to GPU 0, so the pool logic is retained, not
  deleted.
- **Model registry.** Per-target `models.json` (e.g. `etc/models.grimoire.json`,
  `etc/models.mangchi.json`) selected by the same knob, since the two machines
  run disjoint model sets.
- **Device wiring in compose.** `mangchi` uses `runtime: nvidia` (the Jetson
  container toolkit does not support `--gpus all`); `grimoire` keeps
  `count: all`.

Alongside the rename: bump `GRIMOIRE_LLAMA_CPP_PINNED_SHA` from
`2f2f32f5` (2026-08-08) to the current TheTom turboquant HEAD `407f3237bfb3`
(2026-09-06) to pick up `qwen4exp` support needed for Qwen3.8-Flash-Next.

## Context

The operator acquired a Jetson AGX Thor (mangchi) and wants to run large MoE
models on it (Qwen3.8-Flash-Next at UD-IQ4_XS, ~93.7 GB). The existing grimoire
repo is a multi-GPU inference gateway, but 4 of its 5 features are needed on the
Thor too: model load/eviction, KV cache persistence, speculative decoding, and
the OpenAI-compatible API. Only the multi-GPU pool is mangchi-irrelevant.

Investigation on 2026-09-08 established:

- The TheTom `feature/turboquant-kv-cache` branch is a live fork tracking
  upstream closely (HEAD 2026-09-06, 2 days behind upstream master) and already
  carries a full `qwen4exp` model implementation. No forward-port of turboquant
  onto upstream is required.
- Grimoire's *pin* (`2f2f32f5`, 2026-08-08) is a month stale and predates some
  `qwen4exp` fixes; bumping the pin to the fork's own HEAD is sufficient.
- Docker on Jetson Thor is an officially supported path (JetPack 39.x,
  `nvidia-container` package, CUDA 13), not the friction previously assumed.
- Thor hardware: aarch64, Blackwell `sm_110`, CUDA 13, 128 GB unified memory.

The deduplication motive: maintaining a fork (grimoire → mangchi) would split
bug fixes and model-registry improvements across two repos. A single repo with a
target switch keeps one source of truth and lets both machines share KV
persistence, speculative decoding, and gateway improvements.

## Options Considered

### Fork grimoire into a separate mangchi repo

- Upside: mangchi gets a lean single-GPU codebase with no grimoire-specific weight.
- Downside: every future gateway fix (KV persistence, speculative decoding, API)
  must be ported twice; the two drift apart.
- Downside: loses single source of truth, the operator's stated goal.

### Run grimoire as-is on the Thor with gpu-ids: [0]

- Upside: zero code change.
- Downside: pinned llama.cpp SHA cannot load qwen4exp; x86 CUDA base images are
  wrong for ARM64; multi-GPU orchestration is dead weight that complicates the
  single-GPU path.

### Single repo with a target switch (chosen)

- Upside: one source of truth; both machines share improvements; the multi-GPU
  pool is retained for grimoire and simply resolves to GPU 0 on mangchi.
- Upside: the target split lives where the seams already are (Dockerfile CUDA
  args, compose device wiring, per-target models.json).
- Downside: the repo now carries two build targets; CI/build must cover both.

## Rationale

The seams for a per-target split already exist and are small: the Dockerfile
already parameterizes CUDA base, arch, llama.cpp repo/ref/SHA, and patches via
ARGs; compose already centralizes device and resource wiring; the model registry
is already a single JSON file. Making the target a first-class knob costs far
less than forking, and the operator explicitly values deduplication over a lean
per-machine fork. Keeping the GPU allocator (rather than stripping it) avoids a
large, risky refactor of `model_manager.py` (233 GPU/evict/pin/clone
references); on a single-GPU target it degrades gracefully to GPU 0.

## Consequences

- The GitHub repo `LPFchan/grimoire` must be renamed to `LPFchan/inference`
  by the operator (GitHub repo rename is an owner action); git remotes update to
  `git@github.com:LPFchan/inference.git`.
- One build knob (`INFERENCE_TARGET`) selects base image, CUDA arch, GPU wiring,
  and models.json; default stays `grimoire` so the existing box is unaffected.
- `GRIMOIRE_LLAMA_CPP_PINNED_SHA` moves to `407f3237bfb3`; the atomic llama.cpp
  patch set must be validated against that SHA (patches may need re-basing).
- mangchi compose uses `runtime: nvidia`; any `--gpus all` /
  `deploy.resources...devices` usage is gated to the grimoire target.
- Per-target `models.json` files appear under `etc/`; `mangchi` seeds
  Qwen3.8-Flash-Next UD-IQ4_XS at ~256k context with turbo4 KV cache.
- The internal Python package may remain `grimoire` for now to avoid a sweeping
  rename; the *repo* name and image naming move to `inference`. A follow-up DEC
  can decide whether to rename the package.
- Both build targets should be exercised in CI to prevent one target silently
  breaking when shared code changes.

