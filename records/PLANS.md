# Plans

## Qwen3-ASR multilingual adaptation

Follow [DEC-001](decisions/DEC-20261029-001-qwen-asr-autoresearch.md),
[DEC-002](decisions/DEC-20261029-002-four-slot-qwen-asr-autoresearch.md)
and [the four-slot contract](../skills/autoresearch/SKILL.md).

- Wait for the operator to release yeowoolmac before preparing the completed
  AIHub training downloads or starting model jobs. Keep AIHub data off Dumpling.
- Fill free slots with fresh TRAIN-grounded SFT and RL proposals. Preserve
  sealed evaluation inputs, current incumbent capture and source/data hashes;
  no parameter sweeps selected from validation errors.
- Score with optional translator glosses ([DEC-20261030-002](decisions/DEC-20261030-002-qwen-asr-optional-gloss-scoring.md))
  and Korean space-normalized WER ([DEC-20261031-001](decisions/DEC-20261031-001-qwen-asr-korean-space-normalized-wer.md)).
  Adopt recipes by two-seed real-use selection ([DEC-20261101-004](decisions/DEC-20261101-004-qwen-asr-real-use-selection.md)):
  clearly better on Korean real-use speech, clearly worse nowhere.
- The plateau is diagnosed ([RSH-20261031-001](research/RSH-20261031-001-qwen-asr-plateau-diagnosis.md)):
  admit only changes with a plausible effect of 6% or more, not recipe or
  capacity tweaks. Cap-eight GRPO `grpo-ae448ecc75d8e01f` earned a second
  seed under sWER, but its 1.3% lead is inside seed noise and its sibling
  recipe already failed from the same parent seed, so it is not run.
- FLEURS gains do not transfer to Korean real-use speech ([RSH-20261101-001](research/RSH-20261101-001-qwen-asr-korean-real-use-probe.md)).
  Phone-call SFT fixes calls but fails the 8 kHz gate
  ([RSH-20261101-003](research/RSH-20261101-003-qwen-asr-phone-call-sft.md),
  [DEC-20261101-002](decisions/DEC-20261101-002-qwen-asr-8khz-no-regression-gate.md)).
  The Korean-style retrain (`data-v011-calls-ko`, [DEC-20261101-003](decisions/DEC-20261101-003-qwen-asr-korean-number-style.md))
  keeps calls at -25% and ties FLEURS and 8 kHz when its two seeds are
  pooled. It is the incumbent. Evaluation from code `4cf83f9` (PR #44)
  redecodes a looped clip with the adapter off, so loops no longer swing gates.
  JA/ZH/EN real-use probes are still needed; Japanese is the language that
  8 kHz hurts.
- The Korean target is to beat or tie MAI-Transcribe 2 on the Seoul Corpus
  probe at 16 and 8 kHz ([DEC-20261102-001](decisions/DEC-20261102-001-qwen-asr-korean-mai-bar.md)).
  It is not met (+17.8% at 16 kHz). The gap is misheard unscripted speech
  ([RSH-20261102-001](research/RSH-20261102-001-qwen-asr-korean-mai-bar.md)),
  so the next data to try is spontaneous Korean from AIHub training splits,
  used locally. A MAI-safe calls set needs the operator's own recordings.
- Preserve rejected snapshots. Rebuild Common Voice with the merged literal-
  quote parser; admit YODAS only after audio-language/transcript alignment checks.
  AliMeeting stays parked: its meeting-speech conventions differ from v005,
  and every non-FLEURS mix so far lost.
- Admit at most four independently checked workers under the shared memory
  budget. Expand capacity only after simpler gains plateau. Keep existing
  models/data/results and serving paused.

## Steady-State Architecture

Single Bee binary (`Anbeeld/beellama.cpp`) serves DFlash (`--spec-type dflash`), PFlash (via standalone `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse. Legacy `backend:dflash` daemon retired. See git history for the completed migration (Phases 1-7).

## CUDA Flash-Attention Scratch (deployed baseline)

The May Atomic V2 work was retired when Grimoire migrated to the current TheTom engine fork. The current build ports scoped CUDA flash-attention K/V scratch allocation to the pinned engine and keeps CUDA graphs off. Qwen3.8-27B reasoning aliases use a 237,568-token context, a physical prompt batch of 128, the BF16 vision projector on CPU, the MTP head on GPU, and symmetric turbo4 K/V. This configuration maximizes the repeatedly tested long-context boundary on one RTX 3090 while retaining vision and MTP.

The incident audit and OpenCode soak are recorded in `records/research/RSH-20260828-001-qwen38-long-context-crash.md`. The GPU-projector boundary tests are recorded in `records/research/RSH-20260829-001-qwen38-gpu-mmproj-190k.md`. The CPU-projector maximum boundary is recorded in `records/research/RSH-20260829-002-qwen38-cpu-mmproj-237k.md`.

A graph-safe reusable scratch reservation remains optional future work. Revisit it only if CUDA-graph throughput becomes more valuable than the currently verified long-agent-context headroom.

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

The first safe reduction keeps vision: retain FA2 BF16 head-size 96 and 256,
plus BF16 sparse head-size 128, and remove FA3 because it cannot run on SM110.
Unsupported FA2 dtypes and head sizes fail explicitly. Do not remove other
quant/kernel families until import, symbol, startup, text, vision, 262K-context,
and rollback tests prove that neither target model reaches them. The SM110 build
also skips external FA3, FlashMLA, FlashKDA, DeepGEMM, QuTLASS, FMHA-SM100, and
Triton-MLIR FA4 targets because their own configure checks reject SM110; keep
the core stable-ABI CUDA/MoE extensions, FA2, Triton kernels, CUTLASS, and the
separately installed FlashInfer and CuTeDSL paths used by the accepted models.


## DeepSeek-V4.1-Flash-1017: after the recipe upgrade and the keys switch

Shipped 2026-11-13 (STATUS). What is left:

1. Measure the recipe's opt-ins one at a time against the current baseline:
   `TF_FILL_CHUNKS` + `TF_FILL_CUT_IN`, `TF_DS_T25_DRAFT_CAL`, `STEP_AHEAD`.
   `DROP_REASONING_HISTORY` stays off unless an agent loop shows the drift.
2. A short request sent during a 100K-token prompt (the cut-in patches).
3. Drop the rollback once the new setup has served a week: checkout
   `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks` (`27741e1`), image
   `tensorfold-dsv41:exl3-tp2-928adbd0e57d`, the abliterated rank caches, and
   `~/dsv41-tensorfold.service.bak-27741e1`.

## Backlog (Future Interest)


| Priority | Item | Why Deferred | Prerequisite |
|----------|------|-------------|--------------|
| 1 | **DDTree tree-mode verify** — enable `--spec-branch-budget > 0` with larger `GGML_DFLASH_MAX_VERIFY_TOKENS` | Benchmarked (2026-05-18): tree-mode (budget=22) = 62.4 tok/s vs flat = 61.6 tok/s (+1%). The draft is already well-matched to the target for greedy decode (85% acceptance), so tree-mode adds negligible benefit. Deferred unless non-greedy sampling or a lower-quality draft changes the trade-off. | The 25-token cap fix is already deployed. Env var auto-derived by model_manager.py. |
| 2 | **GPU tape recording** — tree-mode DDTree verify using `dflash_tape_*` | Only needed if single-spec throughput becomes a bottleneck | Multi-spec batch decode |

## Not Pursuing

- Multi-spec batched decode — single-spec sufficient for current workload
- Monitoring for 413 path — (user decision)
- VMM park/unpark — measured (RSH-20260518-001): adds 2.5 GB VRAM overhead with no TTFT benefit for single-model. Unnecessary unless multi-model GPU sharing is needed later.
