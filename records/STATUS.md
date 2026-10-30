# Current Status

**Snapshot:** 2026-10-29
**Posture:** The Grimoire gateway serves local llama.cpp models and admits remote vLLM models through Mangchi's residency agent. The local engine remains TheTom's turboquant fork with turbo4 KV. Mangchi model artifacts are managed separately from the gateway.
**Focus:** Serve Qwen3.8 Flash-Next and the grugthink 27B as Mangchi's registered remote models (one resident at a time) and continue Flash-Next's its decode, prefix-cache, and CUDA-graph optimization program.

## Autoresearch — Qwen3-ASR

- **Training incumbent:** gloss-free TRAIN SFT
  `sft-gloss-free-confirm/step-000944`, adapter `f025ecba…c568`, adopted
  2026-10-30 22:13 KST by two-seed recipe comparison
  ([DEC-20261030-003](decisions/DEC-20261030-003-qwen-asr-two-seed-recipe-adoption.md)).
  Its pair `sft-gloss-free`/`sft-gloss-free-confirm` pools to 0.8428 vs the old
  pair's 0.9073, CI95 [-0.1221,-0.0172], and both seeds pass KSS. The
  lower-score seed is the checkpoint. No trained model is deployed; Nemotron
  stays unloaded. Original MLX 8-bit remains the fixed normalization anchor.
- [Confirmed findings](research/RSH-20261030-001-confirmed-qwen-asr-one-pass-sft.md).
  Selection authority: [DEC-001](decisions/DEC-20261029-001-qwen-asr-autoresearch.md),
  [DEC-002](decisions/DEC-20261029-002-four-slot-qwen-asr-autoresearch.md),
  [DEC-20261030-003](decisions/DEC-20261030-003-qwen-asr-two-seed-recipe-adoption.md)
  and [four-slot contract](../skills/autoresearch/SKILL.md).
  Frozen primary set: 1,295 validation clips (EN 394,KO 226,JA 266,ZH 409),
  manifest `c75a4b05069eba7d11a6b50efd7bb11d43800fc66e49b761cd08fb86354acf01`.
  Original 2,624-clip test, nine published predictions and artifacts stay sealed.
- Full validation primary errors, percent (WER EN/KO; CER JA/ZH), with
  translator glosses optional ([DEC-20261030-002](decisions/DEC-20261030-002-qwen-asr-optional-gloss-scoring.md)).
  Scores use the original anchor; passing its screen alone cannot replace the
  incumbent. Trial bullets below keep the raw-reference scores recorded at
  closure.

  | Trial | EN | KO | JA | ZH | CJK score |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Original 8-bit anchor | 4.25 | 10.25 | 4.63 | 3.36 | 1.0000 |
  | Gloss-free TRAIN SFT, seed 21775174, incumbent | 4.06 | 9.59 | 4.47 | 2.10 | 0.8426 |
  | Gloss-free TRAIN SFT, seed 21774173 | 3.90 | 9.89 | 4.24 | 2.18 | 0.8431 |
  | One-pass SFT, seed 21774173, previous incumbent | 3.99 | 9.92 | 4.30 | 2.54 | 0.8841 |
  | One-pass SFT, seed 21775174 | 4.06 | 9.89 | 4.54 | 2.84 | 0.9306 |
  | Late-decay SFT, closed | 4.01 | 9.80 | 4.54 | 2.74 | 0.9173 |
  | Language-mean SFT, closed | 3.84 | 9.92 | 4.27 | 2.57 | 0.8853 |
  | Looped decoder 32 layers `12:16:2`, closed | 4.24 | 10.43 | 4.68 | 2.36 | 0.9102 |
  | Looped decoder 36 layers `10:18:2`, closed | 4.10 | 10.49 | 4.44 | 2.45 | 0.9043 |
  | Looped decoder 42 layers `7:21:2`, closed | 4.17 | 11.00 | 4.79 | 2.91 | 0.9914 |
  | Looped decoder 56 layers `7:21:3`, closed | 4.43 | 11.44 | 5.38 | 3.31 | 1.0872 |
  | 20-second TRAIN extension, closed | 3.96 | 9.77 | 4.36 | 2.45 | 0.8741 |
  | Reference-member GRPO, closed | 4.01 | 9.95 | 4.19 | 2.37 | 0.8605 |
  | SFT-parent cap-eight GRPO, closed | 3.95 | 9.98 | 4.23 | 2.22 | 0.8489 |
  | SFT-parent single-draw GRPO, first seed | 3.99 | 9.83 | 4.26 | 2.21 | 0.8456 |
  | Uniform two-seed SFT average, closed | 3.89 | 10.01 | 4.50 | 2.56 | 0.9032 |
  | Base-init GRPO `bc8f2d8f6f19a369`, first seed | 4.15 | 9.77 | 4.65 | 2.37 | 0.8873 |
  | Base-init GRPO confirmation, closed | 4.25 | 9.51 | 4.64 | 2.30 | 0.8710 |
  | Fixed-scale GRPO `c3b0b430fab2ba0c`, closed | 4.21 | 10.10 | 4.53 | 2.38 | 0.8909 |

- SFT score-delta CI95 vs original anchor: first[-0.1740,-0.0511],
  confirmation[-0.1370,-0.0348]. Every gate passes; all outputs end and none
  are empty. Both 944 × 4 TRAIN sequences match preregistered hashes, each with
  3,776 unique language/source-sentence draws. Sampling, recording choice,
  order, coverage and update count change together versus the rejected
  5,000-update SFT. No isolated repetition/capacity claim.
- KSS Korean guard: 256 digit and 256 other clips, reported separately;
  one voice, oversampled digits, no pooled accuracy. Each row is WER/CER:

  | Model | Digit | Other |
  | --- | ---: | ---: |
  | Original anchor | 32.71% / 11.88% | 8.08% / 1.24% |
  | Selected SFT | 11.05% / 3.58% | 7.20% / 1.35% |
  | SFT confirmation | 12.84% / 4.16% | 7.04% / 1.24% |
  | Base-init GRPO first seed | 14.42% / 4.38% | 7.92% / 1.41% |

  All listed candidate cohorts pass the registered +0.5 pp WER/CER veto,
  with no additional empty/truncated outputs. Other-cohort CER rises 0.1178 pp
  for selected SFT and 0.1766 pp for GRPO. Numerical source stays `code/f119906`;
  each contract matches retained `parallel/kss-v001-base` except adapter SHA.
  Its resume-status reporting bug is frozen: monitor actual admission,
  prediction count and full summary. No baseline predictions are repeated.
- **Closed bounded-redraw GRPO:** `grpo-ae448ecc75d8e01f` finishes 1,000
  updates and all 1,295 validation clips, scoring 0.9094 vs incumbent 0.8839.
  Chinese exceeds its limit; gain, CJK and confidence gates fail. Paired
  score-delta CI95 is [-0.0233,+0.0699], with no empty/truncated outputs.
  All 2,508 rollout groups match the frozen draw rule; 966 updates carry
  reward differences, with 34 exhausting the cap. Checkpoint/source/data
  hashes verify. No confirmation or KSS, and no cap/LR/beta/step/temperature/
  seed variants. Evidence: `parallel/grpo-ae448ecc75d8e01f`.
- **Closed late-decay SFT:** `sft-5866097875feb182` completes 944 updates and
  all 1,295 validation clips, scoring 0.9074 vs incumbent 0.8839. Japanese and
  Chinese exceed their regression limits; gain, CJK and confidence gates fail.
  Paired score-delta CI95 is [+0.0011,+0.0490], with no empty/truncated outputs.
  Exact prefix/native acceptance and all 3,776 unique preregistered TRAIN draws
  verify. No confirmation or KSS run, and no start/end/shape/peak-LR/budget/seed
  variants from this result. Evidence: `parallel/sft-5866097875feb182`.
- **Closed 20-second TRAIN extension:** `sft-f42871766adb064c` completes 944
  updates and all 1,295 validation clips, score 0.8956 vs incumbent 0.8839.
  English/Korean improve, but Chinese exceeds its limit; gain, CJK and
  confidence gates fail. Paired delta CI95 is [-0.0335,+0.0517], with no
  empty/truncated outputs. The fixed 944-update recipe closes without
  confirmation, KSS or cap/budget/LR/seed variants from this result.
  v008's 10,154 clips/30.7208 h and all 3,776 unique preregistered draws verify;
  memory preflight passed at 4.72094 GB. Preserve the data and frozen evidence
  under `proposals/next-data-after-confirmed-20261030` and the trial directory.
- **Closed reference-member GRPO:** `grpo-c34dabc489294800` completes
  1,000 updates and all 1,295 validation clips, score 0.8988 vs incumbent
  0.8839. Chinese exceeds its limit; gain, CJK and confidence gates fail.
  Paired score-delta CI95 is [-0.0280,+0.0541], with no empty/truncated outputs.
  All draws, five-member advantages, TRAIN rewards and final checkpoint/source/
  data hashes verify. No confirmation, KSS or reference-weight/shaping/group/
  LR/beta/temperature/step/seed variants. Evidence: `parallel/grpo-c34dabc489294800`.
- **Closed MLP-LoRA fit trial:** frozen `code/41b7216` passes 169 focused
  tests and exact native default parity (224 adapter/450 optimizer tensors).
  Its ten-update, 40-longest-TRAIN preflight peaks at 4.99139 GB physical,
  exceeding the preregistered 4.75 GB limit; all pressure samples are normal.
  No full-data candidate, confirmation, KSS or accuracy result is produced.
  PR33 closes unmerged; no budget/rank/target-subset/LR/length retuning.
  Evidence: `proposals/next-sft-after-two-losses-20261030/native-verdict.json`.
- **Four-worker throughput:** all 1/2/4-worker phases completed (60 × 4
  microsteps each). Combined training rates: 2.855 / 3.130 / 3.359 clips/s;
  four workers add 17.65%. Pressure stayed normal, with no new swap-outs.
  This is a short workload-specific measurement. [Evidence and limits](research/RSH-20261030-002-qwen-asr-four-worker-throughput.md).
- **RL audit:** token-length reweighting has weak support in the closed TRAIN
  rollouts; no token-sum trial is admitted. Modal-output and re-tokenized-length
  proxies leave actual sampling headroom unmeasured. Evidence and the reported
  short CPU-memory overrun: `proposals/next-rl-after-redraw-20261030`.
- **Optional-gloss scoring** ([DEC-20261030-002](decisions/DEC-20261030-002-qwen-asr-optional-gloss-scoring.md)):
  `selection.py --glosses calibration/glosses-v1.json`, frozen `code/a4559db`
  (PR38), makes 87 bracketed FLEURS translator glosses (KO 19, JA 5, ZH 63)
  optional. Anchor ZH falls from 6.61% to 3.36%. The rescore of every
  1,295-clip evaluation (`calibration/glosses-v1-rescore.json`) passes no
  trial against the incumbent; the confirmation seed's CI95 against the anchor,
  [-0.149,+0.019], now crosses zero. Seed spread is 0.047. Evidence verified on
  Mac/local/Mangchi.
- **Adopted gloss-free TRAIN SFT** (`proposals/gloss-free-train-20261030`,
  `two-seed-evidence.json`, `recipe-selection.json`): the incumbent recipe on v005 with translator
  glosses removed from TRAIN targets (ZH 286, KO 91, JA 25 rows; manifest
  `11c97042…17a6`). Seeds 21774173/21775174 score 0.8431/0.8426 vs 0.8841,
  improving every CJK language: EN/KO/JA/ZH 3.90/9.89/4.24/2.18% and
  4.06/9.59/4.47/2.10%. Seed spread falls from 0.047 to 0.0005. Single-seed
  gates fail: the first seed only on CI95 [-0.1104,+0.0078], the second also on JA.
  Matched seeds against the old data: the second seed passes every gate against
  `sft-a11b015fbeee0082`, CI95 [-0.1508,-0.0308]. Both seeds pass the KSS veto
  (base rerun bound to the new manifest; hypotheses identical to `kss-v001-base`).
  Pooled previews, not gates: both seeds against the incumbent checkpoint
  give CI95 [-0.1053,+0.0061]; both seeds against both incumbent seeds give
  0.8428 vs 0.9073, CI95 [-0.1221,-0.0173], within CJK limits. The official
  `code/523783e` comparison gives CI95 [-0.1221,-0.0172] and passes every gate.
  Adoption is in `parallel/sft-gloss-free-confirm/adoption.json`; the prior
  `parallel-trials.json` is kept as `parallel-trials.before-gloss-free-adoption.json`.
  Evidence verified on Mac/local/Mangchi.
- **Closed single-draw GRPO second seed:** `grpo-76cf17e6620b34fb-confirm`
  (parent `sft-a11b015fbeee0082`, seed 21326232) scores 0.9101 vs 0.8841; gain,
  CJK and confidence gates fail, CI95 [-0.0333,+0.0777]. The first seed's
  0.8456 does not replicate, and the two-seed mean is within 1% of the
  incumbent, so the recipe closes under any pooling rule.
- **Closed decoder MLP LoRA:** `sft-gloss-free-mlp` reruns the gloss-free
  recipe with `--mlp` (LoRA also on MLP gate/up/down, 17.4M trainable weights;
  PR40, frozen `code/0275adf`, train.py identical to merged `55bcab9`) under the
  operator-approved 5.5 GB budget. Preflight on the 40 longest clips peaks at
  4.99 GB; the full run peaks at 5.14 GB, normal pressure. Seed 21774173 scores
  0.8517 vs same-seed 0.8431: EN/KO/JA improve (3.78/9.74/4.18%), ZH worsens
  (2.36%) across scattered number-format and name clips. It fails the
  DEC-20261030-003 screen (pooled 0.8428), so no second seed or KSS. No
  rank/target/LR/budget variants. Evidence verified on Mac/local/Mangchi.
- **Closed GRPO on the gloss-free incumbent:** `grpo-gloss-free` (single-draw
  recipe, parent `sft-gloss-free`, seed 21325231, `code/523783e`, gloss-free
  data) scores 0.8578 vs its parent's 0.8431; KO WER rises to 10.19%. It fails
  the DEC-20261030-003 screen (pooled 0.8428), so no second seed or KSS runs.
  Earlier GRPO gains came mostly from avoiding invented glosses, which the
  gloss-free targets already remove. No LR/beta/step/temperature/seed variants.
  Evidence verified on Mac/local/Mangchi.
- **Closed learning-rate 1e-4 SFT:** `sft-lr-1e-4` reruns the incumbent recipe
  at 1e-4 (`code/a4559db`). Score 0.9783 vs 0.8841; EN improves (3.70%) but ZH
  reaches 3.38%. Gain, CJK and confidence gates fail; delta CI95
  [+0.0055,+0.1838]. Most of the ZH loss is invented bracketed English names,
  which the v005 TRAIN targets teach. Clips with an invented gloss where the
  reference has none: incumbent 7, this trial 7, single-draw GRPO and anchor
  0. No other learning rates. On the 1,245 clips where neither model writes
  bracketed Latin, it still loses on every CJK language (0.8589 vs 0.8073,
  CI95 [+0.0145,+0.0906]), so it is not rerun on gloss-free data. Evidence
  verified on Mac/local/Mangchi.
- **TRAIN sampling diagnostic (complete):** `train-probe-ce33-20261030`
  finishes all 7,200 incumbent decodes (200 TRAIN sentences per language,
  greedy plus eight samples at 0.7), with no empty/truncated outputs. Greedy
  beats the mean sample in every language. Best-of-8 oracle headroom over
  greedy is small: KO 0.028, ZH 0.021, JA 0.013, EN 0.013 reward. Greedy
  errors with a perfect sample: KO 31/105, JA 30/87, ZH 15/77, EN 22/77.
  This is an optimistic TRAIN-only bound, so further RL has little to
  extract; no RL trial is admitted without new evidence. Frozen `code/385880d`;
  evidence verified on Mac/local/Mangchi under the trial directory.
- **Closed looped-decoder arm** ([DEC-20261030-001](decisions/DEC-20261030-001-qwen-asr-looped-decoder-arm.md),
  [findings](research/RSH-20261030-003-qwen-asr-looped-decoder.md)): all four
  levels rerun the incumbent recipe with a looped decoder block (shared frozen
  weights, per-pass LoRA). Scores worsen with depth: 32/36/42/56 layers give
  0.9308/0.9374/0.9605/1.0173 vs 0.8839, each with its paired CI95 above
  zero and no empty/truncated outputs. Frozen `code/4b2a716` (PR36) and
  launcher `code/0116029` (PR37, 5.5 GB for the two deepest levels). No
  level/rank/LR/budget/seed/gate variants. Evidence verified on
  Mac/local/Mangchi under `parallel/sft-loop-*` and `proposals/loop-arm-20261030`.
- **Closed language-mean SFT:** `sft-8f41bf5402204a8f` completes 944 updates
  and all 1,295 validation clips, score 0.8875 vs incumbent 0.8839. Chinese
  exceeds its limit; gain, CJK and confidence gates fail. Paired delta CI95
  [-0.0375,+0.0307], no empty/truncated outputs. The candidate side of the
  TRAIN NLL diagnostic is not run; it never gated adoption. No confirmation,
  KSS or weighting/LR/budget/seed variants. Evidence verified on
  Mac/local/Mangchi under the trial directory.
- **Data intake:** sampled WorldSpeech labels remain unadmitted. AliMeeting
  near-field TRAIN is acquired and audited: 186,605 labels contain no digits;
  conservative segmentation yields 31,029 candidates/25.07 h. Energy checks
  support timing alignment, with word-level accuracy unverified. Raw labels
  have number/filler conventions unlike v005; no data trial is admitted.
  The number-format audit estimates only 4.55% of eligible rows change;
  no format-only trial is proposed. Acoustic/source-diversity benefits remain
  unmeasured. Evidence: `proposals/alimeeting-format-hypothesis-20261030`.
  The 24.53 GB archive stays on Mac; 705 metadata files/TextGrids are verified
  on Mac/local/Mangchi. Evidence: `proposals/alimeeting-label-audit-20261030`.
- **Intake protocol deviation:** the data worker reports displaying one unique
  sealed test reference through two exclusion-related files, without using it.
  The parent did not reopen it. No new test prediction/scoring occurred;
  v008 retains zero held-out ID/text/audio overlap. The deviation and limits
  are recorded in that packet's `parent-protocol-deviation.json`.
- **Closed single-draw SFT-parent GRPO:** `grpo-76cf17e6620b34fb` completes
  1,000 updates and all 1,295 validation clips, score 0.8965 vs parent/current
  incumbent 0.8839. KO/JA errors fall, while ZH rises to 5.2957%, exceeding
  its limit. Gain, CJK and confidence gates fail; delta CI95
  [-0.0327,+0.0555], no empty/truncated outputs. No confirmation/KSS run
  or LR/beta/budget variants. The independently preregistered cap-eight run
  launched before this result and continues. Full frozen evidence is retained
  under `parallel/grpo-76cf17e6620b34fb`.
- **Closed adapter average:** `soup-6bc17d90adcbb97d` completed all 1,295
  validation clips with no empty/truncated outputs. Uniform rank-32
  concatenation of the two confirmed adapters scores 0.9110 against the
  incumbent's 0.8839. Japanese and Chinese exceed their regression limits;
  score-delta CI95 is [+0.0040,+0.0556]. Gain, CJK and confidence gates fail.
  No confirmation ingredients or rank-32 KSS baseline run. The registered
  weights/seed-count/checkpoint/rank variants are closed. Full validation,
  selection, diagnostic and verdict are retained under the trial directory
  on all three hosts; the original two checkpoints remain unchanged.
- Root: `~/models/stt/qwen3-asr-hillclimb-20261029`.
  `parallel-trials.json.incumbent` is current authority; adoption evidence is
  `parallel/sft-ce33a3d7a476e297/adoption.json`. The legacy driver's
  `loop-state.json` is historical and unchanged. Runtime:
  `~/inference-qwen-training/.venv/bin/python`; pins/source/data/results and
  final adapters/optimizers are retained on Mac, local mirror and Mangchi.
- Operator authorized at most four research/model workers, 5e9 physical bytes
  each/20e9 total reservations. Model launches use frozen supervisor
  `code/1ae82c5` in existing `main` tmux and `--memory-gb 4.5` or less.
  Model capacity follows measured reservations; fresh proposals and native
  acceptance checks share admission. The ten-minute heartbeat is `continue`.
- Closed earlier areas: v001 SFT LR 3e-5/1e-5, starter-SFT-parent GRPO,
  base GRPO LR1e-6/1e-5, matched 20k-draw batch 32, numeral-filtered v1,
  expanded filtered/raw v006, 5,000-update written-form SFT and single
  alpha0.5 final-step derivation. Eleven rejected trials retain all full
  outputs. Relevant LOG commits: `200795a`, `649b608`, `31c66bc`, `07110bf`,
  `4d7b9f7`, `a396b8e`; complete per-trial metrics remain under the root.
- Fixed-advantage-scale GRPO `c3b0b430fab2ba0c` completed all 1,000 updates
  and 1,295 validation clips. It passes the original-anchor screen: score
  0.9414, delta CI95 [-0.0922,-0.0245], no empty/truncated outputs. Its
  difference from standard GRPO is +0.0023, CI95 [-0.0190,0.0255], so the
  proposed improvement over that control is unsupported. Recheck against
  the confirmed SFT incumbent fails gain, CJK and confidence gates:
  delta CI95 [-0.0025,0.1106]. It is closed without confirmation or adoption.
  Frozen outputs, controlled comparison and `incumbent-recheck.json` are
  preserved under `parallel/grpo-c3b0b430fab2ba0c` on all three hosts.
- Base-init v005 GRPO first/confirmation scores are 0.9392/0.9270; both
  pass the original-anchor screen, and both fail the current SFT incumbent
  recheck. Confirmation errors: EN 4.25%, KO 10.14%, JA 4.63%, ZH 5.58%;
  no empty/truncated outputs. Its current-incumbent delta CI95 is
  [-0.0127,+0.0948]; gain, CJK and confidence gates fail. The control pair is
  closed, with its first-seed historical KSS retained and no extra KSS run.
  Full confirmation evidence: `parallel/grpo-bc8f2d8f6f19a369-confirm`.
- Intake: v001 9,852 clips/16.51h; raw v002 67,794/108.57h is held for a
  merged-quote Japanese target. Merged literal-quote parser PR23(`c8dec2e`)
  fixes rebuilding; v006 removes the one bad row but does not recover lost
  records. Numeral filtering is imperfect and development-informed. YODAS is
  quarantined for translated captions. v005 written-form FLEURS TRAIN has
  8,966 clips/25.14h and zero held-out sentence-ID/text/audio overlap;
  speaker separation/pretraining exposure remain unestablished.
- Workers stay serial within each process. Native microbatch 2 failed acceptance
  (~9% gradient difference, 1.03× speedup, 5.006 GB peak). Eager-gradient-sum
  PR20(`ba92ec1`) passed exact first-update tensor parity with lower memory;
  existing trials retain their frozen code. Four-slot admission PR21 is
  merged as `6497a26`; numerical external evaluator stays `f119906` although
  main includes its reporting fix. No benchmark CI is configured.
- Preserve all models/audio/results, gateway/tunnel and disabled oMLX settings.
  No Grimoire operations, automatic serving restore or global fleet edits.

## Migration Summary

Bee (`Anbeeld/beellama.cpp`) is the canonical engine. Single binary serves DFlash (`--spec-type dflash`), PFlash (via `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse (1.93x verified). Legacy `backend:dflash` daemon, Lucebox code, and `/opt/dflash` fully retired. For completed phase details see git history (Phases 1-7, commits `f40874c` through `ef4fcb1`).

## Atomic CUDA FA V2 Track

Patch chain in `patches/atomic-llama-cpp/`, applied in order by `Dockerfile`:

| Default-served | File | Concern |
| --- | --- | --- |
| ✓ | `0002-cuda-fa-v2-scratch-owner.patch` | V2 graph-safe FA scratch owner with recoverable failure |
| ✓ | `0004-pool-flush-on-oom.patch` | Legacy pool flush-and-retry on OOM (backport of upstream PR #22155) |
|  | `0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch` | V1, rollback only |
|  | `0003-cuda-fa-view_src-sizing.patch` | view_src sizing (closed upstream PR #23620), kept for slow-VRAM-creep regime if it appears |

`GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON` is the Dockerfile default. The trail of investigation and the decision rationale are in `records/research/RSH-20260523-001`, `RSH-20260526-001` (validation matrix), `RSH-20260526-002` (V3 design, deferred), and `RSH-20260526-003` (V2+ON shipping decision with the bench data and upstream survey).

## Recent Changes

- 2026-10-29: **Qwen3-ASR Mac training is operational** — a separate MLX
  workflow prepares speaker/text/audio-disjoint adaptation data with frozen
  FLEURS exclusions, trains decoder LoRA, resumes checkpoints and refines with
  audio-conditioned GRPO. Actual starter runs complete 200 SFT and 50 GRPO
  updates on the 24 GB M4 Pro, peaking at 4.02 GB MLX allocation. On 105 small
  development clips, SFT improves primary errors; this GRPO run trails SFT.
  No adapted Qwen checkpoint is deployed. Nemotron stays unloaded for the
  ongoing training program. Procedure:
  [`training/qwen_asr`](../training/qwen_asr/README.md); measured evidence:
  [`RSH-20261029-002`](research/RSH-20261029-002-qwen-asr-sft-rl.md).

- 2026-10-29: **Mac transcription is live** — `transcribe.lost.plus` serves
  Nemotron 3.5 Transcribe 30B-A4B as a 19.26 GB MLX INT4 export on yeowoolmac's
  The Rest partition (M4 Pro, 24 GB). The inference Common Auth gateway and a
  dedicated Mac tunnel protect the loopback backend. All four language HTTPS
  smokes pass; missing/invalid credentials return 401 and unsupported language
  returns 400. Twenty preflight clips, including each language's longest
  frozen FLEURS file, pass with measured MLX peak memory of 20.18 GB. oMLX is
  stopped and disabled while transcription uses this Mac. All nine pipelines
  now have 2,624 validated predictions; the original eight files and accuracy
  counts are unchanged. The new INT4 model's primary errors are EN 5.35% WER,
  KO 13.77% WER, JA 4.13% CER and ZH 8.52% CER. It improves Japanese over Qwen,
  while MAI remains best in all four languages. Full methods/results:
  [`RSH-20261029-001`](research/RSH-20261029-001-nemotron-transcribe-mac.md).
  Installation and operation: [`deploy/yeowoolmac`](../deploy/yeowoolmac/README.md).

- 2026-10-28: **Silent streams stay open through Cloudflare** — a cold
  prefill of a long prompt on DeepSeek-V4.1-Flash-1017 sends `message_start`
  and then nothing until the first token, and Cloudflare dropped it after
  ~120 seconds (524), which Claude Code retried for up to half an hour. The
  public proxy now pings any event stream that has been silent for 15 seconds
  (Anthropic's ping event on `/v1/messages`, an SSE comment elsewhere). A cold
  random-text prompt with 190 seconds of prefill answered 200 through
  chat.lost.plus.
- 2026-10-26: **qwen3.6-35B-A3B served from yeowoolmac's The Rest
  partition** — rayray916's oQ3 MLX quant (~3.6 bpw, 15.8 GiB with the
  vision tower, no MTP head) of the Heretic-abliterated Qwen3.6-35B-A3B runs
  in oMLX 0.7.2 (release wheel with its prebuilt Metal kernels, venv
  `~/omlx/.venv`) under the user LaunchAgent `plus.lost.omlx` on
  `100.122.130.86:8000`; Grimoire's entry is a `vllm-remote` without a
  residency agent. oMLX allows keyless inference; its admin API needs
  `~/.omlx/api_key` on that Mac. The root LaunchDaemon
  `plus.lost.iogpu-wired-limit` raises `iogpu.wired_limit_mb` to 23040 at
  boot and oMLX's custom memory guard sits at its 22 GiB maximum (RAM minus
  a fixed 2 GiB), leaving a 20.9 GiB prefill limit. With BF16 KV the window
  is capped at 212,992 tokens: a 210k-token needle prompt answers correctly
  (122 tok/s prefill, 29 tok/s decode), short prompts decode at 82-86 tok/s,
  and 228k is rejected by the guard. dopaemon's oQ4e did not fit usefully:
  20.1 GiB of weights left 32k text-only or 27k with vision, and Affine4 KV
  cut long-context decode to 10 tok/s because its fast kernel needs M5
  TensorOps. Chat, streaming, image input and thinking off
  (`enable_thinking: false`) pass through chat.lost.plus. It serves only
  while The Rest is booted.
- 2026-10-25: **Only administrators push other models out** — any lost.plus account admitted to chat can use chat.lost.plus with its global token, so a request-driven load that would stop another model now needs the Common Auth administrator role, on Grimoire's GPUs and through the Mangchi/spark agents (`load?evict=false`). Everyone else loads cold models only into free room and otherwise gets a 409 naming the model it would unload.
- 2026-10-25: **Restarts keep remote models loaded** — `POST /restart` on the gateway (administrator) and on a residency agent restart them without unloading: the gateway leaves remote models resident and an agent leaves its model containers running for the next agent to adopt. A plain stop (`docker restart`, `systemctl`, reboot) is still a shutdown and unloads. Verified live: gateway and Mangchi agent both restarted while Flash-Next stayed up, and both remote models answered within 24 seconds.
- 2026-10-25: **DeepSeek-V4.1-Flash-1017 served from spark1+spark2** — the
  [MiaAI-Lab two-Spark TensorFold recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks)
  at `27741e1` serves the EXL3 EAQ v1 2.9 bpw checkpoint at TP=2 with the full
  1,048,576-token window, image input and DSpark drafting, with drowzeys' Keys
  abliteration overlay (`ABLIT=1`, `0bdc2036`, 104 attention tensors in layers
  10-35) applied; the stock weight files were removed after it passed refusal,
  arithmetic, tool-call and image checks at 53 (prose) and 72 (code) tok/s
  through chat.lost.plus. spark2 (4 TB) is the
  head and API (`100.120.192.122:8888`); spark1 is the worker over the CX7 link
  (`192.168.100.0/24` and `192.168.101.0/24`, both rails, MTU 9000; NetworkManager
  `cx7-a`/`cx7-b`). spark2 exports its Hugging Face cache read-only to spark1 over
  NFS; Engram's tables and rank 1's weight file sit on spark1's own disk. The
  recipe's systemd user unit `dsv41-tensorfold.service` on spark2 keeps both
  ranks up, so Grimoire's `DeepSeek-V4.1-Flash-1017` is a `vllm-remote` entry
  without a residency agent: unloading it in Grimoire does not stop the pair.
  It needs nearly all memory on both Sparks, so spark1's `spark-agent` is
  stopped and the grugthink 27B entries cannot load while it runs. Chat,
  streaming, thinking off and image input pass through chat.lost.plus.

- 2026-10-20: **Speech benchmark complete** — eight pipelines each have 2,624
  FLEURS test predictions across English, Korean, Japanese and Mandarin.
  Native/local provenance is in
  [`RSH-20261020-001`](research/RSH-20261020-001-eg2-multilingual-stt.md);
  [`RSH-20261020-002`](research/RSH-20261020-002-openrouter-stt.md) adds
  MAI-Transcribe 2 and Scribe v2, costing $2.77 in total.
  [`RSH-20261020-003`](research/RSH-20261020-003-nemotron-stt.md) adds Nemotron
  3.5 ASR, which trails Qwen on all four primary error rates. Data, model assets
  and measured results are retained on Dumpling and Mangchi.

- 2026-10-07: **Presets no longer lock remote models** — a preset governs only Grimoire's own GPUs. Models served by another machine's residency agent (`vllm-remote`, today Flash-Next on Mangchi and grugthink on spark1) can be started and stopped while a locked preset such as `training` is active, and activating a preset no longer stops them. Previously `training` returned `409 Preset 'training' is active` for every cold load, including models that never touch Grimoire's 3090.
- 2026-10-07: **Grugthink 27B served from spark1** — `qwen3.8-27b-grugthink` and `qwen3.8-27b-vanilla` moved from Mangchi to spark1 (ASUS Ascent GX10), so loading them no longer evicts Flash-Next from Mangchi's memory budget. spark1 runs the same residency agent (`docker-compose.spark.yml`, `spark-agent`, port 9700) with `etc/spark-agent.json`, launching the identical spec on spark1's `spark-vllm:pin-5fd5dd5` build (same vLLM commit as Thor). Grimoire reaches it at spark1's Tailscale address `100.94.227.60`, since spark1's LAN address is DHCP. Sibling vLLM containers now name `vllm` as the Docker entrypoint, because spark1's image already starts `vllm serve`. The grugthink D6 sampling server that occupied spark1 was stopped; training is stopped.
- 2026-10-07: **Grugthink 27B moved to the D6 run-3 iteration-4 adapter** — `qwen3.8-27b-grugthink` now applies the grugthink D6 run-3 iteration-4 LoRA (`/home/yeowool/models/grugthink/dense-27b-d6-run3-iter-0004`) instead of D6 iteration 15, at the operator's request. In grugthink's all-checkpoint sweep of run 3 (five harness contexts, 50 math and 24 coding prompts, two samples each), iteration 4 lost at most 4.7 accuracy points in any context and thought 0.37 to 0.57 as much as the untouched model, the closest of the ten checkpoints to the teacher-relative target; it still failed that target in two math contexts and about a fifth of its math runs thought more than 0.77 as much. It has not passed a final exam; `qwen3.8-27b-vanilla` stays beside it.
- 2026-10-03: **Grugthink 27B moved to the D6 iteration-15 adapter** — `qwen3.8-27b-grugthink` now applies the grugthink D6 iteration-15 LoRA (`/home/yeowool/models/grugthink/dense-27b-d6-iter-0015`) instead of D3 round 14. D6 trained with varied system prompts and harness contexts, so its shorter thinking carries over to harnesses it never saw: at its iteration-15 check it thought 0.38 to 0.65 as much as the untouched model under held-out Claude Code, Codex, and Korean prompts. Round 14 compressed mainly under its own training prompt. Iteration 15 has not passed a final exam, and its math accuracy check stopped training 4.2 points down; `qwen3.8-27b-vanilla` stays beside it for comparison.
- 2026-09-30: **Grugthink 27B moved to the round-14 adapter, with the base model beside it** — `qwen3.8-27b-grugthink` now applies the round-14 LoRA (`/home/yeowool/models/grugthink/dense-27b-d3-iter-0014`) instead of round 11. Round 14 thought less on the 50-prompt math probe (think ratio 0.40 against 0.48) but has no coding check or final exam. The same port 8003 container also answers as `qwen3.8-27b-vanilla`, the checkpoint without the adapter, so both can be compared on identical weights; Grimoire lists both, and unloading either stops both.
- 2026-09-30: **Grugthink 27B served from Mangchi** — `qwen3.8-27b-grugthink` runs the vanilla Qwen3.8-27B NVFP4 checkpoint with the grugthink round-11 LoRA (`/home/yeowool/models/grugthink/dense-27b-d4-iter-0011`) applied at runtime, vision enabled, and the shipped MTP head drafting three tokens, on port 8003. Its 62 GiB reservation cannot share Mangchi's budget with Flash-Next's 101 GiB, so loading one evicts the other. Cold start took about 3 minutes; direct text, image, and MTP smokes passed (about 1.7 accepted draft tokens per step). Grimoire's live registry and seed carry the remote entry; the gateway container reaches port 8003. Both Qwen3.8 services now load `templates/qwen3.8-chat_template.jinja`, which renders a later system message in place instead of raising `System message must be at the beginning.`; Codex through OpenCodex sends its later instructions that way (lidge-jun/opencodex#6242).
- 2026-09-15: **Flash-Next warm restart reduced from about 753 to 154-168 seconds** — a prepared vLLM sharded state avoids repeating Thor ModelOpt/CuTe DSL weight conversion, while a persistent cache retains compiled Triton kernels. Direct NVMe reads reached 3.32 GiB/s at queue depth 1 and 5.22 GiB/s at queue depth 16; the old target loader achieved only about 0.19 GiB/s, confirming conversion rather than storage as the bottleneck. Target loading fell from 387.91 to 65.81-68.11 seconds. A truly cold Thor compiler start took about 379 seconds and remains below the gateway's 600-second deadline. The explicit 13.49 GiB FP8 KV reservation provides 844,883 tokens (3.22 native contexts), and text, vision, and four simultaneous request smokes passed. (`RSH-20260915-001`)
- 2026-09-14: **Grimoire now uses common auth for browser and API access** — `auth.lost.plus` validates the shared `lp_auth` browser session with `chat` visibility and validates bearer or `X-API-Key` credentials against `chat-v1`. The public proxy strips client identity headers and passes immutable `sub` plus the shared role to the loopback manager. Chat settings manages common auth's global/per-service token mode directly, while account-specific browser keys and expected-`sub` checks prevent stale tabs from crossing accounts. Shared model, registry, ingest, and plugin mutations require the common-auth administrator role. The browser MCP proxy admits only public targets, pins validated DNS results, isolates target credentials, rejects redirects, and returns an allowlisted response surface. History tree references and imports are account-scoped and cycle-safe. The existing bearer was imported without exposing it, and 104 conversations, 10 settings, 71,132 usage events, and the dashboard layout moved to `sub=1`. Eleven independent GPT-6 Astra high review rounds converged. Production bearer inference, model listing, unauthenticated rejection, browser login redirect, token-management rejection for bearer clients, private MCP rejection, and an exact `AUTH_OK` Flash-Next completion pass after deployment. (`DEC-20260914-001`)
- 2026-09-13: **Flash-Next uses vLLM's default 999-image prompt ceiling** — native vision on the sole Mangchi remote model retains a 2,097,152-pixel cap per image, 262,144-token context, FP8 KV, reduced-vocabulary MTP k=4, and eager mode. Production startup allocated 16.89 GiB / 975,077 KV tokens (3.72 full-context requests), and host availability stayed above 12.44 GiB. Direct five-image inference identified red, green, blue, yellow, and magenta in order; both authenticated Grimoire vision smokes passed, and the live registry matches the tracked multimodal seed.
- 2026-09-13: **Mangchi serving is Flash-Next-only with a 3x KV allocation** — retired the dense Qwen3.8-27B launch spec and its matching Grimoire remote-registry entry while preserving its model artifact, image, native dense optimizations, patches, tests, and research. Flash-Next now reserves 101 GiB at `gpu-memory-utilization=0.82`; production measured 18.8 GiB of KV cache and 1,085,325 cache tokens against 6.13 GiB at 0.72, a 3.07x increase. Host availability remained above 13.89 GiB during startup, and an authenticated Grimoire smoke returned the expected answer.
- 2026-09-12: **Reduced-vocabulary MTP drafting ships; decode +9-15% on top of k=4** — image `mangchi-vllm:thor-dense-candidate-v8-draft-vocab` adds an opt-in FR-Spec draft head (`QWEN4EXP_DRAFT_VOCAB`), now serving at 98,304. `lm_head` is the one unquantized BF16 tensor left (248320x2560 = 1.27 GB) and ~98% of a draft pass's weight traffic, so restricting the draft to frequent token ids removes cost without touching the target, which still verifies over the full vocabulary. Measured at k=4: code 36.30 (+14.7%), arithmetic 35.78 (+11.6%), prose 20.74 (+13.0%), reasoning 32.84 (+8.9%) tok/s, with prose accepted length unchanged at 2.229. 98,304 beats the upstream 65,536 default here: 99.96% of generated tokens fall below it against 97.89%, and it keeps more acceptance at equal speed. Combined with k=4 this is +38% code and +28% reasoning against the previously shipped k=2 full-vocabulary config. Piecewise CUDA graphs remain blocked: the PLE disk gather stays inside capture even as a registered splitting op, so graphs need the staged-gather rework rather than an annotation. (`RSH-20260912-003`)
- 2026-09-12: **Flash-Next MTP depth raised to k=4; measured optimum, not assumed** — a warm four-workload sweep of `num_speculative_tokens` 2/3/4/5 puts the turnover at k=4: code 31.6, reasoning 30.1, arithmetic 32.3, prose 18.5 tok/s, against 25.0/23.9/24.4/17.7 at the shipped k=2 (+21-32% on everything but prose). Mean accepted length rises monotonically through k=5, so deeper speculation never guesses worse; wall-clock stops following it because each extra draft pass costs a fixed increment while the marginal accepted token shrinks (k=5 buys 3% more accepted tokens on arithmetic for ~13% more step cost). k=3 is marginally better on prose alone. Output varies run-to-run at fixed k, so the reported MTP=3 quality cost does not reproduce here. k>=5 cannot boot at the derived block size: the QSA ring capacity (8 for k<=4, 12 for k>=5) never joins vLLM's block-size alignment, so 3184 fails to divide — workaround is `--block-size` at a multiple of 48. The residency agent now captures failed containers' stderr, without which that assertion was invisible. (`RSH-20260912-002`)
- 2026-09-12: **Flash-Next v7 ships MTP=2 and the prefix-caching mamba fix; decode 1.6x on prose, 2.5x on code** — new image `mangchi-vllm:thor-dense-candidate-v7-prefix-mtp` ports blazux/qwen3.8-Flash-DGX's two-line mamba block_size fix and enables MTP from the in-checkpoint draft head. MTP required raising Flash-Next to util 0.72 / 88 GiB (vLLM measured -1.96 GiB KV at 0.68) and disabling the unused multimodal warmup. Measured end-to-end against the 10.56 tok/s pre-MTP baseline: code 26.0 tok/s (2.46x, mean accepted length 2.82/3.00, per-position 95.1/86.6%), prose 16.5-17.5 tok/s (1.56-1.65x, 1.97-2.00/3.00, 61-64/36%), arithmetic 2.95/3.00. Acceptance matches or beats the tonyd2wild Spark reference at every comparable position (93/78% code, 65/39% prose), so the abliteration shift did not hurt the draft head. Step rate holds at ~9/s regardless of workload against 10.56/s pre-MTP, so MTP costs ~15% per step and returns the accepted length. Arithmetic determinism holds (437 x4). Prefix caching is enabled but inert on this vLLM pin — the unannotated MTP draft flags Mamba groups as draft groups, which vLLM documents as disabling cross-request reuse silently, so cached_tokens stays 0 and there is no wrong-restore risk. blazux serves real hits because their image runs the qwen38-flash-next release line, a different codebase from our PR #55557 pin; draft-group annotation is the follow-up that unlocks it. (`RSH-20260912-001`)
- 2026-09-11: **Mangchi co-residency withdrawn; models sized for standalone residency** — after guarded tests at host-memory floors of 6, 4, and 3.5 GiB, Flash-Next's MoE startup spike drove host available memory to ~3.3 GiB regardless of reservation, below any safe floor. 27B is now at util 0.22 / 27 GiB (0.20 left only ~1.07 GiB KV against the 3.35 GiB needed for 100K FP8) and Flash-Next stays at util 0.68 / 83 GiB (its 74.33 GiB weights + ~5.7 GiB non-torch overhead leave no KV room below 0.68). The two models are served one at a time via the agent's LRU eviction; re-enabling co-residency requires shrinking Flash-Next's PLE memory window, not a lower host floor. (`DEC-20260910-001`)
- 2026-09-10: **Remote vLLM startup now remains visibly loading until health passes** — Mangchi reports each resident model's explicit `loading` or `loaded` lifecycle state. Grimoire propagates that state during remote health refreshes instead of preserving a stale loaded state while the model container is alive but its vLLM endpoint is not ready.
- 2026-09-10: **Qwen3.8 reasoning and tool output are parsed separately** — both Mangchi launch specs use vLLM's `qwen3` reasoning parser alongside the `qwen3_xml` tool parser, preventing the tool parser from consuming `</think>` and folding hidden reasoning into visible content.
- 2026-09-10: **The custom Thor vLLM build now keeps vision while dropping unsupported native targets** — pinned and reduced vLLM FlashAttention to 14 BF16 FA2 sources needed by the accepted Qwen3.8 text/vision paths; FA3 and external targets whose own configure checks reject SM110 are no longer built. The retained FA2 library is 52.8 MB, down from 307.8 MB of FA2 plus 39.8 MB of FA3. The native wheel completed in about 21 minutes and now uses persistent BuildKit caches. The residency agent uses the accepted W4A4 checkpoints and v6 image: 27B keeps its measured 2,048-token prefill batch and vision path, while Flash-Next uses an 8,192-token prefill batch at 0.68 GPU-memory utilization. Both retain a 262,144-token request limit, FP8 KV, and Qwen XML automatic tool-call parsing. The prior v5 canary container remains stopped under a rollback name.
- 2026-09-10: **Remote model indicators now refresh when the selector opens** — Grimoire asks Mangchi for residency and process health, then confirms the vLLM `/health` endpoint before showing online. A timeout or ambiguous response keeps the previous indicator; only confirmed health or confirmed absence/death replaces it. Healthy remote processes are rediscovered after a gateway restart, and stale Mangchi residency records are reaped before relaunch. Deployed on Grimoire and Mangchi; the authenticated refresh endpoint reports both registered remote models offline while the separate native vLLM canary remains outside residency-agent control.
- 2026-09-10: **Mangchi remote-backend gateway support deployed** — added `vllm-remote` lifecycle and request routing, registered both Mangchi NVFP4 services in tracked and live Grimoire state, excluded remote models from local GPU allocation and llama.cpp KV slots, and made the registry HTTP admission validator backend-aware. Production health and authenticated management checks pass. A cold-load request reached Mangchi and failed safely because the registered 27B model directory is not present; Mangchi released the reservation and has no resident model. A separate Flash-Next artifact download is still active, so an end-to-end remote chat smoke remains pending. (`DEC-20260909-002`)
- 2026-09-10: **Qwen3.8 Flash-Next now serves native 262K context with FP8 QSA KV on Jetson Thor** — pinned vLLM PR #55557 at `5fd5dd5`, retained the SSD-backed 95.4 GiB PLE table and SM110 top-k fallback, and raised the model allocation to 0.67. Production allocated 5.88 GiB for 434,087 cache tokens (1.66x native context) and passed a deterministic chat smoke. The canary completed an exact 262,144-token request in 1,642.64 seconds with zero preemptions or request errors. Attention KV is FP8; the GDN/Mamba recurrent cache remains float32. (`DEC-20260909-003`, `RSH-20260909-002`)
- 2026-09-09: **Qwen3.8 Flash-Next ported to Jetson Thor with SSD-backed PLE** — ported Blazux's PLE mmap approach to vLLM v0.29's `Qwen4Exp` implementation while retaining the Thor-native `sm_110a` image. The 95.4 GiB BF16 PLE table remains on NVMe; the rest of the model uses 71.11 GiB. Added Thor fallbacks for incompatible FlashInfer autotuning and QSA cooperative top-k kernels. (`RSH-20260909-001`)
- 2026-07-19: **Eastself serving registry restored from retained intake provenance** — reconciled the tracked seed and persistent runtime registry to one byte-identical document, restored all 19 valid manifest-backed Eastself completion aliases in ascending intake `created_at` order, and preserved the fixed embedder/reranker assignments. The failed `eastself-gemma4-12b-dec20260718` alias and pin were removed; its unshared PEFT directory, LoRA GGUF, tokenizer-aligned GGUF, and intake manifest were permanently deleted while the shared 12B base and healthy v2 artifacts were preserved. All 19 restored entries pass real-file registry validation. Live sequential smokes passed for accepted 31B generations (`ck17972-final`, `ck3749-120k-final`) and healthy 12B v2; historical probes remain available as rollback/evidence and are not implied to be quality-accepted (for example, `ck250-dec4` loaded but produced an empty visible response).
- 2026-06-22: **Multi-process gateway + data-parallel GPU replicas** — split the gateway into a stateful manager (internal :9000, owns lifecycle + chat) and N stateless proxy workers (:9001) that round-robin encoder endpoints across per-GPU replicas. Embedder + reranker each get an always-on GPU-0 replica; one model name fans out across both GPUs. Production embeddings 115 → **224 req/s** (full 2x). Cost: ~2.9 GiB/GPU headroom (both GPUs ~21 GiB free), capping large-ctx chat models. (DEC-20260622-002, RSH-20260622-001, commit `8031d6e`)
- 2026-06-22: **Gateway proxy throughput fix** — every proxy path created a fresh per-request `httpx.AsyncClient` (no keepalive), capping high-RPS endpoints at ~30 req/s. Switched to a shared connection-pooled client (`proxy/client.py`); gateway rerank 29.2 → 64.4 req/s. Also confirmed reranker `--parallel` does not help (prefill-only encoder; parallel=1 optimal). Corrects the chat-only conclusion of RSH-20260518-006. (RSH-20260622-001, commit `4c37298`)
- 2026-06-22: **GPU co-location for always-on small models** — replaced strict one-model-per-GPU with a per-model `vram-budget-mib` + live `nvidia-smi` free-VRAM check. Both 0.6B models (embedder + reranker) now co-locate on GPU 1 (~2.9 GiB), freeing GPU 0 for chat. (DEC-20260622-001, commit `b616a99`)
- 2026-06-20/21: **llama.cpp engine + model tuning** — switched to TheTom `llama-cpp-turboquant` fork; turbo4 KV cache with ctx-size tuned to verified maximums across chat models; auto-MTP for nextn models; embeddinggemma-300m registered/tuned. (commits `3ca14cd`, `c59a4ba`, `f2edaac`, `3b16be0`, `e7e8720`)
- 2026-05-30: **vLLM migration prototype — llmcompressor quantization pipeline validated** (RSH-20260529-006, DEC-20260528-001). Driver upgraded to 580.159.04 for CUDA 13. vLLM 0.21.0 serving commercial AWQ model `Qwen3.6-27B-AWQ-INT4` on GPU 1. llmcompressor quantizes BF16 → AWQ (19.17 GB from 52 GB). vLLM loading blocked on post-quantization config patching (Marlin kernel shape requirements, missing multimodal weights).
- 2026-05-26: **Prod Dockerfile default flipped to V2+ON + 0004** (RSH-20260526-003). Pending: rebuild + recreate `grimoire:local` container.


- 2026-05-18: **DFlash MAX_VERIFY_TOKENS cap fixed** — `LLAMA_DFLASH_MAX_VERIFY_TOKENS=25` was silently breaking DDTree tree-mode (3-5% acceptance). Patched with env-var `GGML_DFLASH_MAX_VERIFY_TOKENS` following existing `GGML_DFLASH_MAX_CTX` pattern. Default stays 25. Auto-derived in `model_manager.py` from n_max + branch_budget. (RSH-20260518-005, commit `2f6a345`)
- 2026-05-18: **DDTree tree-mode benchmarked** — Tree-mode works correctly after cap fix (85% acceptance) but adds only +1% throughput over flat mode for greedy decode. Draft is already well-matched; tree branches have nothing to rescue. (RSH-20260518-005)
- 2026-05-18: **Gateway overhead profiled** — Instrumented `proxy/llama.py` with `time.perf_counter()`. Gateway adds 0-60ms per request (<1%). The ~1.3s "gateway overhead" from earlier measurements was a measurement error (comparing warm bench vs production through reasoning mode). (RSH-20260518-006)
- 2026-05-18: **DFlash think-split prototyped** — Streaming SSE parser detects think/answer boundary via `reasoning_content` delta fields. Hybrid split (AR for thinking, DFlash for output) is not beneficial because re-prefilling the assistant message costs more than the DFlash speedup saves. DFlash provides consistent ~1.55x speedup regardless of thinking mode. Per-request `speculative.n_max: 0` toggle already works — no patch needed. (RSH-20260518-006)
- 2026-05-18: **DFlash VRAM overhead measured** — +2,188 MiB (+12%) vs AR-only (17.4 GB → 19.6 GB). Draft model weights (~1.84 GB) + ring buffer + tape buffers. Leaves ~4.4 GB free on 24 GB for KV cache. (RSH-20260518-006)
- 2026-05-18: **PFlash stale-thread deadlock fixed** — `PflashDaemon` replaced `loop.run_in_executor` with a dedicated compressor thread + async Queue. Verified: 100/100 consecutive PFlash-compressing iterations, zero failures, zero VRAM drift, zero deadlocks. (DEC-20260518-001, commit `78cd53b`)
- 2026-05-18: **VRAM drift soak** — `soak_vram_drift.py` added, 100-iteration soak confirmed zero VRAM drift across repeated PFlash compression cycles
- 2026-05-18: **PINNED_SHA enforcement** — Docker build now verifies cloned SHA matches `GRIMOIRE_LLAMA_CPP_PINNED_SHA` (commit `c90d2a0`)
- 2026-05-18: Phase 4 — pflash daemon propagation fixed, catastrophic 413 on compression failure, slot-save-mtmd patch regenerated for Bee HEAD 4db14be0
- 2026-05-18: Phase 3 — Docker rebuild + canary verified (1.93x speedup)
- 2026-05-18: Hygiene cleanup — stale patches deleted, SPEC/README updated, dead BACKEND_DFLASH code removed, .dockerignore expanded
- 2026-05-18: Phase 5 — doc cleanup + speedup verification
- 2026-05-18: Phase 7 — lucebox/ deleted, pflash_daemon extracted, dflash model files removed (3.3 GB), 87 GB disk freed
