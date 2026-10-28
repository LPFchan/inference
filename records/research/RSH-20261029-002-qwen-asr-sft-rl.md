# RSH-20261029-002: Qwen3-ASR Mac SFT and GRPO starter experiment

Opened: 2026-10-29 06-42-50 KST
Recorded by agent: codex-qwen-asr-training

## Question and outcome

Can Qwen3-ASR-1.7B be adapted to Korean, Japanese and Mandarin on the 24 GB
M4 Pro, and does this starter RL stage improve on supervised fine-tuning?

Both training stages run on the actual Mac. Decoder LoRA uses 6,422,528
trainable parameters and peaks at 4.022 GB of MLX allocation in this experiment.
The supervised checkpoint improves all four primary development metrics;
English CER increases slightly. GRPO does not improve on SFT here: Korean WER,
Japanese CER and Mandarin CER worsen slightly. SFT-200 is the better development
candidate. No trained Qwen checkpoint is deployed, and these results do not
establish a FLEURS benchmark gain or a model-capacity ceiling.

## Reproducible setup

Implementation: `060f466`, with per-file source hashes in each run contract.
The procedure and settings live in
[`training/qwen_asr/README.md`](../../training/qwen_asr/README.md).
The base is `mlx-community/Qwen3-ASR-1.7B-8bit` revision
`a8379a2e2f9e313c9292cdf1af4055ab56d50d55` (Apache-2.0).
Every downloaded model file is hashed into the run contract.
MLX 0.32.4, MLX LM 0.31.3 and MLX Audio
`c1e429151c56ce4057c62709f958df998e301d3f` run in a separate repo-local training
venv on yeowoolmac's The Rest partition, M4 Pro 24 GB, macOS 15.7.1,
Python 3.12.13. Production dependencies are untouched.

The [official Qwen SFT recipe](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
is Torch/CUDA oriented. The inspected
[MLX Tune example](https://github.com/ARahim3/mlx-tune/blob/9690fe1fed19c72f0810902b9c188d9a2625eb5b/mlx_tune/stt.py)
builds audio-placeholder-plus-transcript training sequences without Qwen's
native chat/language framing, and masks the first transcript target.
This workflow reuses the native MLX Qwen prompt/audio encoder/decoder for both
training and generation, explicitly supervising the first transcript token
through EOS. The audio encoder, projector and quantized base remain frozen;
rank-16 LoRA targets the 112 decoder attention projections.

SFT uses AdamW, LR 1e-4, four single-clip microsteps per update, gradient clip
1.0, zero weight decay and no dropout. The microstep language cycle is
KO/JA/ZH repeated three times, then English. The pilot separately verifies a
checkpoint at step 2 followed by resume through step 12. The starter run starts
afresh and completes 200 updates.

GRPO starts from the completed SFT checkpoint: four serial completions per
utterance, temperature 0.7 without top-k/nucleus filtering, max 256 tokens,
LR 1e-6, clip epsilon 0.2 and sampled KL beta 0.02 toward the fixed SFT policy.
Reward is `max(-1, 1-error)`, CER for KO/JA/ZH and WER for English; truncated
outputs receive -1. Likelihood temperature matches rollout temperature.
Reference/current policies share frozen weights and store separate adapters.
There is one update per group. Of 50 starter groups, 17 have different rewards
and 33 have zero group-relative advantage; KL/momentum can still update weights.
The RL pilot has 2 different-reward groups out of 8. No external judge/API is used.

## Data and exclusions

The prepared training set has 979 clips (KO 221, JA 355, ZH 286, EN 117),
about 30 minutes per target language and 9 minutes English. Development has
105 clips (KO 24, JA 39, ZH 31, EN 11), about 10.2 minutes total.
Sources are human-transcribed read speech:

- [Zeroth Korean](https://huggingface.co/datasets/kresnik/zeroth_korean), revision
  `1fe937899f828af822293d05e086200946088bdf`, CC BY 4.0; Lucas Jo/Atlas Guide
  and Wonkyum Lee/Gridspace. Development speakers are held out by stable hash.
- [Common Voice 17](https://huggingface.co/datasets/fsicoli/common_voice_17_0),
  revision `8262c16bf297c87a9cd88c51997c4758ed7a8ba2`, CC0 1.0; Mozilla and
  contributors. The mirror is unofficial. Use official train/dev splits and
  exclude dev speakers/sentences from training.

The final set has no shared sample IDs, speaker IDs, normalized text or converted
WAV hashes across train/dev. Normalized text and WAV hashes matching any frozen
FLEURS test clip are excluded from both sets. This does not detect semantic
paraphrases or cross-corpus speakers without shared IDs, and does not establish
that these public datasets were absent from Qwen's original pretraining.
Selection is a seeded candidate pool with duration caps and archive order,
not uniform sampling over the complete corpora.

Train manifest SHA-256: `37f396bc4bf7c4f1983ff5a4a072c15eb84c59cc161572e3ddbff4118b52ee4f`.
Dev manifest SHA-256: `0611499667a698473869949af48731234cf56003776fd7f442a3a616ca8930f3`.
Protected FLEURS manifest SHA-256:
`0c91ba8e164d33576c091385b4ddde130e4220756fe845f57d8736c9f86dd92e`.
All prepared WAV hashes are checked before each run. All nine original
benchmark JSONL files and the frozen test manifest remain byte-identical to
`nemotron-transcribe-20261029/validation-nine.json`; none are rescored or rerun.

## Development results

All three checkpoints use exactly the same 105 development clips. Greedy
transcripts from the SFT final checkpoint and the GRPO initial checkpoint match
exactly after loading from disk. Scoring imports the unchanged shared
`stt-score.py` normalizer and corpus edit-count implementation, jiwer 4.0.0.
These are development results, separate from the 2,624-clip FLEURS comparison.

| Checkpoint | EN WER (11) | KO WER (24) | JA CER (39) | ZH CER (31) |
| --- | ---: | ---: | ---: | ---: |
| Base MLX 8-bit | 7.77% | 15.22% | 13.17% | 3.75% |
| SFT, 200 updates | 6.80% | 11.76% | 10.95% | 3.07% |
| SFT + GRPO, 50 updates | 6.80% | 12.46% | 11.09% | 3.24% |

Character error across all four languages:

| Checkpoint | EN CER | KO CER | JA CER | ZH CER |
| --- | ---: | ---: | ---: | ---: |
| Base MLX 8-bit | 5.57% | 2.56% | 13.17% | 3.75% |
| SFT, 200 updates | 5.81% | 2.10% | 10.95% | 3.07% |
| SFT + GRPO, 50 updates | 5.81% | 2.10% | 11.09% | 3.24% |

Exact edit counts, denominators, per-clip predictions and teacher-forced losses
are retained in `starter-summary.json` and each run's `dev-*.json`.
The English sample is only 11 clips; development improvements are insufficient
for deployment acceptance. No language/script-specific normalization or
post-hoc numeral expansion is added.

## Runtime and service verification

| Run | Updates | Recorded run time | Median update | Peak MLX allocation |
| --- | ---: | ---: | ---: | ---: |
| SFT resume pilot | 12 | 11.61 s | 0.252 s | 3.638 GB |
| GRPO pilot | 8 | 29.58 s | 3.159 s | 3.742 GB |
| Starter SFT | 200 | 296.75 s | 1.015 s | 4.022 GB |
| Starter GRPO | 50 | 249.14 s | 3.052 s | 3.921 GB |

Run time includes model setup, checkpointing and initial/final development
checks after input validation/hashing. Update time includes speech feature
preparation and optimization; the SFT pilot uses one microstep while the starter
uses four. Allocation is MLX allocator memory, not total RSS/OS memory. These
measurements apply to 1–15-second training clips, not arbitrary audio lengths.
No wired-memory, process-priority or JIT changes are made.

The wrapper temporarily unloads only the Nemotron STT backend and restores it
when the bounded job exits. Gateway/tunnel and oMLX settings are preserved.
After training, authenticated HTTPS transcription and discovery both return
200 with production artifact
`7018b1c16891127e389845494e3e762703bbc6605e5c1136610405e053378a1b`.
Production remains Nemotron. No Grimoire operations occur.
Eleven focused tests pass on the actual Mac, including supervision alignment,
data-exclusion guards and GRPO gradient direction. Local CPU tests have ten
passes and one MLX skip. Python compilation and shell syntax checks pass.
Optimizer checkpoint step counters are exactly 12, 200 and 50, confirming the
resumed state and completed starter updates; adapter files contain only the
224 expected LoRA tensors. The production package freeze is byte-identical to
the pre-training served freeze. No benchmark CI is configured.

## Retention and next experiment

The Mac retains model, prepared audio, dependency freeze, frozen source,
checkpoint/optimizer states, metrics, development predictions and source
hashes under `~/models/stt/qwen3-asr-training-20261029`; the executable training
workspace/venv is `~/inference-qwen-training`. Evidence is mirrored to Dumpling
under the same model path and Mangchi under
`~/models/stt-benchmark/qwen3-asr-training-20261029`, without download caches.
The interactive development chart is `starter-development.html`.

Expand and diversify development/training speech before selecting a final
candidate for the untouched FLEURS test. Investigate reward ties and Korean
spacing before spending more on this RL recipe. Additional layers remain an
experiment conditional on a demonstrated capacity constraint; this small
adaptation run provides no evidence that Qwen is saturated.
