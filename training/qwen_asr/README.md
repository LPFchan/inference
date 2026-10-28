# Qwen3-ASR adaptation on Apple Silicon

Adapt Qwen3-ASR-1.7B to Korean, Japanese and Mandarin, retaining English with
replay. Run supervised decoder LoRA first, then experimental GRPO from the SFT
checkpoint. Architecture expansion is deferred until adaptation experiments
show a reason to add capacity. Nothing here deploys a trained checkpoint.

## Environment and inputs

Use a dedicated checkout/worktree and repo-local `.venv`; do not install the
root inference `pyproject.toml` or modify the production `.venv-stt`.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r training/qwen_asr/requirements.txt
.venv/bin/pip freeze > "$HOME/models/stt/qwen-training.freeze.txt"
```

The initial model is `mlx-community/Qwen3-ASR-1.7B-8bit` at
`a8379a2e2f9e313c9292cdf1af4055ab56d50d55` (Apache-2.0). Download with
`huggingface_hub.snapshot_download` at that revision into a new model directory.
Write `training-source.json` there containing exactly:

```json
{"id":"mlx-community/Qwen3-ASR-1.7B-8bit","revision":"a8379a2e2f9e313c9292cdf1af4055ab56d50d55"}
```

The trainer binds every local model file, all Python workflow files, the shared
scorer, both prepared manifests and the protected-test manifest by SHA-256.
Keep the download evidence and the environment freeze with each experiment.

Data preparation uses only training/development speech from pinned sources:

| Language | Source | Revision | License |
| --- | --- | --- | --- |
| Korean | [kresnik/zeroth_korean](https://huggingface.co/datasets/kresnik/zeroth_korean) | `1fe937899f828af822293d05e086200946088bdf` | CC BY 4.0 |
| Japanese, Mandarin, English | [fsicoli/common_voice_17_0](https://huggingface.co/datasets/fsicoli/common_voice_17_0) | `8262c16bf297c87a9cd88c51997c4758ed7a8ba2` | CC0 1.0 |

Zeroth is by Lucas Jo (Atlas Guide) and Wonkyum Lee (Gridspace); Common Voice
is by the Mozilla Foundation and its contributors (Ardila et al., LREC 2020).
These are human-transcribed read-speech corpora. The Common Voice mirror is
unofficial. Do not infer that these corpora were absent from Qwen's original
pretraining. Prepared evaluation is held out from **our** adaptation.

```sh
.venv/bin/python training/qwen_asr/prepare.py "$HOME/models/stt/qwen-data" \
  --protected-test "$HOME/models/stt/fleurs-test/manifest.jsonl"
```

Defaults select approximately 30 minutes per target language plus 9 minutes
English training, and 3 minutes per target plus 54 seconds English development.
Audio is mono PCM16 at 16 kHz, 1–15 seconds per clip. Zeroth development is a
stable hash-based speaker holdout from its training split. Common Voice uses
its official dev/train splits and excludes every dev speaker and sentence
from training. Exact sample IDs, speaker IDs, normalized texts and converted
WAV hashes cannot overlap train/dev. Normalized text and audio hashes matching
any frozen FLEURS test clip are excluded from both splits. This cannot detect
semantic paraphrases or identify cross-corpus speakers without shared IDs.

The seed determines a candidate pool; duration caps and archive order determine
the final Common Voice sample set. This is a reproducible pilot subset, not a
claim of uniform random sampling. `data.json` records exact counts, durations,
source-file hashes and rejected duplicates. Existing completed data is verified
and reused; changed settings require a new output directory. Partial downloads
are cached for retry. Models/audio/checkpoints stay outside Git.

## SFT

```sh
.venv/bin/python training/qwen_asr/train.py sft \
  --model "$HOME/models/stt/qwen-base-8bit" \
  --data "$HOME/models/stt/qwen-data" \
  --protected-test "$HOME/models/stt/fleurs-test/manifest.jsonl" \
  --output "$HOME/models/stt/qwen-sft" --steps 200
```

All 112 attention projections in the 28 decoder blocks receive rank-16 LoRA
(scale 1, equivalent alpha 16). The quantized base, audio encoder, projector,
embeddings and normalization stay frozen. Audio features are detached before
decoder training. Qwen's native chat/audio prompt and known language prefix are
used exactly as in inference. Teacher forcing supervises the first transcript
token through EOS; prompt/audio tokens carry no loss. Frozen FLEURS references
are never model input or used for checkpoint selection.

Batch size is one with four gradient-accumulation microsteps, AdamW with zero
weight decay, LR 1e-4, global gradient clipping at 1.0 and no dropout. Language
selection cycles KO/JA/ZH three times then English: 30/30/30/10 percent of
microsteps. It is independent of corpus size. Cache is capped at 128 MiB and
MLX allocation at 18 GiB by default. These limits do not measure total process
RSS or all other Mac applications. Actual memory/time must be measured.

## GRPO

```sh
.venv/bin/python training/qwen_asr/train.py grpo \
  --model "$HOME/models/stt/qwen-base-8bit" \
  --data "$HOME/models/stt/qwen-data" \
  --protected-test "$HOME/models/stt/fleurs-test/manifest.jsonl" \
  --init-adapter "$HOME/models/stt/qwen-sft/step-000200" \
  --output "$HOME/models/stt/qwen-grpo" --steps 50
```

This experimental audio-specific loop shares the SFT input/likelihood path.
Four on-policy completions are sampled serially at temperature 0.7, without
nucleus/top-k filtering, up to 256 tokens. Reward is `max(-1, 1-error)`: CER for
KO/JA/ZH and WER for English, using the unchanged benchmark normalizer.
Truncated completions receive -1. Rewards are standardized within each group;
identical rewards produce zero advantage and are explicitly logged.

Token likelihoods use the same sampling temperature. The clipped policy
objective (epsilon 0.2) has a nonnegative sampled KL penalty (beta 0.02) toward
the initial SFT policy, and one update per group at LR 1e-6. Reference and
current policies share frozen weights; only reference adapters are copied.
There is no critic, external judge, paid API, or independent rollout server.
Gradients are averaged across group members and globally clipped at 1.0.

[Prior Qwen ASR adaptation work](https://arxiv.org/abs/2607.08208) motivates
SFT followed by RL; this implementation is a small experimental recipe, not a
reproduction of its full training pipeline or a quality guarantee.

## Checkpoints and evaluation

`step-NNNNN/` stores adapter and optimizer safetensors plus a hash-checked
checkpoint manifest. `latest.json` points only to a completed checkpoint.
Repeat the same command with `--resume --steps NEW_TOTAL` to continue; step
limits may increase, while data/model/code/hyperparameters must match. A changed
recipe starts a new run. Sampling and utterance selection use per-step seeds.
The lock refuses duplicate jobs writing the same run directory.

`metrics.jsonl` records losses, gradient norms, time, MLX allocation and GRPO
reward/advantage/completions. Initial/final `dev-*.json` record greedy transcripts
and supervised losses; by default four clips per language. This is a plumbing
check, not evidence of a general accuracy gain. Use a larger independent dev
set for selection, then run the untouched 2,624-clip FLEURS test once for each
accepted final candidate with the shared scorer. Keep English regressions
visible. Do not merge/quantize/deploy adapters on the strength of training loss.

## Sharing the serving Mac

The 20 GB Nemotron service must be unloaded for training. The trainer refuses
while the loopback backend is listening. From the Mac's The Rest partition:

```sh
bash training/qwen_asr/with-training-slot.sh \
  .venv/bin/python training/qwen_asr/train.py sft ...
```

The wrapper pauses only `plus.lost.inference.stt`, leaves gateway/tunnel and
oMLX settings unchanged, runs the bounded command, then restores the backend
and checks health on normal exit or handled interruption. SIGKILL/power loss
can prevent shell cleanup; inspect the slot directory and LaunchAgent before
recovering. This is temporary service downtime, not concurrent serving/training.

## Ongoing single-worker hill-climb

The operator authorized continued work under
[DEC-20261029-001](../../records/decisions/DEC-20261029-001-qwen-asr-autoresearch.md).
The [repo contract](../../skills/autoresearch/SKILL.md) defines the frozen
validation score, per-language safeguards, independent-seed confirmation,
resource bounds and stop behavior. Nemotron stays unloaded during this program.

`prefetch.py` downloads pinned public expansion sources with verified cache
reuse, per-file progress and retryable failures. `freeze_development.py` freezes
FLEURS **validation** while excluding the original test. `assemble.py` attaches
that same validation manifest to each immutable training snapshot. Prepared
training examples remain distinct from validation/test by text and audio hash;
known source speaker holdouts are retained.

`expand_clean.py ROOT --core DATA_V001 --output NEW_DATA --protected-test TEST`
curates already-cached Zeroth Korean, remaining validated Common Voice Japanese
and AISHELL training recordings. It keeps the same frozen development set and
English replay, reserves official source holdouts, verifies every used download
and prepared WAV, and publishes `data.json` after verification. Distinct training
recordings may share a sentence; held-out text remains excluded. It never starts
a model or downloads new files. YODAS captions remain quarantined because the
Japanese pool includes translated lecture material; language/alignment checks
are required before treating it as Japanese speech supervision.

`cycle.py ROOT --runtime PYTHON --initial-root STARTER_ROOT --protected-test TEST`
waits for calibration and the first expanded snapshot, then runs one GPU job at
a time. It starts 5,000-step SFT and 1,000-step GRPO trials, confirms provisional
wins with a second training seed, and updates its internal incumbent only when
both pass `selection.py`. Profiles rotate; closed parent/data/code/profile
combinations are not repeated. A new `latest-data.json` pointing to an immutable
data directory opens new trials. When authorized recipes are exhausted, it
reports `needs-new-hypothesis` for this harness to supply a genuinely new idea.

GRPO can mix Korean WER into CER with `--korean-wer-weight 0.5`, so Korean spacing
errors can generate a learning signal. The default remains CER-only for the
original starter recipe. The fixed starting reference may be an already-adapted
SFT-derived checkpoint. This is a new experimental recipe, not a claim that the
longer RL stage improves quality.

Freeze a source archive for each running controller; do not edit its code in
place. `loop-state.json` and each trial's `verdict.json` are runtime evidence.
The harness records scored trials and accepted state in the repo's LOG history,
STATUS and PLANS. Creating `ROOT/STOP` terminates the owned child and stops new
trials; preserve checkpoints/data and do not restore serving automatically.
