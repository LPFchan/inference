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

## Written-form data and an external Korean probe

`prepare_fleurs_style.py` prepares the pinned FLEURS training split using
`raw_transcription` targets and copies an existing frozen development split.
It excludes held-out sentence IDs across all four languages, normalized text,
audio hashes and any supplied probe. Unknown FLEURS speakers remain unknown;
speaker separation and the base model's pretraining exposure are unestablished.
Training clips default to 1–15 seconds. `train --max-seconds 20` includes clips
up to 20 seconds in a new snapshot and records the cap in `data.json`; it does
not change the KSS probe's 15-second cap.

Prepare the evaluation-only KSS probe first, excluding every adaptation pool:

```sh
.venv/bin/python training/qwen_asr/prepare_fleurs_style.py probe "$PROBE" \
  --cache "$CACHE" --dev "$DEV_DATA" --protected-test "$FROZEN_TEST" \
  --train-pool "$TRAIN_DATA/train.jsonl"
.venv/bin/python training/qwen_asr/prepare_fleurs_style.py train "$STYLE_DATA" \
  --cache "$CACHE" --dev "$DEV_DATA" --protected-test "$FROZEN_TEST" \
  --exclude "$PROBE/probe.jsonl"
```

Repeat `--train-pool` for each other adaptation manifest. KSS uses written
`original_script` targets, a fixed seed and separate digit/non-digit cohorts
(256 clips each by default); an incomplete cohort is refused. Report each
cohort separately. The balanced probe does not estimate natural-population
accuracy. KSS is CC BY-NC-SA 4.0 and stays evaluation-only; FLEURS is CC BY 4.0.

Both paths verify pinned download hashes and converted mono 16 kHz PCM16 WAVs.
Prepared manifests retain source, license, code and exclusion hashes. Keep
cache/audio/output outside Git and use a new directory for changed settings.
Preparation checks free disk and memory pressure between parquet row groups;
a single audio row group can occupy several GB. This recipe establishes no
accuracy improvement until a completed training run passes selection and
independent confirmation.

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

By default each microstep draws a training row with replacement from its
language pool, seeded by the microstep index. `--sampler sentence-epoch` (SFT
only) instead exposes each source sentence at most once within a run: each
language's recordings are grouped by `source_sentence_id` (normalized text when
absent), the sentences are shuffled with a per-language seeded generator, and
one seeded recording is kept per sentence. Microsteps keep the same language
schedule and consume each language's order in turn. The requested
`--steps × --accumulate` must fit the shortest language under that schedule, or
the run is refused before the model loads; the other pools are not exhausted.
The sampler is stored in the exact-match resume contract, and each metrics
event records that update's row `ids`.

`--schedule ko=30,ja=3,zh=3,en=1` (SFT only) replaces the fixed 3:3:3:1
language cycle with these counts per cycle, spread evenly. It lets a large
Korean pool, such as FLEURS plus phone calls, be seen once while the other
languages keep their usual number of draws. The expanded cycle enters the
resume contract; runs without the flag keep the fixed cycle and an unchanged
contract.

The learning rate is constant by default. `--lr-decay-start S` (SFT only, with
optional `--lr-decay-end E`, default `--steps`) keeps the configured rate for
one-based optimizer updates 1..S without touching the optimizer, then uses
`learning_rate × (E − t + 1) / (E − S + 1)` for update t in S+1..E. The rate is
computed from the configured peak and the absolute update index, replacing any
rate restored on resume; update E runs at `learning_rate / (E − S + 1)`, never
zero. S and E are stored in the exact-match resume contract, `0 ≤ S < E` and
`--steps ≤ E` are required, and each metrics event records the `learning_rate`
the optimizer used. `--steps` stays a stop target, so a run can pause before
its first decayed update and resume with the same schedule:

```sh
... sft ... --steps 500 --lr-decay-start 500 --lr-decay-end 944
... sft ... --steps 944 --lr-decay-start 500 --lr-decay-end 944 --resume
```

Each microstep's gradient is added and evaluated immediately. A lazy running
sum would keep every microstep gradient alive until the update ends: about
26 MB each at rank 16, or roughly 0.8 GB at `--accumulate 32`.

### Language-mean token weighting

By default each microstep's loss is the mean NLL over its transcript tokens
(first token through EOS), so every clip counts equally. `--sft-token-weight
language-mean --sft-token-reference-data FULL` (SFT only) instead uses the
clip's summed token NLL divided by the mean token count of its language in
FULL's TRAIN. Within a language a long clip now counts more, while each
language's share stays close to the schedule. Accumulation, clipping and AdamW
are unchanged.

```sh
... sft ... --data SUBSET --sft-token-weight language-mean \
  --sft-token-reference-data FULL
```

FULL is validated like `--data` before the model loads (manifest, audio and
held-out checks). `--data` may be FULL itself or a subset of it, for example a
longest-clip memory preflight: every training row must equal a FULL row except
for its audio location, and both must share the development and protected
manifests. The denominators always come from all of FULL's TRAIN, never from
the subset. FULL's `train.jsonl` and `data.json` hashes and the mode are part of
the resume contract only when the flag is on; the default contract, math and
metrics are unchanged.

After loading, the runtime tokenizer counts each FULL TRAIN transcript plus EOS,
exactly as a microstep supervises it. `token-weight.json` holds the per-language
recordings, tokens and full-precision means. It is written once; on resume the
recount must match it exactly, and a missing file is refused once an update is
published. Each metrics event keeps `loss` as the unweighted mean of per-clip
mean NLL (comparable with default runs) and adds per microstep
`sequence_losses`, `tokens`, `token_normalizers` and `objectives` (sum NLL /
language mean), plus `objective`, their mean, whose gradient was applied before
clipping. Raw loss and objective come from the same forward pass.

Token weighting only approximates the scorer, which counts words or
characters, not BPE tokens. Per-update gradient size now varies; AdamW and
clipping are not invariant to that, so the effective step size can change.

### Looped decoder layers

`--loop START:END:PASSES` runs decoder layers START..END-1 PASSES times in a
row, then continues with END..27. `--loop 7:21:3` turns the 28-layer decoder
into 0–20, 7–20, 7–20, 21–27 (56 layers), the same pattern as stacking a
model's middle blocks to grow it. Repeats are fresh modules over the same
frozen base arrays, so the 8-bit weights stay in memory once; each pass gets
its own rank-`--rank` LoRA on q/k/v/o, and each pass gets its own KV cache
during decoding. Repeats are not gated: the untrained looped model already
differs from the base model, and LoRA has to adapt it.

The loop is part of the resume contract and of every checkpoint; an adapter
loads only into the same loop (`evaluate`, `probe`, external probes and GRPO
need the same flag). Without the flag the model, contract and metrics are
unchanged. Activations and decoding time grow with the added layers, so run
the longest-clip memory preflight for each loop before a full trial.

### Decoder MLP LoRA

`--mlp` also puts a rank-`--rank` LoRA on each decoder layer's MLP
`gate_proj`, `up_proj` and `down_proj`, next to the q/k/v/o adapters. Attention
adapters are created first, so they start from the same values as without the
flag. The base weights stay frozen. Like `--loop`, the flag is part of the
resume contract, and an adapter loads only with the same setting. Without it
the model, contract and metrics are unchanged. It needs more memory than
attention-only LoRA, so run the longest-clip memory preflight first.

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
the fixed starting policy, and one update per group at LR 1e-6. Reference and
current policies share frozen weights; only reference adapters are copied.
There is no critic, external judge, paid API, or independent rollout server.
Gradients are averaged across group members and globally clipped at 1.0.

`--advantage-scale S` (GRPO only, positive and finite) replaces the group
standard deviation with a fixed divisor: each advantage is `(reward - group
mean) / S`, with no epsilon. Groups therefore push in proportion to their
reward spread instead of each being rescaled to unit standard deviation.
Omitted, the default `(reward - mean) / (std + 1e-6)` is unchanged. The value
is stored in the exact-match resume contract and is unrelated to the LoRA
`--scale`.

`--max-draws N` (GRPO only, default 1) redraws a group whose rewards are all
equal, since it carries no policy advantage. Each step keeps its scheduled
language and its single MLX seed. The first draw is the default row; redraws
pick another row of that language with replacement, from a per-attempt seeded
generator. The step stops at the first group whose rewards differ, or at the
Nth group, which is then kept even if its rewards are equal (`exhausted`).
Either way the step makes exactly one normal update, on that stopping group
only. Metrics add `draws`, `exhausted`, and `skipped` (ID and rewards of each
discarded group; no completions). Each redraw costs another full rollout, and
redraws change which clips provide policy gradients, so this is an
experimental recipe with no accuracy claim. The value is stored in the
exact-match resume contract; the default reproduces the previous rows,
rollouts and updates.

`--reference-member` (GRPO only, requires `--max-draws 1`) adds the training
row's own transcript as an additional group member. Only the TRAIN transcript of the
sampled row is used, never development or test text. With the default group size, the four policy samples,
their row, seed and rewards are unchanged; the transcript is tokenized like
SFT targets (one EOS) and scored with the unchanged reward, so it gets 1.
Advantages are standardized and gradients averaged over every member (five
with the default group size). The transcript's behaviour probability is taken as one, so its
ratio is its current token probability at the training temperature: each transcript token is pushed in
proportion to how likely the policy already finds it. Its advantage is never
negative, so the clip cannot bind; the same KL penalty applies to its tokens.
All-perfect groups still give zero advantage. Metrics keep the four sampled
`rewards`/`completions`; `advantages` has five entries (transcript last), plus
`reference_reward` and the transcript's mean token probability before the
update (`reference_mean_probability`). This mixes a supervised term into the
RL objective. It follows the mixed-policy GRPO idea of MP-GRPO and LUFFY
without their exact losses or policy shaping, and carries no accuracy claim.
The flag is stored in the exact-match resume contract.

Use `--base-reference` to start directly from the untouched ASR base. Zero-delta
LoRA initialization preserves that base as the starting policy and fixed KL
reference. This flag is valid for GRPO, is mutually exclusive with
`--init-adapter`, and is stored in the exact-match resume contract.

```sh
.venv/bin/python training/qwen_asr/train.py grpo \
  --model "$HOME/models/stt/qwen-base-8bit" \
  --data "$HOME/models/stt/qwen-data" \
  --protected-test "$HOME/models/stt/fleurs-test/manifest.jsonl" \
  --base-reference --korean-wer-weight 0.5 --skip-initial-eval \
  --output "$HOME/models/stt/qwen-grpo-base" --steps 1000 --eval-clips 100000
```

The frozen base evaluation remains the selection anchor. The controller's
existing profiles continue to choose checkpoint references; select this mode
explicitly for a new trial in the serialized training slot.

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

An accepted resume marks `run.json` as running before model startup and clears
its previous finish time and process peak. The checkpoint contract stays fixed.

On resume, `latest.json` selects the published checkpoint. A run interrupted
before its first checkpoint restarts from the same seeded initialization.
Unpublished checkpoint directories and metrics beyond the saved step are
preserved under `interrupted/` before replay, so a failed save can be retried.

`metrics.jsonl` records losses, gradient norms, time, MLX allocation and GRPO
reward/advantage/completions. Initial/final `dev-*.json` record greedy transcripts
and supervised losses; by default four clips per language. This is a plumbing
check, not evidence of a general accuracy gain. Use a larger independent dev
set for selection, then run the untouched 2,624-clip FLEURS test once for each
accepted final candidate with the shared scorer. Keep English regressions
visible. Do not merge/quantize/deploy adapters on the strength of training loss.

`selection.py CANDIDATE INCUMBENT ANCHOR` gates one evaluation against another.
Korean is scored by space-normalized WER (KsponSpeech's sWER): the hypothesis
takes the reference's word breaks, so spacing alone never counts as an error.
`--glosses` scores FLEURS translator glosses as optional spans.
`--second-seeds CANDIDATE2 INCUMBENT2` compares two recipes instead: each
recipe's two seeds pool their errors per clip, and every gate, including the
paired bootstrap, runs on the pooled errors.

### External probe

`evaluate --external PROBE.jsonl` transcribes a frozen external manifest instead
of development. It never trains, never runs the development evaluation and
never changes the primary FLEURS score. Pass the checkpoint's own training data as `--data`; the
run refuses a checkpoint trained on other data.

```sh
.venv/bin/python training/qwen_asr/train.py evaluate \
  --model "$HOME/models/stt/qwen-base-8bit" --data TRAINING_DATA \
  --protected-test "$HOME/models/stt/fleurs-test/manifest.jsonl" \
  --external PROBE/probe.jsonl \
  --eval-clips 2 --memory-gb 4.5 --output RUN
```

Add `--init-adapter CHECKPOINT` to evaluate a checkpoint.

`evaluate --audio-rate 8000` makes each clip sound like phone audio before
the model hears it. Every clip is downsampled to 8 kHz with an anti-aliasing
filter, then upsampled back to 16 kHz. This works on development and on any
`--external` probe, and the rate is recorded in the run contract. Other
stages refuse the flag.

Before loading the model it checks the manifest against `probe.json`'s
`probe_sha256`, and each WAV's hash, 16 kHz mono PCM16 format and declared
duration. It also rejects any shared clip, text or speaker with `--data` and
the frozen test. The model gets audio and the known language only. Decoding
matches development: greedy, `--max-tokens`, 128 MiB cache.

A greedy decode that reaches `--max-tokens` without EOS is a repetition loop.
Trained adapters loop about once in 2,400 real-use clips, while the base model
has not looped on any probe. Development and external evaluation therefore
redecode such a clip with the adapter switched off (LoRA scale 0). The row
keeps the base transcript as `hypothesis`, and the looped text as `looped`.
GRPO rollouts and the sampling probe keep raw decodes.

`--eval-clips` counts clips per cohort. Every model gets the same seeded subset,
and a larger count keeps the smaller one's clips. Each clip is appended to
`predictions.jsonl` with the run's contract hash as soon as it finishes. Repeat
with `--resume` and a larger `--eval-clips` to continue: saved clips are not
decoded again. A changed model, checkpoint, code, data, probe or setting is
refused, as are foreign or duplicate rows. `summary-N.json` (N clips) reports
WER and CER per language and cohort using the shared scorer, plus empty and
truncated output counts. A finished summary is never overwritten. Cohorts may
be deliberately balanced, so never pool them into one accuracy.

### TRAIN sampling probe

`probe` measures how far one adapter's sampled completions differ from its
greedy completion on its own TRAIN data. It never trains, never computes a
loss, writes no checkpoint and decodes no development, KSS or test clip.

```sh
.venv/bin/python training/qwen_asr/train.py probe \
  --model "$HOME/models/stt/qwen-base-8bit" --data TRAINING_DATA \
  --protected-test "$HOME/models/stt/fleurs-test/manifest.jsonl" \
  --init-adapter CHECKPOINT --memory-gb 4.5 --probe-limit 2 --output RUN
```

The recipe is fixed in `sampling_probe.py` and bound to the resume contract:
the first 200 source sentences per language of `sentence_epoch_order` with
seed 20261030 (one seeded recording each), interleaved KO/JA/ZH/EN; one greedy
and eight temperature-0.7 draws per clip, through the trainer's own `context()`
and `sample()` with `--max-tokens` 256. Each draw sets its own MLX seed from the
clip ID and draw number. Training and evaluation flags are refused, as is an
adapter trained on other data. `probe-plan.json` freezes all 800 clips with
audio and reference hashes before the model loads.

The model gets audio and the known language only. After decoding, the TRAIN
reference scores the text with the frozen reward (Korean WER weight 0.5); a
completion without EOS scores -1, as in GRPO. Every decode is appended to
`decodes.jsonl` and fsynced, with its actual token IDs including EOS, text,
seed, temperature, time, reward, contract hash and a record digest.

`--probe-limit N` stops after the first N plan clips per language and writes
`probe-progress.json`, which holds counts only. Repeat with `--resume` and a
larger or no limit to extend: saved decodes are never repeated. A changed
model, adapter, data, code, plan or setting is refused before loading, as are
edited, duplicate, foreign or corrupt records; only a torn final line is
dropped. `probe-summary.json` is written once, after all 7,200 decodes, and is
never overwritten.

The summary reports, per language: mean greedy and sampled reward, the
reference-ranked best of eight, its positive headroom over greedy, how many
greedy errors have a better or a perfect sample, empty/truncated counts and
plan coverage. For an SFT adapter it also counts plan clips among that run's
own draws, recomputed from its recorded sampler settings. Best-of-eight uses
the TRAIN reference: it is an optimistic TRAIN diagnostic, not deployable
decoding, held-out accuracy or a ceiling for RL.

### TRAIN NLL probe

`evaluate --train-nll-probe PLAN` reads out one SFT adapter's teacher-forced
NLL on frozen TRAIN clips. It uses the trainer's own loader, `context()` and
`log_probs()` at temperature 1, and never decodes, computes a gradient, updates
weights, writes a checkpoint or touches development, KSS or test clips.

```sh
.venv/bin/python training/qwen_asr/train.py evaluate \
  --model "$HOME/models/stt/qwen-base-8bit" --data TRAINING_DATA \
  --protected-test "$HOME/models/stt/fleurs-test/manifest.jsonl" \
  --init-adapter CHECKPOINT --train-nll-probe PLAN --memory-gb 4.5 --output RUN
.venv/bin/python training/qwen_asr/token_weight.py compare \
  CANDIDATE_RUN/train-nll-summary.json TWIN_RUN/train-nll-summary.json --output RESULT
```

PLAN holds per-seed clip IDs and their `ids_sha256`. Before loading, the probe
requires a fresh sentence-epoch SFT checkpoint trained on `--data`, takes the
entry for the checkpoint's seed, and replays that run's draws from its recorded
seed, step and accumulate. Each clip must be a TRAIN row of its language with a
valid 16 kHz WAV of its recorded duration, and its sentence must never have
been drawn in that language. The same FLEURS sentence may have been drawn in
another language; that is recorded per clip as `drawn_in_other_languages`, so
the clips are unseen in their language, not globally unseen. Training and
decoding flags are refused.

Each clip's summed NLL over its reference tokens plus EOS is appended to
`train-nll.jsonl` with its token count, plan fields, contract hash and a record
digest. Resume, refusal and torn-line handling match the sampling probe.
`train-nll-summary.json` is written once and reports token-pooled NLL per
language (sum NLL / sum tokens), every clip record and the full contract.
Empty languages report `null`.

`compare` pairs a candidate with its same-seed twin. The two summaries must
have identical contracts except the adapter, including the memory setting,
and the same clips. Per language,
and for the shortest and longest ZH thirds by tokens, it reports
candidate-minus-twin token-pooled NLL with a 400-resample paired clip bootstrap
(seed 20261030). ZH is primary: CI95 upper < 0 is `supported`, a point
difference >= 0 `falsified`, otherwise `inconclusive`. EN is diagnostic; other
languages, often tiny, are report-only. The result is a TRAIN-only mechanism
check for one seed; it never gates adoption, and an existing result file is
never replaced.

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
original starter recipe. The fixed starting reference may be an existing
SFT/GRPO checkpoint or the untouched ASR base. This is a new experimental
recipe, not a claim that the longer RL stage improves quality.

Freeze a source archive for each running controller; do not edit its code in
place. `loop-state.json` and each trial's `verdict.json` are runtime evidence.
The harness records scored trials and accepted state in the repo's LOG history,
STATUS and PLANS. Creating `ROOT/STOP` terminates the owned child and stops new
trials; preserve checkpoints/data and do not restore serving automatically.
