# DEC-20261029-001: Autonomous Qwen3-ASR multilingual hill-climb

Opened: 2026-10-29 07-27-35 KST
Recorded by agent: codex-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: operator requested more data, more training hours and autonomous hill-climbing; Nemotron need not be restored.
- Related ids: RSH-20261029-002

## Decision

Run an ongoing, single-worker Qwen3-ASR adaptation program on yeowoolmac's
24 GB M4 Pro, The Rest partition. Keep Nemotron unloaded. Continue supervised
adaptation and separate GRPO trials, expanding real-speech data and retaining
confirmed development improvements. The operator's no-subagent instruction
applies: this harness owns proposals, execution and selection.

The following numerical gates are implementation defaults chosen under that
authorization, not operator-specified product requirements:

- Freeze FLEURS **validation**, all four languages, as the primary selection
  set before scoring the new experiments. Its manifest currently has 1,295
  clips after exact test-text/audio exclusions. Never use the 2,624-clip frozen
  FLEURS test set for training, reward tuning, mining or checkpoint selection.
- Primary per-language errors remain KO WER, JA CER and Mandarin CER. The
  internal score is their mean ratio to the original MLX 8-bit anchor's errors.
  Lower is better; report actual per-language WER/CER alongside this dimensionless
  selection score. English WER/CER remain visible.
- A provisional winner must improve the score by more than 1% relative to the
  incumbent, have a paired 95% bootstrap delta interval below zero, keep each
  target language within 2% relative of the better of anchor/incumbent, and keep
  English WER within +0.5 percentage points of both. Empty/truncated outputs
  cannot increase. A tie is a loss.
- Repeat a winning training recipe once with an independent training seed
  before adopting it. Keep the better checkpoint only if both runs pass.
- One GPU process at a time. Start with 5,000-update SFT windows and
  1,000-update GRPO windows; use fixed budgets within each comparison. Increase
  budgets or expand adapter capacity as explicit new hypotheses, recording
  the changed compute. Do not equate a training-loss reduction with a win.
- Keep runtime pins, model inputs, data snapshots, code, seeds, optimizer state
  and measured memory/time. Default MLX allocation cap is 18 GiB; stop intake
  below 100 GiB free. No paid API or rented hardware is authorized by this loop.
- Do not deploy trained checkpoints, restore Nemotron, operate Grimoire, delete
  existing models/audio/results, or change global fleet instructions automatically.

## Context

The starter pipeline ran SFT and GRPO below 4.1 GB MLX allocation. Its 105-clip
development sample favored SFT, while the short GRPO continuation regressed
slightly. More real speech, broader validation and controlled longer trials are
needed before interpreting these results as a capacity limit.

## Options Considered

- Continue the tiny starter experiment alone: fast, with limited acoustic and
  lexical diversity and many equal-reward RL groups.
- Expand data and training under frozen selection gates: enables learning
  curves and repeatable comparisons while preserving the final test.
- Expand the network immediately: changes capacity before establishing a
  data/optimization constraint. Reserve this for measured evidence.

## Rationale

Data diversity and controlled training are the next useful constraints to test.
The single-worker procedure honors the operator's delegation limit. Confirmed
selection and an untouched final test reduce the chance of adopting noise or
formatting improvements that do not generalize.

## Consequences

The training slot remains available and serving stays paused. The loop has no
fixed round count; it continues until the operator stops it or an actual
resource/authority blocker prevents progress. Three losing trials trigger a
review of data quality and new hypothesis areas; do not rerun closed recipes
without changed evidence. Capture each scored trial with a LOG commit, keep
STATUS/PLANS current, and use RSH only for reusable findings. A heartbeat may
wake this existing thread with the single-word prompt `continue`.
