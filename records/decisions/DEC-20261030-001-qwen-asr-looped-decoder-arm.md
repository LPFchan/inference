# DEC-20261030-001: Looped-decoder architecture arm for Qwen3-ASR

Opened: 2026-10-30 11-26-32 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: the operator proposed this arm on 2026-10-30 after the recipe trials plateaued.
- Related ids: DEC-20261029-001, DEC-20261029-002, RSH-20261030-001

## Decision

Open one architecture arm in the four-slot loop. Each trial reruns the
incumbent's exact one-pass SFT recipe (v005 FLEURS TRAIN, seed 21774173,
944 updates of four sentence-epoch microsteps, learning rate 3e-5, rank 16)
with a block of the 28-layer decoder looped, through `train.py --loop`.

- Levels: `12:16:2` (32 layers), `10:18:2` (36), `7:21:2` (42) and `7:21:3`
  (56, the middle-block stacking pattern reported for Qwen3.5-27B). Their
  ten-update longest-clip preflights peak at 4.53, 4.66, 4.88 and 5.11 GB
  physical footprint (28 layers: 4.37 GB), all with normal pressure.
- The first two levels run under the standard 5 GB worker budget. The
  operator authorized a 5.5 GB worker budget for the last two levels only;
  the 20 GB total and every other launcher limit are unchanged.
- Repeats share the frozen 8-bit base weights and get their own q/k/v/o LoRA.
  They are not gated.
- Selection is unchanged: the full 1,295-clip validation against the
  incumbent with `selection.py`. A level that passes repeats with the
  incumbent's confirmation seed 21775174, then the KSS veto, before adoption.
- One run per level. A failing level closes; no level, rank, learning-rate,
  budget or seed variants are chosen from validation errors.

For this arm, this replaces DEC-20261029-001's rule that network expansion
waits for measured evidence: the operator judged the plateau sufficient.

## Context

All seven recipe trials scored since the incumbent's adoption lost. Two seeds of
the same recipe differ by 0.032 in score, and most later trials fall inside
that spread. The operator suspects that 1.7B parameters limit further gains.
A public weight comparison reports that Qwen3.5-27B was grown from Qwen3.5-9B
by stacking its middle layers three times and widening copied neurons.

The orchestrator expects a data limit instead. In the October benchmark,
Qwen3-ASR-1.7B beat Whisper large-v3 and the 4B Voxtral on all four
languages. The 5,000-update control on the incumbent's data kept lowering
TRAIN loss (0.126 to 0.094) while its score worsened to 1.039. This arm tests
the operator's hypothesis directly.

## Options Considered

- **More data only:** remains open (AliMeeting is audited), but does not test
  the capacity question.
- **Untied grown copies with full fine-tuning:** the faithful growth method,
  but full-model optimizer state does not fit the 5 GB worker budget.
- **Zero-gated repeats:** start identical to the base model, but under Adam
  at 3e-5 a gate moves at most about 0.03 in 944 updates, so the loop would
  stay almost switched off.
- **Ungated shared-weight repeats with per-pass LoRA (chosen):** fits the
  worker budget and keeps every other recipe setting fixed.

## Rationale

The chosen form changes one dimension, depth, under the incumbent's recipe and
the unchanged gates. Shared weights keep memory within the existing budget.

## Consequences

Looping adds depth, compute and adapter parameters, not new pretrained
weights. A loss closes deeper decoding under this recipe; it does not rule
out a larger pretrained model. Looped trials train and decode more slowly
than the incumbent, and their adapters need the same `--loop` to evaluate.
