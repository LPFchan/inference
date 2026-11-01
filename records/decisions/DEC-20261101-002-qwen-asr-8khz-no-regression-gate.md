# DEC-20261101-002: 8 kHz no-regression gate for Qwen3-ASR

Opened: 2026-11-01 10-49-18 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: on 2026-11-01 the operator agreed that a new model fails if it
  is clearly worse than the current one at 8 kHz on any test set.
- Related ids: DEC-20261101-001, RSH-20261101-002

## Decision

A candidate cannot be adopted if, at 8 kHz, it is clearly worse than the
incumbent on any evaluation set.

- **Where it is checked:** FLEURS validation per language, the KSS probe per
  cohort, and each real-use probe per cohort, all scored with
  `evaluate --audio-rate 8000`.
- **Clearly worse:** the paired-bootstrap CI95 of the relative error change
  lies entirely above zero. The bootstrap uses 2,000 draws with seed 20261101,
  and real-use probes use the clean rule.
- **Unchanged:** the 16 kHz selection rules, the FLEURS anchor and the sealed
  test set.

## Context

RSH-20261101-002 measured base and incumbent at 8 kHz. The band costs little
except in Japanese, and the incumbent is less robust than base on real-use
Korean. DEC-20261101-001 left the 8 kHz gate open until those baselines
existed.

## Options Considered

- **No-regression gate (chosen):** 8 kHz can veto a candidate but does not
  pick winners. Wins are still judged on the existing scores.
- **Add 8 kHz to the selection score:** this mixes two goals into one number
  and moves the anchor.
- **A fixed margin, such as +5%:** an arbitrary threshold, and blind to how
  noisy each set is.

## Consequences

- Every candidate gets 8 kHz runs on every set before adoption.
- Checking nine languages and cohorts separately raises the chance that one
  of them flags by luck. If that happens often, the operator can revisit the
  gate.
