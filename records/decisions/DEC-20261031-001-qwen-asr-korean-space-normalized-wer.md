# DEC-20261031-001: Score Korean by space-normalized WER for Qwen3-ASR selection

Opened: 2026-10-31 16-04-00 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: the operator made Korean spacing optional on 2026-10-31 after a
  check showed spacing alone was about half of the incumbent's Korean WER.
- Related ids: DEC-20261029-001, DEC-20261030-002, DEC-20261030-003

## Decision

Qwen3-ASR selection scores Korean by space-normalized WER (sWER, from the
KsponSpeech paper, Bang et al. 2020) in `selection.py`, frozen as
`code/d0626bb` (PR #41; sha256 `0b3c31d508be8826…`). The command is unchanged:
`selection.py --glosses calibration/glosses-v1.json [--second-seeds …]`.

- Characters align with spaces removed. Hypothesis characters aligned to the
  reference take its word breaks, inserted characters keep the hypothesis's,
  and a word that loses its first characters still starts at its first
  surviving one. Ordinary WER runs on the result.
- Number forms such as `2` vs `이` still count, as in KsponSpeech. English,
  Japanese and Chinese scoring is unchanged. Glosses stay optional; each
  reference variant respaces against its own spacing.
- Gates, anchor, the 1,295-clip set and the two-seed rule are unchanged. The
  FLEURS test set, published predictions and `stt-score.py` stay sealed.

## Context

Word scoring charges a whole Korean word for a spacing difference such as
슈퍼대회전 vs 슈퍼 대회전, though both are usually correct. With spacing
forgiven, the incumbent's Korean errors fell from 9.59% to about 4.7%.
Scoring numbers against the FLEURS digit-style raw transcripts changed at most
0.1 pp, so number forms stay scored.

Under the rule (`calibration/korean-swer-rescore.json`, sha256 `74f9780f…`),
the anchor's Korean falls from 10.25% to 5.15%. The gloss-free pair pools to
0.8458 vs the old one-pass pair's 0.9098, CI95 [-0.1218,-0.0111], and still
passes every gate.

## Options Considered

- **Keep word WER:** keeps rewarding FLEURS spacing habits, which carried
  nearly all of the Korean gain over the base model.
- **Korean CER:** also ignores spacing, but switches the unit from words to
  syllables, a second change at once.
- **sWER (chosen):** removes only spacing, using the published Korean metric
  for this problem.

## Consequences

Scores under this rule are not comparable with earlier scores. Korean now
barely separates the models: the incumbent seeds score 4.86% and 5.04% vs the
anchor's 5.15%. The two gloss-free seeds now differ by 0.021 (0.8354 vs
0.8562), so by DEC-20261030-003 the lower one, `sft-gloss-free` (adapter
`1a1baedd…9d99`), becomes the checkpoint; both passed KSS. The screen
threshold becomes 0.8458. Three first seeds fall below it:

- Cap-eight GRPO (0.8345): it earns a second seed, not yet run.
- Single-draw GRPO (0.8380): its second seed already ran, the pair pools to
  0.8736, and it stays closed.
- GRPO on gloss-free (0.8405): its preregistered second seed runs.

MLP LoRA (0.8470) stays closed.
