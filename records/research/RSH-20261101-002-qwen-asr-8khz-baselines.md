# RSH-20261101-002: Qwen3-ASR at 8 kHz telephone band

Opened: 2026-11-01 02-37-07 KST
Recorded by agent: claude-qwen-asr-training

## Question

DEC-20261101-001 asks for every evaluation set to be scored at 8 kHz. How much
does the telephone band cost each language and set, for base and for the
incumbent `sft-gloss-free/step-000944`?

## Method

`train.py evaluate --audio-rate 8000` with frozen code `33d4428`. Each clip is
anti-aliased down to 8 kHz and back up to 16 kHz. Clips, data and checkpoints
are the same as in the 16 kHz runs (`parallel/8k-*` vs the existing 16 kHz
runs). Changes are relative, with paired-bootstrap CI95 (2,000 draws, seed
20261101). Scripts and outputs are in `proposals/real-use-ko-20261101`:
`compare8k.py`, `boot8k.py`, `compare-8k-v001.json`, `boot-8k-v001.txt` and
`score-8k-v001-clean-{base,best}.json`. Hashes match on Mac/local/Mangchi.

## Findings

FLEURS validation, primary error in percent (glosses v1):

| Language | Base 16k → 8k | Change | Incumbent 16k → 8k | Change |
| --- | --- | --- | --- | --- |
| EN | 4.25 → 4.47 | +5.1% [-1.6, +12.6] | 3.90 → 4.01 | +2.8% [-2.8, +8.6] |
| KO | 5.15 → 5.36 | +4.0% [-1.5, +10.0] | 4.86 → 4.92 | +1.2% [-4.9, +7.9] |
| JA | 4.63 → 5.28 | +14.0% [+4.9, +24.1] | 4.24 → 4.93 | +16.2% [+5.9, +28.0] |
| ZH | 3.36 → 3.38 | +0.4% [-5.5, +6.4] | 2.18 → 2.22 | +1.6% [-5.2, +8.3] |
| Score vs anchor | 1.000 → 1.062 | | 0.835 → 0.892 | |

KSS probe, verbatim sWER (256 clips per cohort):

| Cohort | Base 16k → 8k | Change | Incumbent 16k → 8k | Change |
| --- | --- | --- | --- | --- |
| digit | 22.24 → 18.29 | -17.7% [-24.4, -11.0] | 6.38 → 5.52 | -13.5% [-28.6, +4.8] |
| other | 2.80 → 3.28 | +17.1% [0.0, +42.9] | 2.80 → 3.04 | +8.6% [-11.4, +35.5] |

Korean real-use probe, clean sWER (400 clips per cohort):

| Cohort | Base 16k → 8k | Change | Incumbent 16k → 8k | Change |
| --- | --- | --- | --- | --- |
| call | 24.34 → 24.28 | -0.2% [-1.9, +1.4] | 26.77 → 26.65 | -0.4% [-2.9, +1.7] |
| broadcast | 20.24 → 21.30 | +5.2% [-0.3, +10.5] | 20.24 → 21.66 | +7.0% [+2.2, +12.2] |
| meeting | 19.01 → 18.43 | -3.1% [-6.3, +0.2] | 19.59 → 21.39 | +9.2% [-0.2, +25.6] |
| all | 20.91 → 20.99 | +0.4% [-1.7, +2.4] | 21.83 → 22.96 | +5.2% [+1.3, +11.2] |

- **The telephone band costs little, except in Japanese.** Japanese FLEURS
  loses 14–16% for both models. EN, KO and ZH changes are inside noise.
- **Calls do not move.** That audio is already 8 kHz, so band-limiting it
  again changes nothing. This confirms the filter itself adds no damage.
- **On FLEURS the incumbent is as robust as base.** Both normalized scores
  rise by about 0.06, and the incumbent still leads at 8 kHz (0.892 vs 1.062).
- **On real-use speech the incumbent is less robust than base.** It loses
  +5.2% [+1.3, +11.2] overall at 8 kHz, while base holds (+0.4%). Broadcast
  carries a real loss. On meeting, one clip (`meeting/DGBEJ21000016.1.1.243`)
  falls into a repetition loop until the token limit and adds 64 errors.
  Without it, the incumbent's meeting change is +2.3% (19.62 → 20.07). Base
  never loops.
- **The KSS digit "gain" is number format, not hearing.** At 8 kHz, base
  writes Arabic numerals in 103 of 256 digit hypotheses instead of 59, and the
  references always use numerals.

## Next

- The operator sets the 8 kHz selection gates (DEC-20261101-001).
- Japanese is the language that 8 kHz hurts. It has no real-use or call probe
  yet.
- Phone-call training can add 8 kHz band-limiting as augmentation on 16 kHz
  data. That needs `--audio-rate` (or an equivalent) in the training stages.
