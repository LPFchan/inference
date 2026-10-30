# RSH-20261030-003: Looped decoder layers do not help Qwen3-ASR LoRA SFT

Opened: 2026-10-30 14-39-23 KST
Recorded by agent: claude-qwen-asr-training

## Question

Does adding depth to Qwen3-ASR-1.7B's 28-layer text decoder beat the
28-layer incumbent under the same one-pass SFT recipe? The operator suspected
that model size limits further gains
([DEC-20261030-001](../decisions/DEC-20261030-001-qwen-asr-looped-decoder-arm.md)).

## Method

`train.py --loop START:END:PASSES` runs a block of decoder layers several
times in a row. Repeats share the frozen 8-bit base weights, so the model
still stores 2.04B parameters. Each pass gets its own rank-16 q/k/v/o LoRA,
and repeats are not gated. Every level reruns the incumbent's exact recipe:
v005 FLEURS TRAIN, seed 21774173, 944 updates of four sentence-epoch
microsteps, learning rate 3e-5. All four were scored with `selection.py`
on the frozen 1,295-clip validation set, using frozen `code/4b2a716`.

## Evidence

Percent errors (WER EN/KO, CER JA/ZH). The score is relative to the original
anchor, and lower is better. "Unrolled" counts each repeat as its own copy.

| Decoder | Loop | Unrolled | EN | KO | JA | ZH | Score | Delta CI95 vs incumbent |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 28 (incumbent) | none | 2.04B | 3.99 | 10.58 | 4.29 | 4.94 | 0.8839 | — |
| 32 | `12:16:2` | 2.24B | 4.24 | 11.09 | 4.66 | 5.02 | 0.9308 | [+0.0140,+0.0799] |
| 36 | `10:18:2` | 2.44B | 4.10 | 11.15 | 4.43 | 5.45 | 0.9374 | [+0.0061,+0.1020] |
| 42 | `7:21:2` | 2.74B | 4.17 | 11.62 | 4.78 | 5.12 | 0.9605 | [+0.0143,+0.1232] |
| 56 | `7:21:3` | 3.45B | 4.43 | 12.01 | 5.37 | 5.17 | 1.0173 | [+0.0683,+0.2142] |

- Every level is significantly worse than the incumbent, and the score gets
  worse as depth grows. The 56-layer model scores worse than the untrained
  anchor (1.0000). No level produced empty or truncated outputs.
- Looping did not break the model before training. On the 40 longest TRAIN
  clips, first-update losses were 0.58 (28 layers) and 0.61/0.54/0.56/0.67
  (32/36/42/56 layers). After those ten updates, every looped model still
  transcribed the four preflight development clips normally.
- Longest-clip preflight memory: 4.37/4.53/4.66/4.88/5.11 GB. Full runs
  peaked at 5.02 GB (42 layers) and 5.17 GB (56 layers) under the operator's
  5.5 GB budget for those two levels.

## Conclusion

Under this recipe, extra decoder depth makes the adapted model worse, not
better. A likely reason: 944 small LoRA updates are enough to adapt the
original 28-layer path, but not to turn extra repeated passes into useful
computation, so each added pass is one more thing to adapt. This agrees
with the earlier evidence that the bottleneck is data, not capacity.
Qwen3-ASR-1.7B already beat the larger Whisper large-v3 and Voxtral 4B on all
four languages, and the 5,000-update control lowered TRAIN loss while its
validation score got worse.

## Limits

- Looping adds compute and adapter parameters, not new pretrained weights.
  These results do not test a genuinely larger pretrained ASR model, and
  they do not test growth followed by large-scale continued training.
- One seed per level and one recipe. Longer budgets, higher learning rates
  and zero-initialized gates were not tried; DEC-20261030-001 closes these
  variants rather than selecting them from validation errors.
