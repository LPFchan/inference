# RSH-20261030-001: Confirmed Qwen3-ASR one-pass SFT

Opened: 2026-10-30 02-41-57 KST
Recorded by agent: codex-qwen-asr-training

## Finding

A preregistered one-sentence-epoch LoRA adaptation of Qwen3-ASR-1.7B
passes the unchanged full development screen on two independent seeds,
and both seeds pass the supplementary Korean KSS veto. The lower-score
first checkpoint is the training-selection incumbent. Serving remains unloaded.

The normalized CJK error score improves 11.61% on the selected seed and
8.42% on confirmation, against the original MLX 8-bit anchor. This is a
confirmed gain with the existing model and rank-16 adapter.

## Method and frozen inputs

- Host: M4 Pro, 24 GiB, macOS 15.7.1. Measured environment: Python 3.12.13,
  MLX 0.32.4, mlx-lm 0.31.3, mlx-audio 0.6.2, jiwer 4.0.0.
- Base: `mlx-community/Qwen3-ASR-1.7B-8bit` revision
  `a8379a2e2f9e313c9292cdf1af4055ab56d50d55`.
- Actual training source: frozen `code/1412c1f`; rank 16, scale 1, LR 3e-5,
  944 optimizer updates with four serial microsteps per update. The base
  remains frozen; only decoder q/k/v/o LoRA parameters train.
- Data v005: 8,966 usable FLEURS TRAIN recordings / 25.14 h. A seeded
  sentence schedule chooses one recording per language/source-sentence,
  with 3,776 unique actual draws: KO/JA/ZH 1,133 each, EN 377. Consumed
  audio is 11.039 h and 11.068 h for seeds 21774173 and 21775174.
- The balanced schedule exhausts Japanese; other language pools retain
  unused sentences. Both actual draw hashes match their preregistrations.
- Primary selection: all 1,295 frozen FLEURS validation clips (EN 394,
  KO 226, JA 266, ZH 409). Manifest SHA256
  `c75a4b05069eba7d11a6b50efd7bb11d43800fc66e49b761cd08fb86354acf01`.
  Data snapshot SHA256 is
  `0fc40070c7545cfccd0eecdf03163f32f59aa7e98dc82182590aa48bdcb2064e`.
- Shared normalization/scorer and selection gates are unchanged: primary
  WER EN/KO, CER JA/ZH; mean CJK error ratios to the original anchor,
  >1% gain, paired 400-resample CI95 upper<0, CJK regression limits,
  English limit and empty/truncation vetoes. See DEC-20261029-001/002.
- The original 2,624-clip test and nine published predictions remain sealed;
  preparation uses exclusion checks only. KSS is evaluation-only.

## Full validation results

Percent; lower is better. Score is dimensionless and uses the fixed anchor.

| Checkpoint | EN WER | KO WER | JA CER | ZH CER | CJK score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 8-bit anchor | 4.2533 | 10.8509 | 4.6178 | 6.6060 | 1.0000 |
| Seed21774173, selected | 3.9889 | 10.5840 | 4.2902 | 4.9358 | 0.883879 |
| Seed21775174, confirmation | 4.0610 | 10.5544 | 4.5320 | 5.2414 | 0.915840 |

Paired score-difference CI95 against anchor: first[-0.174010,-0.051078],
confirmation[-0.137011,-0.034759]. Both pass every primary gate, with
zero empty or truncated outputs. Full-metrics JSONs retain CER for all
four languages and all word/character edit counts. No per-language timing
was inferred from whole-run elapsed time.

## Supplementary Korean veto

KSS has one voice and deliberately oversamples digit clips. Each cohort
has 256 clips; there is no pooled accuracy estimate. Same numerical source
`code/f119906`, source inputs, seed and settings as the retained base run;
only the adapter hash differs. Both WER and CER must be <=base+0.5 pp,
with no extra empty/truncated outputs, separately for both cohorts and seeds.

| Cohort | Model | WER | CER |
| --- | --- | ---: | ---: |
| Digit | Anchor | 32.71% | 11.88% |
| Digit | Selected SFT | 11.05% | 3.58% |
| Digit | Confirmation SFT | 12.84% | 4.16% |
| Other | Anchor | 8.08% | 1.24% |
| Other | Selected SFT | 7.20% | 1.35% |
| Other | Confirmation SFT | 7.04% | 1.24% |

Both seeds pass. Selected-seed other-cohort CER rises 0.1178 pp, within
the veto. All outputs end normally and none are empty.

## Interpretation and limits

The 5,000-update written-form control failed the primary screen (score 1.0390),
while this prospectively registered shorter recipe confirms a gain. Sampling,
recording choice, order, acoustic coverage and update count change together;
the comparison does not isolate a causal effect of repetition. Earlier TRAIN
loss and recording analyses are observational.

FLEURS speaker separation and pretraining exposure are unestablished. KSS
adds a supplementary Korean check, with one speaker and an artificial cohort
mix. These results do not establish a general acoustic-capacity improvement
or a final-test result. The existing 1.7B model supports useful adaptation;
capacity saturation has not been established.

## Reproducibility and next route

Program root: `~/models/stt/qwen3-asr-hillclimb-20261029` on the Mac and
local mirror; Mangchi backup is under `~/models/stt-benchmark/`.

- Runs `parallel/sft-ce33a3d7a476e297` and `parallel/sft-a11b015fbeee0082`
  retain full validation, metrics, training-integrity, contracts, final
  adapters and optimizer states. Their KSS runs have the `kss-v001-` prefix.
- Selected adapter SHA256:
  `ad63fb3868f281315797b6074a8b3f9dc6245b71d496b4ee433ae8e571896e22`.
  Confirmation:
  `a5dc030d9717e50e0e114edb18bd664da9ada31187e4b8589df9312530160d80`.
- `parallel/sft-ce33a3d7a476e297/adoption.json` binds both independent
  validations and guards. `parallel-trials.json.incumbent` is authoritative
  for new trials; the legacy driver's state is retained unchanged.
- Complete evidence is independently hash-verified across the Mac, local
  mirror and Mangchi. Serving and final-test predictions are unchanged.
- Accepted continuation `grpo-76cf17e6620b34fb` trains the registered GRPO
  recipe from the selected SFT adapter and uses it as fixed KL reference,
  with a fresh optimizer. Its own two-seed parent-relative and current-
  incumbent screens, KSS vetoes and final incumbent recheck are required.
