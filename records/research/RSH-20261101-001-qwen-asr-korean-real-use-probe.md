# RSH-20261101-001: Qwen3-ASR on Korean real-use speech

Opened: 2026-11-01 01-50-00 KST
Recorded by agent: claude-qwen-asr-training

## Question

The operator's real use is meetings and calls, videos to subtitle, and their
own voice, in KO/JA/ZH/EN. Does the FLEURS-selected incumbent help on that
kind of speech? Which area is weakest?

## Probe

`proposals/real-use-ko-20261101/probe-v001`: 1,200 Korean clips, 1.93 h,
frozen before any model output. Manifest sha256 `72a756e7…4b65b`, builder
`build_probe.py` sha256 `780ce278…77dc`. The source is AIHub validation splits
(account and terms: AI training/research only, no redistribution, no transfer
abroad). Each cohort has 400 clips:

| Cohort | Source | Notes |
| --- | --- | --- |
| call | 571 저음질 전화망 | 8 kHz phone audio, upsampled to 16 kHz |
| broadcast | 71314 방송콘텐츠 대화체 | 토크쇼 149, 예능 251 |
| meeting | 464 주요 영역별 회의 | 기타녹음 only, so recorded meetings rather than TV/radio debates |

Each clip is 1–15 s with at least 3 words. There are at most 2 clips per call,
episode or meeting, and 350–384 speakers per cohort. Clips are dropped if
another speaker overlaps, if a name is masked (`@`), if anything is marked
unclear, or if the text repeats. Dual transcriptions `(A)/(B)` resolve to the
written form; the spelled side comes first in 571 and second in 71314/464.
Noise tags and the `~` of a lengthened filler are stripped.

`score.py` (sha256 `f969a10d…3212`) scores Korean sWER with the frozen
`code/d0626bb` scorer. The pure interjections 어 음 으 엄 흠 에 아 are dropped
from both reference and hypothesis.

## Findings

Both runs used `evaluate --external` with code `d0626bb` and data
`data-v005-gloss-free`. Neither had empty or truncated outputs. The verbatim
sWER, with paired-bootstrap CI95 for best vs base:

| Cohort | Base | Incumbent | Change |
| --- | --- | --- | --- |
| call | 24.4% | 28.3% | +15.8% [+11.0, +21.0] |
| broadcast | 20.8% | 22.1% | +6.3% [-0.1, +13.0] |
| meeting | 19.6% | 23.2% | +18.3% [+11.5, +25.2] |
| all | 21.4% | 24.3% | +13.8% [+10.3, +17.6] |

- **The incumbent is worse than base on real-use Korean.** FLEURS gains do
  not transfer.
- **Most of the gap is transcript style.** The incumbent's deletions are 3.3×
  base's (766 vs 230), while substitutions are flat. 427 of its deletions are
  discourse words: 그 141, 뭐 76, 네 54, 이제 51, 예 36. FLEURS taught it to
  write clean read-speech text, while these references are verbatim.
- **Calls are a real regression.** With common discourse words also optional
  on both sides, broadcast ties (-1.4% [-6.9, +4.7]) and meeting ties
  (+2.2% [-2.9, +8.1]), but call stays worse (+9.0% [+4.8, +13.6]).
  Training on clean 16 kHz read speech hurt 8 kHz phone audio.
- **Calls are the weakest area.** The base's call sWER of about 24.6% is the
  highest of the three cohorts.
- Real-use Korean errors (21% sWER, 13% CER) are about twice FLEURS Korean
  (10%), so the headroom is here rather than in FLEURS validation.

## Nemotron on this probe (2026-11-02)

The model behind `transcribe.lost.plus`, Nemotron 3.5 Transcribe 30B-A4B
MLX INT4 (artifact `7018b1c1…`), transcribed all 1,200 clips through its
loopback server with language `ko`. The 8 kHz pass uses the same band limit
as `evaluate --audio-rate 8000`. Scoring is clean and number-blind sWER
(DEC-20261101-001, DEC-20261101-003). Evidence is in
`proposals/nemotron-realuse-20261102` (`score_nemotron.py` sha256
`85d031ae…`, `compare_nemotron.py` `40099e18…`, `compare-v001.json`).

| Cohort | Calls-model incumbent | Base Qwen | Nemotron | Nemotron vs incumbent |
| --- | --- | --- | --- | --- |
| call | 19.6% | 23.6% | 28.6% | +45.7% [+34.6, +57.6] |
| broadcast | 19.0% | 19.5% | 22.3% | +17.3% [+9.0, +26.3] |
| meeting | 16.7% | 17.6% | 17.1% | +2.3% [-4.6, +9.9] |
| all | 18.2% | 19.9% | 22.0% | +20.5% [+15.6, +25.8] |

- **Nemotron is clearly worse than both Qwen models on Korean real-use
  speech.** It is worse than base Qwen too (+10.4% [+6.5, +14.6]), and it
  ties only on meetings.
- **Calls are its weakest area.** Its mistakes are acoustic, like 문자 2통 →
  문득 두통, and once an English sentence. They are spread out: the 20
  worst clips explain 22% of the gap. It had no empty outputs.
- **At 8 kHz the gap widens** to +28.9% [+23.1, +34.8] overall and +56.4% on
  calls.
- MAI-Transcribe 2 was not run. The AIHub terms forbid sending this audio
  abroad.

## Next

- Product decision for the operator: should transcripts be verbatim (keep
  그/이제/뭐) or clean? That choice fixes the scoring rule and the training
  targets.
- Train on AIHub training splits aimed at the weakest area, phone calls (571
  TRAIN), and select on this probe. The probe covers Korean only; JA/ZH/EN
  real-use probes are still to build.
