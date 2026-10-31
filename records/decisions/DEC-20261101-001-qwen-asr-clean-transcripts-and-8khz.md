# DEC-20261101-001: Target clean transcripts and test every set at 8 kHz for Qwen3-ASR

Opened: 2026-11-01 02-20-00 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: on 2026-11-01 the operator chose clean transcripts over verbatim
  ones. They also asked that the model be explicitly tested at 8 kHz across
  all evaluation sets.
- Related ids: DEC-20261029-001, DEC-20261031-001, RSH-20261101-001

## Decision

1. **Clean transcripts.** Spoken fillers and discourse words are not part of
   the target transcript.
   - Real-use probes treat 어 음 으 엄 흠 에 아 그 이제 인제 뭐 막 저기 네 예 자
     as optional on both sides. They are dropped from both reference and
     hypothesis before sWER (`proposals/real-use-ko-20261101/score.py`).
   - Training targets built from verbatim corpora drop the same words, plus
     every filler the annotators marked with `~`.
2. **8 kHz on every set.** Each evaluation set is also scored at telephone
   band with `train.py evaluate --audio-rate 8000` (PR #42). This covers
   FLEURS validation in all four languages, the KSS probe and every real-use
   probe. Each clip is anti-aliased down to 8 kHz and back up to 16 kHz.
   The 8 kHz result is reported next to the 16 kHz result. Its selection gates
   are set once base and incumbent 8 kHz baselines exist.

The primary FLEURS selection rule, the anchor, the FLEURS test set,
published predictions and `stt-score.py` are unchanged.

## Context

The operator's real use is meetings and calls, videos to subtitle, and their
own voice. On the Korean real-use probe, the FLEURS incumbent lost to base
mostly by dropping discourse words that the verbatim references keep. Under
clean scoring, broadcast ties (+0.0%) and meeting ties (+3.1%), but calls stay
worse: 26.8% vs 24.3%, CI95 [+5.6%, +14.8%]. Calls are 8 kHz phone audio and
the weakest cohort.

## Options Considered

- **Verbatim:** keeps every spoken word. It suits audits but reads worse for
  the operator's subtitles and notes.
- **Clean (chosen):** neither writing nor omitting a discourse word costs
  errors, and training aims at the readable form.
- **Test 8 kHz only on phone-call sets:** misses regressions in languages that
  have no call data yet. The operator asked for every set.

## Consequences

- Real-use scores use the clean rule. Earlier verbatim numbers in
  RSH-20261101-001 stay as recorded.
- Adoption decisions must weigh 8 kHz as well as 16 kHz once the 8 kHz gates
  are set.
- The next training run targets phone calls (AIHub 571 TRAIN) with clean
  targets.
