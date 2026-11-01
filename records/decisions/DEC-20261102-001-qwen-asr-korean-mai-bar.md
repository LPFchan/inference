# DEC-20261102-001: MAI-Transcribe 2 is the Korean bar for Qwen3-ASR

Opened: 2026-11-02 04-18-59 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: on 2026-11-02 the operator set the bar ("our model should beat
  or tie with MAI-transcribe-2 in korean validation"). They chose the Seoul
  Corpus probe, plus an 8 kHz version of it, as the Korean set to measure it on.
- Related ids: DEC-20261101-004, RSH-20261020-002, RSH-20261101-001,
  RSH-20261102-001

## Decision

The Korean target for the Qwen3-ASR incumbent is to beat or tie
MAI-Transcribe 2.

- **Set:** `proposals/seoul-ko-20261102/probe-v001`. It has 800 interview
  clips: 20 from each of the 40 Seoul Corpus speakers (OpenSLR 113, CC BY-NC
  2.0), 2.15 h in total. Manifest sha256 `9039bda4…8a5c`. It is scored at
  16 kHz and band-limited to 8 kHz, and the bar must hold at both rates.
- **Score:** clean, number-blind sWER (the `gate2.py` loaders).
- **Tie:** the incumbent's change against MAI must not have a paired CI95
  entirely above zero. The bootstrap uses 2,000 draws with seed 20261101.
  `compare_mai.py` reports `bar_met` for each rate.
- **MAI settings:** the same as RSH-20261020-002: language `ko`, json,
  verbatim style, diarization off, one WAV per request. `score_mai.py`
  refuses any probe whose rows are not marked as cleared for hosted APIs.
- **First measurement** (RSH-20261102-001): not met. The incumbent's
  change against MAI is +17.8% [+13.3, +22.6] at 16 kHz and +19.8%
  [+15.5, +24.5] at 8 kHz.
- **Selection is unchanged.** DEC-20261101-004 still picks the incumbent on
  the AIHub probe. This bar decides when Korean is good enough; it does not
  decide between candidates.

## Context

The AIHub real-use probe is the main selection score. Its terms forbid
sending the audio abroad, so MAI cannot be scored on it. MAI led every
language on FLEURS (Korean WER 8.03% against base Qwen's 11.25%), but FLEURS
is read speech. The Seoul Corpus is unscripted interview speech, its licence
allows non-commercial evaluation, and it has no transfer restriction.

## Options Considered

- **Seoul Corpus, 16 and 8 kHz (chosen):** spontaneous speech that MAI may
  receive. It has no real phone calls; the 8 kHz copy only imitates the
  phone band.
- **The operator's own recordings:** the closest match to real use, but
  every transcript needs a hand check. Kept for later, for calls.
- **FLEURS or Zeroth:** ready now, but read aloud.

## Consequences

- MAI is a hosted, unversioned alias. Its outputs are a snapshot from
  2026-11-02. Re-scoring it costs about $0.25 per rate.
- The probe is for evaluation only and is excluded from every training pool.
- References are the corpus's standard-spelling tier. Its rare slips (for
  example 할 릴 for 할 일) count against every model alike.
