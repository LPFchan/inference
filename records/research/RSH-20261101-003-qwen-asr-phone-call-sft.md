# RSH-20261101-003: Phone-call SFT for Qwen3-ASR

Opened: 2026-11-01 14-11-32 KST
Recorded by agent: claude-qwen-asr-training

## Question

Calls are the weakest real-use area, and the FLEURS incumbent made them worse
(RSH-20261101-001). Does adding clean AIHub 571 TRAIN call clips to the
incumbent's recipe fix calls without breaking anything else at 16 or 8 kHz
(DEC-20261101-002)?

## Setup

- **Data** (`proposals/calls-train-20261101`, builder `build_calls.py`
  sha256 `c329eaf1…`):
  - `data-v009-calls` is `data-v005-gloss-free` plus 10,000 call clips:
    14.2 h, 3,393 calls, 5,768 per-call speakers, at most 3 clips per call.
  - Targets follow the probe's markup rules and use the written side of each
    pair. The filler and discourse words of DEC-20261101-001 are dropped.
  - Texts in dev, the protected test, the real-use probe and KSS are
    excluded. TRAIN and validation share no call sessions.
  - `data-v010-calls-band8k` adds one 8 kHz band-limited copy of every FLEURS
    training recording.
- **Recipe:** the incumbent's (base model, lr 3e-5, accumulate 4, rank 16,
  seed 21774173, sentence-epoch), run for 3,436 steps with
  `--schedule ko=30,ja=3,zh=3,en=1` (code `4a2d521`). Every Korean clip is
  seen once. JA/ZH/EN get 1,114/1,114/372 draws, against the incumbent's
  1,133/1,133/377.
- **Scoring:** `gate.py` (sha256 `761c3125…`) compares each candidate with
  the incumbent and base, using paired-bootstrap CI95. The output is
  `gate-v001.json`.

## Findings

Real-use Korean, clean sWER, plain-calls run (`sft-calls-v009`) vs the
incumbent:

| Cohort | 16 kHz | 8 kHz |
| --- | --- | --- |
| call | 26.77 → 20.45, -23.6% [-28.6, -19.1] | 26.65 → 20.42, -23.4% [-28.0, -18.7] |
| broadcast | 20.24 → 26.75 (one loop; 20.08 without) | 21.66 → 21.90 |
| meeting | 19.59 → 18.30 | 21.39 → 19.74 (one loop; 17.92 without) |

The 8 kHz-copies run (`sft-calls-band8k-v010`) is within noise of this on
real-use. On calls, both runs also beat base, by 16.0% and 14.6%.

- **Calls are fixed.** Errors drop by about a quarter at both rates, and
  both runs beat base on calls.
- **Each run loops once in 1,200 clips.** It repeats a word until the token
  limit, which adds 88–246 errors to that cohort. Without the loop clip,
  broadcast ties the incumbent and meeting beats both incumbent and base.
- **FLEURS slips a little.** Scores vs the anchor are 0.881 (v009) and 0.895
  (v010), against the incumbent's 0.835. No v009 language is clearly worse at
  16 kHz; v010 Japanese is (+11.1% [+3.1, +20.3]).
- **Both fail the 8 kHz gate.**
  - v009: FLEURS EN (+6.0% [+0.7, +11.9]), FLEURS KO (+9.1% [+0.6, +20.5])
    and KSS other (+36.8% [+3.8, +93.3]).
  - v010: FLEURS EN (+9.6% [+2.1, +18.4]) and KSS other.
- **The KSS loss is number style, not hearing.**
  - Most of the 14 extra errors on KSS other (8 kHz) are numbers that the
    model now writes in digits, following the call transcripts' written
    side: 십대 → 10대, 만 원 → 10000원, 두 배 → 2배, 한 번 → 1번, 두 명 → 2명.
    KSS writes these in Hangul.
  - The rest are a dropped 저기 or 뭐 and one two-speaker clip cut to its
    first speaker.
  - 17% of call targets contain digits, some unnatural (`2 다` for 둘 다).
- **Number-blind, only English fails.** Rescored with `gate2.py` (sha256
  `2716452c…`, DEC-20261101-003), the KSS and FLEURS KO failures disappear
  for both runs. FLEURS EN at 8 kHz is the only one left. Calls stay -23%
  (v009) and -22% (v010) at both rates (`gate2-v001.json`).
- **The 8 kHz copies did not help 8 kHz.** v010's FLEURS 8 kHz score (0.930)
  equals v009's (0.929). It also cost Japanese at 16 kHz.

## Next

- The operator should choose how numbers are written: digits like the call
  transcripts, or Hangul for native counts (한 번, 두 명) with digits for
  amounts, dates and times. They should also say whether gates should
  score digit and Hangul forms of the same number as equal.
- Keep discourse words that are often real words (그, 이제, 뭐, 저기, 네, 예)
  in training targets, and drop only pure interjections. Scoring already
  treats them as optional.
- Drop 8 kHz copies of FLEURS. Calls already supply 8 kHz audio.
- Repetition loops need a fix, either at decode time or by excluding them in
  training. Base never loops on this probe.
