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

## Korean-style retrain

- **Data:** `data-v011-calls-ko` (builder `build_calls2.py` sha256
  `9bc9d1e4…`, number styling `ko_numbers.py` `65a9731b…`) is v005 plus
  10,000 newly drawn call clips (14.1 h, 3,387 calls, 5,774 speakers). Its
  targets follow DEC-20261101-003. It has 18,966 clips (39.3 h) in all.
- **Runs:** the same recipe at seeds 21774173 (`sft-calls-ko-v011`) and
  21776175 (`sft-calls-ko-v011-s21776175`). The incumbent pair's second seed
  is 21775174, so the second seeds differ.
- **Scoring:** `gate2.py` per seed (`gate2-v011.json`). Both seeds are also
  pooled against the incumbent: per clip at 8 kHz, and with
  `selection.py --second-seeds` for the FLEURS score.

Clean sWER change vs the incumbent, seed 21774173 / seed 21776175:

| Cohort | 16 kHz | 8 kHz |
| --- | --- | --- |
| call | -25.2% / -25.1% | -25.3% / -24.7% |
| meeting | -14.0% / -11.9% | -20.8% / -18.3% |
| broadcast | -4.3% / -1.4% (ties) | -3.1% / +10.6% (ties; one loop) |
| KSS, all | -3.0% / -8.9% (ties) | -4.8% / -7.6% (ties) |

- **Calls hold at -25%** on both seeds and at both rates, and meetings
  improve too (every CI95 below zero). Both seeds beat base on calls by about
  17%.
- **Number style is fixed.** On KSS the models write native counts in
  Hangul as often as the incumbent: 16 Hangul counts, against 16 for the
  incumbent and 3 for v009. KSS ties the incumbent at both rates.
- **FLEURS English at 8 kHz costs a little, but does not veto.** Seed
  21774173 fails alone: +8.7% [+0.3, +17.8]. Seed 21776175 ties: +3.6%
  [-3.3, +10.6]. Pooled, the pair ties: +6.1% [-0.7, +13.3]. All four call
  runs so far are above the incumbent here, so the cost is probably real,
  but it is about the size of seed noise.
- **FLEURS slips about 4% overall, which is a tie.** The two-seed selection
  scores the pair 0.8817, against the incumbent pair's 0.8458 (paired CI95
  [-0.006, +0.074]). Every pooled language ties at both rates. The pair
  still cannot be adopted under DEC-20261030-003, which requires a FLEURS
  gain.
- **Loops: 1 in 4,800 clips.** Seed 21776175 repeats 예를 들어서 on one
  8 kHz broadcast clip (`D221336.1.2.27`), and that one clip accounts for its
  broadcast +10.6%. Seed 21774173 never loops. The incumbent loops on one
  8 kHz meeting clip too, so loops come from training generally, not from
  the call data. Base never loops.

## Next

- Adopting a calls model needs an operator decision on the selection rule.
  The current rule rewards only FLEURS gains. This pair trades a tie-sized
  FLEURS cost for a quarter fewer call errors.
- Repetition loops are fixed at decode time by PR #44 (code `4cf83f9`). A
  greedy decode that never ends is redone with the adapter off. On the 6
  looped clips in all saved evaluations (73 reference words), errors drop from
  916 to 14. Trimming the loop leaves 122. With the fallback, seed
  21776175's 8 kHz broadcast change is -2.2% [-7.4, +3.4] instead of +10.6%.
  Evidence: `proposals/loop-fallback-20261102` and
  `parallel/loopcheck-8k-sft-calls-ko-v011-s21776175`.
- JA/ZH/EN real-use probes are still needed.
