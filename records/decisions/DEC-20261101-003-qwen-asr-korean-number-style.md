# DEC-20261101-003: Korean number style and number-blind gates for Qwen3-ASR

Opened: 2026-11-01 14-59-06 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: on 2026-11-01 the operator chose the usual Korean number style.
  This followed the proposal to score digit and Hangul forms of a number as
  equal in the gates and to keep discourse words in training targets.
- Related ids: DEC-20261101-001, DEC-20261101-002, RSH-20261101-003
- Supersedes: the training-target word list in DEC-20261101-001, item 1

## Decision

1. **Number style in Korean training targets.** Native Korean counts up to
   ten stay in Hangul (한 번, 두 명, 둘 다). Clock hours, dates, minutes,
   amounts and every other number use digits (2시, 10월 6일, 30분), and spoken
   만/억 amounts are grouped by 만 (72만 원, 2만 5000원). Annotated
   (written)/(spoken) pairs are resolved this way by
   `proposals/calls-train-20261101/ko_numbers.py` `style()`.
2. **Number-blind gates.** Korean rows in the gates are compared after
   `ko_numbers.canonical()` rewrites both reference and hypothesis. Any
   spelling of the same number before a counter then matches: 두 명 = 2명,
   만 원 = 1만 원 = 10,000원. This covers FLEURS KO, KSS and real-use probes,
   in `gate2.py`.
3. **Training targets drop only pure interjections** (어 음 으 엄 흠 에 아).
   그, 이제, 뭐, 저기, 네 and 예 stay, because they are often real words.
   Real-use scoring still treats the whole DEC-20261101-001 list as
   optional on both sides.

The FLEURS primary score, its anchor and the sealed test set are unchanged.

## Context

The first phone-call run (RSH-20261101-003) learned the call transcripts'
habit of writing every number in digits (1번 더, 10000원). KSS writes these in
Hangul, so the KSS gate failed on number style rather than on hearing.
Rescored number-blind, that run's KSS and FLEURS KO 8 kHz failures disappear,
and only FLEURS EN remains.

## Options Considered

- **Digits everywhere, like the call transcripts:** unnatural for counts
  (2 다, 1번 더).
- **Usual Korean style (chosen):** reads naturally in subtitles and notes.
- **Scoring with number glosses from each source's dual transcription:**
  KSS has no dual form for Hangul numbers, so it would need a number parser
  anyway.

## Consequences

- Call targets are rebuilt as `data-v011-calls-ko`.
- Gate results before this decision were not number-blind. RSH-20261101-003
  records both versions.
- `canonical()` also rewrites a few non-numbers the same way on both sides,
  for example 이번 → 2번. That leaves error counts unchanged unless only one
  side has the word.
