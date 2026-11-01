# RSH-20261102-001: Qwen3-ASR against MAI-Transcribe 2 on Korean interviews

Opened: 2026-11-02 04-18-59 KST
Recorded by agent: claude-qwen-asr-training

## Question

How far is the Korean incumbent from MAI-Transcribe 2, the operator's bar
(DEC-20261102-001), on unscripted speech that MAI is allowed to receive?

## Probe

`proposals/seoul-ko-20261102/probe-v001` has 800 clips and 2.15 h of audio,
frozen before any model output. The manifest sha256 is `9039bda4…8a5c` and
the builder `build_probe.py` is `835d11dc…8680`. The source is the Seoul
Corpus (OpenSLR 113, Yun et al. 2015, CC BY-NC 2.0): forty Seoul speakers,
ten in each age group from teens to forties, half of them female, each
interviewed for about an hour. Only the interviewee is transcribed. The
archives' sha256 values are `62809cbc…2e30` (sound), `63c3f8ff…d24b` (labels)
and `bc925df3…1d27` (readme).

- **Clip:** a run of the interviewee's utterances in the standard-spelling
  tier (`utt.ortho.`), 1–15 s long, with at least 3 words. It needs a pause of
  at least 0.3 s on both sides and keeps 0.15 s of that pause.
- **Run breaks:** a run ends at interviewer speech, speech while laughing,
  unclear speech, masked personal details, or any text that is not Hangul and
  spaces.
- **Selection:** 20 clips per speaker, taken round-robin over that speaker's
  six files (seed 20261102), from 8,679 eligible clips. The median clip is
  11.2 s.
- **8 kHz:** the same band limit as `train.py evaluate --audio-rate 8000`.

## Results

Clean, number-blind sWER, lower is better. CI95 is the paired bootstrap of
the change against MAI.

| Rate | MAI | Incumbent | Change vs MAI | Base | Change vs MAI |
| --- | ---: | ---: | --- | ---: | --- |
| 16 kHz | 11.35% | 13.37% | +17.8% [+13.3, +22.6] | 13.56% | +19.4% [+15.0, +24.5] |
| 8 kHz | 11.42% | 13.68% | +19.8% [+15.5, +24.5] | 13.81% | +21.0% [+16.3, +26.1] |

- The bar is not met at either rate.
- The calls incumbent ties base on interviews: −1.4% [−4.0, +1.3] at 16 kHz
  and −1.0% [−3.6, +1.7] at 8 kHz. Call training neither helps nor hurts
  this kind of speech.
- The phone band costs MAI almost nothing (11.35% to 11.42%) and costs Qwen
  a little more.
- No Qwen decode looped. MAI returned no empty outputs.

## Where the gap is

At 16 kHz there are 14,000 reference words.

- **Wrong words are most of it.** The incumbent has 1,459 substitutions
  against MAI's 1,203. It has 249 deletions against 190 and 164 insertions
  against 196. Examples: 둘째 → 두 채, 적에 → 쪽에, 구렁이가 → 그러니까,
  모든 선생님을 과외 → 모조선생님과.
- **False starts.** The incumbent drops restarts that the reference keeps
  (살 수 감당을 → 감당을). MAI's verbatim style keeps them.
- **Spread.** The incumbent is worse on 276 clips and MAI on 118. The 40
  clips with the largest gap explain 52% of it.

So the gap is mostly hearing spontaneous speech, not formatting. Fillers,
spacing and number style are already neutralized by the scorer.

## Cost

MAI used 1,643 requests, including 43 that hit 429 or 402 during bursts and
were retried. The accounted cost was $0.46 for both rates. A 402 arrived
with $12 of credit left and cleared on retry, so `score_mai.py` treats it as
retryable. The endpoint was Azure at $0.10/h (`mai-*/endpoints.json`).

## Next

- Spontaneous Korean is the missing training data. AIHub training splits
  (KsponSpeech-style dialogue, broadcast, meetings) can be used locally;
  only their audio must not go to MAI. The AIHub probe still selects
  (DEC-20261101-004), and this probe measures the bar.
- Real phone calls are not in this probe. Own recordings are the route to a
  MAI-safe calls set.

## Evidence

On yeowoolmac under `proposals/seoul-ko-20261102`: probe, `score_mai.py`
(`5b1d88ef…07ef`), `compare_mai.py`, `compare-v001.json`, `gap.py` and
`gap-v001.txt`, MAI outputs in `mai-16k` and `mai-8k`, and the Qwen runs
`parallel/{,8k-}seoul-ko-{sft-calls-ko-v011,base}` (code `4cf83f9`). These
are mirrored to Mangchi with a matching tree hash (`d879dccc…a665`). The
extracted corpus is in `~/models/stt/seoul-corpus-20261102` on yeowoolmac.
