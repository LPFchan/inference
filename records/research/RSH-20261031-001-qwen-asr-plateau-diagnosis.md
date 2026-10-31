# RSH-20261031-001: Why the Qwen3-ASR hill-climb stopped improving

Opened: 2026-10-31 17-24-06 KST
Recorded by agent: claude-qwen-asr-training

## Question

After removing translator glosses from TRAIN targets, every recipe and
capacity change lost. The operator asked why. They offered three theories:
(a) Qwen already trained on the public data, (b) the architecture can't
absorb more, (c) equally correct style variants deflate the score. Three
fresh-agent branches and two supervisor checks tested them, all scored with
Korean sWER and optional glosses
([DEC-20261031-001](../decisions/DEC-20261031-001-qwen-asr-korean-space-normalized-wer.md)).

## Findings

1. **The yardstick can't see small gains.** Reruns of one recipe with new
   seeds move the score by an SD of 0.029 pooled over four recipes (df 5), or
   0.012 over the incumbent recipe's three seeds. Two of six same-recipe pairs
   pass every single-seed gate against themselves. Simulated two-seed
   adoption (DEC-20261030-003) needs about an 8% true gain for 80% power (6%
   with incumbent-recipe noise only), and adopts a no-effect recipe 8–12% of
   the time. Sentence-cluster CIs agree. MLP LoRA and GRPO on gloss-free sit
   inside the no-effect band. Only the gloss fix (about 7%) has cleared it.
   Evidence: `proposals/why-seed-noise-20261031`.
2. **Most remaining errors are a shared floor.** Of the incumbent recipe's
   errors, every other trained run below 0.95 also makes 74/63/54/69%
   (EN/KO/JA/ZH), and 85/78/76/79% were already the base model's. Recipes
   only reach the remaining 31–46% of CJK errors, and each run fixes a
   different part.
   - Korean shared errors are set by the text: number style 22%, names about
     30%, and reference oddities about 24% (hanja/Latin notes, misspellings
     such as 아랫쪽).
   - Chinese shared errors are ordinary single-character mistakes (之/中,
     也/有, dropped 的/了) in words TRAIN contains.
   - Japanese is borderline (52% text-set by hand).
   - One rater, unblinded. Evidence: `proposals/why-sentence-bound-20261031`.
3. **No sign the base model memorized FLEURS TRAIN (theory a).** The base
   model, scored against raw transcripts on 300 TRAIN clips per language and
   1,230 DEV clips, shows no TRAIN advantage in Korean, Japanese or Chinese:
   - The exact-match ratio CI upper bounds are 1.14/1.32/1.17, against a
     memorization bar of 1.5.
   - Every loss-gap CI includes zero.
   - English leans toward TRAIN (exact match 34.3% vs 25.4%), but the test
     can't separate mild exposure from easier sentences.
   - Exposure to both splits would look the same.
   Evidence: `proposals/why-seen-train-20261031`.
4. **Capacity is not the limit (theory b).** One-pass SFT loss on
   not-yet-seen TRAIN clips falls only from 0.155 to 0.136 (100-step means)
   over 944 updates. With MLP LoRA (2.7× trainable weights) it ends at 0.134.
   Looped decoders worsened with depth
   ([RSH-20261030-003](RSH-20261030-003-qwen-asr-looped-decoder.md)).
5. **Style variants were real for Korean spacing (theory c).** Spacing was
   about half of the incumbent's Korean WER and nearly all of its Korean gain;
   sWER removes it. Under sWER, the incumbent recipe's three seeds score
   Korean 4.86/5.04/5.21% vs the anchor's 5.15%. Scoring numbers against
   FLEURS's digit-style raw transcripts changes at most 0.1 pp.

## Conclusion

The wall is mostly measurement: about 150 FLoRes sentences per language
cannot resolve gains under about 6%. Below that sits a floor of base-model
errors that FLEURS-style read speech does not teach. In Korean that floor is
mostly reference text, in Chinese it is the model family. Recipe and
capacity tweaks are below what the loop can detect.

## Routes

- Only admit changes with a plausible effect of 6% or more, such as data
  that adds names, terms or new acoustics, not recipe or capacity tweaks.
- More resolution needs more seeds per arm or a larger selection set beyond
  FLEURS dev. Both are operator decisions.
- Korean number style and reference oddities are a scoring and reference
  question for the operator, like glosses and spacing were.
