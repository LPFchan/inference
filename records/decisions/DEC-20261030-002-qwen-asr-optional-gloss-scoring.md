# DEC-20261030-002: Score FLEURS translator glosses as optional for Qwen3-ASR selection

Opened: 2026-10-30 18-39-16 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: the operator chose to fix the yardstick on 2026-10-30 after the
  prior-art review traced most remaining Chinese errors to translator glosses.
- Related ids: DEC-20261029-001, DEC-20261029-002, DEC-20261030-001

## Decision

Qwen3-ASR selection scores the frozen 1,295-clip validation set with
`selection.py --glosses calibration/glosses-v1.json`, frozen as `code/a4559db`
(PR #38; `selection.py` sha256 `082f8584ce469b8820be…`).

- A gloss is a bracketed span in the FLEURS `raw_transcription` that has a
  Latin letter and no Hangul, kana or Han, such as `罗克汉普顿（Rockhampton）`.
  `glosses-v1.json` (sha256 `c37076af23bd6e0c2015…`) lists 87 of them: KO 19,
  JA 5, ZH 63. English references have none.
- Each gloss is an optional reference span, the NIST sclite convention: a
  clip's errors are the minimum over keeping or dropping each gloss, and its
  length is the reference without any gloss.
- Gates, anchor and incumbent are unchanged; their errors are rescored under
  the rule. The FLEURS test set, published predictions and `stt-score.py`
  stay sealed and untouched.
- Rescoring closed trials reopens none of them. Every closed trial fails at
  least the paired-CI gate against the incumbent.

## Context

The normalized FLEURS references keep translator glosses that readers mostly
do not say. A faithful transcript loses one CER error per Latin letter of each
gloss. Clips with bracketed Latin held 73% of the incumbent's Chinese errors,
36% of its Korean and 17% of its Japanese. Qwen3-ASR issue #211 could not reproduce Qwen's FLEURS Chinese
number, and Whisper strips this bracket text from Korean references.

Under the rule, the anchor scores EN 4.25%, KO 10.25%, JA 4.63% and ZH 3.36%
(ZH was 6.61%). The incumbent scores 0.8841 (was 0.8839), with paired CI95
[-0.190,-0.024] against the anchor.

## Options Considered

- **Keep the raw references:** the score keeps rewarding invented English
  names, the opposite of what the serving model needs.
- **Strip glosses from the references:** penalizes a transcript whenever a
  reader does say the name.
- **Optional spans (chosen):** neither writing nor omitting a gloss costs
  errors, and English scoring is untouched.

## Rationale

The rule removes a scoring artifact without retuning gates or touching the
sealed test set. The gloss list is built from pinned FLEURS rows and checked
against every validation reference, so it cannot drift silently.

## Consequences

Scores under this rule are not comparable with earlier scores. Seed noise
grows: the incumbent's two seeds now differ by 0.047 (0.8841 vs 0.9306), and
the confirmation seed's CI95 against the anchor, [-0.149,+0.019], now crosses
zero. The incumbent stays, because adoption is not re-run retroactively. Three
closed GRPO trials now score 0.8456–0.8605 against the incumbent's 0.8841, yet
each still fails the paired-CI gate. Whether two seeds may pool their errors
for that gate is an open operator question.
