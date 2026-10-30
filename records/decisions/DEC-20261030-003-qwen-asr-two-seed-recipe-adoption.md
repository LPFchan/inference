# DEC-20261030-003: Adopt Qwen3-ASR recipes by two-seed comparison

Opened: 2026-10-30 22-13-21 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: the operator approved the recipe-against-recipe rule and the
  gloss-free TRAIN SFT adoption on 2026-10-30.
- Related ids: DEC-20261029-001, DEC-20261030-002

## Decision

A candidate recipe replaces the incumbent when its two seeds, compared with
the incumbent recipe's two seeds, pass every gate. This replaces
DEC-20261029-001's rule that each seed passes alone.

- Seeds are 21774173 and 21775174. A recipe that starts from the incumbent
  uses each seed's matching incumbent checkpoint as its parent.
- Scoring is `selection.py CANDIDATE INCUMBENT ANCHOR --glosses
  calibration/glosses-v1.json --second-seeds CANDIDATE2 INCUMBENT2`, frozen as
  `code/523783e` (PR #39; `selection.py` sha256 `42856e015ba5a1f7c725…`). Each
  recipe's errors pool per clip, and all gates run on the pooled errors: more
  than 1% gain, CJK within 2% of the better of anchor and incumbent, English
  within +0.5 pp, no added empty or truncated outputs, and a paired CI95
  below zero.
- A first seed earns its second seed when it scores below the incumbent
  recipe's pooled score, with no added empty or truncated outputs. Otherwise
  the recipe closes.
- Both seeds must pass the KSS veto. The lower-score seed becomes the
  incumbent checkpoint.

Adopted under this rule: the gloss-free TRAIN SFT, the incumbent recipe on v005
with translator glosses removed from KO/JA/ZH targets (train manifest
`11c97042…17a6`). Its seeds `sft-gloss-free` and `sft-gloss-free-confirm` score
0.8431 and 0.8426. Pooled, they score 0.8428 against the old pair's 0.9073,
CI95 [-0.1221,-0.0172]. `sft-gloss-free-confirm/step-000944` (adapter
`f025ecbaf6fa…c568`) is the incumbent. Nothing is deployed.

## Context

Under the optional-gloss scoring, the old recipe's two seeds scored 0.8841 and
0.9306. The incumbent checkpoint was the luckier of the two, so single-seed
gates rejected recipes with consistent gains. The gloss-free recipe improved
every CJK language on both seeds, yet each seed failed the CI gate against
that one checkpoint. Pooling only the candidate's seeds still gave CI95
[-0.1053,+0.0061].

## Options Considered

- **Keep single-seed gates:** every gain smaller than the seed spread is
  rejected, and the luckier incumbent seed sets the bar.
- **Pool the candidate's seeds against one incumbent checkpoint:** the
  incumbent's own seed luck stays in the comparison.
- **Pool both recipes' seeds (chosen):** compares like with like and keeps
  every gate.

## Rationale

The question each trial asks is whether a recipe beats another recipe. Pairing
both seeds on both sides answers it with the same gates and the same clips.

## Consequences

Every adoption now costs two seeds. Both incumbent seeds stay retained as the
comparison pair. The gloss-free recipe's seeds differ by 0.0005, which makes
later comparisons sharper than they were against the old pair.
