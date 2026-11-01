# DEC-20261101-004: Real-use speech selects Qwen3-ASR models

Opened: 2026-11-01 18-17-30 KST
Recorded by agent: claude-qwen-asr-training

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: on 2026-11-01 the operator agreed that real-use speech becomes
  the main score, that FLEURS and 8 kHz become checks a model must not fail
  clearly, and that the Korean-style calls model becomes the incumbent.
- Related ids: DEC-20261030-003, DEC-20261101-002, DEC-20261101-003,
  RSH-20261101-003
- Supersedes: the FLEURS gain gates of DEC-20261030-003

## Decision

A recipe replaces the incumbent when its two seeds, pooled, are clearly
better on real-use speech and clearly worse nowhere.

- **Main score:** the 1,200-clip Korean real-use probe at 16 kHz (calls,
  broadcast and meetings together), scored as clean, number-blind sWER. The
  pair must have a paired CI95 entirely below zero against the incumbent
  recipe.
- **Checks:** nothing may be clearly worse (CI95 entirely above zero). This
  covers FLEURS per language, KSS per cohort and real-use per cohort, at both
  16 and 8 kHz. The 8 kHz gate of DEC-20261101-002 is now judged on the
  pooled pair.
- **Pooling:** per-clip error counts are averaged over each recipe's
  evaluated seeds. A set the incumbent has run on one seed only still
  compares fairly. The bootstrap uses 2,000 draws with seed 20261101.
- **Seeds:** a first seed earns its second when its main score is below the
  incumbent recipe's. Of the pair, the seed with the lower main score
  becomes the incumbent checkpoint.
- **Tool:** `proposals/calls-train-20261101/adopt.py` (sha256
  `462111f8854d…`) runs on the `gate2.py` loaders.
- **Unchanged:** the FLEURS normalized score and its anchor are still
  reported, and the sealed test set stays sealed. Number-blind and clean
  rules follow DEC-20261101-003 and DEC-20261101-001.

Adopted under this rule: the incumbent recipe on `data-v011-calls-ko` with
`--schedule ko=30,ja=3,zh=3,en=1` for 3,436 steps (code `4a2d521`). Its seeds
are `sft-calls-ko-v011` (21774173) and `sft-calls-ko-v011-s21776175`. Pooled,
the main score falls 14.2% [-17.1, -11.3] against `sft-gloss-free`, with
calls at -25.1%. No check fails. `sft-calls-ko-v011/step-003436` (adapter
`b54f3ef5…5f25`, real-use 18.24%, FLEURS score 0.8668) is the incumbent.
The adoption is in `parallel/sft-calls-ko-v011/adoption.json`. Nothing is
deployed.

## Context

The FLEURS incumbent made Korean real-use speech worse than base
(RSH-20261101-001). Training on call clips fixes calls, but it costs FLEURS
about 4%, which is within seed noise (RSH-20261101-003). Under
DEC-20261030-003, a recipe was adopted only if it gained on FLEURS, so the
gain on real speech could not count.

## Options Considered

- **Real-use main score, FLEURS as a check (chosen):** this targets the
  operator's actual use: meetings, calls and videos.
- **Keep FLEURS selection:** this keeps a model that is worse than base on
  calls.
- **One blended score:** it hides a trade-off behind weights nobody chose.

## Consequences

- Real-use probes exist only for Korean. Japanese, Chinese and English are
  guarded by FLEURS checks until their probes are built.
- Future candidates compare against the v011 pair. Both of its seeds have
  every evaluation at both rates.
- `gate2.py` still names the old incumbent and remains per-seed evidence
  only. `adopt.py` decides adoption.
