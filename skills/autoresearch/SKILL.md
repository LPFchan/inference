---
name: autoresearch
scope: qwen-asr
description: Four-slot multilingual Qwen3-ASR training and selection contract.
---

# Qwen3-ASR autoresearch contract

Read `records/decisions/DEC-20261029-001-qwen-asr-autoresearch.md`,
`records/decisions/DEC-20261029-002-four-slot-qwen-asr-autoresearch.md`, the
Autoresearch section of STATUS and the Qwen section of PLANS. Four research
agents propose and run trials; the supervisor alone selects, updates truth
docs and commits.

## Slots

- M4 Pro 24 GB: `yeowoolmac-the-rest.tailaa113.ts.net`, `main` tmux session.
- Nemotron backend `plus.lost.inference.stt` stays booted out; no auto-restore.
  Preserve gateway/tunnel, oMLX settings and every existing model/data/result.
- At most four model workers. Slot 1 is `ROOT/.gpu-loop.lock` (serial drivers);
  launch every other model process (`train.py` or a `batching.py` probe) in
  tmux with `parallel.py run ROOT [--job-dir DIR] [--wait N] -- COMMAND`.
  Budget: at most 5,000,000,000 bytes footprint per new worker (5,500,000,000
  only for DEC-20261030-001's 42- and 56-layer runs), 20,000,000,000
  reserved in total; pass `--memory-gb 4.5` or less. Admission denial exits
  75 (retry later); a stopped worker keeps its logs and checkpoints.
- Each job owns a directory under `ROOT/parallel/` and runs a frozen
  `ROOT/code/` source. Trainers use `--output` as the job; probes need
  `--job-dir` with the report inside it. Never write `loop-state.json`,
  `ROOT/trials/` or another agent's job. `parallel.py status ROOT` shows
  slots, workers, reservations, pressure and live launchers (read-only).
- CPU downloads/preparation may overlap GPU jobs. Check existing
  processes/logs before launching; never duplicate an active download or run.
- Runtime Python: `~/inference-qwen-training/.venv/bin/python` (pinned, isolated).
- New work/evidence: `~/models/stt/qwen3-asr-hillclimb-20261029`.

## Try

Write one hypothesis before spending the slot. Freeze source/inputs. Start with
5,000 SFT updates (four single-file microsteps per update) or 1,000 serialized
GRPO updates, rank 16. Change one recipe dimension at a time after the initial
data expansion. A new data snapshot is a new hypothesis; do not mutate one used
by an active run. Source hours are catalogue claims until actual usable hours
are measured. Caption corpora are not guaranteed human transcripts.

Use `training/qwen_asr/train.py`; continue adapters with explicit parent hashes
or exact-matching `--resume`. Record optimizer resets. Intake uses pinned
sources and cached downloads; no paid teacher/API. Larger budgets, better data,
Korean spacing-aware rewards and hard-example curricula are valid new trials.
Architecture changes require evidence that simpler adaptations plateau.

## Score and select

Score every clip in the same frozen 1,295-clip development manifest. Use
`training/qwen_asr/selection.py CANDIDATE INCUMBENT ANCHOR`. It returns the
normalized CJK error score, paired bootstrap interval, per-language errors and
all acceptance gates defined by the charter. Confirm wins with a second
training seed before advancing the incumbent. Keep rejected checkpoints and
measurements as evidence, without deploying them. No changes to normalization
or the selection set within a comparison.

## Sealed

- Original 2,624-clip FLEURS test corpus, nine prediction files and published
  benchmark artifacts. Test references are exclusion-only during preparation.
- Production authentication/tunnel code and unrelated services, all Grimoire
  operations, global AGENTS/fleet notes and other people's processes.
- Source/model/license provenance and every accepted run's inputs/settings.

## Resume and records

On every wake, inspect the named tmux jobs and their logs first. If moving,
leave them running. If failed, capture the cause and resume saved work. Never
substitute catalogue/model-card accuracy for measurements. Keep STATUS short:
incumbent, live jobs, completed trials and closed areas. Record each scored
trial in a contract-compliant LOG commit. Do not use PR polling while T3 watches
that PR; handle its delivered review events separately. The `continue` heartbeat
is for this active training task, not for polling PRs.

Stop all owned hill-climb jobs and the heartbeat when the operator stops
training; preserve checkpoints/data and leave Nemotron unloaded. `/icic stop`
alone disables the heartbeat. If no authorized unsealed experiment remains,
report the actual blocker rather than repeat losing trials indefinitely.
