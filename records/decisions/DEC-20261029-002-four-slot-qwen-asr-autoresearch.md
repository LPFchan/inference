# DEC-20261029-002: Four-slot Qwen3-ASR autoresearch

Opened: 2026-10-29 14-30-00 KST
Recorded by agent: claude-qwen-asr-slot1

## Metadata

- Status: accepted
- Deciders: operator, orchestrator
- Authority: operator: "4gb for rest of the system, four instances of 5GB
  workers, multiple clips together. launch the autoresearch pipeline with
  four agents".
- Supersedes: the single-worker, no-subagent and one-GPU-process clauses of
  DEC-20261029-001. Every other clause of that decision stands.
- Related ids: DEC-20261029-001

## Decision

Allow four research agents and up to four concurrent model workers on
yeowoolmac's The Rest partition. Additional workers start only through
`training/qwen_asr/parallel.py`, and only after its native acceptance.

- Four slots. Slot 1 is the existing `ROOT/.gpu-loop.lock`, so a serial
  driver (`cycle.py`, the batch-32 trial driver) counts as one worker.
  Slots 2–4 are owned by the launcher. Every live `train.py` or `batching.py`
  interpreter counts, with or without a slot.
- Budget: at most 5,000,000,000 bytes physical footprint per new worker and
  20,000,000,000 bytes reserved in total. 4,000,000,000 bytes are set aside
  for macOS and apps; that is an allowance, not a measurement of free memory.
  A worker reserves the larger of its budget and its measured lifetime-peak
  footprint. Processes the launcher did not start reserve at least
  5,000,000,000 bytes. New workers pass `--memory-gb 4.5` or less.
- Each job owns a directory under `ROOT/parallel/` and runs a frozen
  `ROOT/code/` `train.py` or `batching.py`. A trainer's job directory is its
  `--output`; a probe's report is a new file inside its `--job-dir`. The
  worker inherits its slot and job locks, so both stay claimed if the
  launcher dies. Jobs never write `loop-state.json` or `ROOT/trials/`.
- Admission is refused when no slot is free, four workers are live, the
  reservation would exceed 20 GB, kernel memory pressure is not normal,
  `ROOT/STOP` exists, or the job directory is claimed.
- During a run the launcher samples memory every 10 s. A worker above its own
  budget for three samples is stopped. Total live reservation above 20 GB or
  critical pressure for three samples stops the newest worker that still has
  a live launcher, by that launcher only. A sampling failure stops the
  launcher's own workers. Logs and checkpoints are kept. Serial drivers,
  their workers and unrelated processes are never stopped.
- Selection is unchanged: frozen 1,295-clip validation, `selection.py` gates,
  independent-seed confirmation, sealed 2,624-clip test and published files.
  The supervisor (orchestrator) alone records selection, truth docs and commits.

## Context

The batch-4 and batch-32 SFT trials use 4.15–4.81 GB peak MLX allocation.
The live batch-32 worker's measured lifetime-peak footprint is 5.17 GB, above
the per-worker budget. While it runs, only two more 5 GB workers fit in
20 GB; a third needs a budget of at most 4.83 GB. How much memory macOS and
apps actually use outside the workers has not been measured against a
controlled baseline, so the four-worker fit is unproven.

## Options Considered

- Keep one worker: least concurrent memory demand; leaves concurrency gains unmeasured.
- Bypass or delete the serial lock: breaks the live trial's ownership.
- One admission gate plus per-slot and per-job locks inherited by each
  worker, and per-launcher supervision of its own workers: the legacy lock
  keeps its meaning, crashed launchers cannot free capacity still in use and
  no new service is needed.

## Rationale

The operator explicitly allows four agents and four 5 GB workers. MLX's
memory limit is a soft guideline and gives no isolation, so measured
physical footprints decide reservations and live enforcement. Stopping only
the launcher's own newest worker restores the bound without touching runs
that predate admission.

## Consequences

Concurrent runs change wall-clock and per-step timings, so compare speed
only within the same concurrency. Accuracy evidence is unaffected. Native
acceptance is staged: one probe or trainer beside the live batch-32 run
first, with contended timing; the clean 1/2/4-worker throughput benchmark
needs an otherwise idle Mac. Whether four workers fit and whether concurrency
raises total throughput stays unknown until actually measured.
