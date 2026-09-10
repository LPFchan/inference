# DEC-20260910-001: Make Mangchi Model Loading Host-OOM Safe And Order Independent

Opened: 2026-09-10 20-06-56 KST
Recorded by agent: root

## Metadata

- Status: accepted
- Deciders: operator
- Area: Mangchi residency agent, vLLM model loading, unified-memory safety
- Related ids: DEC-20260909-001, DEC-20260909-002, DEC-20260909-003

## Decision

Mangchi must never rely on Linux's host OOM killer to stop an oversized or
temporarily memory-hungry vLLM load. A load that approaches the machine's safe
memory floor must be stopped by the residency agent and reported as a normal,
actionable load failure. The operating system and unrelated services must stay
responsive.

The 27B and Flash-Next models must be able to remain loaded simultaneously, not
merely replace one another. Loading and unloading them must work in every order.
In particular:

- loading 27B and then Flash-Next must leave both models resident, as must
  loading Flash-Next and then 27B;
- unloading either model must work while the other is resident;
- unloading a model that is still loading must cancel and clean up that load,
  rather than waiting for the full startup timeout; and
- a failed or cancelled load must release its process, container, port, and
  memory reservation before another lifecycle operation proceeds.

Static steady-state residency estimates are not sufficient admission control.
The implementation must account for startup peak memory and observe real host
memory pressure while a model loads. Because Thor uses unified memory, CUDA
allocations and CPU allocations compete for the same physical RAM even when a
container's normal cgroup accounting does not show all CUDA usage.

## Context

The steady-state estimates for the current models are 31 GiB for the 27B model
and 83 GiB for Flash-Next. Their 114 GiB sum appears to fit within Thor's 128 GB
unified memory, but loading 27B after Flash-Next drove host memory to 100% and
Linux OOM-killed the 27B container.

The failed load reached safetensors startup with only 10.71 GiB of available
RAM for an 18.36 GiB checkpoint. The loader memory-maps checkpoint tensors,
copies them into CUDA parameter allocations, and the custom dense NVFP4 adapter
creates prepared packed and swizzled tensors before the source tensors are
released. These allocations all consume Thor's shared physical RAM. The
temporary peak is therefore materially larger than the final 31 GiB residency
estimate.

Docker's ordinary memory limit is not a complete safety boundary on unified-
memory NVIDIA systems: CUDA allocations may not be charged to the container's
memory cgroup. The residency agent needs a host-level safety check and active
load supervision even if per-container limits are also used as defense in
depth.

## Options Considered

### Raise the static residency budget and retry

- Upside: smallest configuration change.
- Downside: repeats the unsafe assumption that steady-state size predicts the
  loading peak.
- Downside: does not contain a bad estimate or an unexpected loader regression.

### Require one specific load order

- Upside: loading the smaller model first may leave enough reclaimable memory
  for Flash-Next's loader.
- Downside: makes the public lifecycle API depend on hidden operational trivia.
- Downside: does not satisfy order-independent loading and unloading.

### Add supervised, peak-aware lifecycle control

- Upside: prevents host OOM, preserves system responsiveness, and gives the
  gateway an honest failure state.
- Upside: makes cleanup and cancellation explicit instead of delegating them to
  the kernel OOM killer.
- Downside: requires measuring startup peaks and changing the current lock/task
  structure so unload can preempt an in-progress load.

## Rationale

Mangchi is both an inference host and a general machine. A model load is allowed
to fail; taking the host down with it is not. Unified memory makes a conventional
GPU-only budget misleading because checkpoint pages, CPU preparation buffers,
CUDA weights, activations, and KV cache draw from the same pool.

Order-independent lifecycle behavior is also part of the service contract.
Clients should not need to know which checkpoint happens to have the larger
temporary preparation footprint, and an unload request must remain a reliable
way to recover from a slow or unsafe load.

## Consequences

- The residency admission budget is 115 GiB so the models' 114 GiB combined
  estimate can be admitted, while the host-memory watcher remains the mandatory
  safety boundary during the second startup.
- Each model needs a measured startup-peak allowance in addition to its
  steady-state reservation until the loader no longer duplicates those weights.
- The agent must monitor host available memory during startup and terminate the
  loading model before the configured safety floor is crossed.
- Per-container memory controls may be added as defense in depth, but cannot be
  the only guard on Thor.
- Load cancellation and unload must not be serialized behind the complete
  health-wait interval.
- Simultaneous residency in both load orders, both unload orders, cancellation
  during load, cleanup after failure, and host-memory-floor enforcement become
  deployment acceptance tests.
- If both models cannot reach co-residency within the safety floor, the loader
  or its prepared checkpoint format must be changed; lowering the safety floor
  or accepting host OOM is not a valid workaround.
