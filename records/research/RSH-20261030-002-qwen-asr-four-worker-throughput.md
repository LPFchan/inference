# RSH-20261030-002: Four-worker Qwen3-ASR training fit and throughput

Opened: 2026-10-30 09-41-08 KST
Recorded by agent: codex-qwen-asr-training

## Question and finding

Can four independent Qwen3-ASR workers fit on The Rest's 24 GiB M4 Pro, and
how much combined training throughput does concurrency add?

All seven workers in the sequential 1/2/4-worker test completed. Four workers
increased combined training-only throughput by 17.65% over one; each worker
ran at about 0.840 clips/s versus 2.855 clips/s alone. All sampled memory
pressure stayed normal and swap-out counters did not increase. This is a
short fit/throughput observation, not a multi-hour stability result.

| Workers | Combined training clips/s | Mean per-worker clips/s | Phase-wall clips/s | Maximum reported worker footprint, GB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.8553 | 2.8553 | 2.5594 | 4.3338 |
| 2 | 3.1303 | 1.5651 | 2.9245 | 4.5172 |
| 4 | 3.3592 | 0.8398 | 3.1948 | 4.3735 |

GB is decimal. The producer calls the training-only metric `steady`, but it
includes the first update's compilation; no warm-up was discarded. Phase-wall
time runs from after the worker spawn loop through supervised exit, including
startup, hashing, checkpoint writes and the final four-clip development smoke.
The four-worker phase-wall rate was 24.83% above the one-worker rate.

## Fixed workload and verification

- Host: yeowoolmac, The Rest; M4 Pro, 25,769,803,776 bytes unified RAM.
- Trainer: frozen `code/e49648d`; supervisor: `code/1ae82c5`, SHA256
  `e4e8174f1e4bb51caca89e431a4406e4e32c5c176411f8e9f198d28f0a0480c6`.
- Model: Qwen3-ASR-1.7B MLX 8-bit revision
  `a8379a2e2f9e313c9292cdf1af4055ab56d50d55`; attention LoRA rank 16,
  scale 1, base/encoder frozen. MLX 0.32.4; mlx-lm 0.31.3.
- Data: v005 FLEURS TRAIN, data.json
  `0fc40070c7545cfccd0eecdf03163f32f59aa7e98dc82182590aa48bdcb2064e`.
- Every worker: seed 21774173, 60 updates, four serial microsteps per update,
  replacement sampling, LR 3e-5, identical 240-draw sequence. Seven workers
  produce 420 updates and 1,680 clip draws in total.
- Admission: 5e9 physical bytes per worker, 20e9 aggregate reservations,
  `--memory-gb 4.5`. Existing apps/kernel remained running. The launcher
  started from zero owned model workers and free slots.
- All 60-step sequences, source contracts and final adapter/optimizer hashes
  verify. All worker exits are successful. Development smokes are retained
  without reading their per-clip outputs or using them for selection.
- Maximum sampled system-used memory: 12.282 / 17.029 / 23.315 GB for
  1 / 2 / 4 workers. Existing swap usage stayed about 2.004 GB; new swap-outs
  were zero. Page-out deltas were 32 / 277 / 367; no claim of zero paging.

## Limits and use

The ascending phases were measured once. Warm caches, first-update compilation
and ordinary CPU/network activity can affect rates; a reported roughly
one-second read-only metadata task overlapped the four-worker phase. No other
training/inference worker was admitted. Results apply to this fixed workload,
not arbitrary models or future larger adapters. The reported footprints are
sampled lifetime peaks, and the MLX allocation setting is not hard isolation.

Four workers are supported for independent trials under the existing admission
rules. Combined throughput improves modestly while each trial takes longer.
Every new recipe still needs its own fit check. The confirmed accuracy
incumbent and the sealed 2,624-clip test benchmark are unchanged.

## Retained evidence

Mac/local root: `~/models/stt/qwen3-asr-hillclimb-20261029`.
Mangchi backup: `/home/yeowool/models/stt-benchmark/qwen3-asr-hillclimb-20261029`.

- `parallel/bench-20261030-090639/bench.json` and all seven worker directories.
- `proposals/native-throughput-20261030`: command, initial state, independent
  validation and runtime freeze.
- `snapshots/20261030-native-throughput.tar.gz`, 89 files, 496,369,748 bytes,
  SHA256 `e0ae0f327cc1faa0afacdbbac7a90716df9592a11f72fe99f1e11bf85b660293`.
