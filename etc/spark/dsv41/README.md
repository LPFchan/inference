# DeepSeek-V4.1-Flash-1017 on spark1+spark2: our additions to the MiaAI-Lab recipe

The recipe ([MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks))
is a checkout on spark2, not vendored here. This folder holds what we add to it.

| File | What it is |
| --- | --- |
| `0090-keys-switch.patch` | Engine patch: stock and abliterated from one set of weights, picked per request ([RSH-20261113-001](../../../records/research/RSH-20261113-001-dsv41-keys-overlay-rank1.md)). Made against recipe `c7f2b85` (engine `fbbc971` plus its 54 patches). |
| `keys_dir_1017.safetensors` | The Keys overlay's direction (fp32 5120, unit). `scripts/spark/dsv41_keys_direction.py` recomputes it. |

## Install

On spark2, in the recipe checkout:

1. Copy `0090-keys-switch.patch` into `zig/patches/`. The recipe's image hash covers that folder, so `prepare.sh`
   builds (or finds) the image with it.
2. Put `keys_dir_1017.safetensors` under the HF cache at `tensorfold-dsv41/keys/`. Both ranks see it as
   `/hf/tensorfold-dsv41/keys/keys_dir_1017.safetensors`; rank 1 reads it over NFS.
3. In `scripts/local.sh`, with `ABLIT=0` (stock weights):

   ```bash
   SERVED_NAME=DeepSeek-V4.1-Flash-1017-Stock
   export TF_DS_KEYS_DIR=/hf/tensorfold-dsv41/keys/keys_dir_1017.safetensors
   export TENSORFOLD_KEYS_MODEL=DeepSeek-V4.1-Flash-1017
   ```

   The name chat.lost.plus has served since 2026-10-25 stays the abliterated model; stock gets a name of its own.

   `start.sh` passes `TF_DS_*` and `TENSORFOLD_*` to both ranks. Both ranks must have the same `TF_DS_KEYS_*`.
   Rank 0 sends each stream's flag in `fill_begin`; a rank 1 without `TF_DS_KEYS_DIR` fails on a set flag instead
   of editing half the sum.

Requests that name `TENSORFOLD_KEYS_MODEL` get the edit; `SERVED_NAME` is stock. Optional:
`TF_DS_KEYS_LAMBDA` (default 3.5) and `TF_DS_KEYS_LAYERS` (default `10-35`).

## How it works

- The keys model's requests carry `keys` from the server (`openai.zig` `keysAsked`) through `Request`, `Spec` and
  `Stream` into the dsv41 lanes.
- Rank 0 records the flag per slot and sends it in `fill_begin`. Both ranks build each decode round's row mask from
  their slot table (a `keys` glue column, uploaded with the round's other per-row data). The prompt pass uses the
  filling slot's flag.
- After `wo_b` in layers 10-35, `tf_ds_keys_kernel` applies `y -= lambda * d * (d . y)` to the masked rows of each
  rank's partial, before the ranks' sum. With `TF_DS_KEYS_DIR` unset, none of this runs and the frames stay as they
  were.
- The keys model's cache key has bit 31 set on its first token, so the prompt cache and its joins never resume one
  variant's state for the other. Its states stay off the durable (disk) tier, whose rank 1 keys points from its own
  untagged ids. A picture prompt with `TF_DS_IMAGE_CACHE=0` stays uncached.

## Weights

`ABLIT=0` serves the stock rank caches. spark1's disk cannot hold a second ~106 GB file, so rank 1's lives on spark2
(`~/.cache/huggingface/tensorfold-dsv41/45ee5d0f/rank1-cache`). spark1 reads it through a read-only fstab NFS mount
at `/mnt/spark2-hf` (`x-systemd.automount`), set as `WORKER_CACHE_DIR`. The engine copies the file into memory at
start; nothing reads it afterwards.

## Gate

`scripts/spark/dsv41_keys_gate.py` runs against a live server:

- refusal both ways
- a mixed stock+keys batch matches each request alone
- no cache resume across variants
- arithmetic on both names
