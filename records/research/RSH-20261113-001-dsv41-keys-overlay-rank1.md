# RSH-20261113-001: DeepSeek-V4.1-Flash-1017 Keys Overlay Is One Direction

Opened: 2026-11-13 11-11-08 KST
Recorded by agent: claude-opus-5-5

## Question

Can spark1+spark2 serve stock and abliterated DeepSeek-V4.1-Flash-1017 from
one set of weights, picked per request? And how far has the MiaAI-Lab recipe
moved past our `27741e1` checkout?

## Findings

### The overlay is a rank-1 output projection

drowzeys' Keys overlay (`0bdc2036`, the one `ABLIT=1` applies) replaces
`layers.10-35.attn.wo_b` (26 layers; the EXL3 sidecar is 4 tensors a layer:
`trellis`, `suh`, `svh`, `mul1`, so 104 in all). Its card names the edit "rank-1
`attn.wo_b` projection, recaptured 5120-d, lambda 3.5, k=1".

We checked this numerically. The FP8 sidecar `wo_b_l10_35_1017.safetensors`
was compared against stock `deepseek-ai/DeepSeek-V4.1-Flash-1017` at layers
10, 20 and 35, both dequantized with their UE8M0 32x32 block scales:

| layer | rel. size of change | top-1 singular share | top-2 | fitted lambda |
| --- | --- | --- | --- | --- |
| 10 | 0.059 | 0.891 | 0.003 | 3.38 |
| 20 | 0.050 | 0.868 | 0.004 | 3.32 |
| 35 | 0.050 | 0.866 | 0.004 | 3.32 |

- One direction carries ~88% of the change. The remaining ~12% is spread flat
  across every other direction, which is how FP8 re-rounding noise looks.
- All three layers share the same direction (cosine 0.9999 to their mean).
- So `W_ablit ~= (I - lambda r r^T) W_stock` with one 5120-d unit vector `r`
  and lambda ~3.5. Per row of `wo_b`'s output `y`:
  `y_ablit = y - lambda * r * (r . y)`.
- The inverse also exists: `(I - 3.5 P)^-1 = I - 1.4 P`, so stock can also be
  recovered from abliterated weights.

The extracted `r` (fp32, 20 KB) was computed from those three layers. It needs
the gated overlay (an HF token with access, `infra/HF_TOKEN` in passage) and
three public stock shards (`model-00013/23/38-of-00048`, ~7.4 GB each).

### Consequence for serving both

There is no need to hold two copies of the 104 tensors. Keep stock `wo_b`, and
for rows flagged abliterated, subtract `lambda * r * (r . y)` from `wo_b`'s
fp32 output in layers 10-35. That is a dot product and an axpy over 5120
values per row, negligible next to the matmul. It also avoids a second problem
with swapping tensors: each EXL3 `wo_b` has its own input sign vector `suh`,
which `wo_a`'s epilogue applies when it rotates `wo_b`'s input
(`round.zig` `RotOut.suh`). Swapping `wo_b` per row would have meant patching
`wo_a` too.

The correction is linear. In TP2 each rank holds `wo_b`'s input rows for its
own head groups and produces a partial output, so applying it to each rank's
partial before the reduce is exact. The MTP drafter and layers 0-9 and 36+ are
stock in the overlay, so the drafter needs no change.

### Recipe and engine drift (2026-11-13)

- Recipe `main` is 70 commits past `27741e1` (head `c7f2b85`, 2026-11-10 KST).
  The patch series goes up to 0068, against our 26 patches. The engine pin moved
  `629c524` -> `fbbc971`, which is the fork branch `deepseek-v41-zig`'s current
  tip. The checkpoint, `INPUTS_REF` and the overlay pin are unchanged.
- Worth having: live DSML tool-call streaming (0048); short requests cutting in
  during long prompt passes (0037/0042/0045, 26.7 s -> 0.98 s in their test);
  draft calibration (0050, prose 79.7 -> 85.4 tok/s in their test); weight
  read-ahead at load (0066); replayed `reasoning` reaching the template (0047)
  and the opt-in `DROP_REASONING_HISTORY` (0057); and the CUDA graph cap (0033).
- Behaviour changes at `fbbc971` with every switch off: a call to a tool the
  request did not offer comes back as a `tool_calls` entry; the prompt cache's
  `tail` rule changes; picture prompts keep and resume states by default.
- The fork's `deepseek-v41-zig-tp2` branch (TP2 v0.7.5, 2026-11-04) adds 32
  streams a pool and 64-row rounds. It has diverged ~2,200 commits from the
  recipe's branch and the recipe does not use it. Not a drop-in.

## Rejected Path

Carrying both tensor sets per rank (~340 MB extra a Spark) with a per-row
swap. It costs memory, a second rank-cache layout, and a per-row `wo_a`
epilogue rotation, all for something the rank-1 correction does exactly.
