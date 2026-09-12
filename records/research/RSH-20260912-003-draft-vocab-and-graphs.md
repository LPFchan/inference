# RSH-20260912-003: Reduced-vocabulary MTP drafting and piecewise CUDA graphs
Opened: 2026-09-12 08-00-00 KST
Recorded by agent: claude

## Question

`RSH-20260912-002` left two levers unmeasured: the reduced-vocabulary draft
(FR-Spec) and CUDA graphs. Which is worth more on this stack, and does either
change what the model produces?

## Sizing the draft projection before building

A draft pass reads the MTP head's output projection in full. On this checkpoint
`lm_head` is [248320, 2560] BF16 = 1.27 GB and is *unquantized* -- the one large
BF16 tensor left in an otherwise NVFP4 model. Against it, the routed experts
contribute 0.028 GB per token (10 of 512 experts, 640 intermediate, 4-bit) and
the shared expert 0.003 GB. The projection is therefore ~98% of a draft pass's
weight traffic, which is what makes restricting it worth doing at all.

The method assumes low token ids are frequent (BPE merge order). Measured on
8,110 tokens this model actually generated, rather than assumed:

| draft vocabulary | coverage of generated tokens |
| --- | --- |
| 16,384 | 88.52% |
| 32,768 | 93.61% |
| 65,536 | 97.89% |
| 98,304 | 99.96% |

Prose is the worst case (96.76% at 65,536), arithmetic the best (100%).

## Findings: reduced-vocabulary draft

All legs at k=2, image v8, same session, four passes each with the first
discarded (warm-up is worth ~10-15%, established in `RSH-20260912-002`).

| workload | control | 65,536 | 98,304 |
| --- | --- | --- | --- |
| code | 26.35 | +2.9% | +4.7% |
| reasoning | 25.59 | +5.9% | +5.3% |
| arithmetic | 25.93 | +2.2% | +0.8% |
| prose | 18.76 | +7.2% | **+7.4%** |

Mean accepted length barely moves: prose 1.966 -> 1.966 / 1.956, code
2.810 -> 2.692 / 2.747. So the gain is a cheaper draft, not a different one.

- **98,304 beats the upstream default of 65,536** on this checkpoint: equal or
  faster everywhere that matters, and it keeps more acceptance (code 2.747 vs
  2.692), exactly as the coverage table predicts. The upstream default is not
  wrong, it is tuned for a different token distribution.
- **The gain lands where nothing else did.** Prose resisted deeper speculation
  (k=5 made it worse) because its guesses do not land. This helps precisely
  because it makes each *failed* guess cheaper.
- The gain is smaller than the traffic reduction implies. At k=2 there are only
  two draft passes per step to save on, and GPU occupancy of 63-69% during
  decode says part of the step is latency rather than bytes. The benefit should
  scale with k.

## Method note

The first FR-Spec run was void: a background script left alive by a failed
`pkill` (the pattern matched the command containing it, so the shell killed
itself before killing the target) woke mid-test, reset k and started a second
model load against the same agent. Two loaders raced and the 65,536 leg was
measured through the collision. Every leg was re-run with all variables pinned
explicitly per leg rather than inherited from the config file.

## Findings: the gain scales with speculative depth

Re-measured at k=4, the deployable depth from `RSH-20260912-002`, same image
and session, four passes with the first discarded:

| workload | k=4 control | k=4 + 98,304 | delta |
| --- | --- | --- | --- |
| code | 31.64 | **36.30** | +14.7% |
| arithmetic | 32.07 | **35.78** | +11.6% |
| prose | 18.36 | **20.74** | +13.0% |
| reasoning | 30.16 | **32.84** | +8.9% |

Double the k=2 gain, because there are twice as many draft passes to make
cheaper. Prose accepted length is identical (2.229 -> 2.229): this is pure cost
removal, not a quality trade. Accepted length moves at most 0.13 anywhere.

Against the originally shipped k=2 full-vocabulary configuration, the combined
k=4 + 98,304 stack is +37.8% code, +38.0% arithmetic, +28.3% reasoning and
+10.6% prose, and roughly 3.4x the 10.56 tok/s pre-MTP baseline on code.

## Findings: piecewise CUDA graphs are blocked, and not by the annotation

The PLE gather does a host round trip (ids to CPU, rows off NVMe, staged back),
which cannot run inside CUDA graph capture. This is why the model is served
`--enforce-eager`.

The documented remedy was applied in full: the gather is registered as a custom
op with a fake impl, and `vllm::qwen4_exp_ple_mmap_gather` is added to
`CompilationConfig._attention_ops`, which `set_splitting_ops_for_v1` copies into
`splitting_ops` when the mode is `VLLM_COMPILE`. Both were verified present in
the built image at runtime (`in _attention_ops: True`, `op registered: True`).

It still fails, and the traceback shows the op *is* the call site:

```
vllm_ple_mmap.py:429 patched_forward
torch/_ops.py:1269 __call__                     <- our custom op
vllm_ple_mmap.py:453 _qwen4_exp_ple_mmap_gather
vllm_ple_mmap.py:202 gather
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

Forcing `{"cudagraph_mode":"PIECEWISE"}` did not change it. So on this pin the
split-point annotation does not keep this call site out of capture, and the
remaining route is the staged gather: hoist the disk read into the model state's
`prepare_inputs` ahead of the forward, into a fixed GPU buffer the graph reads.
That is the change tonyd2wild actually made, and it is what let them run
`FULL_DECODE_ONLY`.

Two warnings for whoever picks this up:

- **Do not "fix" the error by pinning the index tensor.** Capture would then
  succeed and silently replay stale rows, because a replayed graph cannot
  re-read data-dependent rows from disk. The error is load-bearing.
- **Iteration costs about an hour per attempt** (torch.compile adds ~28 minutes
  on top of the ~12-minute weight load). A cheaper loop is a prerequisite for
  attacking this seriously.

Separately, `_MmapNgramEmbedding.gather` builds its inverse-index tensor with
`torch.from_numpy(...)` in pageable memory before copying it to the device; the
loader already keeps a pinned staging buffer for the table rows but not for that
index. Pinning it is a small transfer win independent of graphs.

## Conclusion

- Ship k=4 with `QWEN4EXP_DRAFT_VOCAB=98304` on image
  `mangchi-vllm:thor-dense-candidate-v8-draft-vocab`.
- Prefer 98,304 over the upstream 65,536 default for this checkpoint.
- CUDA graphs stay open, re-scoped from "annotate a split point" to "stage the
  gather in prepare_inputs". The measured GPU occupancy of 63-69% still caps the
  prize at roughly 20%, below what the draft-vocabulary change already returned.
