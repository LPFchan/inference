# RSH-20260912-002: MTP speculative depth sweep on Flash-Next (k=2..5)
Opened: 2026-09-12 06-05-00 KST
Recorded by agent: claude

## Question

`RSH-20260912-001` shipped MTP at k=2 without testing other depths. Which
`num_speculative_tokens` is fastest for Flash-Next on Mangchi, does deeper
speculation change what the model says, and where does the curve turn over?

## Method

Fixed four-prompt suite (code, reasoning, arithmetic, prose) at temperature 0,
run through `scripts/mangchi-mtp-sweep.py`, which pairs end-to-end timing with
spec-decode counter deltas and a hash of the exact output. Each k was loaded
through the residency agent and measured with three passes, discarding the
first; reported figures average passes 2-3.

Two measurement traps were found and corrected before any ranking was trusted:

- **Warm-up.** The same k=4 config measured 28.4 tok/s immediately after load
  and 31.5-31.6 tok/s on later passes. The first sweep pass is unreliable, and
  the initial cold-vs-warm comparison was biased against every k except the one
  that happened to be measured an hour into its uptime.
- **Output nondeterminism.** Free-form output already varies run-to-run at a
  *fixed* k, and a k=4 pass reproduced a k=3 pass byte-for-byte. Output hashes
  therefore cannot be used to detect a k-induced quality change.

## Findings

Throughput (tok/s, warm, passes 2-3 averaged):

| workload | k=2 | k=3 | k=4 | k=5 |
| --- | --- | --- | --- | --- |
| code | 24.96 | 30.43 | 31.55 | **32.74** |
| reasoning | 23.86 | 28.38 | **30.07** | 30.05 |
| arithmetic | 24.35 | 28.72 | **32.28** | 30.00 |
| prose | 17.71 | **18.88** | 18.49 | 17.30 |

Mean accepted length rises monotonically with k on every workload (arithmetic
2.95 -> 3.77 -> 4.57 -> 4.71; prose 1.97 -> 2.13 -> 2.23 -> 2.24), so deeper
speculation never guesses *worse*. Wall-clock stops following it after k=4
because each extra draft pass costs a fixed increment while the marginal
accepted token shrinks. On arithmetic, k=5 raised accepted length 3% and step
cost about 13%.

Acceptance by draft position shows why the workloads separate. The fifth guess
lands 48.6% of the time on code and 2.8% on prose; the fourth lands 10.6% on
prose. Predictable text keeps paying for depth, prose stops paying at k=3.

- **k=4 is the operating point.** It wins or ties three of four workloads,
  needs no block-size workaround, and fits the existing 0.72 / 88 GiB budget.
- k=3 is marginally better on prose (+2%) and is the alternative if the served
  mix is dominated by free-form chat.
- Quality is not a k trade-off on this stack, contrary to the blazux report of
  MTP=3 costing an agentic-tournament point. Output varies across repeats at
  fixed k, so any such effect is below the noise floor of exact-match checks.

## Upstream bug: k>=5 cannot boot at the derived block size

k=5 crashes during engine init with
`QSA ring capacity 12 must divide the attention block size 3184`.

The QSA state cache sizes a ring buffer as
`compress_ratio * cdiv(compress_ratio + num_speculative_tokens, compress_ratio)`.
At compress_ratio 4 that is 8 for k=2,3,4 and 12 for k=5,6. The attention block
size is derived independently in
`Platform._align_hybrid_block_size`, whose only extra alignment hook is
`_get_indexer_block_alignment` (the kpool paged-MQA constraint). The ring
capacity never joins that computation, despite the cache's own comment claiming
it "joins the LCM that sets the scheduler block size". 3184 is divisible by 8
and not by 12, so every k below 5 boots and k=5 asserts.

Workaround without an image rebuild: pass `--block-size` set to a multiple of
both the kernel alignment (16) and the ring capacity (12), i.e. a multiple of
**48**. vLLM treats the value as an alignment unit and rounds the true
requirement up to the next multiple, so 48 wastes at most ~1.5%. Passing 3216
directly is wrong: the k=5 requirement exceeds 3216, so it rounds to 6432 and
pads the state cache by 99%. k=5 also needs a slightly larger KV budget
(5.93 GiB required against 5.89 GiB at util 0.72); util 0.74 clears it.

Fixing this upstream means folding the QSA ring capacity into the block-size
alignment, which is the same class of change as the draft-group annotation
tracked in `RSH-20260912-001`.

## Rejected paths

- The Thor NVFP4 MoE kernel was suspected of gating on token count. It is not:
  `permuted_rows` pads any count in 0..262144, so verify shapes are
  unconstrained. The `check.py` gate list is test coverage, not a limit; it was
  extended to tokens 5 and 6 anyway.
- `--mamba-cache-mode align` does not need to be set by hand on this build; the
  engine selects `align` automatically for this architecture when prefix
  caching is on.

## Open questions

- k=5 was measured with the wasteful 6432 block size. Its numbers are mildly
  pessimistic, and a 48-multiple rerun would make the k=4 vs k=5 comparison
  exact. Given k=5 trails by 6-7% on two workloads and block size drives
  memory rather than speed, this is unlikely to change the choice.
- Whether the k=4 optimum moves once CUDA graphs land (step cost drops, which
  should favour deeper speculation).
