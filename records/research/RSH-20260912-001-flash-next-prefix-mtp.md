# RSH-20260912-001: Flash-Next prefix caching and MTP on the v7 image
Opened: 2026-09-12 01-12-00 KST
Recorded by agent: codex

## Question

Can Flash-Next on Mangchi gain working prefix caching and MTP speculative
decoding, per the three-step program greenlit by the operator (prefix caching,
then MTP, then CUDA graphs)?

## Method

Built `mangchi-vllm:thor-dense-candidate-v7-prefix-mtp` with a port of
blazux/qwen3.8-Flash-DGX's two-line mamba block_size fix
(`docker/mangchi-vllm/patch_mamba_block_size.py`, anchors verified present at
the pinned vLLM SHA 5fd5dd5). Enabled `--enable-prefix-caching` and
`--speculative-config {"method":"mtp","num_speculative_tokens":2}`. First
boot failed KV allocation (-1.96 GiB at util 0.68); retried at 0.72 with the
unused multimodal warmup disabled (`--limit-mm-per-prompt {"image":0,"video":0}`).
Validation used a varied-length cold-vs-hit parity harness
(`scripts/mangchi-prefix-cache-parity.py`), an arithmetic determinism probe,
server metrics, and spec-decode counters.

## Findings

- MTP=2 works, and the gain tracks how predictable the text is. End-to-end
  against the 10.56 tok/s pre-MTP decode baseline in PLANS: code 26.0 tok/s
  (2.46x), prose 16.5-17.5 tok/s (1.56-1.65x), with mean accepted length
  2.82/3.00 and 1.97-2.00/3.00 respectively and 2.95/3.00 on arithmetic. The
  first pooled figure recorded here came from vLLM's averaged generation gauge
  over a reasoning-heavy harness, which both understates the rate and hides the
  workload split.
- Per-position acceptance is 95.1/86.6% on code and 61-64/35.5-36.0% on prose,
  matching or beating tonyd2wild's Spark reference (93/78% code, 65/39% prose).
  The worry that refusal-projection would shift draft hidden states and depress
  acceptance did not materialize on either workload.
- Step rate is ~9/s on both workloads against 10.56/s pre-MTP, so the MTP step
  costs about 15% more and returns the accepted length. Per-step overhead is
  therefore not the limiter; acceptance length on prose is.
- GPU SM occupancy during decode is 63-69% mean (median 62-68%, under 20% for
  only 4-8% of samples). There is real host-side headroom in eager mode, but
  decode is not dominated by launch gaps, so CUDA graphs are an incremental
  win rather than the missing multiplier.
- vLLM's own warning explains the prefix-cache behavior: with no KV group
  annotated as the draft's, every group (including Mamba groups 0-3) is
  flagged as a draft group, which the code documents as disabling
  cross-request prefix-cache reuse with no error and no metric. Adding
  `max_model_len` to the speculative config (blazux's serve.sh line 119) did
  not clear the warning on our pin.
- Prefix caching is therefore inert but safe: `cached_tokens` stayed 0 on
  every repeat, so no response ever came from a Mamba state restore, correct
  or otherwise. The blazux two-line fix ships in the image as the necessary
  precondition for when draft-group annotation lands.
- The parity harness's exact-match oracle is too strict for an MTP stack:
  free-form reasoning text varies cold-vs-cold at temperature 0 (3 distinct
  of 4 runs) purely from MTP tie-breaking, with zero cache engagement.
  Short definite-answer determinism (437 x4) is the reliable regression check.
- KV at 0.72: 364-375K tokens, 1.39-1.43x concurrency at 262K context.

## Why blazux gets hits and we do not (yet)

blazux's recipe runs on the `vllm/vllm-openai:qwen38-flash-next` release
(PR #53896, the `release/qwen38next` recipe), a different vLLM codebase from
our PR #55557 pin. Their MTP group is annotatable there. The follow-up that
unlocks prefix caching on our pin is draft-group annotation in the KV cache
group construction (`_warn_if_unannotated_eagle_mamba` in kv_cache_utils.py),
which is vLLM work rather than configuration.

## Conclusion

- Ship v7 with MTP=2 on: the decode gain is real and validated.
- Keep prefix caching enabled in config (harmless, and required for the
  mamba fix to matter later) but do not claim it as a win until draft-group
  annotation lands and `cached_tokens` goes nonzero under the varied-length
  parity harness.
- Step 3 (CUDA graphs) is unchanged in design: tonyd2wild's staged-gather
  (gather in prepare_inputs, fixed GPU buffers) plus FULL_DECODE_ONLY is the
  target, recorded in PLANS.md.
