# Mangchi vLLM Patches

These patches apply to the exact vLLM SHA pinned by
`docker/mangchi-vllm/Dockerfile` before its wheel is built.

| File | Scope | Origin |
| --- | --- | --- |
| `0001-live-prefill-and-generation-timings.patch` | Emit llama.cpp-compatible `prompt_progress` and `timings` fields for live UI statistics, including partial chunked-prefill updates. | local |

The image also applies small Python source transformers from
`docker/mangchi-vllm/`. `patch_sharded_state.py` adds the fast checkpoint path
for prepared Thor MoE weights, excludes the separately mmap-backed PLE table,
and makes MTP honor its explicit draft-model load format.

The timing patch preserves vLLM's standard OpenAI response fields. Its extra
SSE fields are consumed by Grimoire's web UI and ignored by ordinary OpenAI
clients.
