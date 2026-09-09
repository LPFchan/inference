# DEC-20260909-003: Use current vLLM QSA with FP8 cache on Mangchi

Opened: 2026-09-09 21-50-01 KST
Recorded by agent: codex

## Metadata

- Status: accepted
- Deciders: operator, codex
- Area: Mangchi Flash-Next runtime
- Related ids: RSH-20260909-001, UPS-20260909-001, DEC-20260909-001

## Decision

Build Mangchi's Flash-Next image from the exact vLLM PR #55557 head, enable FP8 E4M3 main QSA K/V storage, and raise the served context limit to the model's native 262,144 tokens after SM110 validation. Use a 0.67 memory allocation and an 82 GiB residency reservation.

Retain the SSD-backed PLE table and the SM110 cooperative-top-k exclusion. Keep `mangchi-vllm:thor-v0.29-ple-mmap` as the rollback image.

## Context

vLLM v0.29.0 is the latest stable release, but its release branch lacks several merged QSA rewrites. PR #55557 is written against those newer kernels and does not apply cleanly to v0.29. Its FP8 cache results cover SM120 and SM121, but SM110 had not been tested upstream. Mangchi's initial 0.65 canary provided only 2.66 GiB of the 3.55 GiB required cache, so the final allocation is 0.67.

## Options Considered

### Keep v0.29 with BF16 cache

- Upside: already verified on Thor
- Downside: limited to 131,072 served context at the safe memory setting

### Transplant #55557 into v0.29

- Upside: smaller apparent version change
- Downside: requires manually rebuilding several generations of QSA code and tests

### Build the exact #55557 head and adapt local Thor integrations

- Upside: uses the coherent upstream QSA implementation and keeps local changes narrow
- Downside: carries a pre-merge source pin and requires real SM110 validation

## Rationale

The exact PR head is easier to audit and roll back than a manual QSA backport. A canary answers the only important missing question—whether the FP8 path works correctly on SM110—without discarding the verified v0.29 image.

## Consequences

- Flash-Next requests use `--kv-cache-dtype fp8` and `--max-model-len 262144` after the canary passes.
- The 0.67 canary provided a 5.19 GiB cache for 383,350 tokens. The clean production launch provided 5.88 GiB for 434,087 tokens, or 1.66 concurrent native-length requests.
- Attention K/V is FP8; the hybrid GDN/Mamba recurrent cache remains float32 under vLLM's `auto` setting.
- The PLE mmap patch follows the current `Qwen4ExpNGramEmbedding` interface.
- The Thor top-k patch follows the moved `ops/qsa_indexer.py` path.
- The source pin must be refreshed after #55557 merges and revalidated before replacing this image.
- The checkpoint has no calibrated K/V cache scale tensors, so vLLM uses its default scale and warns that FP8 may reduce accuracy. Capacity and execution are verified; long-context quality still needs normal operational observation.
