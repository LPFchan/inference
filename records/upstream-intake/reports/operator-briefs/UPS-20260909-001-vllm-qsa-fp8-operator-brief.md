# UPS-20260909-001: vLLM QSA FP8 cache — operator brief

## Review Metadata

- Review id: UPS-20260909-001
- Opened: `2026-09-09 21-50-01 KST`
- Recorded by agent: codex
- Review date: 2026-09-09
- Upstream window: vLLM v0.29.0 through PR #55557
- Baseline reviewed against: Mangchi's v0.29 Thor image
- Overall recommendation: ship the validated SM110 adaptation and watch long-context quality

## This Period At A Glance

Current vLLM has several QSA rewrites that are absent from the v0.29 release branch. PR #55557 builds on those changes and stores the main QSA cache in FP8. On SM110, a 0.67 allocation produced 383,350 cache tokens and completed an exact 262,144-token request. The SSD-backed PLE loader and narrow Thor top-k workaround remain required. The old v0.29 image remains available for immediate rollback.

## Decisions Requiring Operator Input

None. The operator approved the canary and rollback plan.

## Watchlist

- Compatibility surfaces to monitor next: FP8 long-context recall with default cache scaling and FlashInfer 0.6.8 compatibility.
- Decisions to carry forward next review: move from the PR head to its merged commit when available.
- Deferred items and revisit date: tile-union prefill remains SM121-only and is not part of this build.

## Decisions Made Autonomously

### Adapt current QSA rather than transplanting one patch into v0.29

The current QSA changes are tightly connected and do not apply cleanly to our release source. The canary therefore pins the exact #55557 head, updates our two necessary local integrations, and leaves unrelated experimental QSA PRs out.
