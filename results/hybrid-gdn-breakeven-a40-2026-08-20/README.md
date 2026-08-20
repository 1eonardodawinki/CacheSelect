# Hybrid GDN A40 break-even evidence

This bundle combines two controlled Qwen3.5-9B break-even runs executed on one
NVIDIA A40 using project commit `264aa284d3fa5e84b691a16b511c0c2e820402ef`.
Every active-reuse output exactly matched its independent full-computation
reference.

## Source runs

- `hybrid-checkpoint-active-1787209417-break-even`: 1, 2, 4, 8 and 16 reused
  64-token blocks, with three repetitions and cache capacity 16.
- `hybrid-checkpoint-active-1787211287-break-even`: 16, 24, 32 and 48 reused
  64-token blocks, with five repetitions and cache capacity 48.

The complete immutable RunPod export is retained locally as
`results/runpod-a40-2026-08-20/cacheselect-runpod-artifacts-2026-08-20.tar.gz`.
Its SHA-256 digest is
`c5f3d447001983c9a504448480cf0957f78eb48073df036c69434152f9c6b683`.

## Method and conclusion

`replication-analysis.json` pools trial-level wall-time and TTFT ratios by
reused block count and reports deterministic 95% percentile-bootstrap
intervals for each median. A speedup is considered replicated only when at
least two source runs contribute and the lower bounds for both metrics exceed
1.0.

The resulting decision is `NO_RELIABLE_SPEEDUP`. At 16 blocks, the only
condition repeated across both runs, the pooled wall-time median is `0.997x`
with interval `[0.989, 1.015]`; three of eight trials were faster. The active
GDN path is therefore correct, but its measured performance is approximately
parity within the current experiment's uncertainty.

These runs use synthetic early-edit prompts. Real edited-prompt quality and
performance remain to be evaluated separately.
