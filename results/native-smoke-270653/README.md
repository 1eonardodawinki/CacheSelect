# Online aligned-block smoke run 270653

This directory archives the first successful end-to-end run of CacheSelect's
online aligned-block locator inside vLLM. The job ran on one Imperial A16 with
`Qwen/Qwen2.5-1.5B-Instruct` at project commit `e2c6b1c`.

## Conditions

Both conditions used automatic prefix caching and a fresh vLLM process:

1. `native-apc`: native vLLM APC;
2. `cacheselect`: native APC plus request-scoped, shadow partial-reuse planning.

Each condition executed the same four-request RAG trace once. CacheSelect did
not execute partial reuse in this run; every reported candidate was still
recomputed normally.

## Result

| Condition | Cached tokens by request | Mean TTFT |
| --- | --- | ---: |
| Native APC | `[0, 80, 32, 160]` | 39.729 ms |
| CacheSelect | `[0, 80, 32, 160]` | 40.063 ms |

All eight responses passed their deterministic quality checks. Both request
ledgers contain four starts, four completions and zero failures. The 0.334 ms
mean TTFT difference is not a performance result; this one-run smoke test
checks parity, candidate discovery and observability.

## Online candidate plans

| Transition | Aligned candidate tokens | Resident tokens | Outcome |
| --- | ---: | ---: | --- |
| Document replacement | 0 | 0 | Requires source gathering/repacking |
| Document reorder | 80 | 80 | Five whole source blocks found |
| Query replacement | 0 | 0 | No post-prefix whole-block match |

For document reorder, the online locator reported these source-to-target block
mappings:

```text
3 -> 5
4 -> 6
5 -> 7
6 -> 8
10 -> 10
```

All five source blocks were still resident in GPU KV cache. Every candidate is
marked `requires_repair=true`: identical token content does not make its KV
state valid under the target prompt's changed preceding context. Native APC
therefore remained the executed policy and preserved the exact baseline cache
hits and outputs.

The earlier offline analyzer found an additional 48 candidate tokens for
document replacement and 16 for document reorder. Those candidates cross
source block boundaries and are intentionally outside this first online
aligned-block milestone.

## Contents

- `results/`: full benchmark result JSON with prompts, token IDs, responses,
  timings, cache hits, quality results and online partial-reuse plans;
- `request-logs/`: append-only full input/output ledgers;
- `server-logs/`: vLLM logs for both fresh server processes;
- `slurm-logs/`: the top-level transcript and successful assertion summary.

The prompts and outputs are synthetic and retained for reproducibility.
