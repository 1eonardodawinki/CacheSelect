# CacheSelect quality-stress matrix 273641

This bundle records the first active partial-reuse experiment in which the
edited prompt changes the correct answer from `NORTH-731` to `SOUTH-913`.
Imperial Slurm job `273641` ran from project commit `690cd85` using
`Qwen/Qwen2.5-1.5B-Instruct` on the `a16` partition.

## Design

The matrix crossed three prompt lengths (256, 1024, and 4096 rendered tokens),
three edit-proximity repair radii (0, 1, and 2), and three answer-fact positions
(early, middle, and late). Each cell contained three paired repetitions of:

1. native vLLM full computation;
2. CacheSelect shadow planning with full computation;
3. CacheSelect active partial reuse.

This produced 27 aggregated cells and 81 measured active edited requests.
Generation used temperature zero. Negative TTFT deltas mean active CacheSelect
was faster than native vLLM.

## Evidence

Active partial reuse executed in all 81 edited requests. All 81 passed the
answer requirement, exactly matched their paired shadow output, and had word
similarity 1.000. The controlled experiment therefore observed no output change.

| Tokens | Radius | Early TTFT | Middle TTFT | Late TTFT | Exact matches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0 | -2.652 ms | +8.113 ms | +24.864 ms | 9/9 |
| 256 | 1 | +0.225 ms | +10.592 ms | +24.101 ms | 9/9 |
| 256 | 2 | +0.404 ms | +12.605 ms | +28.735 ms | 9/9 |
| 1024 | 0 | -166.606 ms | -105.427 ms | -29.513 ms | 9/9 |
| 1024 | 1 | -164.264 ms | -104.899 ms | -28.873 ms | 9/9 |
| 1024 | 2 | -161.554 ms | -103.407 ms | -26.802 ms | 9/9 |
| 4096 | 0 | -844.211 ms | -581.764 ms | -288.293 ms | 9/9 |
| 4096 | 1 | -843.448 ms | -579.804 ms | -287.819 ms | 9/9 |
| 4096 | 2 | -839.639 ms | -579.238 ms | -283.963 ms | 9/9 |

At 256 tokens, planning and copying usually cost more than the saved compute.
At 1024 and 4096 tokens, every tested cell improved TTFT, with larger benefits
for longer reusable suffixes. Increasing the repair radius reduced reuse by one
16-token block per step and produced a small corresponding latency cost.

## Limitations

This is evidence for the controlled transition, model, decoding configuration,
and hardware class tested; it is not a general proof that partial reuse is
quality preserving. The task requires retrieval of one explicit synthetic fact,
and only three repetitions were measured per cell. Harder multi-fact, reasoning,
and natural-language edits are required to search for the quality boundary.

## Contents

- `analysis/summary.csv`: the 27 aggregate rows printed by the nine conditions.

The full per-request JSON, request ledgers, server logs, and generated traces
remain in the Imperial experiment directories rooted at
`/vol/bitbucket/lmw25/cacheselect-{results,request-logs,server-logs}/quality-273641`.
