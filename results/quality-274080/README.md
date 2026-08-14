# CacheSelect quality-boundary matrix 274080

This bundle records the first broad search for cases where CacheSelect's
edit-proximity repair policy stops preserving model behavior. Imperial Slurm
job `274080`, resumed by job `274092`, evaluated seven synthetic dependency
families using `Qwen/Qwen2.5-1.5B-Instruct` on NVIDIA A16 GPUs.

## Design

The matrix crossed three prompt lengths (256, 1,024, and 4,096 rendered
tokens), three repair radii (0, 1, and 2), seven dependency families, and three
edit positions. Every cell requested three paired repetitions of native vLLM,
CacheSelect shadow planning with full computation, and active partial reuse.
This produced 189 aggregate cells and 567 attempted comparisons.

Radius-0 measurements used project commit `7630ea6`; radii 1 and 2 used
`c5b13e9`. The intervening commits changed analysis and resume orchestration,
not partial-reuse execution. All conditions used vLLM build
`0.0.0+d6dbdb9b0`, deterministic decoding, APC enabled, chunked prefill
disabled, and an 8,192-token batching limit.

## Measurement validity

Of 567 attempted comparisons, 513 produced usable timing measurements. The 54
unavailable comparisons are all middle or late cases in the `rule` family.
Another 81 measurable comparisons failed under native or shadow full
computation and therefore cannot evaluate reuse quality. These cases remain in
the matrix as invalid references rather than being silently discarded.

This leaves 432 reference-valid comparisons. Active reuse passed the semantic
answer check in 405 of them (93.75%) and exactly matched shadow output in 396
(91.67%). The 27 semantic failures all belong to the `pointer` family.

| Family | Cells | Reference-valid cells | Active correct | Exact shadow match | Interpretation |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Direct | 27 | 27 | 27/27 | 27/27 | Preserved |
| Composed | 27 | 27 | 27/27 | 27/27 | Preserved |
| Conflict | 27 | 27 | 27/27 | 27/27 | Preserved |
| Conflict 3 | 27 | 27 | 27/27 | 25/27 | Correct with minor wording changes |
| Conflict 5 | 27 | 27 | 27/27 | 26/27 | Correct with minor wording changes |
| Pointer | 27 | 9 | 0/9 | 0/9 | Reuse failure boundary |
| Rule | 27 | 0 | n/a | n/a | Workload must be repaired |

The pointer result is the central negative evidence: positional distance alone
does not identify indirect semantic dependencies. Increasing the radius from 0
to 2 did not repair these failures. This motivates a selector that uses richer
signals than block distance.

## Performance where quality was preserved

The table below covers the five fully reference-valid families where every
active comparison passed the semantic answer check. Negative TTFT deltas mean
active CacheSelect was faster than native vLLM.

| Prompt tokens | Cells | Faster cells | Mean TTFT delta | Observed range |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 45 | 15/45 | +7.784 ms | -3.322 to +27.657 ms |
| 1,024 | 45 | 45/45 | -100.377 ms | -164.160 to -24.365 ms |
| 4,096 | 45 | 45/45 | -591.672 ms | -886.922 to -286.101 ms |

At 256 tokens, preparation and copying usually cost more than the computation
saved. Every quality-preserving 1,024- and 4,096-token cell improved TTFT. Each
additional repair-radius step removed one more 16-token block from reuse, but
the three-repetition latency differences between radii are too noisy to rank
the radii reliably.

## Limitations and next action

These are synthetic traces on one model and one GPU class with three
repetitions per cell. The results establish a controlled failure boundary, not
a general quality rate for real workloads. The `rule` prompts need correction
and revalidation under native computation before they can test reuse. The next
system milestone is to add a dependency-sensitive selection signal and test
whether it rejects or repairs the pointer cases while retaining the long-context
latency gains.

## Contents

- `analysis/quality-matrix.csv`: normalized 189-cell table;
- `analysis/quality-matrix.json`: machine-readable equivalent;
- `analysis/quality-matrix.md`: generated report table.

The complete per-request input/output ledgers, raw JSON, generated traces, and
server logs remain under the Imperial directories rooted at
`/vol/bitbucket/lmw25/cacheselect-{results,request-logs,generated-traces,server-logs}/quality-274080`.

