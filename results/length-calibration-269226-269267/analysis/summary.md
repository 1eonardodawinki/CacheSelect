# Native-vLLM prompt-length calibration

## Validation

- Jobs: 269226, 269267.
- 18 selected conditions, 18 complete request ledgers, and 36 requests validated.
- All selected prompts matched their 256, 1,024, or 4,096 token target exactly.
- APC-off and cold APC-on requests reused zero tokens; every edited APC-on request reused at least one complete cache block.
- All selected outputs finished naturally and passed semantic fact retrieval.
- A failed 4K condition from job 269226 was superseded by the equivalent corrected condition from job 269267; the API prompt itself was unchanged.

## Edited-request results

| Prompt tokens | Edit | Cached/prompt | Hit rate | APC off TTFT | APC on TTFT | Reduction | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | early | 32/256 | 12.5% | 60.80 | 61.51 | -1.2% | 0.99× |
| 256 | middle | 96/256 | 37.5% | 61.65 | 50.03 | 18.8% | 1.23× |
| 256 | late | 160/256 | 62.5% | 61.21 | 35.84 | 41.5% | 1.71× |
| 1024 | early | 32/1024 | 3.1% | 234.16 | 230.51 | 1.6% | 1.02× |
| 1024 | middle | 352/1024 | 34.4% | 232.50 | 169.29 | 27.2% | 1.37× |
| 1024 | late | 672/1024 | 65.6% | 233.20 | 91.39 | 60.8% | 2.55× |
| 4096 | early | 32/4096 | 0.8% | 972.54 | 972.72 | -0.0% | 1.00× |
| 4096 | middle | 1376/4096 | 33.6% | 971.04 | 688.47 | 29.1% | 1.41× |
| 4096 | late | 2720/4096 | 66.4% | 971.35 | 355.15 | 63.4% | 2.74× |

## Cold-request control

Each value is the mean across the three edit-position conditions at that length.

| Prompt tokens | APC off TTFT | APC on TTFT | Difference |
| --- | --- | --- | --- |
| 256 | 59.56 | 59.47 | 0.2% |
| 1024 | 231.39 | 230.88 | 0.2% |
| 4096 | 966.93 | 966.88 | 0.0% |

## Interpretation

- Native prefix caching reuses only the exact token prefix before the changed marker. Early edits therefore reuse only 32 tokens at every tested length.
- Moving the edit later increases the reusable prefix: middle edits reuse approximately one third and late edits approximately two thirds of each prompt.
- TTFT savings grow with both prompt length and the number of cached tokens. This confirms that the final CacheSelect evaluation must include graduated context lengths rather than extrapolating from the initial 100–230-token traces.
- The early 256-token condition shows that a small cache hit can be slower than recomputation because lookup and block-management overhead can exceed saved prefill work.
- This is a one-repetition calibration, not the final performance result. It establishes suitable scales and validates the measurement pipeline; the final matrix needs repeated, order-randomized comparisons.

Generated files: `request_metrics.csv`, `paired_request_comparison.csv`, the plots in this directory, and `provenance.json`.
