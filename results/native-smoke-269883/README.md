# Native CacheSelect smoke run 269883

This directory archives the first end-to-end run of the CacheSelect decision
hook inside vLLM. The job ran on one Imperial A16 using
`Qwen/Qwen2.5-1.5B-Instruct` at project commit `a6a07e9`.

## Conditions

Both conditions used automatic prefix caching and a fresh vLLM process:

1. `native-apc`: unmodified vLLM APC behavior;
2. `cacheselect`: APC with `--enable-cacheselect` decision telemetry.

Each condition executed the same four-request RAG trace once. This is a smoke
test, not a statistically powered performance comparison.

## Result

| Condition | Cached tokens by request | Mean TTFT |
| --- | --- | ---: |
| Native APC | `[0, 80, 32, 160]` | 42.388 ms |
| CacheSelect | `[0, 80, 32, 160]` | 42.030 ms |

The CacheSelect runtime decisions were:

1. `FULL_RECOMPUTE` with no native prefix hit;
2. `VLLM_NATIVE_APC` for an 80-token hit;
3. `VLLM_NATIVE_APC` for a 32-token hit;
4. `VLLM_NATIVE_APC` for a 160-token hit.

This confirms the intended invariant: CacheSelect preserves every safe native
APC hit while recording the policy and reason per request. All eight requests
passed their deterministic answer-quality checks, and both ledgers contain four
starts, four completions, and zero failures.

The mean CacheSelect-minus-native TTFT difference was -0.358 ms in this single
run. It must not be interpreted as a speedup; the purpose of the run is cache
parity and observability correctness.

## Post-prefix reuse opportunity

`analysis/reuse-opportunity.json` applies the deterministic 16-token block
analyzer to the CacheSelect condition. For each full current-position block
that native APC recomputed, it searches the entire previous rendered prompt
for identical token content, independent of location.

| Transition | Native recompute | Candidate blocks | Candidate tokens |
| --- | ---: | ---: | ---: |
| Document replacement | 96 tokens | 3 | 48 (50.0%) |
| Document reorder | 144 tokens | 6 | 96 (66.7%) |
| Query replacement | 17 tokens | 0 | 0 (0.0%) |
| **Total** | **257 tokens** | **9** | **144 (56.0%)** |

Eight candidate blocks moved to a different prompt position. Five have an
identical source occurrence beginning on a previous 16-token block boundary;
four require gathering or repacking tokens across source blocks. The document
reorder transition is therefore the best first backend prototype: it exposes
the largest opportunity and five whole source blocks.

These are content-identical candidates, not valid KV hits. Their KVs were
computed under different preceding context and still require validation or
repair before they can replace normal computation.

## Shutdown log note

`server-logs/vllm-cacheselect.log` contains an `EngineDeadError` after all four
requests returned HTTP 200. It occurs immediately after the harness sends
`SIGTERM` to stop the fresh server and is followed by normal application
shutdown. The Slurm job completed successfully and both ledgers are complete,
so this is shutdown noise rather than an inference failure.

## Contents

- `results/`: full benchmark result JSON, including prompts, responses,
  token IDs, cache counts, runtime decisions, timing, and quality results;
- `request-logs/`: append-only full input/output ledgers;
- `server-logs/`: vLLM logs for both fresh server processes;
- `slurm-logs/`: the top-level job transcript and success marker.
- `analysis/`: deterministic post-prefix token/block opportunity report.

These files deliberately contain complete synthetic benchmark prompts and
outputs for reproducibility. A credential-pattern scan found no secrets.
