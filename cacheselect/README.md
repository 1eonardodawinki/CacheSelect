# CacheSelect runtime layer

CacheSelect has two layers. The external package owns workload metadata,
feature extraction, experiment orchestration and request ledgers. The runtime
decision hook lives inside this repository's vLLM checkout, after vLLM has
tokenized the prompt and looked up its real native prefix-cache hit.

## Policy actions

- `FULL_RECOMPUTE`: do not reuse KV state from the previous request.
- `VLLM_NATIVE_APC`: use vLLM's exact-prefix automatic prefix caching.
- `PARTIAL_KV_REUSE`: reuse KV outside the exact prefix and repair affected
  state. This action is part of the stable interface but has no backend
  implementation yet.

`PrefixHeuristicPlanner` is the first transparent planner. Cold requests use
full recomputation, exact matches and append-only requests use native APC, and
non-prefix edits use native APC only when their exact reusable prefix reaches a
configurable threshold. Its default 64-token threshold is an initial rule
informed by the A16 calibration, not a trained classifier or final result.

## Current execution boundary

The benchmark runner supports `--planner-mode shadow`. In this mode it:

1. renders each request with the model tokenizer before sending it;
2. compares adjacent token sequences and records a recommendation;
3. executes the experiment's forced APC-off or APC-on policy;
4. verifies the preflight token IDs exactly match the prompt rendered by vLLM;
5. records the recommendation, executed policy, full input/output, latency,
   cache counts, and quality.

Shadow mode deliberately does not apply the recommendation. Native APC is
configured for the whole vLLM server. Keeping `planner_decision` separate from
`execution_policy` prevents a recommendation from being mistaken for an
executed optimization.

The runner also supports `--planner-mode vllm`. Start the server with
`--cacheselect-minimum-native-prefix-tokens N`; vLLM then records its candidate
native hit and applies one of the following policies to each request:

- hits below `N`: `FULL_RECOMPUTE`;
- hits of at least `N`: `VLLM_NATIVE_APC`.

The OpenAI-compatible response reports the applied policy, reason, candidate
hit and threshold in its `metrics` object. The benchmark copies these into
`runtime_policy`, alongside the complete request and response ledger.

This threshold policy is an execution and measurement scaffold, not the final
research contribution. Exact native reuse is safe, so deliberately rejecting a
small exact hit is principally useful for verifying per-request control and
collecting controlled comparisons. `PARTIAL_KV_REUSE`, semantic prompt
segments, and a cost- or quality-aware selector are not implemented yet.
