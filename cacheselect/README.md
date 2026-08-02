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

`NativeAPCFallbackPlanner` defines the safe baseline. Cold requests and prompts
without a common prefix require full recomputation. Every non-empty exact
prefix is preserved with native APC. A future partial-reuse policy may improve
on this baseline, but it must fall back to APC rather than discard safe work.

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
`--enable-cacheselect`; vLLM then records its candidate native hit and the safe
fallback policy for each request:

- no native hit: `FULL_RECOMPUTE`;
- any non-zero native hit: `VLLM_NATIVE_APC`.

The OpenAI-compatible response reports the applied policy, reason, and
candidate hit in its `metrics` object. The benchmark copies these into
`runtime_policy`, alongside the complete request and response ledger.

This fallback is an execution and measurement scaffold, not the final research
contribution: its cache behavior intentionally matches native APC.
`PARTIAL_KV_REUSE`, semantic prompt segments, and the cost- or quality-aware
choice between partial reuse and native APC are not implemented yet.
