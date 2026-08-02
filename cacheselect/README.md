# CacheSelect runtime layer

CacheSelect sits between an application and vLLM. Its planner compares the
tokenized current prompt with the previous prompt and returns a structured
policy recommendation before inference begins.

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
currently configured for the whole vLLM server, while partial KV reuse does not
exist yet. Keeping `planner_decision` separate from `execution_policy` prevents
a recommendation from being mistaken for an executed optimization.

The next runtime milestone is a vLLM-facing policy adapter. It will apply full
or native execution per request and later expose the minimal primitive needed
for `PARTIAL_KV_REUSE`. Once all candidate policies can execute, repeated
measurements will provide oracle labels for a learned planner.
