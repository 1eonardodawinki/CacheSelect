# Shadow repair-policy matrix run 271465

This directory archives the first successful comparison of CacheSelect's
configurable repair selectors. The four conditions ran on Imperial NVIDIA A16
GPUs with `Qwen/Qwen2.5-1.5B-Instruct`, project commit `cec6e9d`, vLLM build
`0.0.0+d6dbdb9b0`, APC enabled and a 16-token block size.

## Design

Every condition used a fresh vLLM process and executed the same four-request
RAG trace. The document-reorder transition exposed 80 resident candidate
tokens. CacheSelect then classified those tokens using either full-block repair
or edit-proximity repair with radius 0, 1 or 2.

The selectors ran in shadow mode: they recorded which candidate tokens would
be repaired or skipped, while native vLLM still recomputed the prompt normally.
This run therefore validates selector behavior and observability, not repaired
KV execution or latency improvement.

## Result

| Condition | Candidate tokens | Repair tokens | Skipped tokens |
| --- | ---: | ---: | ---: |
| Full block | 80 | 80 | 0 |
| Edit radius 0 | 80 | 0 | 80 |
| Edit radius 1 | 80 | 48 | 32 |
| Edit radius 2 | 80 | 80 | 0 |

All four conditions produced identical prompts, native cache-hit counts,
outputs and quality results. Every deterministic quality check passed. The
radius-1 policy is the only condition in this controlled trace that creates a
mixed repair/reuse plan: it proposes repairing 48 candidate tokens and reusing
32 without repair.

Because the plans were not executed, output equality does not yet demonstrate
that the skipped tokens are safe to reuse. The next milestone must apply the
candidate KV state and selected repair mask before measuring quality or speed.

## Contents

- `rag-*.json`: full four-request observations for each selector condition;
- `rag-*.manifest.json`: job, model, vLLM, GPU and policy provenance;
- `rag-*.complete`: per-condition validation markers;
- `repair-policy-summary.json`: cross-condition validated counts and parity.

Recreate and validate the combined summary from the repository root with:

```bash
python -m benchmarks.analyze_repair_policy_shadow \
  --input-dir results/repair-shadow-271465 \
  --output results/repair-shadow-271465/repair-policy-summary.json
```
