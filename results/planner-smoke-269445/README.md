# CacheSelect shadow-planner smoke test — Slurm 269445

This immutable bundle records the first GPU execution of the CacheSelect
planner in shadow mode. The planner made recommendations before each request,
while the experiment still forced APC off or on for the complete vLLM process.

## Configuration

- Date: 2 August 2026
- Slurm job: `269445`
- Experiment code commit: `15c5eb7c1373de3963f0018ef811040c7f822a26`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM: `0.0.0+d6dbdb9b0`
- Hardware: NVIDIA A16, 15,356 MiB, driver `595.71.05`
- Planner: `prefix-heuristic-v1`, 64-token native-prefix threshold
- Workload: four-request controlled RAG trace

## Validation

- Both request ledgers contain four starts, four completions, and no failures.
- Planner preflight token IDs matched the prompts rendered by vLLM.
- Every response passed the deterministic quality check.
- APC-off reused zero tokens; APC-on reused `0`, `80`, `32`, and `160` tokens.
- Both forced execution modes produced the same recommendations:

| Request | Transition | Recommendation |
| --- | --- | --- |
| `rag-00` | cold start | `FULL_RECOMPUTE` |
| `rag-01` | document replacement | `VLLM_NATIVE_APC` |
| `rag-02` | document reorder | `FULL_RECOMPUTE` |
| `rag-03` | query replacement | `VLLM_NATIVE_APC` |

The recommendations were recorded but not applied. This bundle validates the
reference planner and observability path; it is not a CacheSelect performance
result.

## Contents

- `results/`: two complete result JSON files.
- `request-logs/`: two append-only full input/output ledgers.
- `server-logs/`: raw vLLM logs for the APC-off and APC-on processes.
- `slurm-logs/`: the top-level job output.

The artifacts contain synthetic prompts and preserve experimental machine
provenance, but contain no credentials or API keys.
