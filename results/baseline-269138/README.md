# Corrected native-vLLM baseline — Slurm 269138

This immutable bundle records the completion-limit correction for the RAG and
periodic-agent workloads from the initial native-vLLM baseline. It supplements
rather than replaces `baseline-268672`.

## Configuration

- Date: 1 August 2026
- Slurm array job: `269138` (tasks 0 and 1)
- Experiment code commit: `2c034ff380d3f4ba27fa094441c8a5d81d89e781`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM: `0.0.0+d6dbdb9b0`
- Hardware: NVIDIA A16, 15,356 MiB, driver `595.71.05`
- Workloads: controlled RAG and periodic Data Analyst
- Policies: full recomputation and native vLLM automatic prefix caching
- Repetitions: three per workload/policy pair
- Completion limit: 96 tokens

The 12 conditions each started a fresh vLLM process. RAG and periodic-agent
tasks used separate GPUs, while APC-off/on comparisons within a workload used
the same GPU.

## Contents

- `results/`: 12 result JSON files and 12 run manifests.
- `request-logs/`: 12 complete append-only full input/output ledgers.
- `server-logs/`: 12 raw vLLM server logs.
- `slurm-logs/`: two top-level array-task logs.

The raw artifacts contain synthetic prompts and outputs. They also preserve
experimental machine provenance—including Imperial hostnames, GPU UUIDs,
filesystem paths, and server addresses—but contain no credentials or API keys.

## Outcome

Every response finished naturally; none reached the 96-token limit. RAG passed
all deterministic answer checks in both APC modes. The periodic-agent model
still made arithmetic or exact-format errors in both modes, with a mean
requirement score of `0.625`. Those failures are therefore model/workload
limitations rather than truncation or an observed cache regression.

Use the combined analysis in `../baseline-268672-269138/analysis`, which retains
chat from job `268672` and selects these corrected RAG and periodic-agent
conditions.
