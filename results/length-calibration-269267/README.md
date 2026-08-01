# 4K prompt-length calibration continuation — Slurm 269267

This immutable bundle completes the 4,096-token part of the native-vLLM
prompt-length calibration after the answer check was corrected to measure
semantic fact retrieval rather than citation formatting.

## Configuration

- Date: 1 August 2026
- Slurm array job: `269267` (task 2)
- Experiment code commit: `ba133f5819712e27ec05ffea92f878afbe9925c3`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM: `0.0.0+d6dbdb9b0`
- Hardware: NVIDIA A16, 15,356 MiB, driver `595.71.05`
- Prompt length: 4,096 rendered tokens
- Edit positions: early, middle, and late
- Policies: full recomputation and native vLLM automatic prefix caching
- Repetitions: one per condition

The API prompts are unchanged from the original calibration design. Only the
deterministic benchmark-side answer requirement changed.

## Contents and outcome

- `results/`: six result JSON files and six run manifests.
- `request-logs/`: six complete append-only full input/output ledgers.
- `server-logs/`: six raw vLLM server logs.
- `slurm-logs/`: one top-level array-task log.
- `generated-traces/`: the three regenerated 4,096-token traces.

All six conditions completed successfully, every prompt contained exactly
4,096 rendered tokens, and every response finished naturally and passed the
semantic fact-retrieval check.
