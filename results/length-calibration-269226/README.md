# Initial prompt-length calibration — Slurm 269226

This immutable bundle records the first native-vLLM prompt-length calibration
run at 256, 1,024, and 4,096 rendered prompt tokens.

## Configuration

- Date: 1 August 2026
- Slurm array job: `269226`
- Experiment code commit: `2c034ff380d3f4ba27fa094441c8a5d81d89e781`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM: `0.0.0+d6dbdb9b0`
- Hardware: NVIDIA A16, 15,356 MiB, driver `595.71.05`
- Edit positions: early, middle, and late
- Policies: full recomputation and native vLLM automatic prefix caching
- Repetitions: one per condition

Each condition started a fresh vLLM process and sent a cold donor request
followed by one edited request.

## Contents and interruption

- `results/`: 13 result JSON files and 13 run manifests.
- `request-logs/`: 13 complete append-only full input/output ledgers.
- `server-logs/`: 13 raw vLLM server logs.
- `slurm-logs/`: three top-level array-task logs.
- `generated-traces/`: tokenizer-aware traces created by the three array tasks.

All six 256-token and all six 1,024-token conditions completed. The first
4,096-token condition also produced a valid result and complete ledger, but the
task stopped because its original quality gate required a citation even though
the response retrieved the correct `NORTH-731` fact and finished naturally.
The remaining five 4,096-token conditions were therefore not run in this job.

The raw failed-gate condition is retained for provenance but is superseded in
the combined analysis by job `269267`.
