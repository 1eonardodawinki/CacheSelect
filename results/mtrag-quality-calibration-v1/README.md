# MTRAG reference-quality calibration set

This artifact freezes 20 training requests—five from each MTRAG collection—
before their Qwen outputs are observed. It is used only to choose conservative
token-recall, ROUGE-L, and maximum-regression limits for natural answers.

The tasks come from the same real, naturally aligned workload pool as the
causal pilot, but no KV block is tested during calibration. Validation and test
conversations remain unopened. The raw source file is verified by SHA-256 both
when this manifest is selected and when the GPU command runs.

## Reproduction

The frozen manifest contains the 20 selected training task IDs. With the raw
MTRAG release staged locally, reproduce their uncached reference answers with:

```bash
python -m benchmarks.run_mtrag_calibration \
  --input /path/to/RAG.jsonl \
  --manifest results/mtrag-quality-calibration-v1/manifest.json \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-completion-tokens 384 \
  --request-log-dir /path/to/request-logs \
  --output /tmp/reference-calibration.json
```

The next experiment computes each selected request from scratch, records its
full input/output ledger, and saves the two lexical metrics. Thresholds are
frozen only after those reference outputs are audited; this manifest contains
no model outcomes or causal labels.

## Recorded run

Slurm job `274842` completed all 20 uncached requests. Its full request ledger
and reference outputs are stored under `run-274842/`.

`manual-audit-v1.json` records a pre-intervention semantic review of every
answer: 8 pass, 8 concern, and 4 fail. The review showed that absolute lexical
scores overlap heavily between correct and incorrect answers. Therefore a task
must be manually approved before it can create a causal label; token recall and
ROUGE-L are retained as repeatability and pairwise-regression checks, not as a
standalone proof of correctness.
