# Expanded audited MTRAG counterfactual experiment

This experiment expands the seven-block feasibility pilot without changing its
quality rules. It uses all eight training transitions whose full-compute answers
were approved before any partial-reuse output was inspected.

The live vLLM plan must still contain all 147 recorded testable blocks. Only 53
blocks are selected for isolated interventions: vLLM reuses one selected block
and repairs every other candidate in that trial. Seven targets repeat the first
pilot as stability checks, leaving 46 new causal block experiments.

## Selection

| Measure | Value |
| --- | ---: |
| Approved transitions | 8 |
| MTRAG collections | 4 |
| Full testable block pool | 147 |
| Selected target blocks | 53 |
| Repeated pilot targets | 7 |
| New target blocks | 46 |
| Maximum targets per transition | 12 |

Targets are ranked deterministically from task and block identities before
their counterfactual outputs are observed. Conversations remain in the training
split; validation and test conversations are untouched.

## Reproduction

```bash
python -m benchmarks.select_mtrag_counterfactual \
  --coverage /path/to/mtrag-coverage.json \
  --audit results/mtrag-quality-calibration-v1/manual-audit-v1.json \
  --reference-artifact results/mtrag-quality-calibration-v1/run-274842/reference-calibration.json \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output results/mtrag-expanded-v1/manifest.json \
  --per-collection 5 \
  --max-target-blocks 12
```

The manifest binds the full coverage artifact, manual audit, reference answers,
raw MTRAG revision, model, tokenizer, prompt template, and split seed by value
or SHA-256 hash.

## Cluster submission

The larger run should receive a five-hour allocation because model startup took
about 17 minutes in the pilot and every block requires an isolated reference,
donor, and intervention sequence.

```bash
export CACHESELECT_MTRAG_PILOT_MANIFEST=results/mtrag-expanded-v1/manifest.json

EXPANDED_MTRAG_JOB_ID=$(sbatch --parsable \
  --time=05:00:00 \
  benchmarks/run_mtrag_counterfactual_pilot.slurm)

unset CACHESELECT_MTRAG_PILOT_MANIFEST
echo "$EXPANDED_MTRAG_JOB_ID"
```
