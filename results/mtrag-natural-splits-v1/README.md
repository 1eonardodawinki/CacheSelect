# Natural MTRAG selector splits v1

This bundle freezes complete MTRAG conversations before collecting new model
outputs or causal block labels. It combines seven existing training transitions
with 33 new transitions to produce an exact 70/15/15 split.

| Split | Existing | New | Final |
|---|---:|---:|---:|
| Train | 7 | 21 | 28 |
| Validation | 0 | 6 | 6 |
| Test | 0 | 6 | 6 |
| Total | 7 | 33 | 40 |

The 33 new tasks come from different conversations. All 18 conversations used
by the earlier MTRAG reference calibration are excluded. Selection is balanced
across the four collections where the real candidate pool permits it.

`test-manifest.json` is frozen but unopened. It must not be submitted to the
GPU runner until model architecture, features, and thresholds are final.

## Reproduction

```bash
python -m benchmarks.freeze_mtrag_data_splits \
  --coverage /tmp/mtrag-coverage.json \
  --source-dataset /tmp/mtrag-rag.jsonl \
  --exclude-manifest results/mtrag-quality-calibration-v1/manifest.json \
  --output-dir results/mtrag-natural-splits-v1
```

SHA-256:

- Train manifest: `60d7b9363a38d228d256920f96e5fcbcb021f4558b0228e62dd179baf4bc2fc5`
- Validation manifest: `8d531b80e99fb72d45164c22f48825d83d28c8e77d53c743abb983d5340d3eb8`
- Test manifest: `1e74a4229fb364bc06856862e803f16731c7279409dc36f8f435883332e36397`
- Split plan: `4fe9af032a7a38ad8a6eb351afbaa25a28555fbac8fc1daa4d888c06775f59e9`

## Training and validation reference run

The reference launcher starts vLLM once and records the 21 new training answers
followed by the six validation answers. It never reads `test-manifest.json`.
The 8,192-token server window leaves room for the 384-token completion after the
longest selected prompt (3,853 tokens).

```bash
REFERENCE_JOB_ID=$(sbatch --parsable \
  benchmarks/run_mtrag_reference_splits.slurm)
echo "$REFERENCE_JOB_ID"
```

Monitor it with:

```bash
squeue -j "$REFERENCE_JOB_ID"
tail -f \
  "/vol/bitbucket/$USER/cacheselect-server-logs/mtrag-reference-$REFERENCE_JOB_ID.out"
```
