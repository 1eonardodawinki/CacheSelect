# Combined prompt-length calibration analysis

This directory contains analysis only. It selects the completed 256- and
1,024-token conditions from Slurm job `269226` and the six 4,096-token
conditions from job `269267`. The later 4,096-token early/APC-off result
supersedes its duplicate from the interrupted first job.

Recreate it from the repository root:

```bash
python -m pip install -r benchmarks/requirements-analysis.txt
python -m benchmarks.analyze_length_calibration \
  --input-root results/length-calibration-269226 \
  --input-root results/length-calibration-269267 \
  --output-dir results/length-calibration-269226-269267/analysis
```

The analyzer validates all 18 selected conditions, 18 complete full-request
ledgers, 36 requests, exact rendered prompt lengths, cache invariants, natural
completions, semantic answer quality, and APC-off/on prompt-token identity.
