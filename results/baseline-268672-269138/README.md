# Combined corrected baseline analysis

This directory contains analysis only. It selects the three original chat
repetitions from Slurm job `268672` and the corrected 96-completion-token RAG
and periodic-agent repetitions from job `269138`. Later input roots supersede
matching workload/APC/repetition conditions without modifying either raw
bundle.

Recreate it from the repository root:

```bash
python -m pip install -r benchmarks/requirements-analysis.txt
python -m benchmarks.analyze_baseline \
  --input-root results/baseline-268672 \
  --input-root results/baseline-269138 \
  --output-dir results/baseline-268672-269138/analysis
```

The analyzer validates the selected 18 result files, 18 complete full-request
ledgers, cache invariants, and APC-off/on request pairing before producing the
CSV tables, Markdown summary, provenance record, and PNG/PDF plots.
