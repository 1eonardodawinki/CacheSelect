# Initial native-vLLM baseline — Slurm 268672

This immutable bundle records the first complete native-vLLM baseline matrix.
It is retained as evidence of the initial benchmark infrastructure and must not
be replaced by later corrected runs.

## Configuration

- Date: 31 July 2026
- Slurm array job: `268672`
- Experiment code commit: `137d7f1ee0a870999921d92f09058946dbe8ab10`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM: `0.0.0+d6dbdb9b0`
- Hardware: NVIDIA A16, 15,356 MiB, driver `595.71.05`
- Workloads: controlled RAG, periodic Data Analyst, and fixed chat replay
- Policies: full recomputation and native vLLM automatic prefix caching
- Repetitions: three per workload/policy pair
- Completion limit: 48 tokens for every request

Every one of the 18 conditions started a fresh vLLM process. The three workload
tasks used separate GPUs, while APC-off/on comparisons within a workload used
the same GPU.

## Contents

- `results/`: 18 result JSON files and 18 run manifests.
- `request-logs/`: 18 complete append-only full input/output ledgers.
- `server-logs/`: 18 raw vLLM server logs.
- `slurm-logs/`: three top-level array-task logs.
- `analysis/`: validated CSV tables, Markdown summary, provenance, and PNG/PDF
  plots.

The raw artifacts contain synthetic prompts and outputs. They also preserve
experimental machine provenance—including Imperial hostnames, GPU UUIDs,
filesystem paths, and server addresses—but contain no credentials or API keys.

## Main observations

For requests eligible for reuse, native APC reduced mean TTFT relative to the
APC-off baseline by:

- 28.4% for RAG (`53.12 ms` to `38.03 ms`);
- 16.4% for the periodic agent (`60.15 ms` to `50.28 ms`);
- 31.2% for chat (`47.74 ms` to `32.82 ms`).

Cold-request APC-off/on differences were approximately zero, supporting the
interpretation that the warm-request improvements came from KV-cache reuse.
See `analysis/summary.md` and the request-level CSVs for the complete results.

## Limitations

- The prompts are small: approximately 97–230 rendered tokens.
- The 48-token completion cap truncated all periodic-agent outputs and the
  first RAG output in every condition. Their TTFT and cache measurements remain
  valid, but their absolute output-quality results are not final.
- Chat responses finished naturally, although the small model missed some
  required constraints in both cache modes.
- There are only three repetitions.
- APC-off always ran before APC-on for each workload.
- Each workload ran on one GPU, so comparisons across workloads also include a
  device assignment difference.
- This tests native exact-prefix caching only; it does not test CacheSelect or
  non-prefix KV reuse.

Corrected completion-limit measurements and prompt-length calibration results
must be stored as separate experiment bundles.

## Reproduce the analysis

From the repository root:

```bash
python -m pip install -r benchmarks/requirements-analysis.txt
python -m benchmarks.analyze_baseline \
  --input-root results/baseline-268672
```

The analyzer validates all 18 results and ledgers before recreating the tables,
summary, and plots under `analysis/`.
