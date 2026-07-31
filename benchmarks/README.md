# Initial CacheSelect baselines

This experiment records the information available at vLLM's standard
OpenAI-compatible boundary before implementing a reuse planner.

It creates three controlled workloads:

- RAG document replacement, reordering and query changes.
- A periodic Data Analyst with a sliding data window.
- Append-only chat followed by an edited-history request.

All three are defined from scratch in `benchmarks/workloads.py`; there is no
legacy agent harness. RAG has document-level identities, the periodic agent has
row-level identities, and chat has turn-level identities. The chat workload is
a fixed conversation replay rather than a live conversation so every cache
policy receives exactly the same prompts. Its append transitions are positive
controls for native vLLM prefix caching.

Each request records the original structured API payload, a text-free structural
summary, vLLM's rendered prompt, its prompt token IDs, cached-token counts,
per-request timing, usage, deterministic answer score and the raw response. The
generated traces contain benchmark-only segment, change and answer labels;
these labels are not sent to vLLM.

The runner also writes an append-only request ledger to
`request_logs/<run-id>.jsonl`. It persists each full request before contacting
vLLM, then appends the full response or failure. Use `--request-log-dir` to
place ledgers on experiment storage and `--run-id` to supply a stable run name.

## 1. Generate traces

From the repository root:

```bash
python -m benchmarks.generate_traces
```

This writes `benchmarks/traces/{rag,periodic_agent,chat}.json`.

## GPU smoke experiment

Before launching the full baseline matrix, run the four-request RAG trace once
with APC disabled and once with APC enabled. The Slurm job starts a fresh vLLM
server for each condition, waits for its health endpoint, records the complete
requests and responses, and shuts the server down before changing cache mode.

The default smoke model is `Qwen/Qwen2.5-1.5B-Instruct`, which fits an Imperial
A16. From the Imperial submission host:

```bash
mkdir -p /vol/bitbucket/$USER/cacheselect-server-logs
cd ~/DeltaCache
sbatch benchmarks/run_rag_smoke.slurm
```

The job prints the result, request-ledger and vLLM server-log directories when
it finishes. Find and follow its top-level log with:

```bash
squeue -u "$USER"
ls -lt /vol/bitbucket/$USER/cacheselect-server-logs/rag-smoke-*.out
tail -f /vol/bitbucket/$USER/cacheselect-server-logs/rag-smoke-<job-id>.out
```

`tail -f` only follows the log; stopping it does not stop the Slurm job. The
smoke run is successful when both conditions save four observations, both
request ledgers report four completed requests, APC-off reports zero cached
tokens, and the final summary prints `RAG smoke experiment completed
successfully`.

## Full baseline matrix

After the smoke experiment passes, submit the complete controlled baseline as
one Slurm array:

```bash
cd ~/DeltaCache
MATRIX_JOB_ID=$(sbatch --parsable benchmarks/run_baseline_matrix.slurm)
echo "$MATRIX_JOB_ID"
```

The array contains three submitted tasks to stay below Imperial's per-user job
submission quota. Each task owns one workload (`rag`, `periodic_agent`, or
`chat`) and sequentially runs two APC modes times three repetitions. The tasks
use at most three GPUs concurrently, while every one of the 18 measured
conditions still starts with a fresh vLLM process and writes a result JSON, a
hardware/run manifest, a complete request ledger, and a vLLM log.

Check progress without attaching to a live log:

```bash
squeue -j "$MATRIX_JOB_ID"
grep -h "Baseline condition completed successfully" \
  /vol/bitbucket/$USER/cacheselect-server-logs/baseline-"$MATRIX_JOB_ID"_*.out \
  2>/dev/null | wc -l
```

The successful-run count reaches 18 when the matrix finishes. Results are under
`/vol/bitbucket/$USER/cacheselect-results/baseline-$MATRIX_JOB_ID`, with
corresponding request and server logs under their `cacheselect-request-logs`
and `cacheselect-server-logs` roots.

## 2. Start vLLM with APC enabled

Use the vLLM checkout in this repository. The two observability flags are
required for cached-token counts and server-side TTFT:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics
```

Run one trace:

```bash
python -m benchmarks.run_vllm_baseline \
  --trace benchmarks/traces/rag.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --apc-label on \
  --output benchmarks/results/rag_apc-on.json
```

Repeat the runner for `periodic_agent.json` and `chat.json`.

Audit the resulting ledger before analysing a run:

```bash
python -m observability.validate_ledger request_logs/<run-id>.jsonl
```

## 3. Restart vLLM with APC disabled

APC is enabled by default in this vLLM version, so the negative form must be
explicit:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --no-enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics
```

Rerun each trace with `--apc-label off` and a distinct output filename.

## Privacy

The diagnostic files deliberately retain complete prompts and responses. API
keys are never recorded. Use only synthetic or public benchmark data. A
production deployment should default to structural summaries, hashes and
lengths rather than raw content, but CacheSelect research runs retain the raw
input and output for reproducibility.
