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
