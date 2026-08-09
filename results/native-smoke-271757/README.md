# Guarded KV-copy smoke run 271757

This directory archives the first successful physical KV-copy execution by
CacheSelect inside vLLM. The experiment ran on one Imperial NVIDIA A16 with
`Qwen/Qwen2.5-1.5B-Instruct`, project commit `7b04478`, vLLM build
`0.0.0+d6dbdb9b0`, APC enabled and a 16-token block size.

## Design

The job started a fresh vLLM process for each condition:

1. `native-apc`: unmodified native prefix-cache execution;
2. `cacheselect`: CacheSelect planning and repair selection in shadow mode;
3. `cacheselect-copy`: the same plan with guarded physical KV copying enabled.

Each server executed the same four-request RAG trace. The document-reorder
transition exposed five resident, block-aligned candidates representing 80
tokens. The configured conservative selector still marked all 80 tokens for
normal recomputation.

## Result

| Condition | Cached tokens by request | Copied on reorder | Reorder TTFT |
| --- | --- | ---: | ---: |
| Native APC | `[0, 80, 32, 160]` | — | 49.807 ms |
| CacheSelect shadow | `[0, 80, 32, 160]` | 0 blocks / 0 tokens | 49.656 ms |
| CacheSelect copy | `[0, 80, 32, 160]` | 5 blocks / 80 tokens | 57.133 ms |

All 12 requests passed their deterministic quality checks. Prompt token IDs,
native APC hits, generated outputs and quality results were identical across
all three conditions. Each request ledger contains four starts, four
completions and zero failures.

The copy-enabled condition proves that the scheduler retained the source
blocks and the GPU worker submitted the expected five physical source-to-target
copies. It does not demonstrate an inference speedup: vLLM still ran full
prefill and overwrote every copied block. The extra copy therefore added work;
the single-run TTFT difference is not a performance comparison.

## Contents

- `results/`: full prompts, responses, timings, runtime plans and copy counts;
- `request-logs/`: append-only full request/response ledgers;
- `server-logs/`: one vLLM log for each fresh server;
- `slurm-logs/`: the top-level experiment transcript and assertions.

Validate each ledger from the repository root with:

```bash
python -m observability.validate_ledger \
  results/native-smoke-271757/request-logs/<ledger>.jsonl
```
