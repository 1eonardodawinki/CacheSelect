# CacheSelect baseline matrix 268672, 269138

## Validation

- 18 result files, 18 complete request ledgers, and 84 request observations (42 APC-off/on pairs) validated.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- CacheSelect commit: `137d7f1ee0a8, 2c034ff380d3`; vLLM: `0.0.0+d6dbdb9b0`.
- APC-off cached-token counts are all zero; every cold APC-on request also has zero cached tokens.
- Full request inputs and raw outputs are present in every ledger.

## Reuse-eligible TTFT

The first request from each fresh server is excluded here because no cache can
exist yet. Values are means of three repetition means.

| Workload | APC off (ms) | APC on (ms) | Reduction | Speedup | APC-on hit rate |
| --- | --- | --- | --- | --- | --- |
| rag | 51.83 | 37.49 | 27.7% | 1.38× | 51.3% |
| periodic_agent | 61.62 | 51.00 | 17.2% | 1.21× | 34.8% |
| chat | 47.74 | 32.82 | 31.2% | 1.45× | 45.1% |

## Cold-start control

| Workload | APC off (ms) | APC on (ms) | Difference |
| --- | --- | --- | --- |
| rag | 49.49 | 49.52 | -0.1% |
| periodic_agent | 60.00 | 60.38 | -0.6% |
| chat | 29.13 | 29.11 | 0.1% |

The near-zero cold-start differences support attributing the warm-request
improvements to cache reuse rather than a general difference between the two
server configurations.

## Request-level behavior

| Request | Transition | Cached/prompt | Hit rate | Off TTFT | On TTFT | Reduction |
| --- | --- | --- | --- | --- | --- | --- |
| rag-01 | document_replacement | 80/176 | 45.5% | 52.25 | 37.28 | 28.7% |
| rag-02 | document_reorder | 32/176 | 18.2% | 51.91 | 49.54 | 4.6% |
| rag-03 | query_replacement | 160/177 | 90.4% | 51.32 | 25.64 | 50.0% |
| periodic-01 | sliding_window | 80/230 | 34.8% | 60.88 | 50.18 | 17.6% |
| periodic-02 | sliding_window | 80/230 | 34.8% | 61.79 | 51.12 | 17.3% |
| periodic-03 | sliding_window | 80/230 | 34.8% | 61.83 | 51.23 | 17.1% |
| periodic-04 | sliding_window | 80/230 | 34.8% | 62.09 | 51.10 | 17.7% |
| periodic-05 | sliding_window | 80/230 | 34.8% | 61.49 | 51.35 | 16.5% |
| chat-01 | append_turn | 48/97 | 49.5% | 36.74 | 29.43 | 19.9% |
| chat-02 | append_turn | 96/149 | 64.4% | 56.52 | 31.33 | 44.6% |
| chat-03 | edit_history | 32/149 | 21.5% | 49.95 | 37.71 | 24.5% |

## Quality as recorded

| Workload | Pass off | Pass on | Mean score off | Mean score on | Length-stop rate |
| --- | --- | --- | --- | --- | --- |
| rag | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% |
| periodic_agent | 0.0% | 0.0% | 0.625 | 0.625 | 0.0% |
| chat | 50.0% | 50.0% | 0.888 | 0.888 | 0.0% |

No paired request changed its recorded quality score between APC off and on.

No response ended because of the configured completion-token limit, so truncation does not qualify these quality scores. The chat failures are missing required constraints and should be inspected separately from truncation. Failures that occur identically
in both APC modes are model, workload, or evaluation-configuration limitations
rather than observed APC regressions.

## Interpretation and limitations

- Native vLLM APC works best when changes occur late in the prompt. The RAG
  query replacement reused 160 tokens, while document reordering reused only
  32 despite retaining the same documents.
- Sliding periodic windows reused only the stable 80-token prefix; unchanged
  rows that moved position were not recovered by native APC.
- Append-only chat reused 48 and then 96 tokens. Editing early history reduced
  reuse to 32 tokens, even though most later text remained identical.
- Only three repetitions were collected, APC-off always ran before APC-on
  within each workload job, and each workload ran on one GPU. The paired cold
  controls are stable, but final experiments should randomize mode order and
  include more repetitions and model/context scales.
- These measurements establish the native-vLLM baseline. They do not yet test
  CacheSelect or approximate/non-prefix reuse.

Generated files: `request_metrics.csv`, `run_summary.csv`,
`paired_request_comparison.csv`, `workload_summary.csv`,
`request_summary.csv`, and the plots in this directory.
