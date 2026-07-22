# What's in this folder

Output from running `periodic_agents_approach/harness/run_baseline.py` for each of the three agents, once with
vLLM's prefix caching on and once off, so the two can be compared. Every filename
starts with `{agent}_{tag}` (currently `run1` for all three) so results for different
agents or runs never overwrite each other.

## File types

**`{agent}_trace_{tag}.json`** — the generated trace itself: the fixed system
prompt, the per-activation user messages (e.g. each price/coverage check window),
and the ground truth (which activation should trigger an alert, and on what).
Saved by the `--apc on` run and reloaded by the `--apc off` run via `--trace-file`,
so both runs see byte-identical prompts — any difference between them is down to
caching, not the input changing.

**`{agent}_{tag}_apc-{on|off}.csv`** and **`.json`** — the actual measurements, one
row per activation, same data in both formats (CSV to eyeball, JSON to reload for
plotting). Columns:

| column | meaning |
|---|---|
| `activation_index` | which activation in the trace this row is (0, 1, 2, …) |
| `apc` | `"on"` or `"off"` — which run produced this row |
| `prompt_token_count` | prompt length in the real model's tokens |
| `num_cached_tokens` | how many of those tokens vLLM served from cache instead of recomputing — the direct explanation for any TTFT difference, and the number DeltaCache is trying to raise |
| `ttft_seconds` | time to first token — the latency number that actually matters |
| `output_text` | the model's raw response, kept for spot-checking |
| `expect_flag` / `anomaly_entity` | the ground truth for this activation — should it have alerted, and about what |
| `predicted_flagged_entities` | what the model actually flagged |
| `correct` | whether the model's answer matched the ground truth |

**`{agent}_{tag}_apc-{on|off}_transcript.txt`** — human-readable version of the
same run: the exact system + user prompt sent to the model and its raw output,
written out in full for every activation, so you can read what actually happened
without cross-referencing the CSV against the trace JSON by hand.
