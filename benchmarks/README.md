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

Before launching a full experiment, run the four-request RAG trace once with
APC disabled and once with APC enabled. The Slurm job starts a fresh vLLM
server for each condition, waits for its health endpoint, records the complete
requests and responses, and shuts the server down before changing cache mode.
It also runs the CacheSelect planner in shadow mode and checks its four expected
recommendations without applying them to either forced baseline condition.

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
request ledgers report four completed requests, the planner preflight matches
vLLM's prompt tokens, the expected recommendations are recorded, APC-off
reports zero cached tokens, and the final summary prints `RAG smoke experiment
completed successfully`.

## Native CacheSelect smoke experiment

After the baseline smoke passes, test the planner that now executes inside
vLLM. This job compares a fresh native-APC server with a fresh
CacheSelect-enabled server:

```bash
mkdir -p /vol/bitbucket/$USER/cacheselect-server-logs
cd ~/DeltaCache
git pull --ff-only
sbatch benchmarks/run_cacheselect_smoke.slurm
```

The job checks that every request records a runtime decision, every non-zero
native hit is preserved, CacheSelect and native APC report identical cache-hit
counts, all ledgers complete, and answer quality still passes. It also passes
each transition's previous request ID to vLLM and verifies the online shadow
locator's aligned block mappings. For the current RAG trace it expects 80
resident candidate tokens in the document-reorder transition and none in the
other two transitions. This is a correctness and observability smoke test, not
an expected speedup: every candidate is marked as requiring KV repair and is
still recomputed by native vLLM.

The online locator currently covers the deliberately narrow first milestone:
one full-attention KV group where scheduler, hash and physical block sizes are
equal. It indexes only full source blocks, requires an explicit source request
ID, checks cache-salt and LoRA compatibility, and confirms that each source
block is still resident. Candidates that cross source block boundaries remain
visible only to the offline analyzer until gathering/repacking is implemented.

Analyze the full blocks whose token content exists elsewhere in the previous
prompt but falls outside APC's exact-prefix hit:

```bash
python -m benchmarks.analyze_reuse_opportunities \
  --input results/native-smoke-269883/results/rag-cacheselect.json \
  --output results/native-smoke-269883/analysis/reuse-opportunity.json \
  --block-size 16
```

The analyzer reports content opportunity rather than safe KV reuse. An
identical block can have context-dependent KV state and may require selective
repair. It also distinguishes whole source blocks from candidates that need
token gathering or repacking.

## Shadow repair-policy matrix

The next experiment compares four repair policies on the controlled RAG trace:
full-block repair and edit-proximity repair with radii 0, 1 and 2. These policies
currently run in shadow mode. They record which candidate tokens would be
repaired or skipped, while vLLM still performs its normal exact computation.
Consequently, this experiment validates policy decisions and unchanged output
quality; it does not yet measure a CacheSelect speedup.

Submit the four-condition Slurm array from the Imperial submission host:

```bash
cd ~/DeltaCache
git pull --ff-only
REPAIR_JOB_ID=$(sbatch --parsable benchmarks/run_repair_policy_shadow.slurm)
echo "$REPAIR_JOB_ID"
```

Check the queue and count successful conditions:

```bash
squeue -j "$REPAIR_JOB_ID"
grep -h "Repair policy shadow condition completed successfully" \
  /vol/bitbucket/$USER/cacheselect-server-logs/repair-shadow-"$REPAIR_JOB_ID"_*.out \
  2>/dev/null | wc -l
```

The count reaches 4 when the array finishes. The controlled trace expects the
following shadow decisions:

| Condition | Candidate tokens | Repair tokens | Skipped tokens |
| --- | ---: | ---: | ---: |
| `full-block` | 80 | 80 | 0 |
| `edit-radius-0` | 80 | 0 | 80 |
| `edit-radius-1` | 80 | 48 | 32 |
| `edit-radius-2` | 80 | 80 | 0 |

Artifacts are written to
`/vol/bitbucket/$USER/cacheselect-results/repair-shadow-$REPAIR_JOB_ID`. Once
all four conditions have passed their individual checks, one array task writes
`repair-policy-summary.json` there. If that final aggregation is interrupted,
rerun it manually:

```bash
python -m benchmarks.analyze_repair_policy_shadow \
  --input-dir /vol/bitbucket/$USER/cacheselect-results/repair-shadow-"$REPAIR_JOB_ID" \
  --output /vol/bitbucket/$USER/cacheselect-results/repair-shadow-"$REPAIR_JOB_ID"/repair-policy-summary.json
```

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

The matrix sets completion limits explicitly: 96 tokens for RAG and the
periodic agent, and 48 for chat. The earlier `268672` baseline used the old
48-token default for all workloads. To preserve that dataset while correcting
only the affected RAG and periodic-agent quality runs, submit array tasks 0 and
1 as a new experiment:

```bash
cd ~/DeltaCache
CORRECTION_JOB_ID=$(sbatch --parsable --array=0-1%2 \
  benchmarks/run_baseline_matrix.slurm)
echo "$CORRECTION_JOB_ID"
```

This runs 12 conditions: two affected workloads, two APC modes, and three
repetitions. Chat does not need correction because all of its recorded
responses finished naturally below 48 tokens. Keep the old and corrected
artifact directories separate.

After downloading the artifact directories, build the validated tables,
Markdown summary, and report-ready PNG/PDF plots. A corrected partial run can
be layered over the original matrix by repeating `--input-root`; later roots
replace duplicate workload/APC/repetition conditions:

```bash
python -m pip install -r benchmarks/requirements-analysis.txt
python -m benchmarks.analyze_baseline \
  --input-root results/baseline-268672 \
  --input-root results/baseline-269138 \
  --output-dir results/baseline-268672-269138/analysis
```

For a single root, `--output-dir` defaults to `<input-root>/analysis`; it is
required when combining roots. The analyzer validates that all 18 selected
matrix conditions and full request ledgers are present, pairs APC-off and
APC-on requests, and reports cold-start controls separately from requests
eligible for reuse.

## Prompt-length calibration

The initial traces are deliberately small infrastructure tests. Before building
the new reuse policy, run a one-repetition native-vLLM calibration at
approximately 256, 1,024, and 4,096 rendered prompt tokens:

```bash
cd ~/DeltaCache
CALIBRATION_JOB_ID=$(sbatch --parsable \
  benchmarks/run_length_calibration.slurm)
echo "$CALIBRATION_JOB_ID"
```

The three array tasks each own one target length. For that length, the task
generates tokenizer-aware traces and runs early, middle, and late document edits
with APC off and on. Every condition starts a fresh vLLM server and contains a
cold donor request followed by one edited request. This produces 18 conditions
in total while using at most three GPUs concurrently.

Check progress with:

```bash
squeue -j "$CALIBRATION_JOB_ID"
grep -h "Length calibration condition completed successfully" \
  /vol/bitbucket/$USER/cacheselect-server-logs/length-calibration-"$CALIBRATION_JOB_ID"_*.out \
  2>/dev/null | wc -l
```

The count reaches 18 when all conditions finish. The job rejects prompts more
than 16 tokens from their target, truncated outputs, incomplete request ledgers,
APC-off cache hits, missing APC-on warm hits, and failed answer checks. Artifacts
are stored under:

- `cacheselect-results/length-calibration-<job-id>`
- `cacheselect-request-logs/length-calibration-<job-id>`
- `cacheselect-server-logs/length-calibration-<job-id>`
- `cacheselect-generated-traces/length-calibration-<job-id>`

This calibration is not the final evaluation. It checks that prompt-length and
edit-position scaling work before the implementation determines the definitive
lengths, repetitions, policy order, and workload matrix.

## Online locator calibration

After the online aligned-block smoke passes, run the same 256, 1,024 and 4,096
token early/middle/late edits through the CacheSelect-enabled vLLM server:

```bash
cd ~/DeltaCache
LOCATOR_JOB_ID=$(sbatch --parsable benchmarks/run_locator_calibration.slurm)
echo "$LOCATOR_JOB_ID"
```

The three array tasks each own one prompt length and execute its three edit
positions sequentially, using at most three GPUs. Every condition gets a fresh
server, a two-request cold-source/edit trace, a manifest, a complete request
ledger and full server metrics. The job verifies safe APC execution, answer
quality, stable request-scoped source lookup, agreement between the online
aligned locator and the offline opportunity analyzer, full candidate residency
and absence of physical GPU block IDs from API responses.

Check progress with:

```bash
squeue -j "$LOCATOR_JOB_ID"
grep -h "Locator calibration condition completed successfully" \
  /vol/bitbucket/$USER/cacheselect-server-logs/locator-calibration-"$LOCATOR_JOB_ID"_*.out \
  2>/dev/null | wc -l
```

The successful condition count reaches 9. Artifacts are stored below the
`locator-calibration-<job-id>` result, request-log, server-log and generated
trace directories under `/vol/bitbucket/$USER`.

After downloading those directories into
`results/locator-calibration-<job-id>`, validate and summarize them with:

```bash
python -m benchmarks.analyze_locator_calibration \
  --input-root results/locator-calibration-<job-id>
```

The analyzer requires all nine manifests, results and complete request ledgers.
It produces `analysis/locator-calibration.json`, a condition-level CSV and a
Markdown table comparing native recomputation with offline content matches,
online aligned candidates and resident candidates.

Analyse one complete run, or combine an interrupted initial run with a later
continuation in the same way:

```bash
python -m benchmarks.analyze_length_calibration \
  --input-root results/length-calibration-269226 \
  --input-root results/length-calibration-269267 \
  --output-dir results/length-calibration-269226-269267/analysis
```

The analyzer selects the later copy of duplicate conditions and validates the
complete 18-condition matrix, exact rendered prompt lengths, full request
ledgers, cache invariants, natural completions, and semantic answer quality.

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

To evaluate the initial CacheSelect planner without changing the forced
baseline policy, enable shadow mode:

```bash
python -m benchmarks.run_vllm_baseline \
  --trace benchmarks/traces/rag.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --apc-label on \
  --planner-mode shadow \
  --output benchmarks/results/rag_apc-on-shadow.json
```

Shadow mode tokenizes every request before execution, records the planner's
recommendation separately from the policy actually executed, and fails if the
local token IDs differ from vLLM's rendered prompt. It does not yet switch the
backend policy.

To record the safe fallback inside vLLM, start this repository's vLLM checkout
with:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-prefix-caching \
  --enable-cacheselect \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics
```

Then run the trace with `--planner-mode vllm`. The runner requires the server's
per-request CacheSelect fields and records them as `runtime_policy`. See
`cacheselect/README.md` for current limitations.

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
