#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-native-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/native-smoke-%j.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_SMOKE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${CACHESELECT_SMOKE_PORT:-8000}"
JOB_TAG="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"

RESULT_DIR="$STORAGE_ROOT/cacheselect-results/native-smoke-$JOB_TAG"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/native-smoke-$JOB_TAG"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/native-smoke-$JOB_TAG"
TMP_ROOT="$STORAGE_ROOT/tmp"

mkdir -p \
  "$RESULT_DIR" \
  "$REQUEST_LOG_DIR" \
  "$SERVER_LOG_DIR" \
  "$TMP_ROOT" \
  "$STORAGE_ROOT/hf-cache" \
  "$STORAGE_ROOT/uv-cache"

source "$CUDA_SETUP"
source "$VENV_ROOT/bin/activate"
cd "$PROJECT_ROOT"

export TMPDIR="$TMP_ROOT"
export HF_HOME="$STORAGE_ROOT/hf-cache"
export UV_CACHE_DIR="$STORAGE_ROOT/uv-cache"
export PYTHONUNBUFFERED=1

command -v nvidia-smi >/dev/null
command -v curl >/dev/null
command -v vllm >/dev/null
test -f benchmarks/traces/rag.json

echo "CacheSelect native execution smoke experiment"
echo "job_id=$JOB_TAG"
echo "host=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "project_commit=$(git rev-parse HEAD)"
echo "model=$MODEL"
echo "vllm=$(command -v vllm)"
echo "vllm_version=$(vllm --version)"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader

SERVER_PID=""

stop_server() {
  if [[ -z "$SERVER_PID" ]]; then
    return
  fi

  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null \
      || kill -TERM "$SERVER_PID" 2>/dev/null \
      || true
    for _ in {1..30}; do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi

  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -KILL -- "-$SERVER_PID" 2>/dev/null \
      || kill -KILL "$SERVER_PID" 2>/dev/null \
      || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

trap stop_server EXIT INT TERM

wait_for_server() {
  local server_log="$1"
  for _ in {1..600}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited before becoming healthy" >&2
      tail -n 200 "$server_log" >&2 || true
      return 1
    fi
    if curl --fail --silent --show-error \
      "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  echo "Timed out waiting for vLLM health endpoint" >&2
  tail -n 200 "$server_log" >&2 || true
  return 1
}

# Run one isolated server condition with optional CacheSelect copy execution.
run_condition() {
  local condition="$1"
  local planner_mode="$2"
  local execute_partial_reuse="$3"
  local server_log="$SERVER_LOG_DIR/vllm-$condition.log"
  local result_path="$RESULT_DIR/rag-$condition.json"
  local run_id="native-smoke-$JOB_TAG-$condition"
  local -a cacheselect_args=()

  if [[ "$planner_mode" == "vllm" ]]; then
    # Radius one leaves the calibrated reorder transition with 32 reusable rows.
    cacheselect_args=(
      --enable-cacheselect
      --cacheselect-repair-selector edit_proximity
      --cacheselect-edit-radius 1
    )
  fi
  if [[ "$execute_partial_reuse" == "true" ]]; then
    # The first partial execution path deliberately excludes compiled graphs.
    cacheselect_args+=(--cacheselect-execute-partial-reuse --enforce-eager)
  fi

  echo "Starting fresh vLLM server: condition=$condition"
  setsid vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-per-request-metrics \
    "${cacheselect_args[@]}" \
    >"$server_log" 2>&1 &
  SERVER_PID=$!

  wait_for_server "$server_log"
  echo "vLLM is healthy: condition=$condition pid=$SERVER_PID"

  python -m benchmarks.run_vllm_baseline \
    --trace benchmarks/traces/rag.json \
    --model "$MODEL" \
    --apc-label on \
    --planner-mode "$planner_mode" \
    --run-id "$run_id" \
    --request-log-dir "$REQUEST_LOG_DIR" \
    --output "$result_path"

  stop_server
  echo "Completed condition: condition=$condition result=$result_path"
}

# Each condition gets a fresh vLLM process so it cannot inherit GPU KV state.
run_condition native-apc off false
run_condition cacheselect vllm false
run_condition cacheselect-copy vllm true

python - \
  "$RESULT_DIR/rag-native-apc.json" \
  "$RESULT_DIR/rag-cacheselect.json" \
  "$RESULT_DIR/rag-cacheselect-copy.json" <<'PY'
import json
import sys
from pathlib import Path

native_path = Path(sys.argv[1])
cacheselect_path = Path(sys.argv[2])
copy_path = Path(sys.argv[3])
native = json.loads(native_path.read_text())
cacheselect = json.loads(cacheselect_path.read_text())
copy_enabled = json.loads(copy_path.read_text())

results = (
    (native_path, native),
    (cacheselect_path, cacheselect),
    (copy_path, copy_enabled),
)
for path, result in results:
    observations = result["observations"]
    ledger = result["request_ledger_summary"]
    if len(observations) != 4:
        raise SystemExit(f"{path}: expected 4 observations, got {len(observations)}")
    if ledger != {"started": 4, "completed": 4, "failed": 0}:
        raise SystemExit(f"{path}: unexpected ledger summary {ledger}")
    if not all(item["quality"]["passed"] for item in observations):
        raise SystemExit(f"{path}: at least one quality check failed")

observations = cacheselect["observations"]
policies = [item["runtime_policy"]["policy"] for item in observations]
candidates = [
    item["runtime_policy"]["native_cached_tokens"] for item in observations
]
selected = [item["cached_tokens"] for item in observations]
expected_policies = [
    "FULL_RECOMPUTE",
    "VLLM_NATIVE_APC",
    "VLLM_NATIVE_APC",
    "VLLM_NATIVE_APC",
]
if policies != expected_policies:
    raise SystemExit(
        f"{cacheselect_path}: expected policies {expected_policies}, got {policies}"
    )

for policy, candidate, cached in zip(policies, candidates, selected):
    if candidate == 0 and policy != "FULL_RECOMPUTE":
        raise SystemExit(
            f"{cacheselect_path}: no hit should select full recomputation"
        )
    if candidate > 0 and policy != "VLLM_NATIVE_APC":
        raise SystemExit(
            f"{cacheselect_path}: native hit {candidate} was not preserved"
        )
    if cached != candidate:
        raise SystemExit(
            f"{cacheselect_path}: selected cached={cached}, candidate={candidate}"
        )

native_cached = [item["cached_tokens"] for item in native["observations"]]
if selected != native_cached:
    raise SystemExit(
        f"CacheSelect changed APC hits: native={native_cached}, selected={selected}"
    )

copy_cached = [item["cached_tokens"] for item in copy_enabled["observations"]]
if copy_cached != native_cached:
    raise SystemExit(
        f"CacheSelect copies changed APC hits: native={native_cached}, "
        f"copy={copy_cached}"
    )

behavior = [
    (
        item["prompt_token_ids"],
        item["cached_tokens"],
        item["output_text"],
        item["quality"],
    )
    for item in native["observations"]
]
for path, result in ((cacheselect_path, cacheselect), (copy_path, copy_enabled)):
    observed_behavior = [
        (
            item["prompt_token_ids"],
            item["cached_tokens"],
            item["output_text"],
            item["quality"],
        )
        for item in result["observations"]
    ]
    if observed_behavior != behavior:
        raise SystemExit(f"{path}: CacheSelect changed request behavior")

transition_observations = observations[1:]
plans = [
    (item.get("runtime_policy") or {}).get("partial_reuse_plan")
    for item in transition_observations
]
if any(plan is None for plan in plans):
    raise SystemExit(
        f"{cacheselect_path}: every transition must report a partial reuse plan"
    )

expected_candidate_tokens = [0, 80, 0]
candidate_tokens = [plan["candidate_token_count"] for plan in plans]
resident_tokens = [plan["resident_candidate_token_count"] for plan in plans]
if candidate_tokens != expected_candidate_tokens:
    raise SystemExit(
        f"{cacheselect_path}: expected aligned candidate tokens "
        f"{expected_candidate_tokens}, got {candidate_tokens}"
    )
if resident_tokens != expected_candidate_tokens:
    raise SystemExit(
        f"{cacheselect_path}: expected all candidates to remain resident, "
        f"got resident tokens {resident_tokens}"
    )

expected_mappings = [
    [],
    [(3, 5), (4, 6), (5, 7), (6, 8), (10, 10)],
    [],
]
actual_mappings = [
    [
        (candidate["source_block_index"], candidate["target_block_index"])
        for candidate in plan["candidates"]
    ]
    for plan in plans
]
if actual_mappings != expected_mappings:
    raise SystemExit(
        f"{cacheselect_path}: expected block mappings {expected_mappings}, "
        f"got {actual_mappings}"
    )
if "source_block_id" in json.dumps(plans):
    raise SystemExit(f"{cacheselect_path}: internal GPU block IDs leaked")

shadow_metrics = cacheselect["observations"][2]["server_metrics"]
copy_metrics = copy_enabled["observations"][2]["server_metrics"]
expected_repair_metrics = {
    "cacheselect_candidate_tokens": 80,
    "cacheselect_repair_tokens": 48,
    "cacheselect_skipped_repair_tokens": 32,
}
for name, expected in expected_repair_metrics.items():
    if shadow_metrics.get(name) != expected:
        raise SystemExit(
            f"{cacheselect_path}: expected {name}={expected}, "
            f"got {shadow_metrics.get(name)}"
        )
    if copy_metrics.get(name) != expected:
        raise SystemExit(
            f"{copy_path}: expected {name}={expected}, "
            f"got {copy_metrics.get(name)}"
        )

expected_copy_counts = {
    cacheselect_path: (shadow_metrics, 0, 0),
    copy_path: (copy_metrics, 5, 80),
}
for path, (metrics, expected_blocks, expected_tokens) in expected_copy_counts.items():
    actual = (
        metrics.get("cacheselect_copied_blocks"),
        metrics.get("cacheselect_copied_tokens"),
    )
    expected = (expected_blocks, expected_tokens)
    if actual != expected:
        raise SystemExit(f"{path}: expected copy counts {expected}, got {actual}")

# Shadow mode stays disabled while the enabled condition runs compacted spans.
expected_execution_decisions = {
    cacheselect_path: (
        shadow_metrics,
        {
            "cacheselect_execution_eligible": False,
            "cacheselect_execution_reason": "execution_disabled",
            "cacheselect_reused_batch_rows": 32,
            "cacheselect_compute_batch_rows": 144,
            "cacheselect_compute_span_count": 0,
            "cacheselect_compacted_batch_built": False,
            "cacheselect_compacted_batch_executed": False,
            "cacheselect_span_metadata_built": False,
            "cacheselect_span_metadata_count": 0,
        },
    ),
    copy_path: (
        copy_metrics,
        {
            "cacheselect_execution_eligible": True,
            "cacheselect_execution_reason": "eligible",
            "cacheselect_reused_batch_rows": 32,
            "cacheselect_compute_batch_rows": 112,
            "cacheselect_compute_span_count": 2,
            "cacheselect_compacted_batch_built": True,
            "cacheselect_compacted_batch_executed": True,
            "cacheselect_span_metadata_built": True,
            "cacheselect_span_metadata_count": 2,
        },
    ),
}
for path, (metrics, expected) in expected_execution_decisions.items():
    actual = {name: metrics.get(name) for name in expected}
    if actual != expected:
        raise SystemExit(
            f"{path}: expected execution decision {expected}, got {actual}"
        )

# Enabled execution must partition every original row into compute or reuse.
enabled_row_total = (
    copy_metrics["cacheselect_compute_batch_rows"]
    + copy_metrics["cacheselect_reused_batch_rows"]
)
if enabled_row_total != 144:
    raise SystemExit(
        f"{copy_path}: expected 144 accounted batch rows, got {enabled_row_total}"
    )

for path, result in results:
    observations = result["observations"]
    cached = [item["cached_tokens"] for item in observations]
    ttft = [
        (item.get("server_metrics") or {}).get("time_to_first_token_ms")
        for item in observations
    ]
    runtime = [item.get("runtime_policy") for item in observations]
    print(f"{path.name}: cached_tokens={cached} ttft_ms={ttft} runtime={runtime}")
print(
    "online_aligned_candidates="
    f"{candidate_tokens} resident_tokens={resident_tokens} "
    f"mappings={actual_mappings}"
)
print("repair_selection=48_repair_tokens/32_reused_rows")
print(
    "execution_gate=shadow:144_compute/32_candidate_reuse "
    "enabled:112_compute/32_reused"
)
print("compact_batch=shadow:not_built enabled:built_and_executed")
print("compute_spans=shadow:0 enabled:2")
print("span_metadata=shadow:0 enabled:2")
print("copy_execution=shadow:0_blocks/0_tokens enabled:5_blocks/80_tokens")
PY

echo "CacheSelect native execution smoke completed successfully"
echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_logs=$SERVER_LOG_DIR"
