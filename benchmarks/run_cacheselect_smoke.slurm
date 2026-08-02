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
MINIMUM_NATIVE_PREFIX_TOKENS="${CACHESELECT_NATIVE_PREFIX_MIN_TOKENS:-64}"
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
echo "minimum_native_prefix_tokens=$MINIMUM_NATIVE_PREFIX_TOKENS"
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

run_condition() {
  local condition="$1"
  local planner_mode="$2"
  local server_log="$SERVER_LOG_DIR/vllm-$condition.log"
  local result_path="$RESULT_DIR/rag-$condition.json"
  local run_id="native-smoke-$JOB_TAG-$condition"
  local -a cacheselect_args=()

  if [[ "$planner_mode" == "vllm" ]]; then
    cacheselect_args=(
      --cacheselect-minimum-native-prefix-tokens
      "$MINIMUM_NATIVE_PREFIX_TOKENS"
    )
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
    --minimum-native-prefix-tokens "$MINIMUM_NATIVE_PREFIX_TOKENS" \
    --run-id "$run_id" \
    --request-log-dir "$REQUEST_LOG_DIR" \
    --output "$result_path"

  stop_server
  echo "Completed condition: condition=$condition result=$result_path"
}

# Each condition gets a fresh vLLM process so it cannot inherit GPU KV state.
run_condition native-apc off
run_condition cacheselect vllm

python - "$RESULT_DIR/rag-native-apc.json" "$RESULT_DIR/rag-cacheselect.json" <<'PY'
import json
import sys
from pathlib import Path

native_path = Path(sys.argv[1])
cacheselect_path = Path(sys.argv[2])
native = json.loads(native_path.read_text())
cacheselect = json.loads(cacheselect_path.read_text())

for path, result in ((native_path, native), (cacheselect_path, cacheselect)):
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
    "FULL_RECOMPUTE",
    "VLLM_NATIVE_APC",
]
if policies != expected_policies:
    raise SystemExit(
        f"{cacheselect_path}: expected policies {expected_policies}, got {policies}"
    )

for policy, candidate, cached in zip(policies, candidates, selected):
    if policy == "FULL_RECOMPUTE" and cached != 0:
        raise SystemExit(
            f"{cacheselect_path}: full recomputation selected but cached={cached}"
        )
    if policy == "VLLM_NATIVE_APC" and cached != candidate:
        raise SystemExit(
            f"{cacheselect_path}: native APC selected {cached}, candidate={candidate}"
        )

if not any(
    policy == "FULL_RECOMPUTE" and candidate > 0
    for policy, candidate in zip(policies, candidates)
):
    raise SystemExit(
        f"{cacheselect_path}: no non-zero native hit was rejected by CacheSelect"
    )

for path, result in ((native_path, native), (cacheselect_path, cacheselect)):
    observations = result["observations"]
    cached = [item["cached_tokens"] for item in observations]
    ttft = [
        (item.get("server_metrics") or {}).get("time_to_first_token_ms")
        for item in observations
    ]
    runtime = [item.get("runtime_policy") for item in observations]
    print(f"{path.name}: cached_tokens={cached} ttft_ms={ttft} runtime={runtime}")
PY

echo "CacheSelect native execution smoke completed successfully"
echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_logs=$SERVER_LOG_DIR"
