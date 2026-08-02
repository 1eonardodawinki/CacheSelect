#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-rag-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/rag-smoke-%j.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_SMOKE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${CACHESELECT_SMOKE_PORT:-8000}"
JOB_TAG="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"

RESULT_DIR="$STORAGE_ROOT/cacheselect-results/rag-smoke-$JOB_TAG"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/rag-smoke-$JOB_TAG"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/rag-smoke-$JOB_TAG"
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

echo "CacheSelect RAG smoke experiment"
echo "job_id=$JOB_TAG"
echo "host=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "project_commit=$(git rev-parse HEAD)"
echo "model=$MODEL"
echo "planner_mode=shadow"
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
  local apc_label="$1"
  local cache_flag="$2"
  local server_log="$SERVER_LOG_DIR/vllm-apc-$apc_label.log"
  local result_path="$RESULT_DIR/rag-apc-$apc_label.json"
  local run_id="rag-smoke-$JOB_TAG-apc-$apc_label"

  echo "Starting fresh vLLM server: apc=$apc_label"
  setsid vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-prompt-tokens-details \
    --enable-per-request-metrics \
    "$cache_flag" \
    >"$server_log" 2>&1 &
  SERVER_PID=$!

  wait_for_server "$server_log"
  echo "vLLM is healthy: apc=$apc_label pid=$SERVER_PID"

  python -m benchmarks.run_vllm_baseline \
    --trace benchmarks/traces/rag.json \
    --model "$MODEL" \
    --apc-label "$apc_label" \
    --planner-mode shadow \
    --run-id "$run_id" \
    --request-log-dir "$REQUEST_LOG_DIR" \
    --output "$result_path"

  stop_server
  echo "Completed condition: apc=$apc_label result=$result_path"
}

# A fresh process is required between conditions so APC-on cannot inherit any
# GPU KV state from APC-off (or vice versa).
run_condition off --no-enable-prefix-caching
run_condition on --enable-prefix-caching

python - "$RESULT_DIR/rag-apc-off.json" "$RESULT_DIR/rag-apc-on.json" <<'PY'
import json
import sys
from pathlib import Path

for result_name in sys.argv[1:]:
    path = Path(result_name)
    result = json.loads(path.read_text())
    observations = result["observations"]
    cached = [item["cached_tokens"] for item in observations]
    ttft = [
        (item.get("server_metrics") or {}).get("time_to_first_token_ms")
        for item in observations
    ]
    quality = [item["quality"]["passed"] for item in observations]
    decisions = [
        item["policy_metadata"]["planner_decision"]["policy"]
        for item in observations
    ]
    ledger = result["request_ledger_summary"]
    if len(observations) != 4:
        raise SystemExit(f"{path}: expected 4 observations, got {len(observations)}")
    if ledger != {"started": 4, "completed": 4, "failed": 0}:
        raise SystemExit(f"{path}: unexpected ledger summary {ledger}")
    if result.get("planner", {}).get("mode") != "shadow":
        raise SystemExit(f"{path}: shadow planner metadata is missing")
    expected_decisions = [
        "FULL_RECOMPUTE",
        "VLLM_NATIVE_APC",
        "VLLM_NATIVE_APC",
        "VLLM_NATIVE_APC",
    ]
    if decisions != expected_decisions:
        raise SystemExit(
            f"{path}: expected planner decisions {expected_decisions}, got {decisions}"
        )
    print(
        f"{path.name}: cached_tokens={cached} ttft_ms={ttft} "
        f"quality_passed={quality} planner_decisions={decisions}"
    )
PY

echo "RAG smoke experiment completed successfully"
echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_logs=$SERVER_LOG_DIR"
