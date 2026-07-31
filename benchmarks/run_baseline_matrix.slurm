#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-baseline
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/baseline-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Run this file as a Slurm array job}"
MATRIX_ID="${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"

# Imperial limits the number of submitted jobs per user. Each of these three
# tasks owns one workload and runs its six conditions sequentially, while the
# three tasks can use three GPUs concurrently.
WORKLOADS=(rag periodic_agent chat)
APC_LABELS=(off on)
APC_FLAGS=(--no-enable-prefix-caching --enable-prefix-caching)
WORKLOAD="${WORKLOADS[$TASK_ID]}"

# Array tasks can share a physical host, so they must not share a TCP port.
PORT=$((18000 + (MATRIX_ID % 1000) * 20 + TASK_ID))
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/baseline-$MATRIX_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/baseline-$MATRIX_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/baseline-$MATRIX_ID"
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
test -f "benchmarks/traces/$WORKLOAD.json"

PROJECT_COMMIT="$(git rev-parse HEAD)"
VLLM_VERSION="$(vllm --version)"
GPU_DESCRIPTION="$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)"

echo "CacheSelect baseline workload task"
echo "matrix_id=$MATRIX_ID task_id=$TASK_ID workload=$WORKLOAD"
echo "host=$(hostname) port=$PORT"
echo "project_commit=$PROJECT_COMMIT"
echo "model=$MODEL vllm_version=$VLLM_VERSION"
echo "gpu=$GPU_DESCRIPTION"

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

write_manifest() {
  local manifest_path="$1"
  local apc_label="$2"
  local repetition="$3"

  python - \
    "$manifest_path" \
    "$MATRIX_ID" \
    "$TASK_ID" \
    "${SLURM_JOB_ID:-}" \
    "$PROJECT_COMMIT" \
    "$MODEL" \
    "$VLLM_VERSION" \
    "$GPU_DESCRIPTION" \
    "$WORKLOAD" \
    "$apc_label" \
    "$repetition" \
    "$PORT" <<'PY'
import json
import socket
import sys

(
    manifest_path,
    matrix_job_id,
    array_task_id,
    slurm_job_id,
    project_commit,
    model,
    vllm_version,
    gpu,
    workload,
    apc,
    repetition,
    port,
) = sys.argv[1:]
manifest = {
    "matrix_job_id": matrix_job_id,
    "array_task_id": int(array_task_id),
    "slurm_job_id": slurm_job_id,
    "host": socket.gethostname(),
    "project_commit": project_commit,
    "model": model,
    "vllm_version": vllm_version,
    "gpu": gpu,
    "workload": workload,
    "apc": apc,
    "repetition": int(repetition),
    "port": int(port),
}
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
PY
}

run_condition() {
  local apc_label="$1"
  local apc_flag="$2"
  local repetition="$3"
  local run_key="${WORKLOAD}-apc-${apc_label}-rep-${repetition}"
  local result_path="$RESULT_DIR/$run_key.json"
  local manifest_path="$RESULT_DIR/$run_key.manifest.json"
  local server_log="$SERVER_LOG_DIR/$run_key.vllm.log"

  write_manifest "$manifest_path" "$apc_label" "$repetition"

  echo "Starting condition: $run_key"
  setsid vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-prompt-tokens-details \
    --enable-per-request-metrics \
    "$apc_flag" \
    >"$server_log" 2>&1 &
  SERVER_PID=$!

  wait_for_server "$server_log"
  echo "vLLM is healthy pid=$SERVER_PID"

  python -m benchmarks.run_vllm_baseline \
    --trace "benchmarks/traces/$WORKLOAD.json" \
    --model "$MODEL" \
    --base-url "http://127.0.0.1:$PORT" \
    --apc-label "$apc_label" \
    --run-id "baseline-$MATRIX_ID-$run_key" \
    --request-log-dir "$REQUEST_LOG_DIR" \
    --output "$result_path"

  stop_server
  sleep 2

  python - "$result_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text())
observations = result["observations"]
cached = [item["cached_tokens"] for item in observations]
ttft = [
    (item.get("server_metrics") or {}).get("time_to_first_token_ms")
    for item in observations
]
quality = [item["quality"]["passed"] for item in observations]
ledger = result["request_ledger_summary"]
if ledger["started"] != len(observations):
    raise SystemExit(f"request/ledger count mismatch: {ledger}")
if ledger["completed"] != len(observations) or ledger["failed"] != 0:
    raise SystemExit(f"incomplete request ledger: {ledger}")
print(f"cached_tokens={cached}")
print(f"ttft_ms={ttft}")
print(f"quality_passed={quality}")
PY

  echo "Baseline condition completed successfully: $run_key"
  echo "result=$result_path"
  echo "manifest=$manifest_path"
  echo "server_log=$server_log"
}

for apc_index in 0 1; do
  for repetition in 1 2 3; do
    run_condition \
      "${APC_LABELS[$apc_index]}" \
      "${APC_FLAGS[$apc_index]}" \
      "$repetition"
  done
done

echo "Baseline workload task completed successfully: $WORKLOAD"
echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_logs=$SERVER_LOG_DIR"
