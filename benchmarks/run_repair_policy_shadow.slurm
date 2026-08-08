#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-repair-shadow
#SBATCH --array=0-3%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/repair-shadow-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_SMOKE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Run this file as a Slurm array job}"
MATRIX_ID="${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"
CONDITIONS=(full-block edit-radius-0 edit-radius-1 edit-radius-2)
SELECTORS=(full_block edit_proximity edit_proximity edit_proximity)
RADII=(1 0 1 2)
CONDITION="${CONDITIONS[$TASK_ID]}"
SELECTOR="${SELECTORS[$TASK_ID]}"
RADIUS="${RADII[$TASK_ID]}"

# Array tasks can share one host, so derive a distinct port for every task.
PORT=$((22000 + (MATRIX_ID % 1000) * 10 + TASK_ID))
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/repair-shadow-$MATRIX_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/repair-shadow-$MATRIX_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/repair-shadow-$MATRIX_ID"
TMP_ROOT="$STORAGE_ROOT/tmp"
RESULT_PATH="$RESULT_DIR/rag-$CONDITION.json"
MANIFEST_PATH="$RESULT_DIR/rag-$CONDITION.manifest.json"
SERVER_LOG="$SERVER_LOG_DIR/rag-$CONDITION.vllm.log"

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

PROJECT_COMMIT="$(git rev-parse HEAD)"
VLLM_VERSION="$(vllm --version)"
GPU_DESCRIPTION="$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)"

echo "CacheSelect shadow repair-policy condition"
echo "matrix_id=$MATRIX_ID task_id=$TASK_ID condition=$CONDITION"
echo "selector=$SELECTOR edit_radius=$RADIUS"
echo "host=$(hostname) port=$PORT"
echo "project_commit=$PROJECT_COMMIT"
echo "model=$MODEL vllm_version=$VLLM_VERSION"
echo "gpu=$GPU_DESCRIPTION"

SERVER_PID=""

# Stop the condition's vLLM process without affecting other array tasks.
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

# Wait until the server is healthy or report its startup failure.
wait_for_server() {
  for _ in {1..600}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited before becoming healthy" >&2
      tail -n 200 "$SERVER_LOG" >&2 || true
      return 1
    fi
    if curl --fail --silent --show-error \
      "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  echo "Timed out waiting for vLLM health endpoint" >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  return 1
}

# Record enough environment detail to reproduce this policy condition.
write_manifest() {
  python - \
    "$MANIFEST_PATH" \
    "$MATRIX_ID" \
    "$TASK_ID" \
    "${SLURM_JOB_ID:-}" \
    "$PROJECT_COMMIT" \
    "$MODEL" \
    "$VLLM_VERSION" \
    "$GPU_DESCRIPTION" \
    "$CONDITION" \
    "$SELECTOR" \
    "$RADIUS" \
    "$PORT" <<'PY'
import json
import socket
import sys

(
    path,
    matrix_id,
    task_id,
    slurm_job_id,
    project_commit,
    model,
    vllm_version,
    gpu,
    condition,
    selector,
    edit_radius,
    port,
) = sys.argv[1:]
manifest = {
    "matrix_job_id": matrix_id,
    "array_task_id": int(task_id),
    "slurm_job_id": slurm_job_id,
    "host": socket.gethostname(),
    "project_commit": project_commit,
    "model": model,
    "vllm_version": vllm_version,
    "gpu": gpu,
    "condition": condition,
    "repair_selector": selector,
    "edit_radius": int(edit_radius),
    "port": int(port),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
PY
}

# Validate this task immediately, then analyze the matrix when all tasks exist.
validate_result() {
  python - "$RESULT_PATH" "$CONDITION" <<'PY'
import json
import sys
from pathlib import Path

from benchmarks.analyze_repair_policy_shadow import (
    validate_repair_policy_condition,
)

path = Path(sys.argv[1])
condition = sys.argv[2]
result = json.loads(path.read_text())
actual = validate_repair_policy_condition(result, condition)
print(f"validated_repair_metrics={actual}")
PY
  touch "$RESULT_DIR/rag-$CONDITION.complete"

  local all_results=true
  local expected_condition
  for expected_condition in "${CONDITIONS[@]}"; do
    if [[ ! -f "$RESULT_DIR/rag-$expected_condition.complete" ]]; then
      all_results=false
    fi
  done
  if [[ "$all_results" == true ]] \
    && mkdir "$RESULT_DIR/.analysis-lock" 2>/dev/null; then
    python -m benchmarks.analyze_repair_policy_shadow \
      --input-dir "$RESULT_DIR" \
      --output "$RESULT_DIR/repair-policy-summary.json"
  fi
}

write_manifest

setsid vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-cacheselect \
  --cacheselect-repair-selector "$SELECTOR" \
  --cacheselect-edit-radius "$RADIUS" \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server
echo "vLLM is healthy pid=$SERVER_PID"

python -m benchmarks.run_vllm_baseline \
  --trace benchmarks/traces/rag.json \
  --model "$MODEL" \
  --base-url "http://127.0.0.1:$PORT" \
  --apc-label on \
  --planner-mode vllm \
  --max-completion-tokens 96 \
  --run-id "repair-shadow-$MATRIX_ID-$CONDITION" \
  --request-log-dir "$REQUEST_LOG_DIR" \
  --output "$RESULT_PATH"

stop_server
validate_result

echo "Repair policy shadow condition completed successfully: $CONDITION"
echo "result=$RESULT_PATH"
echo "manifest=$MANIFEST_PATH"
echo "server_log=$SERVER_LOG"
echo "matrix_results=$RESULT_DIR"
