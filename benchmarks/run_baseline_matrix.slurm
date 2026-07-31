#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-baseline
#SBATCH --array=0-17%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/baseline-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Run this file as a Slurm array job}"
MATRIX_ID="${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"

WORKLOADS=(rag periodic_agent chat)
APC_LABELS=(off on)
APC_FLAGS=(--no-enable-prefix-caching --enable-prefix-caching)

# Six conditions per workload: APC off/on, with three repetitions each.
WORKLOAD_INDEX=$((TASK_ID / 6))
WITHIN_WORKLOAD=$((TASK_ID % 6))
APC_INDEX=$((WITHIN_WORKLOAD / 3))
REPETITION=$((WITHIN_WORKLOAD % 3 + 1))

WORKLOAD="${WORKLOADS[$WORKLOAD_INDEX]}"
APC_LABEL="${APC_LABELS[$APC_INDEX]}"
APC_FLAG="${APC_FLAGS[$APC_INDEX]}"
RUN_KEY="${WORKLOAD}-apc-${APC_LABEL}-rep-${REPETITION}"

# Array tasks can share a physical host, so they must not share a TCP port.
PORT=$((18000 + (MATRIX_ID % 1000) * 20 + TASK_ID))
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/baseline-$MATRIX_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/baseline-$MATRIX_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/baseline-$MATRIX_ID"
RESULT_PATH="$RESULT_DIR/$RUN_KEY.json"
MANIFEST_PATH="$RESULT_DIR/$RUN_KEY.manifest.json"
SERVER_LOG="$SERVER_LOG_DIR/$RUN_KEY.vllm.log"
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

python - "$MANIFEST_PATH" <<PY
import json
import os
import socket
import sys

manifest = {
    "matrix_job_id": "$MATRIX_ID",
    "array_task_id": $TASK_ID,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "host": socket.gethostname(),
    "project_commit": "$PROJECT_COMMIT",
    "model": "$MODEL",
    "vllm_version": "$VLLM_VERSION",
    "gpu": """$GPU_DESCRIPTION""",
    "workload": "$WORKLOAD",
    "apc": "$APC_LABEL",
    "repetition": $REPETITION,
    "port": $PORT,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
PY

echo "CacheSelect baseline condition"
echo "matrix_id=$MATRIX_ID task_id=$TASK_ID run_key=$RUN_KEY"
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

setsid vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics \
  "$APC_FLAG" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in {1..600}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before becoming healthy" >&2
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if curl --fail --silent --show-error \
    "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl --fail --silent --show-error \
  "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "Timed out waiting for vLLM health endpoint" >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "vLLM is healthy pid=$SERVER_PID"

python -m benchmarks.run_vllm_baseline \
  --trace "benchmarks/traces/$WORKLOAD.json" \
  --model "$MODEL" \
  --base-url "http://127.0.0.1:$PORT" \
  --apc-label "$APC_LABEL" \
  --run-id "baseline-$MATRIX_ID-$RUN_KEY" \
  --request-log-dir "$REQUEST_LOG_DIR" \
  --output "$RESULT_PATH"

stop_server

python - "$RESULT_PATH" <<'PY'
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

echo "Baseline condition completed successfully: $RUN_KEY"
echo "result=$RESULT_PATH"
echo "manifest=$MANIFEST_PATH"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_log=$SERVER_LOG"
