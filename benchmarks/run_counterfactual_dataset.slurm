#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-counterfactual
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/counterfactual-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_COUNTERFACTUAL_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET_SPLIT="${CACHESELECT_COUNTERFACTUAL_SPLIT:-train}"
HF_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Run this file as a Slurm array job}"
EXPERIMENT_ID="${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"
TRANSITIONS=(rag-00-to-01 rag-01-to-02 rag-02-to-03)
if (( TASK_ID < 0 || TASK_ID >= ${#TRANSITIONS[@]} )); then
  echo "Invalid counterfactual array task: $TASK_ID" >&2
  exit 2
fi
TRANSITION_ID="${TRANSITIONS[$TASK_ID]}"
# Array tasks can share a host, so each receives a deterministic unique port.
PORT=$((27000 + (EXPERIMENT_ID % 500) * 10 + TASK_ID))

EXPERIMENT_ROOT="$STORAGE_ROOT/cacheselect-results/counterfactual-$EXPERIMENT_ID"
RESULT_DIR="$EXPERIMENT_ROOT/$TRANSITION_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/counterfactual-$EXPERIMENT_ID/$TRANSITION_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/counterfactual-$EXPERIMENT_ID/$TRANSITION_ID"
TMP_ROOT="$STORAGE_ROOT/tmp"
DATASET_PATH="$RESULT_DIR/counterfactual-blocks.csv"
SUMMARY_PATH="$RESULT_DIR/counterfactual-blocks.summary.json"
SERVER_LOG="$SERVER_LOG_DIR/vllm.log"

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
export HF_HUB_OFFLINE="$HF_OFFLINE"
export TRANSFORMERS_OFFLINE="$HF_OFFLINE"
export PYTHONUNBUFFERED=1

command -v nvidia-smi >/dev/null
command -v curl >/dev/null
command -v setsid >/dev/null
command -v vllm >/dev/null
test -f benchmarks/traces/rag.json

PROJECT_COMMIT="$(git rev-parse HEAD)"
VLLM_VERSION="$(vllm --version)"
GPU_DESCRIPTION="$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)"

echo "CacheSelect counterfactual dataset experiment"
echo "experiment_id=$EXPERIMENT_ID task_id=$TASK_ID transition=$TRANSITION_ID"
echo "host=$(hostname) port=$PORT split=$DATASET_SPLIT"
echo "project_commit=$PROJECT_COMMIT"
echo "model=$MODEL vllm_version=$VLLM_VERSION"
echo "gpu=$GPU_DESCRIPTION"

# Preserve hardware and software provenance beside the resulting dataset.
{
  echo "experiment=counterfactual_block_dataset"
  echo "experiment_id=$EXPERIMENT_ID"
  echo "task_id=$TASK_ID"
  echo "transition_id=$TRANSITION_ID"
  echo "dataset_split=$DATASET_SPLIT"
  echo "project_commit=$PROJECT_COMMIT"
  echo "model=$MODEL"
  echo "vllm_version=$VLLM_VERSION"
  echo "gpu=$GPU_DESCRIPTION"
  echo "repair_selector=full_block"
  echo "partial_reuse_execution=true"
  echo "chunked_prefill=false"
  echo "eager_execution=true"
} >"$RESULT_DIR/metadata.env"

SERVER_PID=""

# Stop vLLM and its process group when the task succeeds, fails, or is cancelled.
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

# Wait for the serving endpoint, exposing its log if startup fails.
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

# Start the conservative default policy plus request-scoped interventions.
setsid vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-cacheselect \
  --cacheselect-repair-selector full_block \
  --cacheselect-execute-partial-reuse \
  --no-enable-chunked-prefill \
  --enforce-eager \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server
echo "vLLM is healthy pid=$SERVER_PID"

python -m benchmarks.run_counterfactual_dataset \
  --trace benchmarks/traces/rag.json \
  --transition-id "$TRANSITION_ID" \
  --split "$DATASET_SPLIT" \
  --model "$MODEL" \
  --base-url "http://127.0.0.1:$PORT" \
  --max-completion-tokens 96 \
  --run-id "counterfactual-$EXPERIMENT_ID-$TASK_ID-$TRANSITION_ID" \
  --request-log-dir "$REQUEST_LOG_DIR" \
  --output "$DATASET_PATH" \
  --summary-output "$SUMMARY_PATH"

stop_server
test -s "$DATASET_PATH"
test -s "$SUMMARY_PATH"

echo "Counterfactual dataset task completed successfully: $TRANSITION_ID"
echo "dataset=$DATASET_PATH"
echo "summary=$SUMMARY_PATH"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_log=$SERVER_LOG"
