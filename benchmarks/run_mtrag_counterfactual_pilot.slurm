#!/usr/bin/env bash
#SBATCH --job-name=mtrag-counterfactual
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/mtrag-counterfactual-%j.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_MTRAG_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MTRAG_INPUT="${CACHESELECT_MTRAG_INPUT:-$STORAGE_ROOT/cacheselect-data/mtrag/RAG.jsonl}"
MANIFEST="${CACHESELECT_MTRAG_PILOT_MANIFEST:-results/mtrag-pilot-v2/manifest.json}"
AUDIT="${CACHESELECT_MTRAG_AUDIT:-results/mtrag-quality-calibration-v1/manual-audit-v1.json}"
REFERENCE_ARTIFACT="${CACHESELECT_MTRAG_REFERENCE_ARTIFACT:-results/mtrag-quality-calibration-v1/run-274842/reference-calibration.json}"
HF_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"
SERVER_START_TIMEOUT_SECONDS="${CACHESELECT_SERVER_START_TIMEOUT_SECONDS:-3600}"

if ! [[ "$SERVER_START_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CACHESELECT_SERVER_START_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi

EXPERIMENT_ID="${SLURM_JOB_ID:?Run this file with sbatch}"
PORT=$((29000 + (EXPERIMENT_ID % 1000)))
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/mtrag-counterfactual-$EXPERIMENT_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/mtrag-counterfactual-$EXPERIMENT_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/mtrag-counterfactual-$EXPERIMENT_ID"
TMP_ROOT="$STORAGE_ROOT/tmp"
SUMMARY="$RESULT_DIR/summary.json"
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
test -s "$MTRAG_INPUT"
test -s "$MANIFEST"
test -s "$AUDIT"
test -s "$REFERENCE_ARTIFACT"

PROJECT_COMMIT="$(git rev-parse HEAD)"
VLLM_VERSION="$(vllm --version)"
GPU_DESCRIPTION="$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)"

echo "Audited MTRAG counterfactual pilot"
echo "experiment_id=$EXPERIMENT_ID host=$(hostname) port=$PORT"
echo "project_commit=$PROJECT_COMMIT"
echo "model=$MODEL vllm_version=$VLLM_VERSION"
echo "gpu=$GPU_DESCRIPTION"

# Preserve software, hardware, source, and execution-policy provenance.
{
  echo "experiment=mtrag_counterfactual_pilot"
  echo "experiment_id=$EXPERIMENT_ID"
  echo "project_commit=$PROJECT_COMMIT"
  echo "model=$MODEL"
  echo "vllm_version=$VLLM_VERSION"
  echo "gpu=$GPU_DESCRIPTION"
  echo "mtrag_input=$MTRAG_INPUT"
  echo "manifest=$MANIFEST"
  echo "audit=$AUDIT"
  echo "reference_artifact=$REFERENCE_ARTIFACT"
  echo "repair_selector=full_block"
  echo "partial_reuse_execution=true"
  echo "max_completion_tokens=384"
  echo "server_start_timeout_seconds=$SERVER_START_TIMEOUT_SECONDS"
} >"$RESULT_DIR/metadata.env"

SERVER_PID=""

# Stop vLLM and release the allocated GPU on every exit path.
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

# Wait for the local endpoint or expose its startup log.
wait_for_server() {
  local start_seconds=$SECONDS
  local elapsed_seconds=0
  local next_progress_seconds=60
  while ((elapsed_seconds < SERVER_START_TIMEOUT_SECONDS)); do
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
    elapsed_seconds=$((SECONDS - start_seconds))
    if ((elapsed_seconds >= next_progress_seconds)); then
      echo "Still waiting for vLLM: ${elapsed_seconds}s elapsed"
      tail -n 1 "$SERVER_LOG" 2>/dev/null || true
      next_progress_seconds=$((next_progress_seconds + 60))
    fi
  done
  echo "Timed out after ${elapsed_seconds}s waiting for vLLM health" >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  return 1
}

# Start one conservative server; request metadata activates isolated reuse.
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

python -m benchmarks.run_mtrag_counterfactual_pilot \
  --input "$MTRAG_INPUT" \
  --manifest "$MANIFEST" \
  --audit "$AUDIT" \
  --reference-artifact "$REFERENCE_ARTIFACT" \
  --model "$MODEL" \
  --base-url "http://127.0.0.1:$PORT" \
  --max-completion-tokens 384 \
  --run-id "mtrag-counterfactual-$EXPERIMENT_ID" \
  --request-log-dir "$REQUEST_LOG_DIR" \
  --output-dir "$RESULT_DIR" \
  --summary-output "$SUMMARY"

stop_server
test -s "$SUMMARY"

echo "MTRAG counterfactual pilot completed successfully"
echo "summary=$SUMMARY"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_log=$SERVER_LOG"
