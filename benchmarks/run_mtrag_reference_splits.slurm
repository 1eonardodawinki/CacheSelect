#!/usr/bin/env bash
#SBATCH --job-name=mtrag-reference-splits
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/mtrag-reference-%j.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_MTRAG_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MANIFEST_MODEL="${CACHESELECT_MTRAG_MANIFEST_MODEL:-$MODEL}"
MODEL_DTYPE="${CACHESELECT_MODEL_DTYPE:-auto}"
MTRAG_INPUT="${CACHESELECT_MTRAG_INPUT:-$STORAGE_ROOT/cacheselect-data/mtrag/RAG.jsonl}"
SPLIT_ROOT="${CACHESELECT_MTRAG_SPLIT_ROOT:-results/mtrag-natural-splits-v1}"
START_TIMEOUT="${CACHESELECT_SERVER_START_TIMEOUT_SECONDS:-3600}"
SPLITS=(train validation)

if ! [[ "$START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "CACHESELECT_SERVER_START_TIMEOUT_SECONDS must be positive" >&2
  exit 1
fi

EXPERIMENT_ID="${SLURM_JOB_ID:?Run this file with sbatch}"
PORT=$((30000 + (EXPERIMENT_ID % 1000)))
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/mtrag-reference-$EXPERIMENT_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/mtrag-reference-$EXPERIMENT_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/mtrag-reference-$EXPERIMENT_ID"
TMP_ROOT="$STORAGE_ROOT/tmp"
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
export HF_HUB_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE"
export PYTHONUNBUFFERED=1

command -v nvidia-smi >/dev/null
command -v curl >/dev/null
command -v setsid >/dev/null
command -v vllm >/dev/null
test -s "$MTRAG_INPUT"
for SPLIT in "${SPLITS[@]}"; do
  test -s "$SPLIT_ROOT/$SPLIT-manifest.json"
done

PROJECT_COMMIT="$(git rev-parse HEAD)"
VLLM_VERSION="$(vllm --version)"
GPU_DESCRIPTION="$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)"

{
  echo "experiment=mtrag_reference_splits"
  echo "experiment_id=$EXPERIMENT_ID"
  echo "project_commit=$PROJECT_COMMIT"
  echo "model=$MODEL"
  echo "manifest_selection_model=$MANIFEST_MODEL"
  echo "model_dtype=$MODEL_DTYPE"
  echo "vllm_version=$VLLM_VERSION"
  echo "gpu=$GPU_DESCRIPTION"
  echo "mtrag_input=$MTRAG_INPUT"
  echo "split_root=$SPLIT_ROOT"
  echo "executed_splits=${SPLITS[*]}"
  echo "max_model_len=8192"
  echo "max_completion_tokens=384"
} >"$RESULT_DIR/metadata.env"

SERVER_PID=""

# Stop the server process group and release the allocated GPU on every exit.
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null \
      || kill -TERM "$SERVER_PID" 2>/dev/null \
      || true
    for _ in {1..30}; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL -- "-$SERVER_PID" 2>/dev/null \
        || kill -KILL "$SERVER_PID" 2>/dev/null \
        || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap stop_server EXIT INT TERM

# Wait for the local endpoint without hiding a failed server startup.
wait_for_server() {
  local start_seconds=$SECONDS
  while ((SECONDS - start_seconds < START_TIMEOUT)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited before becoming healthy" >&2
      tail -n 200 "$SERVER_LOG" >&2 || true
      return 1
    fi
    if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for vLLM" >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  return 1
}

# Match the later causal environment while references remain uncached.
setsid vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype "$MODEL_DTYPE" \
  --max-model-len 8192 \
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

# Record training and validation separately; the sealed test manifest is absent.
for SPLIT in "${SPLITS[@]}"; do
  python -m benchmarks.run_mtrag_calibration \
    --input "$MTRAG_INPUT" \
    --manifest "$SPLIT_ROOT/$SPLIT-manifest.json" \
    --model "$MODEL" \
    --manifest-model "$MANIFEST_MODEL" \
    --base-url "http://127.0.0.1:$PORT" \
    --max-completion-tokens 384 \
    --run-id "mtrag-reference-$EXPERIMENT_ID-$SPLIT" \
    --request-log-dir "$REQUEST_LOG_DIR" \
    --output "$RESULT_DIR/$SPLIT-reference-calibration.json"
done

stop_server
test -s "$RESULT_DIR/train-reference-calibration.json"
test -s "$RESULT_DIR/validation-reference-calibration.json"

echo "MTRAG training and validation references completed successfully"
echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_log=$SERVER_LOG"
