#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-performance
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/performance-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_PERFORMANCE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
REPETITIONS="${CACHESELECT_PERFORMANCE_REPETITIONS:-5}"
EXPERIMENT_KIND="${CACHESELECT_EXPERIMENT_KIND:-performance}"
EDIT_RADIUS="${CACHESELECT_EDIT_RADIUS:-1}"
ANSWER_SENSITIVE="${CACHESELECT_ANSWER_SENSITIVE:-0}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Run this file as a Slurm array job}"
EXPERIMENT_ID="${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"
TARGETS=(256 1024 4096)
POSITIONS=(early middle late)
TARGET_INDEX="${CACHESELECT_TARGET_INDEX:-$TASK_ID}"
MODE_ORDER_INDEX="${CACHESELECT_MODE_ORDER_INDEX:-$((TASK_ID % 3))}"
TARGET="${TARGETS[$TARGET_INDEX]}"
CONDITION_PATH="${CACHESELECT_CONDITION_PATH:-tokens-$TARGET}"
PORT=$((23000 + (EXPERIMENT_ID % 500) * 10 + TASK_ID))

EXPERIMENT_ROOT="$STORAGE_ROOT/cacheselect-results/$EXPERIMENT_KIND-$EXPERIMENT_ID"
RESULT_DIR="$EXPERIMENT_ROOT/$CONDITION_PATH"
MEASURED_DIR="$RESULT_DIR/measured"
WARMUP_DIR="$RESULT_DIR/warmup"
ANALYSIS_DIR="$RESULT_DIR/analysis"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/$EXPERIMENT_KIND-$EXPERIMENT_ID/$CONDITION_PATH"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/$EXPERIMENT_KIND-$EXPERIMENT_ID/$CONDITION_PATH"
TRACE_DIR="$STORAGE_ROOT/cacheselect-generated-traces/$EXPERIMENT_KIND-$EXPERIMENT_ID/$CONDITION_PATH"
TMP_ROOT="$STORAGE_ROOT/tmp"

mkdir -p \
  "$MEASURED_DIR" \
  "$WARMUP_DIR" \
  "$ANALYSIS_DIR" \
  "$REQUEST_LOG_DIR" \
  "$SERVER_LOG_DIR" \
  "$TRACE_DIR" \
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

PROJECT_COMMIT="$(git rev-parse HEAD)"
VLLM_VERSION="$(vllm --version)"
GPU_DESCRIPTION="$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)"

echo "CacheSelect $EXPERIMENT_KIND experiment"
echo "experiment_id=$EXPERIMENT_ID task_id=$TASK_ID target_tokens=$TARGET"
echo "edit_radius=$EDIT_RADIUS answer_sensitive=$ANSWER_SENSITIVE"
echo "host=$(hostname) port=$PORT repetitions=$REPETITIONS"
echo "project_commit=$PROJECT_COMMIT"
echo "model=$MODEL vllm_version=$VLLM_VERSION"
echo "gpu=$GPU_DESCRIPTION"

# Preserve the essential run provenance beside the measured results.
{
  echo "experiment=cacheselect_$EXPERIMENT_KIND"
  echo "experiment_id=$EXPERIMENT_ID"
  echo "target_tokens=$TARGET"
  echo "repetitions=$REPETITIONS"
  echo "project_commit=$PROJECT_COMMIT"
  echo "model=$MODEL"
  echo "vllm_version=$VLLM_VERSION"
  echo "gpu=$GPU_DESCRIPTION"
  echo "selector=edit_proximity"
  echo "edit_radius=$EDIT_RADIUS"
  echo "answer_sensitive=$ANSWER_SENSITIVE"
  echo "chunked_prefill=false"
  echo "max_num_batched_tokens=8192"
} >"$RESULT_DIR/metadata.env"

TRACE_ARGS=()
if [[ "$ANSWER_SENSITIVE" == "1" ]]; then
  TRACE_ARGS+=(--answer-sensitive)
fi
python -m benchmarks.generate_length_calibration \
  --model "$MODEL" \
  --target-prompt-tokens "$TARGET" \
  --output-dir "$TRACE_DIR" \
  "${TRACE_ARGS[@]}"

SERVER_PID=""

# Stop the current vLLM process and wait until its GPU resources are released.
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

# Wait for the OpenAI-compatible endpoint or expose its startup failure.
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

# Start one native, shadow, or active server with comparable execution settings.
start_server() {
  local mode="$1"
  local server_log="$SERVER_LOG_DIR/vllm-$mode.log"
  local -a cacheselect_args=()
  if [[ "$mode" != "native" ]]; then
    cacheselect_args+=(
      --enable-cacheselect
      --cacheselect-repair-selector edit_proximity
      --cacheselect-edit-radius "$EDIT_RADIUS"
    )
    if [[ "$mode" == "active" ]]; then
      cacheselect_args+=(--cacheselect-execute-partial-reuse)
    fi
  fi

  echo "Starting vLLM mode=$mode"
  setsid vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-per-request-metrics \
    --enforce-eager \
    "${cacheselect_args[@]}" \
    >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$server_log"
  echo "vLLM is healthy mode=$mode pid=$SERVER_PID"
}

# Submit one source/edit pair with a cache namespace unique to this repetition.
run_trial() {
  local mode="$1"
  local position="$2"
  local repetition="$3"
  local output_dir="$4"
  local label="$5"
  local run_key="tokens-${TARGET}-${position}-${mode}-rep-${repetition}"
  local cache_salt="${EXPERIMENT_KIND}-${EXPERIMENT_ID}-${TASK_ID}-${mode}-${position}-${label}"
  local planner_mode="vllm"
  if [[ "$mode" == "native" ]]; then
    planner_mode="off"
  fi

  python -m benchmarks.run_vllm_baseline \
    --trace "$TRACE_DIR/tokens-${TARGET}-${position}.json" \
    --model "$MODEL" \
    --base-url "http://127.0.0.1:$PORT" \
    --apc-label on \
    --planner-mode "$planner_mode" \
    --cache-salt "$cache_salt" \
    --max-completion-tokens 96 \
    --run-id "$EXPERIMENT_KIND-$EXPERIMENT_ID-$TASK_ID-$mode-$position-$label" \
    --request-log-dir "$REQUEST_LOG_DIR" \
    --output "$output_dir/$run_key.json"
}

# Warm kernels once, then collect all measured edit positions for one mode.
run_mode() {
  local mode="$1"
  start_server "$mode"
  for position in "${POSITIONS[@]}"; do
    run_trial "$mode" "$position" 0 "$WARMUP_DIR" "warmup"
  done
  for repetition in $(seq 1 "$REPETITIONS"); do
    for position in "${POSITIONS[@]}"; do
      echo "Measuring mode=$mode position=$position repetition=$repetition"
      run_trial \
        "$mode" "$position" "$repetition" "$MEASURED_DIR" "rep-$repetition"
    done
  done
  stop_server
}

# Rotate condition order across GPUs to reduce systematic warm-machine bias.
case "$MODE_ORDER_INDEX" in
  0) MODES=(native shadow active) ;;
  1) MODES=(shadow active native) ;;
  2) MODES=(active native shadow) ;;
esac
for mode in "${MODES[@]}"; do
  run_mode "$mode"
done

python -m benchmarks.analyze_cacheselect_performance \
  --input-dir "$MEASURED_DIR" \
  --output-dir "$ANALYSIS_DIR" \
  --target-tokens "$TARGET" \
  --repetitions "$REPETITIONS"

echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_logs=$SERVER_LOG_DIR"
echo "CacheSelect $EXPERIMENT_KIND task completed successfully"
