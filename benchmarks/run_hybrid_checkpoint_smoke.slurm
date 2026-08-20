#!/usr/bin/env bash
#SBATCH --job-name=hybrid-checkpoint
#SBATCH --partition=a30
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/hybrid-checkpoint-%j.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_HYBRID_MODEL:-Qwen/Qwen3.5-9B}"
START_TIMEOUT="${CACHESELECT_SERVER_START_TIMEOUT_SECONDS:-3600}"
DELTA_CACHE_CAPACITY="${CACHESELECT_GDN_DELTA_CACHE_CAPACITY:-2}"
DELTA_BLOCK_SIZE="${CACHESELECT_GDN_DELTA_BLOCK_SIZE:-64}"
EXECUTION_MODE="${CACHESELECT_GDN_DELTA_EXECUTION_MODE:-shadow}"
RUN_ACTIVE_MATRIX="${CACHESELECT_RUN_ACTIVE_MATRIX:-0}"
RUN_BREAK_EVEN_MATRIX="${CACHESELECT_RUN_BREAK_EVEN_MATRIX:-0}"
RUN_COMPONENT_PROFILE="${CACHESELECT_RUN_COMPONENT_PROFILE:-0}"
GPU_MEMORY_UTILIZATION="${CACHESELECT_GPU_MEMORY_UTILIZATION:-0.96}"
MATRIX_RECORD_COUNTS="${CACHESELECT_MATRIX_RECORD_COUNTS:-64 112 160}"
MATRIX_REPETITIONS="${CACHESELECT_MATRIX_REPETITIONS:-3}"
BREAK_EVEN_BLOCK_COUNTS="${CACHESELECT_BREAK_EVEN_BLOCK_COUNTS:-1 2 4 8 16}"
BREAK_EVEN_REPETITIONS="${CACHESELECT_BREAK_EVEN_REPETITIONS:-3}"
PROFILE_REUSED_BLOCK_COUNT="${CACHESELECT_PROFILE_REUSED_BLOCK_COUNT:-32}"

if [[ "$EXECUTION_MODE" != "shadow" && "$EXECUTION_MODE" != "active" ]]; then
  echo "Unsupported GDN delta execution mode: $EXECUTION_MODE" >&2
  exit 2
fi
if [[ "$RUN_ACTIVE_MATRIX" != "0" && "$RUN_ACTIVE_MATRIX" != "1" ]]; then
  echo "CACHESELECT_RUN_ACTIVE_MATRIX must be 0 or 1" >&2
  exit 2
fi
if [[ "$RUN_ACTIVE_MATRIX" == "1" && "$EXECUTION_MODE" != "active" ]]; then
  echo "The graduated matrix requires active GDN execution mode" >&2
  exit 2
fi
if [[ "$RUN_BREAK_EVEN_MATRIX" != "0" && "$RUN_BREAK_EVEN_MATRIX" != "1" ]]; then
  echo "CACHESELECT_RUN_BREAK_EVEN_MATRIX must be 0 or 1" >&2
  exit 2
fi
if [[ "$RUN_BREAK_EVEN_MATRIX" == "1" && "$EXECUTION_MODE" != "active" ]]; then
  echo "The break-even matrix requires active GDN execution mode" >&2
  exit 2
fi
if [[ "$RUN_COMPONENT_PROFILE" != "0" && "$RUN_COMPONENT_PROFILE" != "1" ]]; then
  echo "CACHESELECT_RUN_COMPONENT_PROFILE must be 0 or 1" >&2
  exit 2
fi
if [[ "$RUN_COMPONENT_PROFILE" == "1" ]]; then
  if [[ "$EXECUTION_MODE" != "active" ]]; then
    echo "Component profiling requires active GDN execution mode" >&2
    exit 2
  fi
  if [[ ! "$PROFILE_REUSED_BLOCK_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Profile reused block count must be a positive integer" >&2
    exit 2
  fi
  if [[ ! "$DELTA_CACHE_CAPACITY" =~ ^[1-9][0-9]*$ ]]; then
    echo "GDN delta cache capacity must be a positive integer" >&2
    exit 2
  fi
  if ((DELTA_CACHE_CAPACITY < PROFILE_REUSED_BLOCK_COUNT)); then
    echo "GDN delta cache capacity must cover the profile case" >&2
    exit 2
  fi
fi

# Fail before loading the model if the sidecar cannot retain the largest case.
if [[ "$RUN_BREAK_EVEN_MATRIX" == "1" ]]; then
  read -r -a BREAK_EVEN_BLOCK_COUNT_ARGS <<<"$BREAK_EVEN_BLOCK_COUNTS"
  if ((${#BREAK_EVEN_BLOCK_COUNT_ARGS[@]} == 0)); then
    echo "At least one break-even block count is required" >&2
    exit 2
  fi
  if [[ ! "$DELTA_CACHE_CAPACITY" =~ ^[1-9][0-9]*$ ]]; then
    echo "GDN delta cache capacity must be a positive integer" >&2
    exit 2
  fi
  MAX_BREAK_EVEN_BLOCK_COUNT=0
  for block_count in "${BREAK_EVEN_BLOCK_COUNT_ARGS[@]}"; do
    if [[ ! "$block_count" =~ ^[1-9][0-9]*$ ]]; then
      echo "Break-even block counts must be positive integers" >&2
      exit 2
    fi
    if ((block_count > MAX_BREAK_EVEN_BLOCK_COUNT)); then
      MAX_BREAK_EVEN_BLOCK_COUNT=$block_count
    fi
  done
  if ((DELTA_CACHE_CAPACITY < MAX_BREAK_EVEN_BLOCK_COUNT)); then
    echo "GDN delta cache capacity must cover every break-even case" >&2
    exit 2
  fi
fi

EXPERIMENT_ID="${SLURM_JOB_ID:?Run this file with sbatch}"
PORT=$((32000 + (EXPERIMENT_ID % 1000)))
RUN_ID="hybrid-checkpoint-$EXECUTION_MODE-$EXPERIMENT_ID"
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/$RUN_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/$RUN_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/$RUN_ID"
SERVER_LOG="$SERVER_LOG_DIR/vllm.log"
SUMMARY_PATH="$RESULT_DIR/summary.json"
SHADOW_ANALYSIS_PATH="$RESULT_DIR/gdn-shadow-analysis.json"
ACTIVE_ANALYSIS_PATH="$RESULT_DIR/gdn-active-analysis.json"
SERVER_INFO_PATH="$RESULT_DIR/server-info.json"
COMPONENT_PROFILE_DIR="$RESULT_DIR/component-profile"
COMPONENT_PROFILE_PATH="$COMPONENT_PROFILE_DIR/component-profile.json"
TORCH_PROFILE_DIR="$COMPONENT_PROFILE_DIR/torch-traces"

mkdir -p \
  "$RESULT_DIR" \
  "$REQUEST_LOG_DIR" \
  "$SERVER_LOG_DIR" \
  "$STORAGE_ROOT/tmp" \
  "$STORAGE_ROOT/hf-cache"

source "$CUDA_SETUP"
source "$VENV_ROOT/bin/activate"
cd "$PROJECT_ROOT"

export TMPDIR="$STORAGE_ROOT/tmp"
export HF_HOME="$STORAGE_ROOT/hf-cache"
export HF_HUB_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE"
export PYTHONUNBUFFERED=1
# The server-info endpoint is bound to localhost inside this isolated job.
export VLLM_SERVER_DEV_MODE=1
# Hybrid models default to V1, while GDN operator transport is implemented in V2.
export VLLM_USE_V2_MODEL_RUNNER=1

command -v curl >/dev/null
command -v nvidia-smi >/dev/null
command -v setsid >/dev/null
command -v vllm >/dev/null

PROJECT_COMMIT="$(git rev-parse HEAD)"
VLLM_VERSION="$(vllm --version)"
GPU_DESCRIPTION="$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)"

{
  echo "experiment=hybrid_checkpoint_smoke"
  echo "experiment_id=$EXPERIMENT_ID"
  echo "project_commit=$PROJECT_COMMIT"
  echo "model=$MODEL"
  echo "vllm_version=$VLLM_VERSION"
  echo "gpu=$GPU_DESCRIPTION"
  echo "mamba_cache_mode=all"
  echo "gdn_prefill_backend=triton"
  echo "gdn_delta_cache_capacity=$DELTA_CACHE_CAPACITY"
  echo "gdn_delta_block_size=$DELTA_BLOCK_SIZE"
  echo "gdn_delta_execution_mode=$EXECUTION_MODE"
  echo "model_runner=v2"
  echo "attention_reuse_policy=native_apc_only"
  echo "run_active_matrix=$RUN_ACTIVE_MATRIX"
  echo "run_break_even_matrix=$RUN_BREAK_EVEN_MATRIX"
  echo "run_component_profile=$RUN_COMPONENT_PROFILE"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "matrix_record_counts=$MATRIX_RECORD_COUNTS"
  echo "matrix_repetitions=$MATRIX_REPETITIONS"
  echo "break_even_block_counts=$BREAK_EVEN_BLOCK_COUNTS"
  echo "break_even_repetitions=$BREAK_EVEN_REPETITIONS"
  echo "profile_reused_block_count=$PROFILE_REUSED_BLOCK_COUNT"
  echo "max_model_len=4096"
  echo "max_completion_tokens=64"
} >"$RESULT_DIR/metadata.env"

SERVER_PID=""

# Stop the complete vLLM process group when the job finishes or fails.
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

# Enable Torch profiling only for the explicit component experiment.
PROFILER_ARGS=()
if [[ "$RUN_COMPONENT_PROFILE" == "1" ]]; then
  export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
  mkdir -p "$TORCH_PROFILE_DIR"
  PROFILER_ARGS=(
    "--profiler-config.profiler=torch"
    "--profiler-config.torch_profiler_dir=$TORCH_PROFILE_DIR"
    "--profiler-config.torch_profiler_with_stack=false"
    "--profiler-config.ignore_frontend=true"
  )
fi

trap stop_server EXIT INT TERM

# Wait for the endpoint while surfacing an early server failure.
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

# Start Qwen3.5 with one recurrent checkpoint per cache block.
setsid vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  --mamba-cache-mode all \
  --gdn-delta-cache-capacity "$DELTA_CACHE_CAPACITY" \
  --gdn-delta-block-size "$DELTA_BLOCK_SIZE" \
  --gdn-delta-execution-mode "$EXECUTION_MODE" \
  --gdn-prefill-backend triton \
  --no-enable-chunked-prefill \
  --enforce-eager \
  --enable-per-request-metrics \
  "${PROFILER_ARGS[@]}" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server
echo "vLLM is healthy pid=$SERVER_PID"
grep -q "Using V2 Model Runner" "$SERVER_LOG" || {
  echo "vLLM did not start the required V2 model runner" >&2
  exit 1
}

# Record and validate the effective configuration after vLLM fallback handling.
curl --fail --silent \
  "http://127.0.0.1:$PORT/server_info?config_format=json" \
  --output "$SERVER_INFO_PATH"
python - "$SERVER_INFO_PATH" "$DELTA_CACHE_CAPACITY" "$DELTA_BLOCK_SIZE" "$EXECUTION_MODE" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))["vllm_config"]
cache = config["cache_config"]
additional = config["additional_config"]
assert cache["enable_prefix_caching"] is True
assert cache["enable_cacheselect"] is False
assert cache["cacheselect_execute_partial_reuse"] is False
assert cache["mamba_cache_mode"] == "all"
assert cache["gdn_delta_cache_capacity"] == int(sys.argv[2])
assert cache["gdn_delta_block_size"] == int(sys.argv[3])
assert cache["gdn_delta_execution_mode"] == sys.argv[4]
assert additional["gdn_prefill_backend"] == "triton"
PY
nvidia-smi \
  --query-gpu=name,uuid,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$RESULT_DIR/gpu-after-load.csv"

# Select the execution proof required for the configured experiment mode.
if [[ "$EXECUTION_MODE" == "active" ]]; then
  EXECUTION_REQUIREMENT=(--require-gdn-active-execution)
else
  EXECUTION_REQUIREMENT=(--require-gdn-shadow-execution)
fi

# Compare checkpointed requests with independent full-computation controls.
python -m benchmarks.run_hybrid_apc_baseline \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --run-id "$RUN_ID" \
  --request-log-dir "$REQUEST_LOG_DIR" \
  --output "$SUMMARY_PATH" \
  --max-completion-tokens 64 \
  --require-token-aligned-edits \
  --require-gdn-preflight-observability \
  "${EXECUTION_REQUIREMENT[@]}" \
  --validate-against-reference

if [[ "$EXECUTION_MODE" == "shadow" ]]; then
  python -m benchmarks.analyze_hybrid_gdn_shadow \
    --input "$SUMMARY_PATH" \
    --output "$SHADOW_ANALYSIS_PATH"
else
  python -m benchmarks.analyze_hybrid_gdn_active \
    --input "$SUMMARY_PATH" \
    --output "$ACTIVE_ANALYSIS_PATH"
fi

# Continue with repeated graduated contexts only after the strict smoke passes.
if [[ "$RUN_ACTIVE_MATRIX" == "1" ]]; then
  read -r -a MATRIX_RECORD_COUNT_ARGS <<<"$MATRIX_RECORD_COUNTS"
  python -m benchmarks.run_hybrid_gdn_matrix \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --run-id "$RUN_ID-matrix" \
    --output-root "$RESULT_DIR/matrix" \
    --request-log-root "$REQUEST_LOG_DIR/matrix" \
    --record-counts "${MATRIX_RECORD_COUNT_ARGS[@]}" \
    --repetitions "$MATRIX_REPETITIONS" \
    --max-completion-tokens 64
  test -s "$RESULT_DIR/matrix/matrix-summary.json"
fi

# Measure the smallest reused fraction that beats full GDN recomputation.
if [[ "$RUN_BREAK_EVEN_MATRIX" == "1" ]]; then
  python -m benchmarks.run_hybrid_gdn_breakeven \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --run-id "$RUN_ID-break-even" \
    --output "$RESULT_DIR/break-even/break-even-summary.json" \
    --request-log-dir "$REQUEST_LOG_DIR/break-even" \
    --reused-block-counts "${BREAK_EVEN_BLOCK_COUNT_ARGS[@]}" \
    --repetitions "$BREAK_EVEN_REPETITIONS" \
    --block-size "$DELTA_BLOCK_SIZE" \
    --cache-capacity "$DELTA_CACHE_CAPACITY" \
    --max-completion-tokens 64
  test -s "$RESULT_DIR/break-even/break-even-summary.json"
fi

# Attribute one full and one actively reused prefill to model components.
if [[ "$RUN_COMPONENT_PROFILE" == "1" ]]; then
  python -m benchmarks.run_hybrid_gdn_profile \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --run-id "$RUN_ID-component-profile" \
    --output "$COMPONENT_PROFILE_PATH" \
    --request-log-dir "$REQUEST_LOG_DIR/component-profile" \
    --profile-dir "$TORCH_PROFILE_DIR" \
    --reused-block-count "$PROFILE_REUSED_BLOCK_COUNT" \
    --block-size "$DELTA_BLOCK_SIZE" \
    --max-completion-tokens 64
  test -s "$COMPONENT_PROFILE_PATH"
fi

stop_server
test -s "$SUMMARY_PATH"
if [[ "$EXECUTION_MODE" == "shadow" ]]; then
  test -s "$SHADOW_ANALYSIS_PATH"
else
  test -s "$ACTIVE_ANALYSIS_PATH"
fi
test -s "$SERVER_INFO_PATH"
grep -q "GDN delta sidecar populated" "$SERVER_LOG"

echo "Hybrid checkpoint smoke completed successfully"
echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_log=$SERVER_LOG"
