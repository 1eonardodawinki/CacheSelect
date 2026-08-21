#!/usr/bin/env bash
# Load Qwen3-14B BF16 on a RunPod A40 and validate it before cache experiments.

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-/workspace/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/workspace/cacheselect-env-cu130}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/workspace}"
MODEL="${CACHESELECT_FULL_ATTENTION_MODEL:-Qwen/Qwen3-14B}"
EXPECTED_GPU="${CACHESELECT_EXPECTED_GPU_NAME:-NVIDIA A40}"
MINIMUM_GPU_MEMORY_MIB="${CACHESELECT_MINIMUM_GPU_MEMORY_MIB:-45000}"
GPU_MEMORY_UTILIZATION="${CACHESELECT_GPU_MEMORY_UTILIZATION:-0.90}"
START_TIMEOUT="${CACHESELECT_SERVER_START_TIMEOUT_SECONDS:-3600}"
EXPERIMENT_ID="${CACHESELECT_EXPERIMENT_ID:-$(date -u +%s)}"
PORT="${CACHESELECT_SERVER_PORT:-8000}"
RUN_ID="qwen3-14b-bf16-validation-$EXPERIMENT_ID"
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/$RUN_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/$RUN_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/$RUN_ID"
SERVER_LOG="$SERVER_LOG_DIR/vllm.log"
SERVER_INFO="$RESULT_DIR/server-info.json"
HF_CONFIG="$RESULT_DIR/hf-config.json"
SUMMARY="$RESULT_DIR/summary.json"

# Stop early when the persistent RunPod environment is incomplete.
require_command() {
  command -v "$1" >/dev/null || {
    echo "Required command is unavailable: $1" >&2
    exit 2
  }
}

require_command curl
require_command git
require_command nvidia-smi
require_command setsid

[[ -d "$PROJECT_ROOT/.git" ]] || {
  echo "DeltaCache repository not found at $PROJECT_ROOT" >&2
  exit 2
}
[[ -x "$VENV_ROOT/bin/python" && -x "$VENV_ROOT/bin/vllm" ]] || {
  echo "CacheSelect vLLM environment not found at $VENV_ROOT" >&2
  exit 2
}

cd "$PROJECT_ROOT"
git diff --quiet && git diff --cached --quiet || {
  echo "Tracked repository files must be clean before an experiment" >&2
  exit 2
}
PROJECT_COMMIT="$(git rev-parse HEAD)"
ORIGIN_COMMIT="$(git rev-parse origin/main)"
[[ "$PROJECT_COMMIT" == "$ORIGIN_COMMIT" ]] || {
  echo "RunPod checkout must match the pushed origin/main commit" >&2
  exit 2
}

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_MEMORY_MIB="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1
)"
[[ "$GPU_NAME" == *"$EXPECTED_GPU"* ]] || {
  echo "Expected $EXPECTED_GPU but found $GPU_NAME" >&2
  exit 2
}
[[ "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] \
  && ((GPU_MEMORY_MIB >= MINIMUM_GPU_MEMORY_MIB)) || {
  echo "GPU memory is below the required ${MINIMUM_GPU_MEMORY_MIB} MiB" >&2
  exit 2
}

mkdir -p \
  "$RESULT_DIR" \
  "$REQUEST_LOG_DIR" \
  "$SERVER_LOG_DIR" \
  "$STORAGE_ROOT/hf-cache" \
  "$STORAGE_ROOT/tmp"

# Use the editable vLLM fork and the persistent model cache.
export PYTHONPATH="$PROJECT_ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"
source "$VENV_ROOT/bin/activate"
VLLM_SOURCE="$(python -c 'import vllm; print(vllm.__file__)')"
[[ "$VLLM_SOURCE" == "$PROJECT_ROOT"/vllm/* ]] || {
  echo "vLLM is not imported from the DeltaCache checkout: $VLLM_SOURCE" >&2
  exit 2
}
export HF_HOME="$STORAGE_ROOT/hf-cache"
export TMPDIR="$STORAGE_ROOT/tmp"
export HF_HUB_OFFLINE="${CACHESELECT_HF_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE"
export PYTHONUNBUFFERED=1
export VLLM_SERVER_DEV_MODE=1

{
  echo "experiment=full_attention_model_validation"
  echo "experiment_id=$EXPERIMENT_ID"
  echo "project_commit=$PROJECT_COMMIT"
  echo "model=$MODEL"
  echo "dtype=bfloat16"
  echo "quantization=none"
  echo "gpu=$GPU_NAME"
  echo "gpu_memory_mib=$GPU_MEMORY_MIB"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "max_model_len=8192"
  echo "kv_block_size=16"
  echo "cacheselect=disabled"
} >"$RESULT_DIR/metadata.env"

SERVER_PID=""

# Terminate the complete vLLM process group on success, failure, or interruption.
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
}
trap stop_server EXIT INT TERM

# Surface model-download, compatibility, or out-of-memory failures immediately.
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

# Keep CacheSelect disabled so this gate isolates the new model and precision.
setsid vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --block-size 16 \
  --enable-prefix-caching \
  --no-enable-chunked-prefill \
  --enforce-eager \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server
echo "Qwen3-14B BF16 is healthy pid=$SERVER_PID"

# Preserve the effective vLLM and Hugging Face configurations as evidence.
curl --fail --silent \
  "http://127.0.0.1:$PORT/server_info?config_format=json" \
  --output "$SERVER_INFO"
python - "$SERVER_INFO" "$HF_CONFIG" <<'PY'
import json
import sys
from transformers import AutoConfig

server_info = json.load(open(sys.argv[1], encoding="utf-8"))
model_path = server_info["vllm_config"]["model_config"]["model"]
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump(config.to_dict(), output, indent=2)
    output.write("\n")
PY
nvidia-smi \
  --query-gpu=name,uuid,memory.total,memory.used,memory.free,driver_version \
  --format=csv,noheader >"$RESULT_DIR/gpu-after-load.csv"

python -m benchmarks.run_full_attention_model_validation \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --run-id "$RUN_ID" \
  --request-log-dir "$REQUEST_LOG_DIR" \
  --server-info "$SERVER_INFO" \
  --hf-config "$HF_CONFIG" \
  --output "$SUMMARY"

python - "$SUMMARY" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["passed"] is True
assert summary["configuration"]["runtime_dtype"] == "torch.bfloat16"
assert summary["configuration"]["kv_block_size"] == 16
assert summary["uncached_stability"]["passed"] is True
PY

echo "RunPod Qwen3-14B BF16 validation completed successfully"
echo "commit=$PROJECT_COMMIT"
echo "gpu=$GPU_NAME"
echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_log=$SERVER_LOG"
