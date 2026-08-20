#!/usr/bin/env bash
# Launch and validate the hybrid GDN break-even experiment on RunPod.

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-/workspace/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/workspace/cacheselect-env-cu130}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/workspace}"
MODEL="${CACHESELECT_HYBRID_MODEL:-Qwen/Qwen3.5-9B}"
EXPECTED_GPU="${CACHESELECT_EXPECTED_GPU_NAME:-NVIDIA A40}"
MINIMUM_GPU_MEMORY_MIB="${CACHESELECT_MINIMUM_GPU_MEMORY_MIB:-45000}"
EXPERIMENT_ID="${CACHESELECT_EXPERIMENT_ID:-$(date -u +%s)}"
REPETITIONS="${CACHESELECT_BREAK_EVEN_REPETITIONS:-3}"
BLOCK_COUNTS="${CACHESELECT_BREAK_EVEN_BLOCK_COUNTS:-1 2 4 8 16}"
CACHE_CAPACITY="${CACHESELECT_GDN_DELTA_CACHE_CAPACITY:-16}"
BLOCK_SIZE="${CACHESELECT_GDN_DELTA_BLOCK_SIZE:-64}"

# Stop with a clear message when one required local program is unavailable.
require_command() {
  command -v "$1" >/dev/null || {
    echo "Required command is unavailable: $1" >&2
    exit 2
  }
}

require_command git
require_command nvidia-smi
require_command tee

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

# Put the nested source package before the repository's namespace directory.
export PYTHONPATH="$PROJECT_ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"
source "$VENV_ROOT/bin/activate"
VLLM_SOURCE="$(python -c 'import vllm; print(vllm.__file__)')"
[[ "$VLLM_SOURCE" == "$PROJECT_ROOT"/vllm/* ]] || {
  echo "vLLM is not imported from the DeltaCache checkout: $VLLM_SOURCE" >&2
  exit 2
}

export SLURM_JOB_ID="$EXPERIMENT_ID"
export CACHESELECT_PROJECT_ROOT="$PROJECT_ROOT"
export CACHESELECT_VENV_ROOT="$VENV_ROOT"
export CACHESELECT_STORAGE_ROOT="$STORAGE_ROOT"
export CACHESELECT_CUDA_SETUP="/dev/null"
export CACHESELECT_HF_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"
export CACHESELECT_HYBRID_MODEL="$MODEL"
export CACHESELECT_GDN_DELTA_EXECUTION_MODE=active
export CACHESELECT_GDN_DELTA_CACHE_CAPACITY="$CACHE_CAPACITY"
export CACHESELECT_GDN_DELTA_BLOCK_SIZE="$BLOCK_SIZE"
export CACHESELECT_RUN_ACTIVE_MATRIX=0
export CACHESELECT_RUN_BREAK_EVEN_MATRIX=1
export CACHESELECT_BREAK_EVEN_BLOCK_COUNTS="$BLOCK_COUNTS"
export CACHESELECT_BREAK_EVEN_REPETITIONS="$REPETITIONS"

RUN_ID="hybrid-checkpoint-active-$EXPERIMENT_ID"
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/$RUN_ID"
SERVER_LOG_ROOT="$STORAGE_ROOT/cacheselect-server-logs"
WRAPPER_LOG="$SERVER_LOG_ROOT/runpod-break-even-$EXPERIMENT_ID.out"
mkdir -p "$SERVER_LOG_ROOT"

set -o pipefail
bash benchmarks/run_hybrid_checkpoint_smoke.slurm 2>&1 | tee "$WRAPPER_LOG"

METADATA="$RESULT_DIR/metadata.env"
SUMMARY="$RESULT_DIR/break-even/break-even-summary.json"
test -s "$METADATA"
test -s "$SUMMARY"
grep -Fx "project_commit=$PROJECT_COMMIT" "$METADATA" >/dev/null
grep -Fx "model=$MODEL" "$METADATA" >/dev/null
grep -Fx "gdn_delta_cache_capacity=$CACHE_CAPACITY" "$METADATA" >/dev/null
grep -Fx "gdn_delta_block_size=$BLOCK_SIZE" "$METADATA" >/dev/null
grep -Fx "gdn_delta_execution_mode=active" "$METADATA" >/dev/null
grep -Fx "run_break_even_matrix=1" "$METADATA" >/dev/null
grep -F "$EXPECTED_GPU" "$METADATA" >/dev/null

python -m benchmarks.analyze_hybrid_gdn_breakeven \
  --input "$SUMMARY" \
  --output "$RESULT_DIR/break-even/break-even-analysis.json" \
  --markdown-output "$RESULT_DIR/break-even/break-even-report.md"

echo "RunPod hybrid GDN break-even experiment completed"
echo "commit=$PROJECT_COMMIT"
echo "gpu=$GPU_NAME"
echo "results=$RESULT_DIR"
echo "log=$WRAPPER_LOG"
