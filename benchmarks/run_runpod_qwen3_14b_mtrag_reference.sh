#!/usr/bin/env bash
# Collect uncached MTRAG train/validation answers from Qwen3-14B on RunPod.

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-/workspace/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/workspace/cacheselect-env-cu130}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/workspace}"
MODEL="${CACHESELECT_MTRAG_MODEL:-Qwen/Qwen3-14B}"
MANIFEST_MODEL="${CACHESELECT_MTRAG_MANIFEST_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
EXPAND_REFERENCES="${CACHESELECT_MTRAG_EXPAND_REFERENCES:-0}"
REMAINING_REFERENCES="${CACHESELECT_MTRAG_REMAINING_REFERENCES:-0}"
EXPECTED_GPU="${CACHESELECT_EXPECTED_GPU_NAME:-NVIDIA A40}"
MINIMUM_GPU_MEMORY_MIB="${CACHESELECT_MINIMUM_GPU_MEMORY_MIB:-45000}"
EXPERIMENT_ID="${CACHESELECT_EXPERIMENT_ID:-$(date -u +%s)}"
MTRAG_INPUT="$STORAGE_ROOT/cacheselect-data/mtrag/RAG.jsonl"
MTRAG_SHA256="5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873"
MTRAG_URL="https://raw.githubusercontent.com/IBM/mt-rag-benchmark/cc5b1d481b391181b89f7ced860308482e785463/mtrag-human/generation_tasks/RAG.jsonl"
RUN_ID="mtrag-reference-$EXPERIMENT_ID"
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/$RUN_ID"
SERVER_LOG_ROOT="$STORAGE_ROOT/cacheselect-server-logs"
WRAPPER_LOG="$SERVER_LOG_ROOT/runpod-$RUN_ID.out"
SPLIT_ROOT="results/mtrag-natural-splits-v1"
EXPECTED_TRAIN_COUNT=21
EXPECTED_VALIDATION_COUNT=6

[[ "$EXPAND_REFERENCES" == 0 || "$EXPAND_REFERENCES" == 1 ]] || {
  echo "CACHESELECT_MTRAG_EXPAND_REFERENCES must be 0 or 1" >&2
  exit 2
}
[[ "$REMAINING_REFERENCES" == 0 || "$REMAINING_REFERENCES" == 1 ]] || {
  echo "CACHESELECT_MTRAG_REMAINING_REFERENCES must be 0 or 1" >&2
  exit 2
}
((EXPAND_REFERENCES + REMAINING_REFERENCES <= 1)) || {
  echo "Choose only one MTRAG reference expansion mode" >&2
  exit 2
}

# Stop before paid work when the persistent environment is incomplete.
require_command() {
  command -v "$1" >/dev/null || {
    echo "Required command is unavailable: $1" >&2
    exit 2
  }
}

for command in curl git nvidia-smi sha256sum tar tee; do
  require_command "$command"
done
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

# Download only the frozen public source and reject any content drift.
mkdir -p "$(dirname "$MTRAG_INPUT")" "$SERVER_LOG_ROOT"
if [[ ! -s "$MTRAG_INPUT" ]]; then
  curl --location --fail --retry 3 \
    --output "$MTRAG_INPUT.download" \
    "$MTRAG_URL"
  mv "$MTRAG_INPUT.download" "$MTRAG_INPUT"
fi

# Use the frozen final 36 tasks without recomputing their selection.
if [[ "$REMAINING_REFERENCES" == 1 ]]; then
  SPLIT_ROOT="results/qwen3-14b-mtrag-reference-expansion-v2"
  EXPECTED_TRAIN_COUNT=21
  EXPECTED_VALIDATION_COUNT=15
  MANIFEST_MODEL="$MODEL"
fi
ACTUAL_MTRAG_SHA256="$(sha256sum "$MTRAG_INPUT" | cut -d ' ' -f 1)"
[[ "$ACTUAL_MTRAG_SHA256" == "$MTRAG_SHA256" ]] || {
  echo "Staged MTRAG source has the wrong SHA-256" >&2
  exit 2
}

# Require the editable vLLM fork before delegating to the shared split job.
export PYTHONPATH="$PROJECT_ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"
source "$VENV_ROOT/bin/activate"
VLLM_SOURCE="$(python -c 'import vllm; print(vllm.__file__)')"
[[ "$VLLM_SOURCE" == "$PROJECT_ROOT"/vllm/* ]] || {
  echo "vLLM is not imported from the DeltaCache checkout: $VLLM_SOURCE" >&2
  exit 2
}

# Freeze 75 unseen tasks from exact Qwen3 prompt geometry before inference.
if [[ "$EXPAND_REFERENCES" == 1 ]]; then
  COVERAGE="$RESULT_DIR/qwen3-mtrag-coverage.json"
  SPLIT_ROOT="$RESULT_DIR/manifests"
  EXPECTED_TRAIN_COUNT=60
  EXPECTED_VALIDATION_COUNT=15
  MANIFEST_MODEL="$MODEL"
  mkdir -p "$RESULT_DIR"
  export HF_HOME="$STORAGE_ROOT/hf-cache"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  python -m benchmarks.analyze_mtrag_coverage \
    --input "$MTRAG_INPUT" \
    --output "$COVERAGE" \
    --model "$MODEL" \
    --block-size 16 \
    --local-files-only
  python -m benchmarks.freeze_mtrag_reference_expansion \
    --coverage "$COVERAGE" \
    --source-dataset "$MTRAG_INPUT" \
    --existing-manifest results/mtrag-natural-splits-v1/train-manifest.json \
    --existing-manifest results/mtrag-natural-splits-v1/validation-manifest.json \
    --output-dir "$SPLIT_ROOT" \
    --train-count "$EXPECTED_TRAIN_COUNT" \
    --validation-count "$EXPECTED_VALIDATION_COUNT"
fi

export SLURM_JOB_ID="$EXPERIMENT_ID"
export CACHESELECT_PROJECT_ROOT="$PROJECT_ROOT"
export CACHESELECT_VENV_ROOT="$VENV_ROOT"
export CACHESELECT_STORAGE_ROOT="$STORAGE_ROOT"
export CACHESELECT_CUDA_SETUP="/dev/null"
export CACHESELECT_HF_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"
export CACHESELECT_MTRAG_MODEL="$MODEL"
export CACHESELECT_MTRAG_MANIFEST_MODEL="$MANIFEST_MODEL"
export CACHESELECT_MODEL_DTYPE=bfloat16
export CACHESELECT_MTRAG_MAX_COMPLETION_TOKENS=768
export CACHESELECT_MTRAG_INPUT="$MTRAG_INPUT"
export CACHESELECT_MTRAG_SPLIT_ROOT="$SPLIT_ROOT"

set -o pipefail
bash benchmarks/run_mtrag_reference_splits.slurm 2>&1 | tee "$WRAPPER_LOG"

TRAIN="$RESULT_DIR/train-reference-calibration.json"
VALIDATION="$RESULT_DIR/validation-reference-calibration.json"
test -s "$TRAIN"
test -s "$VALIDATION"

# Reject missing, cached, truncated, or provenance-mismatched answers.
python - \
  "$TRAIN" "$VALIDATION" "$MODEL" "$MANIFEST_MODEL" \
  "$EXPECTED_TRAIN_COUNT" "$EXPECTED_VALIDATION_COUNT" <<'PY'
import json
import sys

expected_counts = {"train": int(sys.argv[5]), "validation": int(sys.argv[6])}
for path, split in zip(sys.argv[1:3], expected_counts):
    artifact = json.load(open(path, encoding="utf-8"))
    assert artifact["model"] == sys.argv[3]
    assert artifact["manifest_selection_model"] == sys.argv[4]
    assert artifact["request_count"] == expected_counts[split]
    assert all(row["split"] == split for row in artifact["rows"])
    assert all(row["cached_tokens"] == 0 for row in artifact["rows"])
    assert all(row["finish_reason"] == "stop" for row in artifact["rows"])
    assert all(row["output_text"] for row in artifact["rows"])
PY

# Prepare answers for a later blinded manual quality comparison.
python -m benchmarks.prepare_mtrag_reference_review \
  --input "$TRAIN" "$VALIDATION" \
  --review-output "$RESULT_DIR/blinded-reference-review.json" \
  --key-output "$RESULT_DIR/blinded-reference-key.json"

ARCHIVE="$STORAGE_ROOT/$RUN_ID-artifacts.tar.gz"
tar -czf "$ARCHIVE" -C "$STORAGE_ROOT" \
  "cacheselect-results/$RUN_ID" \
  "cacheselect-request-logs/$RUN_ID" \
  "cacheselect-server-logs/$RUN_ID" \
  "cacheselect-server-logs/runpod-$RUN_ID.out"

echo "RunPod Qwen3-14B MTRAG references completed successfully"
echo "commit=$PROJECT_COMMIT"
echo "gpu=$GPU_NAME"
echo "results=$RESULT_DIR"
echo "log=$WRAPPER_LOG"
echo "archive=$ARCHIVE"
echo "archive_sha256=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
