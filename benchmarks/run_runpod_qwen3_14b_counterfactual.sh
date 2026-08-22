#!/usr/bin/env bash
# Run an exhaustive reviewed Qwen3-14B MTRAG experiment on a RunPod A40.

set -Eeuo pipefail

ROOT="${CACHESELECT_PROJECT_ROOT:-/workspace/DeltaCache}"
VENV="${CACHESELECT_VENV_ROOT:-/workspace/cacheselect-env-cu130}"
STORAGE="${CACHESELECT_STORAGE_ROOT:-/workspace}"
MODEL="${CACHESELECT_MTRAG_MODEL:-Qwen/Qwen3-14B}"
PORT="${CACHESELECT_SERVER_PORT:-8000}"
START_CASE="${CACHESELECT_COUNTERFACTUAL_START_CASE:-1}"
MAX_CASES="${CACHESELECT_COUNTERFACTUAL_MAX_CASES:-}"
INPUTS="${CACHESELECT_COUNTERFACTUAL_INPUT_ROOT:-$STORAGE/cacheselect-inputs/qwen3-mtrag-v1}"
PLAN="${CACHESELECT_COUNTERFACTUAL_PLAN:-$INPUTS/counterfactual-plan.json}"
REFERENCES="${CACHESELECT_COUNTERFACTUAL_REFERENCES:-$INPUTS/references.json}"
MTRAG="$STORAGE/cacheselect-data/mtrag/RAG.jsonl"
MTRAG_SHA="5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873"
MTRAG_URL="https://raw.githubusercontent.com/IBM/mt-rag-benchmark/cc5b1d481b391181b89f7ced860308482e785463/mtrag-human/generation_tasks/RAG.jsonl"
RUN_ID="qwen3-mtrag-counterfactual-${CACHESELECT_EXPERIMENT_ID:-$(date -u +%s)}"
RESULT="$STORAGE/cacheselect-results/$RUN_ID"
SERVER_LOGS="$STORAGE/cacheselect-server-logs/$RUN_ID"
SERVER_LOG="$SERVER_LOGS/vllm.log"

cd "$ROOT"
source "$VENV/bin/activate"
test -s "$PLAN" && test -s "$REFERENCES"
COMMIT="$(git rev-parse HEAD)"
git diff --quiet && git diff --cached --quiet || {
  echo "RunPod checkout must be clean" >&2
  exit 2
}
GPU_MEMORY="$({ nvidia-smi --query-gpu=memory.total \
  --format=csv,noheader,nounits | head -n 1; })"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "$GPU" == *"A40"* \
  && "$GPU_MEMORY" =~ ^[0-9]+$ && "$GPU_MEMORY" -ge 45000 ]] || {
  echo "This experiment requires a 48 GB A40" >&2
  exit 2
}
[[ "$START_CASE" =~ ^[1-9][0-9]*$ ]] || {
  echo "CACHESELECT_COUNTERFACTUAL_START_CASE must be positive" >&2
  exit 2
}

mkdir -p "$(dirname "$MTRAG")" "$RESULT/request-logs" "$SERVER_LOGS" \
  "$STORAGE/hf-cache" "$STORAGE/tmp"
if [[ ! -s "$MTRAG" ]]; then
  curl --location --fail --retry 3 --output "$MTRAG.download" "$MTRAG_URL"
  mv "$MTRAG.download" "$MTRAG"
fi
[[ "$(sha256sum "$MTRAG" | cut -d ' ' -f 1)" == "$MTRAG_SHA" ]] || {
  echo "MTRAG source hash mismatch" >&2
  exit 2
}

export PYTHONPATH="$ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$STORAGE/hf-cache"
export TMPDIR="$STORAGE/tmp"
export HF_HUB_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE"
[[ "$(python -c 'import vllm; print(vllm.__file__)')" == "$ROOT"/vllm/* ]] || {
  echo "vLLM is not imported from this checkout" >&2
  exit 2
}
printf 'project_commit=%s\nmodel=%s\ngpu=%s\n' "$COMMIT" "$MODEL" "$GPU" \
  >"$RESULT/metadata.env"
printf 'start_case=%s\n' "$START_CASE" >>"$RESULT/metadata.env"

# Bind the exact reviewed answers to a plan containing every testable block.
python - "$PLAN" "$REFERENCES" "$MODEL" "$MTRAG_SHA" <<'PY'
import hashlib, json, sys

plan, refs = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:3])
assert plan["source_model"] == refs["model"] == sys.argv[3]
assert plan["source_dataset_sha256"] == refs["source_dataset_sha256"] == sys.argv[4]
assert plan["source_reference_manifest_sha256"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert refs["accepted_task_count"] == plan["transition_count"] == len(plan["transitions"])
assert plan["block_size"] == 16
assert isinstance(plan.get("repacking_enabled", False), bool)
assert plan["total_target_blocks"] == plan["total_testable_blocks"] > 0
assert all(row["target_block_indices"] == row["testable_block_indices"] for row in plan["transitions"])
PY

REPACK_ARGS=()
if python -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("repacking_enabled", False))' "$PLAN"; then
  REPACK_ARGS+=(--cacheselect-repack-partial-reuse)
else
  REPACK_ARGS+=(--no-cacheselect-repack-partial-reuse)
fi
CASE_LIMIT_ARGS=()
if [[ -n "$MAX_CASES" ]]; then
  [[ "$MAX_CASES" =~ ^[1-9][0-9]*$ ]] || exit 2
  CASE_LIMIT_ARGS+=(--max-cases "$MAX_CASES")
fi

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$SERVER_PID" 2>/dev/null || return
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM

setsid vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --block-size 16 \
  --enable-prefix-caching --enable-cacheselect \
  --cacheselect-repair-selector full_block \
  --cacheselect-execute-partial-reuse \
  "${REPACK_ARGS[@]}" \
  --no-enable-chunked-prefill --enforce-eager \
  --enable-prompt-tokens-details --enable-per-request-metrics \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in {1..1800}; do
  kill -0 "$SERVER_PID" 2>/dev/null || {
    tail -n 100 "$SERVER_LOG" >&2
    exit 1
  }
  curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 2
done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null

python -m benchmarks.run_mtrag_counterfactual_pilot \
  --input "$MTRAG" --manifest "$PLAN" --references "$REFERENCES" \
  --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
  --max-completion-tokens 768 --timeout-seconds 900 \
  --start-case "$START_CASE" \
  "${CASE_LIMIT_ARGS[@]}" \
  --run-id "$RUN_ID" --request-log-dir "$RESULT/request-logs" \
  --output-dir "$RESULT" --summary-output "$RESULT/summary.json"

python - "$RESULT/summary.json" "$PLAN" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
plan = json.load(open(sys.argv[2], encoding="utf-8"))
selected = plan["transitions"][summary["source_case_start"] - 1:summary["source_case_end"]]
assert summary["case_count"] == len(selected)
assert summary["completed_case_count"] + summary["skipped_reference_case_count"] == summary["case_count"]
assert summary["planned_target_blocks"] == sum(len(row["target_block_indices"]) for row in selected)
assert summary["trial_count"] + summary["skipped_target_blocks"] == summary["planned_target_blocks"]
assert summary["invalid_trials"] == 0
PY
python -m benchmarks.prepare_mtrag_counterfactual_review --result-dir "$RESULT"

stop_server
SERVER_PID=""
ARCHIVE="$STORAGE/$RUN_ID-artifacts.tar.gz"
tar -czf "$ARCHIVE" -C "$STORAGE" \
  "cacheselect-results/$RUN_ID" "cacheselect-server-logs/$RUN_ID"
echo "Completed $RUN_ID"
echo "archive=$ARCHIVE"
echo "archive_sha256=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
