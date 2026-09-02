#!/usr/bin/env bash
# Run an exhaustive reviewed Qwen3-14B MTRAG experiment on a RunPod A40.

set -Eeuo pipefail

ROOT="${CACHESELECT_PROJECT_ROOT:-/workspace/DeltaCache}"
VENV="${CACHESELECT_VENV_ROOT:-/workspace/cacheselect-env-cu130}"
STORAGE="${CACHESELECT_STORAGE_ROOT:-/workspace}"
PLATFORM="${CACHESELECT_EXECUTION_PLATFORM:-runpod-a40}"
MODEL="${CACHESELECT_MTRAG_MODEL:-Qwen/Qwen3-14B}"
PORT="${CACHESELECT_SERVER_PORT:-8000}"
START_CASE="${CACHESELECT_COUNTERFACTUAL_START_CASE:-1}"
MAX_CASES="${CACHESELECT_COUNTERFACTUAL_MAX_CASES:-}"
FULL_DATASET="${CACHESELECT_FULL_MTRAG_COUNTERFACTUAL:-0}"
START_BATCH="${CACHESELECT_COUNTERFACTUAL_START_BATCH:-1}"
BLOCKS_PER_BATCH="${CACHESELECT_COUNTERFACTUAL_BLOCKS_PER_BATCH:-900}"
EXCLUDE_PLAN="${CACHESELECT_COUNTERFACTUAL_EXCLUDE_PLAN:-$ROOT/benchmarks/mtrag_completed_transitions.json}"
EXCLUDE_COMPLETED="${CACHESELECT_COUNTERFACTUAL_EXCLUDE_COMPLETED:-1}"
MLP_SMOKE="${CACHESELECT_MLP_SMOKE:-0}"
MLP_EVALUATION="${CACHESELECT_MLP_EVALUATION:-0}"
CHATRAG_EVALUATION="${CACHESELECT_CHATRAG_EVALUATION:-0}"
NATIVE_APC_EVALUATION="${CACHESELECT_NATIVE_APC_EVALUATION:-0}"
POLICY_SPLIT="${CACHESELECT_POLICY_SPLIT:-validation}"
MLP_MODEL="${CACHESELECT_MLP_MODEL:-}"
CORRECT_KV_POSITIONS="${CACHESELECT_CORRECT_KV_POSITIONS:-0}"
MIN_REUSE_SPAN_BLOCKS="${CACHESELECT_MIN_REUSE_SPAN_BLOCKS:-1}"
if [[ "$MLP_EVALUATION" == 1 || "$CHATRAG_EVALUATION" == 1 ]]; then
  MIN_REUSE_SPAN_BLOCKS="${CACHESELECT_MIN_REUSE_SPAN_BLOCKS:-8}"
fi
DEFAULT_INPUTS=qwen3-mtrag-v1
[[ "$FULL_DATASET" == 1 ]] && DEFAULT_INPUTS=qwen3-mtrag-full-v1
INPUTS="${CACHESELECT_COUNTERFACTUAL_INPUT_ROOT:-$STORAGE/cacheselect-inputs/$DEFAULT_INPUTS}"
PLAN="${CACHESELECT_COUNTERFACTUAL_PLAN:-$INPUTS/counterfactual-plan.json}"
REFERENCES="${CACHESELECT_COUNTERFACTUAL_REFERENCES:-$INPUTS/references.json}"
MTRAG="$STORAGE/cacheselect-data/mtrag/RAG.jsonl"
MTRAG_SHA="5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873"
MTRAG_URL="https://raw.githubusercontent.com/IBM/mt-rag-benchmark/cc5b1d481b391181b89f7ced860308482e785463/mtrag-human/generation_tasks/RAG.jsonl"
CHATRAG="${CACHESELECT_CHATRAG_INPUT:-$STORAGE/cacheselect-data/chatrag/doc2dial-test.json}"
CHATRAG_MANIFEST="${CACHESELECT_CHATRAG_MANIFEST:-$STORAGE/cacheselect-inputs/chatrag-doc2dial-policy-v1/policy-manifest.json}"
RUN_PREFIX=qwen3-mtrag-counterfactual
[[ "$CHATRAG_EVALUATION" == 1 ]] && RUN_PREFIX=qwen3-chatrag-policy
[[ "$MLP_EVALUATION" == 1 ]] && RUN_PREFIX=qwen3-mtrag-policy
[[ "$NATIVE_APC_EVALUATION" == 1 ]] && RUN_PREFIX=qwen3-mtrag-native-apc
[[ "$CHATRAG_EVALUATION$NATIVE_APC_EVALUATION" == 11 ]] \
  && RUN_PREFIX=qwen3-chatrag-native-apc
RUN_ID="$RUN_PREFIX-${CACHESELECT_EXPERIMENT_ID:-$(date -u +%s)}"
RESULT="$STORAGE/cacheselect-results/$RUN_ID"
SERVER_LOGS="$STORAGE/cacheselect-server-logs/$RUN_ID"
SERVER_LOG="$SERVER_LOGS/vllm.log"

cd "$ROOT"
source "$VENV/bin/activate"
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
[[ "$FULL_DATASET" == 0 || "$FULL_DATASET" == 1 ]] || exit 2
[[ "$START_BATCH" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$BLOCKS_PER_BATCH" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$EXCLUDE_COMPLETED" == 0 || "$EXCLUDE_COMPLETED" == 1 ]] || exit 2
[[ "$MLP_SMOKE" == 0 || "$MLP_SMOKE" == 1 ]] || exit 2
[[ "$MLP_EVALUATION" == 0 || "$MLP_EVALUATION" == 1 ]] || exit 2
[[ "$CHATRAG_EVALUATION" == 0 || "$CHATRAG_EVALUATION" == 1 ]] || exit 2
[[ "$NATIVE_APC_EVALUATION" == 0 || "$NATIVE_APC_EVALUATION" == 1 ]] || exit 2
[[ "$POLICY_SPLIT" == validation || "$POLICY_SPLIT" == test ]] || exit 2
MODE_COUNT=$((MLP_SMOKE + MLP_EVALUATION + CHATRAG_EVALUATION + NATIVE_APC_EVALUATION))
[[ "$MODE_COUNT" -le 1 || "$MODE_COUNT$CHATRAG_EVALUATION$NATIVE_APC_EVALUATION" == 211 ]] || exit 2
[[ "$CORRECT_KV_POSITIONS" == 0 || "$CORRECT_KV_POSITIONS" == 1 ]] || exit 2
[[ "$MIN_REUSE_SPAN_BLOCKS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$FULL_DATASET" != 1 || "$MLP_SMOKE$MLP_EVALUATION$NATIVE_APC_EVALUATION" == 000 ]] || exit 2
[[ "$CHATRAG_EVALUATION" != 1 || "$FULL_DATASET$CORRECT_KV_POSITIONS" == 01 ]] || exit 2
if [[ "$MLP_SMOKE" == 1 || "$MLP_EVALUATION" == 1 || "$CHATRAG_EVALUATION" == 1 ]]; then
  test -s "$MLP_MODEL" || { echo "CACHESELECT_MLP_MODEL is required" >&2; exit 2; }
fi

mkdir -p "$(dirname "$MTRAG")" "$RESULT/request-logs" "$SERVER_LOGS" \
  "$STORAGE/hf-cache" "$STORAGE/tmp"
if [[ "$CHATRAG_EVALUATION" == 1 ]]; then
  test -s "$CHATRAG" && test -s "$CHATRAG_MANIFEST"
else
  if [[ ! -s "$MTRAG" ]]; then
    curl --location --fail --retry 3 --output "$MTRAG.download" "$MTRAG_URL"
    mv "$MTRAG.download" "$MTRAG"
  fi
  [[ "$(sha256sum "$MTRAG" | cut -d ' ' -f 1)" == "$MTRAG_SHA" ]] || {
    echo "MTRAG source hash mismatch" >&2
    exit 2
  }
fi

export PYTHONPATH="$ROOT:$ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$STORAGE/hf-cache"
export TMPDIR="$STORAGE/tmp"
export HF_HUB_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE"
[[ "$(python -c 'import vllm; print(vllm.__file__)')" == "$ROOT"/vllm/* ]] || {
  echo "vLLM is not imported from this checkout" >&2
  exit 2
}
if [[ "$CHATRAG_EVALUATION" == 1 ]]; then
  :
elif [[ "$FULL_DATASET" == 1 ]]; then
  EXCLUDE_ARGS=()
  if [[ "$EXCLUDE_COMPLETED" == 1 ]]; then
    test -s "$EXCLUDE_PLAN" || { echo "completed counterfactual plan not found: $EXCLUDE_PLAN" >&2; exit 2; }
    EXCLUDE_ARGS+=(--exclude-plan "$EXCLUDE_PLAN")
  fi
  COVERAGE="$INPUTS/qwen3-mtrag-coverage.json"
  mkdir -p "$INPUTS"
  python -m benchmarks.analyze_mtrag_coverage \
    --input "$MTRAG" --output "$COVERAGE" --model "$MODEL" \
    --block-size 16 --local-files-only
  python -m benchmarks.freeze_mtrag_reference_expansion \
    --coverage "$COVERAGE" --source-dataset "$MTRAG" \
    --output-dir "$INPUTS" --full-repacking-coverage \
    "${EXCLUDE_ARGS[@]}" \
    --target-blocks-per-batch "$BLOCKS_PER_BATCH"
else
  test -s "$REFERENCES"
fi
test -s "$PLAN"
printf 'project_commit=%s\nmodel=%s\ngpu=%s\nexecution_platform=%s\n' \
  "$COMMIT" "$MODEL" "$GPU" "$PLATFORM" \
  >"$RESULT/metadata.env"
printf 'start_case=%s\n' "$START_CASE" >>"$RESULT/metadata.env"
printf 'max_cases=%s\n' "${MAX_CASES:-all}" >>"$RESULT/metadata.env"
printf 'correct_kv_positions=%s\n' "$CORRECT_KV_POSITIONS" >>"$RESULT/metadata.env"
printf 'min_reuse_span_blocks=%s\n' "$MIN_REUSE_SPAN_BLOCKS" >>"$RESULT/metadata.env"
printf 'policy_split=%s\n' "$POLICY_SPLIT" >>"$RESULT/metadata.env"
printf 'native_apc_evaluation=%s\n' "$NATIVE_APC_EVALUATION" >>"$RESULT/metadata.env"
if [[ "$MLP_EVALUATION" == 1 || "$CHATRAG_EVALUATION" == 1 ]]; then
  cp "$MLP_MODEL" "$RESULT/mlp-model.json"
fi

# Validate either reviewed inputs or the frozen live-reference full plan.
if [[ "$CHATRAG_EVALUATION" != 1 ]]; then
python - "$PLAN" "$REFERENCES" "$MODEL" "$MTRAG_SHA" "$FULL_DATASET" <<'PY'
import hashlib, json, sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
assert plan["source_model"] == sys.argv[3]
assert plan["source_dataset_sha256"] == sys.argv[4]
assert plan["transition_count"] == len(plan["transitions"])
assert plan["block_size"] == 16
assert isinstance(plan.get("repacking_enabled", False), bool)
assert plan["total_target_blocks"] == plan["total_testable_blocks"] > 0
assert all(row["target_block_indices"] == row["testable_block_indices"] for row in plan["transitions"])
if sys.argv[5] == "1":
    assert plan["reference_status"] == "generated_in_trial_pending_review"
    assert plan["batch_count"] == len(plan["batches"])
else:
    refs = json.load(open(sys.argv[2], encoding="utf-8"))
    assert refs["model"] == sys.argv[3]
    assert refs["source_dataset_sha256"] == sys.argv[4]
    assert plan["source_reference_manifest_sha256"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
    assert refs["accepted_task_count"] == plan["transition_count"]
PY
fi

REPACK_ARGS=()
if [[ "$CHATRAG_EVALUATION" == 1 ]] || python -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("repacking_enabled", False))' "$PLAN"; then
  REPACK_ARGS+=(--cacheselect-repack-partial-reuse)
else
  REPACK_ARGS+=(--no-cacheselect-repack-partial-reuse)
fi
POSITION_ARGS=()
[[ "$CORRECT_KV_POSITIONS" == 1 ]] \
  && POSITION_ARGS+=(--cacheselect-correct-kv-positions)
CASE_LIMIT_ARGS=()
if [[ -n "$MAX_CASES" ]]; then
  [[ "$MAX_CASES" =~ ^[1-9][0-9]*$ ]] || exit 2
  CASE_LIMIT_ARGS+=(--max-cases "$MAX_CASES")
fi
REPAIR_ARGS=(--cacheselect-repair-selector full_block)
if [[ "$MLP_SMOKE" == 1 || "$MLP_EVALUATION" == 1 || "$CHATRAG_EVALUATION" == 1 ]]; then
  REPAIR_ARGS=(--cacheselect-repair-selector mlp --cacheselect-mlp-model "$MLP_MODEL")
fi
CACHESELECT_ARGS=()
if [[ "$NATIVE_APC_EVALUATION" != 1 ]]; then
  CACHESELECT_ARGS=(
    --enable-cacheselect
    "${REPAIR_ARGS[@]}"
    --cacheselect-execute-partial-reuse
    --cacheselect-min-reuse-span-blocks "$MIN_REUSE_SPAN_BLOCKS"
    "${REPACK_ARGS[@]}"
    "${POSITION_ARGS[@]}"
  )
fi

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$SERVER_PID" 2>/dev/null || return 0
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
  --enable-prefix-caching \
  "${CACHESELECT_ARGS[@]}" \
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

POLICY_ARGS=()
[[ "$NATIVE_APC_EVALUATION" == 1 ]] && POLICY_ARGS+=(--native-apc-policy)
if [[ "$MLP_SMOKE" == 1 ]]; then
  python -m benchmarks.run_hybrid_apc_baseline \
    --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --run-id "$RUN_ID" --request-log-dir "$RESULT/request-logs" \
    --output "$RESULT/summary.json" --record-count 40 \
    --max-completion-tokens 16 --timeout-seconds 900 \
    --validate-against-reference
  stop_server
  SERVER_PID=""
  ARCHIVE="$STORAGE/$RUN_ID-artifacts.tar.gz"
  tar -czf "$ARCHIVE" -C "$STORAGE" \
    "cacheselect-results/$RUN_ID" "cacheselect-server-logs/$RUN_ID"
  echo "Completed $RUN_ID"
  echo "archive=$ARCHIVE"
  echo "archive_sha256=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
  exit 0
fi

if [[ "$CHATRAG_EVALUATION" == 1 ]]; then
  python -m benchmarks.run_mtrag_counterfactual_pilot \
    --input "$CHATRAG" --input-format chatrag --subset doc2dial \
    --manifest "$CHATRAG_MANIFEST" --live-references \
    --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
    --max-completion-tokens 2048 --timeout-seconds 900 \
    --policy-evaluation --policy-split test --start-case "$START_CASE" \
    "${POLICY_ARGS[@]}" "${CASE_LIMIT_ARGS[@]}" --run-id "$RUN_ID" \
    --request-log-dir "$RESULT/request-logs" --output-dir "$RESULT" \
    --summary-output "$RESULT/summary.json"
  stop_server
  SERVER_PID=""
  ARCHIVE="$STORAGE/$RUN_ID-artifacts.tar.gz"
  tar -czf "$ARCHIVE" -C "$STORAGE" \
    "cacheselect-results/$RUN_ID" "cacheselect-server-logs/$RUN_ID"
  echo "Completed $RUN_ID"
  echo "archive=$ARCHIVE"
  echo "archive_sha256=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
  exit 0
fi

if [[ "$MLP_EVALUATION" == 1 || "$NATIVE_APC_EVALUATION" == 1 ]]; then
  python -m benchmarks.run_mtrag_counterfactual_pilot \
    --input "$MTRAG" --manifest "$PLAN" --references "$REFERENCES" \
    --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
    --max-completion-tokens 2048 --timeout-seconds 900 \
    --policy-evaluation --policy-split "$POLICY_SPLIT" \
    "${POLICY_ARGS[@]}" --run-id "$RUN_ID" \
    --request-log-dir "$RESULT/request-logs" --output-dir "$RESULT" \
    --summary-output "$RESULT/summary.json"
  stop_server
  SERVER_PID=""
  ARCHIVE="$STORAGE/$RUN_ID-artifacts.tar.gz"
  tar -czf "$ARCHIVE" -C "$STORAGE" \
    "cacheselect-results/$RUN_ID" "cacheselect-server-logs/$RUN_ID"
  echo "Completed $RUN_ID"
  echo "archive=$ARCHIVE"
  echo "archive_sha256=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
  exit 0
fi

if [[ "$FULL_DATASET" == 1 ]]; then
  BATCH_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_count"])' "$PLAN")"
  ((START_BATCH <= BATCH_COUNT)) || { echo "start batch exceeds $BATCH_COUNT" >&2; exit 2; }
  CASES_LEFT="$MAX_CASES"
  while IFS=$'\t' read -r BATCH_INDEX BATCH_SPLIT BATCH_START BATCH_CASES BATCH_BLOCKS; do
    ((BATCH_INDEX >= START_BATCH)) || continue
    if [[ -n "$CASES_LEFT" ]]; then
      ((CASES_LEFT > 0)) || break
      RUN_CASES=$((BATCH_CASES < CASES_LEFT ? BATCH_CASES : CASES_LEFT))
      RUN_BLOCKS="$(python - "$PLAN" "$BATCH_START" "$RUN_CASES" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))["transitions"]
start = int(sys.argv[2]) - 1
print(sum(len(row["target_block_indices"]) for row in rows[start:start + int(sys.argv[3])]))
PY
)"
    else
      RUN_CASES="$BATCH_CASES"
      RUN_BLOCKS="$BATCH_BLOCKS"
    fi
    BATCH_TAG="$(printf '%02d' "$BATCH_INDEX")"
    BATCH_RUN_ID="$RUN_ID-batch-$BATCH_TAG"
    BATCH_RESULT="$STORAGE/cacheselect-results/$BATCH_RUN_ID"
    mkdir -p "$BATCH_RESULT/request-logs"
    printf 'project_commit=%s\nmodel=%s\ngpu=%s\nsplit=%s\nbatch=%s\n' \
      "$COMMIT" "$MODEL" "$GPU" "$BATCH_SPLIT" "$BATCH_INDEX" \
      >"$BATCH_RESULT/metadata.env"
    cp "$PLAN" "$BATCH_RESULT/counterfactual-plan.json"

    python -m benchmarks.run_mtrag_counterfactual_pilot \
      --input "$MTRAG" --manifest "$PLAN" --live-references \
      --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
      --max-completion-tokens 2048 --timeout-seconds 900 \
      --start-case "$BATCH_START" --max-cases "$RUN_CASES" \
      --run-id "$BATCH_RUN_ID" --request-log-dir "$BATCH_RESULT/request-logs" \
      --output-dir "$BATCH_RESULT" --summary-output "$BATCH_RESULT/summary.json"

    python - "$BATCH_RESULT/summary.json" "$RUN_BLOCKS" "$MODEL" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
assert summary["case_count"] == summary["completed_case_count"]
assert summary["planned_target_blocks"] == int(sys.argv[2])
assert summary["trial_count"] == summary["planned_target_blocks"]
assert summary["skipped_target_blocks"] == 0
rows = []
for case in summary["cases"]:
    rows.append({
        "task_id": case["current_task_id"],
        "split": case["split"],
        "collection": case["collection"],
        "prompt_token_count": case["reference_prompt_token_count"],
        "cached_tokens": case["reference_cached_tokens"],
        "finish_reason": case["reference_finish_reason"],
        "output_text": case["reference_output"],
        "expected_answer": case["expected_answer"],
        "quality": case["reference_quality"],
    })
artifact = {
    "schema_version": 1,
    "analysis": "mtrag-reference-quality-calibration",
    "gate_status": "pending_manual_review",
    "model": sys.argv[3],
    "source_sha256": summary["source_sha256"],
    "request_count": len(rows),
    "rows": rows,
}
(path.parent / "live-reference-calibration.json").write_text(
    json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
)
PY
    python -m benchmarks.prepare_mtrag_reference_review \
      --input "$BATCH_RESULT/live-reference-calibration.json" \
      --review-output "$BATCH_RESULT/blinded-reference-review.json" \
      --key-output "$BATCH_RESULT/blinded-reference-key.json"
    if python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["invalid_trials"] != 0)' \
      "$BATCH_RESULT/summary.json"; then
      python -m benchmarks.prepare_mtrag_counterfactual_review \
        --result-dir "$BATCH_RESULT"
    fi
    tail -n 1000 "$SERVER_LOG" >"$BATCH_RESULT/vllm-tail.log"
    BATCH_ARCHIVE="$STORAGE/$BATCH_RUN_ID-artifacts.tar.gz"
    tar -czf "$BATCH_ARCHIVE" -C "$STORAGE" \
      "cacheselect-results/$BATCH_RUN_ID"
    [[ -z "$CASES_LEFT" ]] || CASES_LEFT=$((CASES_LEFT - RUN_CASES))
    echo "Completed batch $BATCH_INDEX/$BATCH_COUNT ($RUN_BLOCKS blocks)"
    echo "archive=$BATCH_ARCHIVE"
    echo "archive_sha256=$(sha256sum "$BATCH_ARCHIVE" | cut -d ' ' -f 1)"
  done < <(python - "$PLAN" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
for row in plan["batches"]:
    print(row["batch"], row["split"], row["start_case"], row["case_count"], row["target_block_count"], sep="\t")
PY
  )
  [[ -z "$CASES_LEFT" || "$CASES_LEFT" == 0 ]] || {
    echo "$CASES_LEFT requested cases were not run" >&2
    exit 2
  }
  stop_server
  SERVER_PID=""
  echo "Completed all full-MTRAG batches from batch $START_BATCH"
  exit 0
fi

python -m benchmarks.run_mtrag_counterfactual_pilot \
  --input "$MTRAG" --manifest "$PLAN" --references "$REFERENCES" \
  --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
  --max-completion-tokens 2048 --timeout-seconds 900 \
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
