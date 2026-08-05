#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-locator-calibration
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/locator-calibration-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
CUDA_SETUP="${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
MODEL="${CACHESELECT_BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Run this file as a Slurm array job}"
CALIBRATION_ID="${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"

TARGET_PROMPT_TOKENS=(256 1024 4096)
EDIT_POSITIONS=(early middle late)
TARGET="${TARGET_PROMPT_TOKENS[$TASK_ID]}"
MAX_COMPLETION_TOKENS=96
BLOCK_SIZE=16

# Array tasks can share a host, so assign a task-specific port.
PORT=$((26000 + (CALIBRATION_ID % 1000) * 20 + TASK_ID))
RESULT_DIR="$STORAGE_ROOT/cacheselect-results/locator-calibration-$CALIBRATION_ID"
REQUEST_LOG_DIR="$STORAGE_ROOT/cacheselect-request-logs/locator-calibration-$CALIBRATION_ID"
SERVER_LOG_DIR="$STORAGE_ROOT/cacheselect-server-logs/locator-calibration-$CALIBRATION_ID"
TRACE_DIR="$STORAGE_ROOT/cacheselect-generated-traces/locator-calibration-$CALIBRATION_ID"
TMP_ROOT="$STORAGE_ROOT/tmp"

mkdir -p \
  "$RESULT_DIR" \
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

echo "CacheSelect online locator prompt-length calibration"
echo "calibration_id=$CALIBRATION_ID task_id=$TASK_ID target_prompt_tokens=$TARGET"
echo "host=$(hostname) port=$PORT"
echo "project_commit=$PROJECT_COMMIT"
echo "model=$MODEL vllm_version=$VLLM_VERSION"
echo "gpu=$GPU_DESCRIPTION"

python -m benchmarks.generate_length_calibration \
  --model "$MODEL" \
  --target-prompt-tokens "$TARGET" \
  --output-dir "$TRACE_DIR"

SERVER_PID=""

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

write_manifest() {
  local manifest_path="$1"
  local edit_position="$2"

  python - \
    "$manifest_path" \
    "$CALIBRATION_ID" \
    "$TASK_ID" \
    "${SLURM_JOB_ID:-}" \
    "$PROJECT_COMMIT" \
    "$MODEL" \
    "$VLLM_VERSION" \
    "$GPU_DESCRIPTION" \
    "$TARGET" \
    "$edit_position" \
    "$MAX_COMPLETION_TOKENS" \
    "$BLOCK_SIZE" \
    "$PORT" <<'PY'
import json
import socket
import sys

(
    manifest_path,
    calibration_job_id,
    array_task_id,
    slurm_job_id,
    project_commit,
    model,
    vllm_version,
    gpu,
    target_prompt_tokens,
    edit_position,
    max_completion_tokens,
    block_size,
    port,
) = sys.argv[1:]
manifest = {
    "experiment": "online_locator_length_calibration",
    "calibration_job_id": calibration_job_id,
    "array_task_id": int(array_task_id),
    "slurm_job_id": slurm_job_id,
    "host": socket.gethostname(),
    "project_commit": project_commit,
    "model": model,
    "vllm_version": vllm_version,
    "gpu": gpu,
    "target_prompt_tokens": int(target_prompt_tokens),
    "edit_position": edit_position,
    "apc": "on",
    "planner_mode": "vllm",
    "repetition": 1,
    "max_completion_tokens": int(max_completion_tokens),
    "block_size": int(block_size),
    "port": int(port),
}
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
PY
}

validate_result() {
  local result_path="$1"

  python - "$result_path" "$TARGET" "$BLOCK_SIZE" <<'PY'
import json
import sys
from pathlib import Path

from benchmarks.analyze_reuse_opportunities import analyze_benchmark_result

path = Path(sys.argv[1])
target = int(sys.argv[2])
block_size = int(sys.argv[3])
result = json.loads(path.read_text())
observations = result["observations"]
if len(observations) != 2:
    raise SystemExit(f"expected two observations, found {len(observations)}")

prompt_tokens = [item["prompt_token_count"] for item in observations]
if any(abs(count - target) > 16 for count in prompt_tokens):
    raise SystemExit(
        f"rendered prompts outside target tolerance: target={target}, "
        f"actual={prompt_tokens}"
    )

finish_reasons = [
    item["raw_response"]["choices"][0]["finish_reason"] for item in observations
]
if finish_reasons != ["stop", "stop"]:
    raise SystemExit(f"truncated calibration output: {finish_reasons}")

cached = [item["cached_tokens"] for item in observations]
if cached[0] != 0 or cached[1] <= 0:
    raise SystemExit(f"APC did not show cold-then-warm reuse: {cached}")
if [item["quality"]["passed"] for item in observations] != [True, True]:
    raise SystemExit("calibration quality failed")
if result["request_ledger_summary"] != {
    "started": 2,
    "completed": 2,
    "failed": 0,
}:
    raise SystemExit(f"incomplete request ledger: {result['request_ledger_summary']}")

runtime = observations[1].get("runtime_policy") or {}
plan = runtime.get("partial_reuse_plan")
if plan is None:
    raise SystemExit("edited request omitted its online partial-reuse plan")
transition = result["transitions"][0]
if (
    plan["transition_id"] != transition["transition_id"]
    or plan["source_request_id"] != transition["previous_request_id"]
    or plan["target_request_id"] != transition["current_request_id"]
):
    raise SystemExit(f"online plan references the wrong transition: {plan}")
if plan["block_size"] != block_size:
    raise SystemExit(f"unexpected online block size: {plan['block_size']}")
if plan["native_cached_tokens"] != cached[1]:
    raise SystemExit("online plan and executed APC disagree on cached tokens")
if runtime.get("policy") != "VLLM_NATIVE_APC":
    raise SystemExit(f"safe APC fallback was not preserved: {runtime}")

offline = analyze_benchmark_result(result, block_size=block_size)
opportunity = offline["transitions"][0]["opportunity"]
expected_aligned_tokens = opportunity["whole_source_block_count"] * block_size
if plan["candidate_token_count"] != expected_aligned_tokens:
    raise SystemExit(
        "online/offline aligned candidate mismatch: "
        f"online={plan['candidate_token_count']} "
        f"offline={expected_aligned_tokens}"
    )
if plan["resident_candidate_token_count"] != plan["candidate_token_count"]:
    raise SystemExit(
        "not all aligned source blocks remained resident: "
        f"{plan['resident_candidate_token_count']}/"
        f"{plan['candidate_token_count']}"
    )
if "source_block_id" in json.dumps(plan):
    raise SystemExit("internal GPU block IDs leaked into response metrics")

print(f"prompt_tokens={prompt_tokens}")
print(f"cached_tokens={cached}")
print(
    "candidate_tokens="
    f"{plan['candidate_token_count']} "
    f"resident_tokens={plan['resident_candidate_token_count']} "
    f"offline_all_candidates={opportunity['candidate_token_count']}"
)
print(f"finish_reasons={finish_reasons}")
print("quality_passed=[True, True]")
PY
}

run_condition() {
  local edit_position="$1"
  local run_key="tokens-${TARGET}-${edit_position}-cacheselect-rep-1"
  local trace_path="$TRACE_DIR/tokens-${TARGET}-${edit_position}.json"
  local result_path="$RESULT_DIR/$run_key.json"
  local manifest_path="$RESULT_DIR/$run_key.manifest.json"
  local server_log="$SERVER_LOG_DIR/$run_key.vllm.log"

  test -f "$trace_path"
  write_manifest "$manifest_path" "$edit_position"

  echo "Starting condition: $run_key"
  setsid vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-cacheselect \
    --enable-prompt-tokens-details \
    --enable-per-request-metrics \
    >"$server_log" 2>&1 &
  SERVER_PID=$!

  wait_for_server "$server_log"
  echo "vLLM is healthy pid=$SERVER_PID"

  python -m benchmarks.run_vllm_baseline \
    --trace "$trace_path" \
    --model "$MODEL" \
    --base-url "http://127.0.0.1:$PORT" \
    --apc-label on \
    --planner-mode vllm \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --run-id "locator-calibration-$CALIBRATION_ID-$run_key" \
    --request-log-dir "$REQUEST_LOG_DIR" \
    --output "$result_path"

  stop_server
  sleep 2
  validate_result "$result_path"

  echo "Locator calibration condition completed successfully: $run_key"
  echo "result=$result_path"
  echo "manifest=$manifest_path"
  echo "server_log=$server_log"
}

for edit_position in "${EDIT_POSITIONS[@]}"; do
  run_condition "$edit_position"
done

echo "Online locator calibration task completed successfully: target=$TARGET"
echo "results=$RESULT_DIR"
echo "request_logs=$REQUEST_LOG_DIR"
echo "server_logs=$SERVER_LOG_DIR"
echo "generated_traces=$TRACE_DIR"
