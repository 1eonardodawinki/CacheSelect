#!/usr/bin/env bash
# Run the controlled Qwen3-14B span break-even matrix on an A40.

set -Eeuo pipefail

ROOT="${CACHESELECT_PROJECT_ROOT:-/workspace/DeltaCache}"
VENV="${CACHESELECT_VENV_ROOT:-/workspace/cacheselect-env-cu130}"
STORAGE="${CACHESELECT_STORAGE_ROOT:-/workspace}"
MODEL="${CACHESELECT_MODEL:-Qwen/Qwen3-14B}"
PORT="${CACHESELECT_SERVER_PORT:-8000}"
SOURCE_OFFSET_TOKENS="${CACHESELECT_SPAN_SOURCE_OFFSET_TOKENS:-0}"
RUN_PREFIX=span-breakeven
[[ "$SOURCE_OFFSET_TOKENS" == 0 ]] || RUN_PREFIX=span-repack-breakeven
RUN_ID="$RUN_PREFIX-${CACHESELECT_EXPERIMENT_ID:-$(date -u +%s)}"
RESULT="$STORAGE/cacheselect-results/$RUN_ID"
LOGS="$STORAGE/cacheselect-server-logs/$RUN_ID"

cd "$ROOT"
source "$VENV/bin/activate"
export PYTHONPATH="$ROOT:$ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$STORAGE/hf-cache"
export HF_HUB_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE"
export TMPDIR="$STORAGE/tmp"
mkdir -p "$RESULT" "$LOGS" "$TMPDIR"
[[ "$SOURCE_OFFSET_TOKENS" =~ ^[0-9]+$ && "$SOURCE_OFFSET_TOKENS" -lt 16 ]] || exit 2

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "$GPU" == *A40* ]] || { echo "This benchmark requires an A40" >&2; exit 2; }
printf 'commit=%s\nmodel=%s\ngpu=%s\nsource_offset_tokens=%s\n' \
  "$(git rev-parse HEAD)" "$MODEL" "$GPU" "$SOURCE_OFFSET_TOKENS" \
  >"$RESULT/metadata.env"

REPACK_ARG=--no-cacheselect-repack-partial-reuse
[[ "$SOURCE_OFFSET_TOKENS" == 0 ]] || REPACK_ARG=--cacheselect-repack-partial-reuse

SERVER_PID=""
stop_server() {
  [[ -z "$SERVER_PID" ]] || kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  [[ -z "$SERVER_PID" ]] || wait "$SERVER_PID" 2>/dev/null || true
}
trap stop_server EXIT INT TERM

setsid vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --block-size 16 --enable-prefix-caching \
  --enable-cacheselect --cacheselect-repair-selector edit_proximity \
  --cacheselect-edit-radius 0 --cacheselect-execute-partial-reuse \
  "$REPACK_ARG" --cacheselect-correct-kv-positions \
  --no-enable-chunked-prefill --enforce-eager \
  --enable-prompt-tokens-details --enable-per-request-metrics \
  >"$LOGS/vllm.log" 2>&1 &
SERVER_PID=$!

for _ in {1..1800}; do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 100 "$LOGS/vllm.log" >&2; exit 1; }
  curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 2
done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null

python -m benchmarks.run_span_breakeven \
  --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
  --gap-blocks 1 2 4 8 16 --flank-blocks 16 --repetitions 7 \
  --source-offset-tokens "$SOURCE_OFFSET_TOKENS" \
  --output "$RESULT/summary.json"

stop_server
SERVER_PID=""
ARCHIVE="$STORAGE/$RUN_ID-artifacts.tar.gz"
tar -czf "$ARCHIVE" -C "$STORAGE" \
  "cacheselect-results/$RUN_ID" "cacheselect-server-logs/$RUN_ID"
echo "Completed span break-even benchmark"
echo "results=$RESULT"
echo "archive=$ARCHIVE"
echo "archive_sha256=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
