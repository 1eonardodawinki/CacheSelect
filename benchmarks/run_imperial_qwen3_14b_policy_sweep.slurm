#!/usr/bin/env bash
#SBATCH --job-name=qwen3-policy-sweep
#SBATCH --partition=a40
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/qwen3-policy-sweep-%j.out

set -Eeuo pipefail

JOB_ID="${SLURM_JOB_ID:?Run this file with sbatch}"
ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
STORAGE="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
MODEL_DIR="${CACHESELECT_REUSE_BUDGET_MODEL_DIR:-$STORAGE/cacheselect-models/logistic-reuse-budget-v1}"
PORT="${CACHESELECT_SERVER_PORT:-$((30000 + JOB_ID % 1000))}"
RUNNER="$ROOT/benchmarks/run_runpod_qwen3_14b_counterfactual.sh"
SWEEP_RESULT="$STORAGE/cacheselect-results/qwen3-mtrag-policy-sweep-$JOB_ID"

source "${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
mkdir -p "$SWEEP_RESULT"
COMMON=(
  CACHESELECT_PROJECT_ROOT="$ROOT"
  CACHESELECT_VENV_ROOT="${CACHESELECT_VENV_ROOT:-$STORAGE/cacheselect-env}"
  CACHESELECT_STORAGE_ROOT="$STORAGE"
  CACHESELECT_COUNTERFACTUAL_INPUT_ROOT="${CACHESELECT_COUNTERFACTUAL_INPUT_ROOT:-$STORAGE/cacheselect-inputs/qwen3-mtrag-v3}"
  CACHESELECT_EXECUTION_PLATFORM="${CACHESELECT_EXECUTION_PLATFORM:-imperial-a40}"
  CACHESELECT_HF_OFFLINE=1
  CACHESELECT_POLICY_SPLIT=test
  CACHESELECT_SERVER_PORT="$PORT"
)

env "${COMMON[@]}" \
  CACHESELECT_EXPERIMENT_ID="$JOB_ID-native-apc" \
  CACHESELECT_NATIVE_APC_EVALUATION=1 \
  bash "$RUNNER"
BASELINE="$STORAGE/cacheselect-results/qwen3-mtrag-native-apc-$JOB_ID-native-apc"
ARCHIVE_PATHS=(
  "cacheselect-results/qwen3-mtrag-native-apc-$JOB_ID-native-apc"
  "cacheselect-server-logs/qwen3-mtrag-native-apc-$JOB_ID-native-apc"
)

RESULT_ARGS=()
for BUDGET in 005 010 025 050 075 100; do
  MODEL="$MODEL_DIR/logistic-selector-runtime-reuse-$BUDGET.json"
  test -s "$MODEL"
  env "${COMMON[@]}" \
    CACHESELECT_EXPERIMENT_ID="$JOB_ID-reuse-$BUDGET" \
    CACHESELECT_MLP_EVALUATION=1 \
    CACHESELECT_MLP_MODEL="$MODEL" \
    CACHESELECT_CORRECT_KV_POSITIONS=1 \
    bash "$RUNNER"
  RESULT_ARGS+=(
    --result-dir
    "$STORAGE/cacheselect-results/qwen3-mtrag-policy-$JOB_ID-reuse-$BUDGET"
  )
  ARCHIVE_PATHS+=(
    "cacheselect-results/qwen3-mtrag-policy-$JOB_ID-reuse-$BUDGET"
    "cacheselect-server-logs/qwen3-mtrag-policy-$JOB_ID-reuse-$BUDGET"
  )
done

python -m benchmarks.analyze_policy_reuse_sweep \
  --baseline-result-dir "$BASELINE" \
  "${RESULT_ARGS[@]}" \
  --output "$SWEEP_RESULT/summary.json"

ARCHIVE="$STORAGE/qwen3-mtrag-policy-sweep-$JOB_ID-artifacts.tar.gz"
tar -czf "$ARCHIVE" -C "$STORAGE" \
  "cacheselect-results/qwen3-mtrag-policy-sweep-$JOB_ID" \
  "${ARCHIVE_PATHS[@]}"
echo "Completed native APC plus six CacheSelect logistic policies"
echo "results=$SWEEP_RESULT"
echo "archive=$ARCHIVE"
echo "archive_sha256=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)"
