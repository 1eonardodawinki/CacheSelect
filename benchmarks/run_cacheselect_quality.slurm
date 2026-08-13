#!/usr/bin/env bash
#SBATCH --job-name=cacheselect-quality
#SBATCH --array=0-8%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/quality-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Run this file as a Slurm array job}"
TARGETS=(256 1024 4096)
RADII=(0 1 2)

# Map one array index to one prompt-length and repair-radius condition.
TARGET_INDEX=$((TASK_ID / ${#RADII[@]}))
RADIUS_INDEX=$((TASK_ID % ${#RADII[@]}))
TARGET="${TARGETS[$TARGET_INDEX]}"
RADIUS="${RADII[$RADIUS_INDEX]}"

export CACHESELECT_EXPERIMENT_KIND="quality"
export CACHESELECT_TARGET_INDEX="$TARGET_INDEX"
export CACHESELECT_MODE_ORDER_INDEX="$RADIUS_INDEX"
export CACHESELECT_EDIT_RADIUS="$RADIUS"
export CACHESELECT_ANSWER_SENSITIVE=1
export CACHESELECT_CONDITION_PATH="tokens-$TARGET/radius-$RADIUS"
export CACHESELECT_PERFORMANCE_REPETITIONS="${CACHESELECT_QUALITY_REPETITIONS:-3}"

echo "Quality matrix task=$TASK_ID target_tokens=$TARGET edit_radius=$RADIUS"
exec bash "$PROJECT_ROOT/benchmarks/run_cacheselect_performance.slurm"
