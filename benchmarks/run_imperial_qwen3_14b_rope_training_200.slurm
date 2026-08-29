#!/usr/bin/env bash
#SBATCH --job-name=qwen3-rope-200
#SBATCH --partition=a40
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/qwen3-rope-200-%j.out

set -Eeuo pipefail

export CACHESELECT_FULL_MTRAG_COUNTERFACTUAL=1
export CACHESELECT_COUNTERFACTUAL_MAX_CASES=200
export CACHESELECT_COUNTERFACTUAL_EXCLUDE_COMPLETED=0
export CACHESELECT_CORRECT_KV_POSITIONS=1
export CACHESELECT_COUNTERFACTUAL_BLOCKS_PER_BATCH=300
export CACHESELECT_COUNTERFACTUAL_INPUT_ROOT="/vol/bitbucket/$USER/cacheselect-inputs/qwen3-mtrag-rope-training-200-v1"

exec bash "${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}/benchmarks/run_imperial_qwen3_14b_counterfactual.slurm"
