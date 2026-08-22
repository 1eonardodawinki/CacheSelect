#!/usr/bin/env bash
#SBATCH --job-name=qwen3-mtrag-cf
#SBATCH --partition=a40
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=/vol/bitbucket/%u/cacheselect-server-logs/qwen3-mtrag-cf-%j.out

set -Eeuo pipefail

EXPERIMENT_ID="${SLURM_JOB_ID:?Run this file with sbatch}"
source "${CACHESELECT_CUDA_SETUP:-/vol/cuda/13.0.0/setup.sh}"
export CACHESELECT_PROJECT_ROOT="${CACHESELECT_PROJECT_ROOT:-$HOME/DeltaCache}"
export CACHESELECT_VENV_ROOT="${CACHESELECT_VENV_ROOT:-/vol/bitbucket/$USER/cacheselect-env}"
export CACHESELECT_STORAGE_ROOT="${CACHESELECT_STORAGE_ROOT:-/vol/bitbucket/$USER}"
export CACHESELECT_COUNTERFACTUAL_INPUT_ROOT="${CACHESELECT_COUNTERFACTUAL_INPUT_ROOT:-/vol/bitbucket/$USER/cacheselect-inputs/qwen3-mtrag-v3}"
export CACHESELECT_COUNTERFACTUAL_START_CASE="${CACHESELECT_COUNTERFACTUAL_START_CASE:-1}"
export CACHESELECT_EXPERIMENT_ID="${CACHESELECT_EXPERIMENT_ID:-$EXPERIMENT_ID}"
export CACHESELECT_SERVER_PORT="${CACHESELECT_SERVER_PORT:-$((30000 + EXPERIMENT_ID % 1000))}"
export CACHESELECT_EXECUTION_PLATFORM="imperial-a40"
export CACHESELECT_HF_OFFLINE="${CACHESELECT_HF_OFFLINE:-1}"

exec bash "$CACHESELECT_PROJECT_ROOT/benchmarks/run_runpod_qwen3_14b_counterfactual.sh"
