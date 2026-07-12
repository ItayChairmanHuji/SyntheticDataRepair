#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_ROOT="/u4/ichairman/private_synthetic_data_repair"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

python synthetic_data_sampling/synthetic_data_sampling.py \
  --multirun \
  data_name=adult \
  model=mst \
  epsilon=1.0 \
  seed=0 \
  hydra.job.chdir=false