#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/u4/ichairman/private_synthetic_data_repair"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

"$PROJECT_ROOT/.venv/bin/python" synthetic_data_training/synthetic_data_training.py -m \
    data_name=census \
    model=aim \
    epsilon=0.7,0.8,0.9,1.0 \
    input_dir=resources/data \
    output_dir=resources/models
