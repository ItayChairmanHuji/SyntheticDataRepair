#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/u4/ichairman/private_synthetic_data_repair"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

python marginals_generation/marginals_generation.py \
  --multirun \
  data_name=compas,tax \
  rho=1 \
  seed=0,1,2,3,4,5,6,7,8,9 \
  num_of_marginals=10,20,30,40,50,60,70,80,90,100 \