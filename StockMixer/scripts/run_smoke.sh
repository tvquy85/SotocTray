#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

python -m src_tc.sanity_checks --config configs/nasdaq_baseline.yaml
python -m src_tc.run_experiment --config configs/nasdaq_smoke.yaml --seed 1
