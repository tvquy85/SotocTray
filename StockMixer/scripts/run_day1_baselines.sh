#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

python -m src_tc.run_experiment --config configs/nasdaq_baseline.yaml --seed 1 | tee logs/nasdaq_baseline_seed1.log
python -m src_tc.run_experiment --config configs/nyse_baseline.yaml --seed 1 | tee logs/nyse_baseline_seed1.log
python -m src_tc.run_experiment --config configs/sp500_baseline.yaml --seed 1 | tee logs/sp500_baseline_seed1.log
