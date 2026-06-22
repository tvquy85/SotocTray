#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

python -m src_tc.run_experiment --config configs/nasdaq_tc.yaml --seed 1 | tee logs/nasdaq_tc_seed1.log
python -m src_tc.run_experiment --config configs/sp500_tc.yaml --seed 1 | tee logs/sp500_tc_seed1.log
python -m src_tc.run_experiment --config configs/nyse_tc.yaml --seed 1 | tee logs/nyse_tc_seed1.log
