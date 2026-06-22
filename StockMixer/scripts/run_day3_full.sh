#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

for seed in 1 2 3; do
  python -m src_tc.run_experiment --config configs/nasdaq_tc.yaml --seed $seed | tee logs/nasdaq_tc_seed${seed}.log
  python -m src_tc.run_experiment --config configs/sp500_tc.yaml --seed $seed | tee logs/sp500_tc_seed${seed}.log
  python -m src_tc.run_experiment --config configs/nyse_tc.yaml --seed $seed | tee logs/nyse_tc_seed${seed}.log
done

python -m src_tc.aggregate_results --dataset NASDAQ --method tc_stockmixer --seeds 1 2 3
python -m src_tc.aggregate_results --dataset SP500 --method tc_stockmixer --seeds 1 2 3
python -m src_tc.aggregate_results --dataset NYSE --method tc_stockmixer --seeds 1 2 3
