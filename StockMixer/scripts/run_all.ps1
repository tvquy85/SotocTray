$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES="0"

Write-Host "Running Day 1 Baselines..."
python -m src_tc.run_experiment --config configs/nasdaq_baseline.yaml --seed 1
python -m src_tc.run_experiment --config configs/sp500_baseline.yaml --seed 1
python -m src_tc.run_experiment --config configs/nyse_baseline.yaml --seed 1

Write-Host "Running Day 3 Full TC-StockMixer..."
foreach ($seed in 1..3) {
  Write-Host "Running seed $seed..."
  python -m src_tc.run_experiment --config configs/nasdaq_tc.yaml --seed $seed
  python -m src_tc.run_experiment --config configs/sp500_tc.yaml --seed $seed
  python -m src_tc.run_experiment --config configs/nyse_tc.yaml --seed $seed
}

Write-Host "Aggregating results..."
python -m src_tc.aggregate_results --dataset NASDAQ --method tc_stockmixer --seeds 1 2 3
python -m src_tc.aggregate_results --dataset SP500 --method tc_stockmixer --seeds 1 2 3
python -m src_tc.aggregate_results --dataset NYSE --method tc_stockmixer --seeds 1 2 3

Write-Host "All experiments finished!"

Write-Host 'Generating reports...'
python -m src_tc.generate_report
