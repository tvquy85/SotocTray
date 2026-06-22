$ErrorActionPreference = "Stop"

$seeds = @(1, 2, 3)
$configs = @("nasdaq_tc", "sp500_tc", "nyse_tc")

foreach ($seed in $seeds) {
    foreach ($config in $configs) {
        Write-Host "Running $config for seed $seed"
        python -m src_tc.run_experiment --config configs/${config}.yaml --seed $seed | Tee-Object -FilePath "logs/${config}_seed${seed}.log"
    }
}

python -m src_tc.aggregate_results --dataset NASDAQ --method tc_stockmixer --seeds 1 2 3
python -m src_tc.aggregate_results --dataset SP500 --method tc_stockmixer --seeds 1 2 3
python -m src_tc.aggregate_results --dataset NYSE --method tc_stockmixer --seeds 1 2 3
python -m src_tc.generate_report
