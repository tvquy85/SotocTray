$ErrorActionPreference = "Stop"

Write-Host "Running S_CGRAN on NASDAQ..."
python -u scgran_eval_tc.py --dataset NASDAQ --seed 123456789 | Tee-Object -FilePath "scgran_eval_nasdaq.log"

Write-Host "Running S_CGRAN on SP500..."
python -u scgran_eval_tc.py --dataset SP500 --seed 123456789 | Tee-Object -FilePath "scgran_eval_sp500.log"

Write-Host "Done!"
