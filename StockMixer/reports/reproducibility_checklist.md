# Reproducibility Checklist (TC-StockMixer)

## 1. Environment
- `python >= 3.8` (Tested on `3.10`)
- `torch >= 2.0` (Tested on `cu118`)
- Dependencies installed via `requirements_tc.txt`

## 2. Dataset Setup
- Datasets placed in `dataset/NASDAQ`, `dataset/SP500`. 
- NYSE dataset is intentionally ignored due to missing data.
- Ensure the data paths match the `dataset_root` in the YAML configs.

## 3. Training & Evaluation
To reproduce the main results presented in `PRICAI2026_experiment_summary.md`:
1. Run parallel training (will take a few minutes on RTX 3090, requires ~12GB VRAM):
   ```bash
   python -m src_tc.run_parallel
   ```
2. The script will automatically:
   - Run baseline StockMixer (seed 1) for all datasets.
   - Run TC-StockMixer (seeds 1, 2, 3) for all datasets.
   - Aggregate seeds and pick the best ensemble kappa on the validation set.
   - Output final metrics to `tables/main_results.csv` and `tables/ensemble_*.csv`.
   - Generate equity curve comparison plots in `figures/`.

## 4. Source Code Location
All custom logic for TC-StockMixer is located in `src_tc/`:
- `losses.py`: Contains `netrank_loss`.
- `backtest.py`: Computes net sharpe, turnover, drawdowns.
- `trainer.py`: Uses sequential historical windows for the turnover logic (`prev_w`).
