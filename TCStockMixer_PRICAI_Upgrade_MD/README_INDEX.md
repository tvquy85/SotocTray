# README_INDEX — How to Use These Antigravity Tasks

## Purpose
This folder contains micro-task Markdown files for upgrading TC-StockMixer toward a PRICAI 2026-ready submission. Each task is intentionally short enough for Antigravity to execute with low risk.

## Recommended execution order

Run in this order:

```text
Goal.md
01_Reproducibility_Baseline_Audit.md
02_Global_ContextNet_Regime.md
03_Softmax_Backtest_Alignment.md
04_Inference_Diagnostics.md
05_Ablation_Controls.md
06_MultiSeed_Runner_Aggregation.md
07_WalkForward_Current_Data.md
08_Statistical_Testing.md
09_Cost_Robustness.md
10_Baselines_Current_Data.md
11_Paper_Claim_Control.md
12_Final_Acceptance_Checklist.md
```

## Execution policy

For each file:

1. Complete only the specified files.
2. Run the listed test cases.
3. Save command outputs.
4. Do not move to the next task until pass conditions are satisfied.
5. Prefer current dataset and current configs first.
6. Do not expand to full-scale datasets until current-data evidence improves.

## Current high-risk issues addressed

The tasks directly address these issues:

```text
Formulation-code mismatch in ContextNet
Training objective versus Top-K backtest mismatch
No inference diagnostics for alpha/lambda
Insufficient ablations
Too few seeds
No walk-forward validation
No confidence intervals or paired tests
Transaction-cost inconsistency
Weak baselines
Overclaiming in paper text
```

## Minimum PRICAI-ready evidence package

At the end, the repo should contain:

```text
results/.../metrics.json
results/.../history.csv
results/.../test_predictions.npz
results/.../test_backtest_returns_top5.npz
reports/final_multiseed/summary.csv
reports/final_stats/stat_tests.json
reports/cost_grid/*/cost_grid.csv
reports/simple_baselines_*/simple_baselines.json
paper/*.md with claim-controlled text
```

## Important constraint

These tasks are designed for the dataset and configs that already exist in the repository. The goal is to improve correctness and empirical credibility first, not to scale prematurely.

