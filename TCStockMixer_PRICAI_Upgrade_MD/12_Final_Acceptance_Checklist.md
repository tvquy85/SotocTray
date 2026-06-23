# 12 — Final Acceptance Checklist before PRICAI Submission

## Objective
Provide a hard gate for deciding whether TC-StockMixer is ready for PRICAI 2026 submission.

Do not submit until every mandatory item below is checked or explicitly disclosed as a limitation.

## A. Methodology gates

- [ ] `ContextNet` uses a global market-regime representation, not accidental feature averaging per stock.
- [ ] Training objective and reported backtest are aligned or both Top-K and softmax execution are reported separately.
- [ ] `alpha_t` and `lambda_t` diagnostics are exported and interpretable.
- [ ] The paper states whether context acts as training-time regularization or execution-time allocation control.
- [ ] Transaction cost is defined consistently across config, text, tables, and reports.

## B. Ablation gates

- [ ] `no_penalty` baseline completed.
- [ ] `static_alpha_only` completed.
- [ ] `static_lambda_only` completed.
- [ ] `static_both` completed.
- [ ] `dynamic_alpha_only` completed.
- [ ] `dynamic_lambda_only` completed.
- [ ] `dynamic_both` completed.
- [ ] `random_context` completed.
- [ ] `dynamic_both` beats `random_context` on the main current dataset.
- [ ] Results reported as mean ± std across seeds.

## C. Empirical robustness gates

- [ ] At least 10 seeds completed on the current dataset.
- [ ] At least 3 walk-forward folds completed on the current dataset.
- [ ] Bootstrap CI for Sharpe is reported.
- [ ] Paired bootstrap comparison against strongest baseline is reported.
- [ ] Cost grid completed for 0, 5, 10, 15, 25, and 50 bps.
- [ ] No best-seed number is used as the main result.

## D. Baseline gates

Minimum current-data baselines:

- [ ] Equal-weight all valid assets.
- [ ] Random Top-K mean over at least 100 random portfolios.
- [ ] Original Top-K score strategy.

Preferred PRICAI baselines:

- [ ] Original StockMixer under identical split and cost.
- [ ] Buy-and-hold market proxy.
- [ ] Markowitz or minimum-variance baseline if covariance estimation is stable.
- [ ] Hierarchical Risk Parity if feasible.
- [ ] At least one recent DRL portfolio baseline if implementation time allows.

If preferred baselines are absent, add a limitation and avoid SOTA language.

## E. Reproducibility gates

- [ ] `config_used.yaml` saved for every run.
- [ ] `metrics.json` saved for every run.
- [ ] `history.csv` saved for every run.
- [ ] `test_predictions.npz` saved for every run.
- [ ] `test_backtest_returns_top5.npz` saved for every run.
- [ ] Git commit hash saved in run metadata.
- [ ] Environment file saved: `requirements.txt` or `environment.yml`.
- [ ] All scripts run from a clean checkout.

## F. Paper gates

- [ ] Introduction claim is narrow and defensible.
- [ ] Related Work cites StockMixer and portfolio-learning literature.
- [ ] Methodology matches code.
- [ ] Experiment tables use current-data results and clearly state dataset/split/cost/seeds.
- [ ] Discussion includes failure cases.
- [ ] Conclusion avoids universal dominance claims.
- [ ] Double-anonymous formatting checked.
- [ ] LLM assistance documented according to submission policy.
- [ ] Page limit checked.

## G. Strong-accept threshold

Use this threshold for internal decision:

```text
Submit target: Weak Accept / Accept
- All methodology gates pass.
- Current dataset shows positive test Net Sharpe in majority of walk-forward folds.
- dynamic_both beats static_both and random_context in mean test Net Sharpe or risk-adjusted utility.
- Cost sensitivity remains acceptable at 15 bps and preferably 25 bps.
- Paper removes SOTA/first/perfect language.
```

```text
Strong Accept target
- Above conditions pass.
- Significant paired improvement against strongest baseline.
- Interpretable alpha/lambda regime behavior.
- Robust results across 10–30 seeds and walk-forward folds.
- Reviewer can reproduce all current-data tables from scripts.
```

## H. Stop conditions

Do not submit as PRICAI main-track if any of these remains true:

- [ ] ContextNet formulation and implementation mismatch remains unresolved.
- [ ] Backtest ignores the key claimed mechanism and no clarification is provided.
- [ ] Only one split and three seeds are reported.
- [ ] Transaction-cost values contradict across sections.
- [ ] Paper claims SOTA without adequate baselines.
- [ ] Code cannot reproduce the main table.

## Final command checklist

Run this before manuscript freeze:

```bash
cd StockMixer
pytest -q
python scripts/audit_run_outputs.py --root results/crypto_tc_v2
python scripts/audit_paper_claims.py --root paper
python scripts/aggregate_multiseed.py --root results/crypto_tc_v2 --out reports/final_multiseed
python scripts/run_stat_tests.py --proposed_root results/crypto_tc_v2 --out reports/final_stats --n_boot 2000
```

Expected final artifacts:

```text
reports/final_multiseed/all_runs.csv
reports/final_multiseed/summary.csv
reports/final_stats/stat_tests.json
reports/cost_grid/*/cost_grid.csv
reports/simple_baselines_*/simple_baselines.json
paper/*.md
```

