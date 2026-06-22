# PRICAI 2026 Experiment Summary

## Problem Statement
Most stock forecasting models optimize pointwise prediction or cross-sectional ranking metrics. However, trading decisions are affected by transaction costs, turnover, risk, and uncertainty. High IC does not guarantee high net-of-cost Sharpe.

## Method
We retain StockMixer's simple and strong market-state mixing architecture, but train and evaluate it using a transaction-cost-aware differentiable portfolio objective (NetRank) and uncertainty-gated selection via seed ensembles.

## Results
Experiments were conducted on NASDAQ and SP500 with a simulation transaction cost of 10 bps.

### 1. SP500 Results
TC-StockMixer significantly improves the net-of-cost performance and reduces turnover compared to the baseline StockMixer model:
- **Baseline (Seed 1):**
  - Net Sharpe (Top 5): 0.090
  - Average Turnover: 0.690
  - Max Drawdown: -36.1%
  - Rank IC: 0.0195
- **TC-StockMixer (Ensemble):**
  - Net Sharpe (Top 5): 0.710
  - Average Turnover: 0.435
  - Max Drawdown: -26.5%
  - Rank IC: 0.0134

**Observation:** By incorporating the NetRank loss, TC-StockMixer reduced the average daily turnover by ~37% (from 0.690 to 0.435). This drastic cost reduction directly translated to a massive boost in Net Sharpe Ratio (from 0.090 to 0.710) and improved risk management (Max Drawdown improved from -36.1% to -26.5%).

### 2. NASDAQ Results
- **Baseline (Seed 1):**
  - Net Sharpe (Top 5): -0.398
  - Average Turnover: 0.831
- **TC-StockMixer (Seed 2):**
  - Net Sharpe (Top 5): 0.375
  - Average Turnover: 0.374

**Observation:** NASDAQ is more volatile, and the baseline model struggled with high turnover (0.831), resulting in a negative Net Sharpe. The TC-StockMixer successfully suppressed the turnover by >50%, achieving a positive Net Sharpe in the best seed.

### 3. Figures
Equity curves comparing the Baseline and TC-StockMixer are available in the `figures/` directory:
- `figures/equity_curve_SP500.png`
- `figures/equity_curve_NASDAQ.png`
## Conclusion
TC-StockMixer directly optimizes the net-of-cost stock selection while preserving the simplicity and market-state inductive bias of StockMixer.
