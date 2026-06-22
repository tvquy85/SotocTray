import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Set visual style for academic paper (AAAI)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'font.family': 'serif',
})

# 1. Generate synthetic timeline and market data (Simulating a Crypto Crash)
np.random.seed(42)
days = 150
dates = pd.date_range(start='2025-01-01', periods=days)

# Market Regime simulation: Bull (0-60), Sideways/Noisy (60-100), Crash (100-120), Recovery (120-150)
market_trend = np.concatenate([
    np.linspace(1, 1.5, 60),  # Bull
    1.5 + np.random.normal(0, 0.05, 40), # Noisy
    np.linspace(1.5, 0.8, 20), # Crash
    np.linspace(0.8, 0.9, 30)  # Recovery
])
market_price = market_trend + np.random.normal(0, 0.02, days)

# 2. Simulate Portfolio Performances
# Baseline (Static Penalty): Fails to cut losses during crash
baseline_port = market_price * 0.9 + np.random.normal(0, 0.03, days)
baseline_port[:60] = market_price[:60] * 1.05 # Good in bull
baseline_port[100:120] = market_price[100:120] * 0.8 # Terrible in crash (Max Drawdown high)

# ContextNet (Dynamic Penalty): Survives crash
dynamic_port = np.copy(baseline_port)
dynamic_port[60:100] = 1.6 + np.random.normal(0, 0.01, 40) # Stable during noise (alpha penalty high)
dynamic_port[100:150] = np.linspace(1.6, 1.55, 50) + np.random.normal(0, 0.01, 50) # Safe during crash (lambda penalty high)

# 3. Simulate ContextNet's Penalty Signals
# Alpha (Turnover penalty): Spikes during noise (60-100)
alpha_penalty = np.zeros(days)
alpha_penalty[50:110] = np.interp(np.arange(60), [0, 10, 50, 60], [0.1, 0.8, 0.8, 0.1])
alpha_penalty += np.random.normal(0, 0.02, days)
alpha_penalty = np.clip(alpha_penalty, 0.1, 0.9)

# Lambda (Downside penalty): Spikes right before and during crash (90-120)
lambda_penalty = np.zeros(days)
lambda_penalty[85:125] = np.interp(np.arange(40), [0, 10, 30, 40], [0.1, 0.95, 0.95, 0.1])
lambda_penalty += np.random.normal(0, 0.02, days)
lambda_penalty = np.clip(lambda_penalty, 0.1, 0.95)

# 4. Plotting the Teaser Figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)

# Top Subplot: Portfolio Performance
ax1.plot(dates, market_price, color='gray', linestyle='--', alpha=0.6, label='Market Index (Crypto)')
ax1.plot(dates, baseline_port, color='#E63946', label='Baseline (Static Penalty)', alpha=0.8)
ax1.plot(dates, dynamic_port, color='#2A9D8F', label='Unified Regime-Aware (Ours)', linewidth=3.5)

# Highlight Regimes
ax1.axvspan(dates[60], dates[100], color='orange', alpha=0.1, label='Noisy Regime')
ax1.axvspan(dates[100], dates[120], color='red', alpha=0.15, label='Bear/Crash Regime')

ax1.set_ylabel('Normalized Portfolio Value')
ax1.set_title('Regime-Aware Portfolio Adaptation vs. Static Baselines', pad=15)
ax1.legend(loc='upper left', frameon=True, shadow=True)

# Bottom Subplot: Dynamic Penalties (ContextNet Outputs)
ax2.plot(dates, alpha_penalty, color='orange', linestyle='-', label=r'Turnover Penalty ($\alpha_t$)')
ax2.plot(dates, lambda_penalty, color='red', linestyle='-', label=r'Downside Penalty ($\lambda_t$)')

ax2.set_ylabel('Penalty Weight')
ax2.set_xlabel('Time')
ax2.legend(loc='upper left', frameon=True)
ax2.set_ylim(0, 1.1)

# Format x-axis dates
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('teaser_figure_concept.png', dpi=300, bbox_inches='tight')
print("Teaser figure saved as 'teaser_figure_concept.png'")
