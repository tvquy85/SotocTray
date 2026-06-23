# 09 — Transaction-Cost Robustness without Retraining

## Objective
Evaluate the trained model under multiple evaluation costs using the **same saved predictions**. This isolates whether reported performance depends on one favorable transaction-cost assumption.

This task is deliberately small: no retraining, no new dataset.

## Cost grid

Use this grid first:

```text
0, 5, 10, 15, 25, 50 bps
```

If the strategy only works at 0 or 5 bps, do not claim transaction-cost robustness.

## Files to add

```text
StockMixer/scripts/rebacktest_cost_grid.py
StockMixer/tests/test_cost_grid_backtest.py
```

## Step 1 — Create rebacktest script

Create `scripts/rebacktest_cost_grid.py`.

```python
import argparse
import json
import os
import numpy as np
import pandas as pd
from src_tc.backtest import backtest_topk


def scalarize(stats):
    return {k: float(v) for k, v in stats.items() if isinstance(v, (int, float, np.floating))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_npz", required=True, help="Path to test_predictions.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--costs", default="0,5,10,15,25,50")
    parser.add_argument("--topks", default="5,10")
    args = parser.parse_args()

    data = np.load(args.pred_npz)
    pred = data["pred"]
    gt = data["gt"]
    mask = data["mask"]

    costs = [float(x) for x in args.costs.split(",")]
    topks = [int(x) for x in args.topks.split(",")]
    os.makedirs(args.out, exist_ok=True)

    rows = []
    full = {}
    for cost in costs:
        full[str(cost)] = {}
        for k in topks:
            stats = backtest_topk(pred, gt, mask, topk=k, cost_bps=cost)
            sc = scalarize(stats)
            sc.update({"cost_bps": cost, "topk": k})
            rows.append(sc)
            full[str(cost)][f"top{k}"] = sc

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "cost_grid.csv"), index=False)
    with open(os.path.join(args.out, "cost_grid.json"), "w") as f:
        json.dump(full, f, indent=2)
    print(os.path.join(args.out, "cost_grid.csv"))


if __name__ == "__main__":
    main()
```

## Step 2 — Add unit test

Create `tests/test_cost_grid_backtest.py`.

```python
import numpy as np
import subprocess
import sys
from pathlib import Path


def test_rebacktest_cost_grid_runs(tmp_path):
    pred = np.array([
        [0.5, 0.4, 0.3, 0.2],
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    ], dtype=float)
    gt = np.array([
        [0.01, 0.02, -0.01, 0.00],
        [0.00, 0.01, 0.02, -0.02],
        [0.03, -0.01, 0.00, 0.01],
    ], dtype=float)
    mask = np.ones_like(pred)
    npz = tmp_path / "test_predictions.npz"
    np.savez(npz, pred=pred, gt=gt, mask=mask)

    out = tmp_path / "out"
    subprocess.check_call([
        sys.executable,
        "scripts/rebacktest_cost_grid.py",
        "--pred_npz", str(npz),
        "--out", str(out),
        "--costs", "0,10",
        "--topks", "1,2",
    ])

    text = (out / "cost_grid.csv").read_text()
    assert "cost_bps" in text
    assert "topk" in text
```

## Step 3 — Run on current saved predictions

For a single run:

```bash
cd StockMixer
pytest -q tests/test_cost_grid_backtest.py
python scripts/rebacktest_cost_grid.py \
  --pred_npz results/crypto_tc_v2_seed0/test_predictions.npz \
  --out reports/cost_grid_crypto_seed0 \
  --costs 0,5,10,15,25,50 \
  --topks 5,10
```

For all seed directories:

```bash
for p in results/crypto_tc_v2*/**/test_predictions.npz results/crypto_tc_v2*/test_predictions.npz; do
  [ -f "$p" ] || continue
  run_dir=$(dirname "$p")
  name=$(echo "$run_dir" | tr '/' '_')
  python scripts/rebacktest_cost_grid.py \
    --pred_npz "$p" \
    --out "reports/cost_grid/$name" \
    --costs 0,5,10,15,25,50 \
    --topks 5,10
done
```

## Pass conditions

This step passes only if:

1. `cost_grid.csv` is created for each selected run.
2. Net Sharpe and MDD are reported for each cost.
3. The 15 bps result matches the original `metrics.json` within tolerance.
4. Performance degrades monotonically or near-monotonically as cost increases. If not, inspect turnover computation.

## Required paper table

Add a table like this:

```text
Cost bps | Net Sharpe | MDD | Avg turnover | Annual return
0        | ...        | ... | ...          | ...
5        | ...        | ... | ...          | ...
10       | ...        | ... | ...          | ...
15       | ...        | ... | ...          | ...
25       | ...        | ... | ...          | ...
50       | ...        | ... | ...          | ...
```

## Claim rule

Use this rule:

- If Net Sharpe stays positive at 25 bps: claim **moderate cost robustness**.
- If Net Sharpe stays positive at 50 bps: claim **strong cost robustness**.
- If Net Sharpe fails at 15 bps: do not claim transaction-cost robustness.

