# 06 — Multi-Seed Runner and Aggregation on Current Dataset

## Objective
Replace 3-seed anecdotal reporting with a reproducible multi-seed protocol on the existing dataset.

Minimum target for internal decision: **10 seeds**.  
PRICAI-ready target: **30 seeds** if compute permits.

## Files to add

```text
StockMixer/scripts/run_multiseed.py
StockMixer/scripts/aggregate_multiseed.py
StockMixer/tests/test_aggregate_multiseed.py
```

## Step 1 — Create multi-seed launcher

Create `scripts/run_multiseed.py`.

```python
import argparse
import os
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", required=True, help="Comma list, e.g. 0,1,2,3,4")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base = yaml.safe_load(f)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    base_out = base["output_dir"]
    tag = args.tag or os.path.basename(args.config).replace(".yaml", "")

    for seed in seeds:
        cfg = dict(base)
        cfg["seed"] = seed
        cfg["output_dir"] = os.path.join(base_out, tag)
        tmp_path = f"configs/tmp_{tag}_seed{seed}.yaml"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        cmd = [sys.executable, "-m", "src_tc.run_experiment_v2", "--config", tmp_path, "--seed", str(seed)]
        print("RUN", " ".join(cmd))
        if not args.dry_run:
            subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
```

## Step 2 — Create aggregation script

Create `scripts/aggregate_multiseed.py`.

```python
import argparse
import glob
import json
import os
import pandas as pd


def flatten_metrics(path):
    with open(path, "r") as f:
        m = json.load(f)
    row = {"metrics_path": path, "run_dir": os.path.dirname(path)}

    for split in ["valid_backtest", "test_backtest"]:
        for kname, stats in m.get(split, {}).items():
            for key, val in stats.items():
                if isinstance(val, (int, float)):
                    row[f"{split}_{kname}_{key}"] = val

    for split in ["valid_prediction_metrics", "test_prediction_metrics"]:
        for key, val in m.get(split, {}).items():
            if isinstance(val, (int, float)):
                row[f"{split}_{key}"] = val
    return row


def summarize(df, metric_cols):
    rows = []
    for c in metric_cols:
        s = df[c].dropna()
        rows.append({
            "metric": c,
            "n": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "metrics.json"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No metrics.json under {args.root}")

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame([flatten_metrics(p) for p in paths])
    df.to_csv(os.path.join(args.out, "all_runs.csv"), index=False)

    metric_cols = [c for c in df.columns if c.startswith("test_backtest") or c.startswith("valid_backtest")]
    summary = summarize(df, metric_cols)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    print(f"Wrote {len(df)} runs to {args.out}")


if __name__ == "__main__":
    main()
```

## Step 3 — Unit test aggregation

Create `tests/test_aggregate_multiseed.py`.

```python
import json
import os
import subprocess
import sys
from pathlib import Path


def write_metrics(path, sharpe):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "valid_backtest": {"top5": {"net_sharpe": sharpe, "avg_turnover": 0.2, "max_drawdown": -0.1}},
        "test_backtest": {"top5": {"net_sharpe": sharpe + 1, "avg_turnover": 0.3, "max_drawdown": -0.2}},
        "valid_prediction_metrics": {"mse": 0.01},
        "test_prediction_metrics": {"mse": 0.02},
    }
    path.write_text(json.dumps(payload))


def test_aggregate_multiseed(tmp_path):
    root = tmp_path / "results"
    write_metrics(root / "run_seed0" / "metrics.json", 1.0)
    write_metrics(root / "run_seed1" / "metrics.json", 3.0)
    out = tmp_path / "agg"

    subprocess.check_call([
        sys.executable,
        "scripts/aggregate_multiseed.py",
        "--root", str(root),
        "--out", str(out),
    ])

    assert (out / "all_runs.csv").exists()
    summary = (out / "summary.csv").read_text()
    assert "test_backtest_top5_net_sharpe" in summary
```

## Step 4 — Run commands

Smoke test:

```bash
cd StockMixer
pytest -q tests/test_aggregate_multiseed.py
python scripts/run_multiseed.py --config configs/crypto_tc_v2.yaml --seeds 0,1 --tag smoke --dry_run
```

Actual current-data run:

```bash
python scripts/run_multiseed.py --config configs/crypto_tc_v2.yaml --seeds 0,1,2,3,4,5,6,7,8,9 --tag crypto_dynamic_both_10seed
python scripts/aggregate_multiseed.py --root results/crypto_tc_v2 --out reports/crypto_dynamic_both_10seed
```

## Pass conditions

This step passes only if:

1. 10 `metrics.json` files exist for the chosen config.
2. `reports/crypto_dynamic_both_10seed/all_runs.csv` has 10 rows.
3. `summary.csv` reports mean and std for at least:
   - `test_backtest_top5_net_sharpe`
   - `test_backtest_top5_max_drawdown`
   - `test_backtest_top5_avg_turnover`
4. No seed silently overwrites another seed directory.

## PRICAI reporting requirement

The paper must report:

```text
mean ± std across seeds
number of seeds
best seed not emphasized as main result
```

If only 10 seeds are available, write:

> We report 10-seed statistics on the current dataset and reserve larger-scale validation for future work.

Do not claim robustness from a single best seed.

