import argparse
import glob
import json
import os
import numpy as np
from src_tc.stat_tests import bootstrap_metric_ci, paired_bootstrap_delta_ci, max_drawdown_from_returns


def load_returns(path):
    data = np.load(path)
    return data["returns"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed_root", required=True)
    parser.add_argument("--baseline_root", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prop_paths = sorted(glob.glob(os.path.join(args.proposed_root, "**", "test_backtest_returns_top5.npz"), recursive=True))
    if not prop_paths:
        raise FileNotFoundError("No proposed return files found")

    rows = []
    for p in prop_paths:
        r = load_returns(p)
        ci = bootstrap_metric_ci(r, n_boot=args.n_boot, seed=0)
        rows.append({
            "run": p,
            "sharpe": ci,
            "mdd": max_drawdown_from_returns(r),
        })

    result = {"proposed": rows}

    if args.baseline_root:
        base_paths = sorted(glob.glob(os.path.join(args.baseline_root, "**", "test_backtest_returns_top5.npz"), recursive=True))
        paired = []
        for p, b in zip(prop_paths, base_paths):
            paired.append({
                "proposed": p,
                "baseline": b,
                "delta_sharpe": paired_bootstrap_delta_ci(load_returns(p), load_returns(b), n_boot=args.n_boot, seed=1),
            })
        result["paired_vs_baseline"] = paired

    with open(os.path.join(args.out, "stat_tests.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(os.path.join(args.out, "stat_tests.json"))


if __name__ == "__main__":
    main()
