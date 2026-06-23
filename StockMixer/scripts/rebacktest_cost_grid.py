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
