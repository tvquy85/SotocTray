import argparse
import json
import os
import numpy as np
from src_tc.simple_baselines import equal_weight_all, random_topk_mean
from src_tc.backtest import backtest_topk


def scalarize(stats):
    return {k: float(v) for k, v in stats.items() if isinstance(v, (int, float, np.floating))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cost_bps", type=float, default=15.0)
    parser.add_argument("--n_random", type=int, default=100)
    args = parser.parse_args()

    data = np.load(args.pred_npz)
    pred, gt, mask = data["pred"], data["gt"], data["mask"]
    os.makedirs(args.out, exist_ok=True)

    result = {
        "score_topk": scalarize(backtest_topk(pred, gt, mask, topk=args.topk, cost_bps=args.cost_bps)),
        "equal_weight_all": equal_weight_all(gt, mask),
        "random_topk_mean": random_topk_mean(gt, mask, topk=args.topk, n_random=args.n_random, cost_bps=args.cost_bps, seed=0),
    }

    with open(os.path.join(args.out, "simple_baselines.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(os.path.join(args.out, "simple_baselines.json"))


if __name__ == "__main__":
    main()
