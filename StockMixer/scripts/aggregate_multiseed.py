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

    metric_cols = [c for c in df.columns if c.startswith("test_") or c.startswith("valid_")]
    summary = summarize(df, metric_cols)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    print(f"Wrote {len(df)} runs to {args.out}")


if __name__ == "__main__":
    main()
