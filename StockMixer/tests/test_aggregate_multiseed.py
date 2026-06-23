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
