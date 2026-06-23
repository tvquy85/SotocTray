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
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.check_call([
        sys.executable,
        "scripts/rebacktest_cost_grid.py",
        "--pred_npz", str(npz),
        "--out", str(out),
        "--costs", "0,10",
        "--topks", "1,2",
    ], env=env)

    text = (out / "cost_grid.csv").read_text()
    assert "cost_bps" in text
    assert "topk" in text
