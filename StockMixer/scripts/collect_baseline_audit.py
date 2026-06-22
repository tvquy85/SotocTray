import argparse
import json
from pathlib import Path
import torch

from src_tc.config import load_config
from src_tc.data import StockDatasetGPU, make_offsets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="audit/baseline_audit.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StockDatasetGPU(cfg, device)
    train, valid, test = make_offsets(
        cfg.valid_index, cfg.test_index, dataset.trade_dates, cfg.lookback_length, cfg.steps
    )

    audit = {
        "config_path": args.config,
        "config": vars(cfg),
        "device": str(device),
        "trade_dates": int(dataset.trade_dates),
        "eod_shape": list(dataset.eod.shape),
        "mask_shape": list(dataset.mask.shape),
        "gt_shape": list(dataset.gt.shape),
        "price_shape": list(dataset.price.shape),
        "offset_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "offset_ranges": {
            "train": [int(train.min()), int(train.max())],
            "valid": [int(valid.min()), int(valid.max())],
            "test": [int(test.min()), int(test.max())],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
