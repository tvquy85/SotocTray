import argparse
import json
import os
import yaml
import numpy as np
import torch
from torch.cuda.amp import GradScaler

from src_tc.config import load_config
from src_tc.data import StockDatasetGPU, make_offsets, assert_no_target_leakage
from src_tc.model_tc_v2 import StockMixerBackboneV2
from src_tc.trainer_v2 import train_one_epoch, evaluate_model
from src_tc.utils import set_seed, configure_torch


def save_all_results(cfg, history, valid_pred, valid_gt, valid_mask, valid_pm, valid_bt, test_pred, test_gt, test_mask, test_pm, test_bt):
    import pandas as pd
    
    if history:
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(cfg.output_dir, 'history.csv'), index=False)

    def get_scalars(bt_dict):
        res = {}
        for topk, stats in bt_dict.items():
            res[topk] = {k: v for k, v in stats.items() if not isinstance(v, np.ndarray)}
        return res

    metrics = {
        'valid_prediction_metrics': valid_pm,
        'valid_backtest': get_scalars(valid_bt),
        'test_prediction_metrics': test_pm,
        'test_backtest': get_scalars(test_bt),
    }
    with open(os.path.join(cfg.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    np.savez(os.path.join(cfg.output_dir, 'valid_predictions.npz'), pred=valid_pred, gt=valid_gt, mask=valid_mask)
    np.savez(os.path.join(cfg.output_dir, 'test_predictions.npz'), pred=test_pred, gt=test_gt, mask=test_mask)

    equity_df = pd.DataFrame({
        'valid_equity': valid_bt['top5']['equity'],
    })
    equity_df.to_csv(os.path.join(cfg.output_dir, 'valid_equity_curve_top5.csv'), index=False)
    
    equity_df_test = pd.DataFrame({
        'test_equity': test_bt['top5']['equity'],
    })
    equity_df_test.to_csv(os.path.join(cfg.output_dir, 'test_equity_curve_top5.csv'), index=False)

    with open(os.path.join(cfg.output_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(vars(cfg), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.output_dir = f"{cfg.output_dir}_seed{cfg.seed}"

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    configure_torch(cfg.speed_mode)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = StockDatasetGPU(cfg, device)
    train_offsets, valid_offsets, test_offsets = make_offsets(
        cfg.valid_index, cfg.test_index, dataset.trade_dates,
        cfg.lookback_length, cfg.steps
    )

    # Assert no leakage.
    assert_no_target_leakage(train_offsets, cfg.valid_index, cfg.test_index, cfg.lookback_length, cfg.steps, 'train')
    assert_no_target_leakage(valid_offsets, cfg.valid_index, cfg.test_index, cfg.lookback_length, cfg.steps, 'valid')
    assert_no_target_leakage(test_offsets, cfg.test_index, dataset.trade_dates, cfg.lookback_length, cfg.steps, 'test')

    # Model
    model = StockMixerBackboneV2(
        cfg.stock_num, cfg.lookback_length, cfg.fea_num, cfg.market_dim, cfg.scale_factor
    ).to(device)

    if cfg.torch_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp and device.type == 'cuda')

    best_score = -1e18
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(model, dataset, train_offsets, optimizer, scaler, cfg, device)
        _, _, _, valid_pred_metrics, valid_bt = evaluate_model(model, dataset, valid_offsets, cfg)
        valid_score = valid_bt['top5']['net_sharpe']

        row = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'valid_{k}': v for k, v in valid_pred_metrics.items()},
            'valid_top5_net_sharpe': valid_bt['top5']['net_sharpe'],
            'valid_top5_avg_turnover': valid_bt['top5']['avg_turnover'],
            'valid_top5_mdd': valid_bt['top5']['max_drawdown'],
        }
        history.append(row)
        print(row)

        if valid_score > best_score:
            best_score = valid_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
            torch.save(best_state, os.path.join(cfg.output_dir, 'best_model.pt'))
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    valid_pred, valid_gt, valid_mask, valid_pm, valid_bt = evaluate_model(model, dataset, valid_offsets, cfg)
    test_pred, test_gt, test_mask, test_pm, test_bt = evaluate_model(model, dataset, test_offsets, cfg)

    save_all_results(cfg, history, valid_pred, valid_gt, valid_mask, valid_pm, valid_bt, test_pred, test_gt, test_mask, test_pm, test_bt)


if __name__ == '__main__':
    main()
