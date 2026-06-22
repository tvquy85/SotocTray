import argparse
import torch
import numpy as np
from src_tc.config import load_config
from src_tc.data import StockDatasetGPU, make_offsets, assert_no_target_leakage
from src_tc.model_tc import StockMixerBackbone
from src_tc.utils import set_seed, configure_torch
from src_tc.losses import compute_tc_loss
from src_tc.trainer import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    configure_torch(cfg.speed_mode)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading dataset...')
    dataset = StockDatasetGPU(cfg, device)
    
    print('Making offsets...')
    train_offsets, valid_offsets, test_offsets = make_offsets(
        cfg.valid_index, cfg.test_index, dataset.trade_dates,
        cfg.lookback_length, cfg.steps
    )

    print('Checking leakage...')
    assert_no_target_leakage(train_offsets, cfg.valid_index, cfg.test_index, cfg.lookback_length, cfg.steps, 'train')
    assert_no_target_leakage(valid_offsets, cfg.valid_index, cfg.test_index, cfg.lookback_length, cfg.steps, 'valid')
    assert_no_target_leakage(test_offsets, cfg.test_index, dataset.trade_dates, cfg.lookback_length, cfg.steps, 'test')
    assert len(train_offsets) > 0

    print('Initializing model...')
    model = StockMixerBackbone(
        stocks=cfg.stock_num,
        time_steps=cfg.lookback_length,
        channels=cfg.fea_num,
        market=cfg.market_dim,
        scale=cfg.scale_factor,
    ).to(device)

    print('Testing forward pass...')
    model.train()
    data_batch, mask_batch, price_batch, gt_batch = dataset.get_batch(int(train_offsets[0]))
    prediction = model(data_batch)
    assert prediction.shape == (cfg.stock_num, 1)
    assert torch.isfinite(prediction).all()

    print('Testing loss and backward pass...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer.zero_grad()
    loss, return_ratio, w, stats = compute_tc_loss(
        prediction, gt_batch, price_batch, mask_batch, None, cfg
    )
    assert torch.isfinite(loss).all()
    loss.backward()
    optimizer.step()

    print('Testing evaluation...')
    pred, gt, mask, pred_metrics, bt = evaluate_model(model, dataset, valid_offsets[:2], cfg)
    assert not np.isnan(pred_metrics['IC'])
    assert np.isfinite(bt['top5']['net_sharpe'])

    print('Sanity checks passed!')

if __name__ == '__main__':
    main()
