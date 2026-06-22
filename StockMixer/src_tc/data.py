import os
import pickle
import numpy as np
import torch


def make_offsets(valid_index: int, test_index: int, trade_dates: int, lookback: int, steps: int):
    train_start = 0
    train_end = valid_index - lookback - steps + 1
    valid_start = valid_index - lookback - steps + 1
    valid_end = test_index - lookback - steps + 1
    test_start = test_index - lookback - steps + 1
    test_end = trade_dates - lookback - steps + 1

    assert train_end > train_start, 'empty train offsets'
    assert valid_end > valid_start, 'empty valid offsets'
    assert test_end > test_start, 'empty test offsets'

    train_offsets = np.arange(train_start, train_end, dtype=int)
    valid_offsets = np.arange(valid_start, valid_end, dtype=int)
    test_offsets = np.arange(test_start, test_end, dtype=int)

    return train_offsets, valid_offsets, test_offsets


def assert_no_target_leakage(offsets, split_start, split_end, lookback, steps, split_name):
    target_days = offsets + lookback + steps - 1
    if split_name == 'train':
        assert target_days.max() < split_start
    else:
        assert target_days.min() >= split_start
        assert target_days.max() < split_end


def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class StockDatasetGPU:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        if cfg.market_name == 'SP500':
            data = np.load(os.path.join(cfg.dataset_root, 'SP500', 'SP500.npy'))
            data = data[:, 915:, :]
            price_data = data[:, :, -1]
            mask_data = np.ones((data.shape[0], data.shape[1]), dtype=np.float32)
            eod_data = data.astype(np.float32)
            gt_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
            for ticket in range(data.shape[0]):
                for row in range(1, data.shape[1]):
                    denom = data[ticket][row - cfg.steps][-1]
                    gt_data[ticket][row] = (data[ticket][row][-1] - denom) / denom if denom != 0 else 0.0
        else:
            eod_data = _load_pickle(os.path.join(cfg.dataset_root, cfg.market_name, 'eod_data.pkl'))
            mask_data = _load_pickle(os.path.join(cfg.dataset_root, cfg.market_name, 'mask_data.pkl'))
            gt_data = _load_pickle(os.path.join(cfg.dataset_root, cfg.market_name, 'gt_data.pkl'))
            price_data = _load_pickle(os.path.join(cfg.dataset_root, cfg.market_name, 'price_data.pkl'))

        self.trade_dates = eod_data.shape[1]
        self.eod = torch.as_tensor(eod_data, dtype=torch.float32, device=device)
        self.mask = torch.as_tensor(mask_data, dtype=torch.float32, device=device)
        self.gt = torch.as_tensor(gt_data, dtype=torch.float32, device=device)
        self.price = torch.as_tensor(price_data, dtype=torch.float32, device=device)

    def get_batch(self, offset: int):
        L = self.cfg.lookback_length
        steps = self.cfg.steps
        mask_batch = self.mask[:, offset: offset + L + steps].min(dim=1).values.unsqueeze(1)
        data_batch = self.eod[:, offset: offset + L, :]
        price_batch = self.price[:, offset + L - 1].unsqueeze(1)
        gt_batch = self.gt[:, offset + L + steps - 1].unsqueeze(1)
        return data_batch, mask_batch, price_batch, gt_batch
