import numpy as np
import torch
from src_tc.trainer_v2 import predict_over_offsets


class DummyCfg:
    tau = 0.1


class DummyDataset:
    def get_batch(self, offset):
        x = torch.ones(4, 3, 2) * float(offset + 1)
        mask = torch.tensor([[1.0], [1.0], [0.0], [1.0]])
        price = torch.ones(4, 1)
        gt = torch.tensor([[0.01], [0.02], [0.99], [-0.01]])
        return x, mask, price, gt


class DummyModel(torch.nn.Module):
    def forward(self, x, mask=None):
        pred = torch.arange(x.shape[0], dtype=torch.float32).view(-1, 1)
        return pred, torch.tensor([[0.7]]), torch.tensor([[1.2]])


def test_predict_over_offsets_returns_diagnostics():
    pred, gt, mask, diag = predict_over_offsets(
        DummyModel(), DummyDataset(), offsets=np.array([0, 1, 2]), cfg=DummyCfg(), return_diagnostics=True
    )
    assert pred.shape == (4, 3)
    assert len(diag) == 3
    assert diag[0]['valid_count'] == 3
    assert abs(diag[0]['alpha_t'] - 0.7) < 1e-6
    assert abs(diag[0]['lambda_t'] - 1.2) < 1e-6
