import numpy as np
import torch

from src_tc.losses_v2 import compute_tc_loss_v2
from src_tc.metrics import evaluate_prediction_metrics
from src_tc.backtest import backtest_topk, backtest_softmax

def average_stats(stats_list):
    if not stats_list:
        return {}
    keys = stats_list[0].keys()
    return {k: float(np.mean([s[k] for s in stats_list])) for k in keys}


def train_one_epoch(model, dataset, offsets, optimizer, scaler, cfg, device):
    model.train()
    total_stats = []
    prev_w = None

    if cfg.gamma_net > 0:
        iter_offsets = offsets
    else:
        iter_offsets = np.random.permutation(offsets)

    for offset in iter_offsets:
        data_batch, mask_batch, price_batch, gt_batch = dataset.get_batch(int(offset))
        optimizer.zero_grad(set_to_none=True)

        if cfg.use_amp:
            with torch.cuda.amp.autocast():
                out = model(
                    data_batch,
                    mask_batch,
                    ablation_mode=getattr(cfg, "ablation_mode", "dynamic_both"),
                    random_context_seed=getattr(cfg, "random_context_seed", 12345),
                )
                prediction, alpha_t, lambda_t = out if isinstance(out, tuple) and len(out) == 3 else (out, None, None)
                if not getattr(cfg, 'dynamic_netrank', True):
                    alpha_t = None
                    lambda_t = None
                loss, return_ratio, w, stats = compute_tc_loss_v2(
                    prediction, gt_batch, price_batch, mask_batch, prev_w, cfg, alpha_t=alpha_t, lambda_t=lambda_t
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(
                data_batch,
                mask_batch,
                ablation_mode=getattr(cfg, "ablation_mode", "dynamic_both"),
                random_context_seed=getattr(cfg, "random_context_seed", 12345),
            )
            prediction, alpha_t, lambda_t = out if isinstance(out, tuple) and len(out) == 3 else (out, None, None)
            if not getattr(cfg, 'dynamic_netrank', True):
                alpha_t = None
                lambda_t = None
            loss, return_ratio, w, stats = compute_tc_loss_v2(
                prediction, gt_batch, price_batch, mask_batch, prev_w, cfg, alpha_t=alpha_t, lambda_t=lambda_t
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        prev_w = w.detach()
        total_stats.append({k: float(v.detach().cpu()) for k, v in stats.items()})

    return average_stats(total_stats)


@torch.no_grad()
def predict_over_offsets(model, dataset, offsets, cfg, return_diagnostics=False):
    model.eval()
    preds = []
    gts = []
    masks = []
    diag_rows = []

    for offset in offsets:
        data_batch, mask_batch, price_batch, gt_batch = dataset.get_batch(int(offset))
        out = model(
            data_batch,
            mask_batch,
            ablation_mode=getattr(cfg, "ablation_mode", "dynamic_both"),
            random_context_seed=getattr(cfg, "random_context_seed", 12345),
        )
        prediction, alpha_t, lambda_t = out if isinstance(out, tuple) and len(out) == 3 else (out, None, None)
        return_ratio = (prediction - price_batch) / (price_batch + 1e-12)
        preds.append(return_ratio.squeeze(-1).detach().cpu().numpy())
        gts.append(gt_batch.squeeze(-1).detach().cpu().numpy())
        masks.append(mask_batch.squeeze(-1).detach().cpu().numpy())

        if return_diagnostics:
            valid_bool = mask_batch.squeeze(-1) > 0.5
            valid_count = int(valid_bool.sum().detach().cpu().item())
            market_return = float(gt_batch.squeeze(-1)[valid_bool].mean().detach().cpu().item()) if valid_count > 0 else 0.0
            diag_rows.append({
                "offset": int(offset),
                "alpha_t": float(alpha_t.mean().detach().cpu().item()) if alpha_t is not None else float("nan"),
                "lambda_t": float(lambda_t.mean().detach().cpu().item()) if lambda_t is not None else float("nan"),
                "valid_count": valid_count,
                "market_return": market_return,
            })

    prediction_np = np.stack(preds, axis=1)
    gt_np = np.stack(gts, axis=1)
    mask_np = np.stack(masks, axis=1)
    if return_diagnostics:
        return prediction_np, gt_np, mask_np, diag_rows
    return prediction_np, gt_np, mask_np


def evaluate_model(model, dataset, offsets, cfg):
    pred, gt, mask, diag_rows = predict_over_offsets(model, dataset, offsets, cfg, return_diagnostics=True)
    pred_metrics = evaluate_prediction_metrics(pred, gt, mask, precision_k=10)
    bt_softmax = backtest_softmax(pred, gt, mask, tau=cfg.tau, cost_bps=cfg.eval_cost_bps)
    bt5 = backtest_topk(pred, gt, mask, topk=5, cost_bps=cfg.eval_cost_bps)
    bt10 = backtest_topk(pred, gt, mask, topk=10, cost_bps=cfg.eval_cost_bps)
    return pred, gt, mask, pred_metrics, {'softmax': bt_softmax, 'top5': bt5, 'top10': bt10}, diag_rows
