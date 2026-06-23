import os
import sys
import numpy as np
import torch
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scgran_model import SCGRAN

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stockmixer_dir = os.path.join(base_dir, "StockMixer")
sys.path.append(stockmixer_dir)

from src_tc.metrics import evaluate_prediction_metrics
from src_tc.backtest import backtest_topk

def load_data(dataset):
    dataset_path = os.path.join(stockmixer_dir, "dataset", dataset)
    if dataset == 'SP500':
        data = np.load(os.path.join(dataset_path, 'SP500.npy'))
        data = data[:, 915:, :]
        eod_data = data.astype(np.float32)
        price_data = data[:, :, -1].astype(np.float32)
        mask_data = np.ones((data.shape[0], data.shape[1]), dtype=np.float32)
        gt_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        steps = 1
        for ticket in range(data.shape[0]):
            for row in range(1, data.shape[1]):
                denom = data[ticket][row - steps][-1]
                gt_data[ticket][row] = (data[ticket][row][-1] - denom) / denom if denom != 0 else 0.0
    else:
        with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
            price_data = pickle.load(f)
    return eod_data, mask_data, price_data, gt_data

def get_predictions(model, eod_data, mask_data, price_data, gt_data, start_index, end_index, device):
    lookback_length = 16
    steps = 1
    stock_num = eod_data.shape[0]
    
    cur_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
    cur_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
    cur_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
    
    model.eval()
    with torch.no_grad():
        for cur_offset in range(
                start_index - lookback_length - steps + 1,
                end_index - lookback_length - steps + 1,
        ):
            m_batch = mask_data[:, cur_offset: cur_offset + lookback_length + steps]
            m_batch = np.min(m_batch, axis=1)
            
            data_batch = torch.Tensor(eod_data[:, cur_offset: cur_offset + lookback_length, :]).to(device)
            base_price = torch.Tensor(np.expand_dims(price_data[:, cur_offset + lookback_length - 1], axis=1)).to(device)
            mask_b = torch.Tensor(np.expand_dims(m_batch, axis=1)).to(device)
            gt_b = torch.Tensor(np.expand_dims(gt_data[:, cur_offset + lookback_length + steps - 1], axis=1)).to(device)
            
            pred, alpha_pred, lambda_alpha, lambda_beta = model(data_batch)
            
            return_ratio = torch.div(torch.sub(pred.squeeze(), base_price.squeeze()), base_price.squeeze() + 1e-12)
            
            idx = cur_offset - (start_index - lookback_length - steps + 1)
            cur_pred[:, idx] = return_ratio.cpu().numpy()
            cur_gt[:, idx] = gt_b[:, 0].cpu().numpy()
            cur_mask[:, idx] = mask_b[:, 0].cpu().numpy()
            
    return cur_pred, cur_gt, cur_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("tc_scgran_final_results.md", "w", encoding='utf-8') as f_out:
        f_out.write("# TC-SCGRAN Final Ensemble Results\n\n")
        
        for dataset in ['NASDAQ', 'SP500']:
            eod_data, mask_data, price_data, gt_data = load_data(dataset)
            stock_num = eod_data.shape[0]
            fea_num = eod_data.shape[2]
            
            if dataset == 'SP500':
                valid_index = 1006
                test_index = 1259
            else:
                valid_index = 756
                test_index = 1008
            trade_dates = mask_data.shape[1]
            
            valid_preds, test_preds = [], []
            valid_gt_out, valid_mask_out = None, None
            test_gt_out, test_mask_out = None, None
            
            for seed in [1, 2, 3]:
                model = SCGRAN(stocks=stock_num, time_steps=16, channels=fea_num, latent_dim=32).to(device)
                model.load_state_dict(torch.load(f"tc_scgran_best_{dataset}_{seed}.pt", weights_only=True))
                
                vp, vgt, vmask = get_predictions(model, eod_data, mask_data, price_data, gt_data, valid_index, test_index, device)
                tp, tgt, tmask = get_predictions(model, eod_data, mask_data, price_data, gt_data, test_index, trade_dates, device)
                
                valid_preds.append(vp)
                test_preds.append(tp)
                valid_gt_out, valid_mask_out = vgt, vmask
                test_gt_out, test_mask_out = tgt, tmask
                
            mu_v = np.mean(valid_preds, axis=0)
            sigma_v = np.std(valid_preds, axis=0)
            
            best_kappa = 0.0
            best_score = -1e18
            for kappa in [0.0, 0.25, 0.5, 1.0]:
                adj_score = mu_v - kappa * sigma_v
                bt = backtest_topk(adj_score, valid_gt_out, valid_mask_out, topk=5, cost_bps=10.0)
                if bt['net_sharpe'] > best_score:
                    best_score = bt['net_sharpe']
                    best_kappa = kappa
                    
            print(f"[{dataset}] Selected best kappa: {best_kappa}")
            
            mu_t = np.mean(test_preds, axis=0)
            sigma_t = np.std(test_preds, axis=0)
            
            adj_t_0 = mu_t
            pm_0 = evaluate_prediction_metrics(adj_t_0, test_gt_out, test_mask_out)
            bt_0 = backtest_topk(adj_t_0, test_gt_out, test_mask_out, topk=5, cost_bps=10.0)
            
            adj_t_k = mu_t - best_kappa * sigma_t
            pm_k = evaluate_prediction_metrics(adj_t_k, test_gt_out, test_mask_out)
            bt_k = backtest_topk(adj_t_k, test_gt_out, test_mask_out, topk=5, cost_bps=10.0)
            
            print(f"====== TC-SCGRAN ({dataset}) ======")
            print(f"Kappa 0: RIC {pm_0['RIC']:.4f}, Net Sharpe: {bt_0['net_sharpe']:.4f}, Turnover: {bt_0['avg_turnover']:.4f}, MDD: {bt_0['max_drawdown']:.4f}")
            print(f"Kappa {best_kappa}: RIC {pm_k['RIC']:.4f}, Net Sharpe: {bt_k['net_sharpe']:.4f}, Turnover: {bt_k['avg_turnover']:.4f}, MDD: {bt_k['max_drawdown']:.4f}")
            print("===================================\n")
            
            f_out.write(f"## Dataset: {dataset}\n")
            f_out.write(f"- **Selected Kappa**: {best_kappa}\n")
            f_out.write(f"- **Kappa 0 (No Gating)**: Net Sharpe: {bt_0['net_sharpe']:.4f}, Turnover: {bt_0['avg_turnover']:.4f}, RIC: {pm_0['RIC']:.4f}\n")
            f_out.write(f"- **Kappa {best_kappa} (With Gating)**: Net Sharpe: {bt_k['net_sharpe']:.4f}, Turnover: {bt_k['avg_turnover']:.4f}, RIC: {pm_k['RIC']:.4f}\n\n")

if __name__ == '__main__':
    main()
