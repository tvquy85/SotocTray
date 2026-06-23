import argparse
import random
import numpy as np
import os
import torch
import pickle
import json
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scgran_model import get_scgran_loss, SCGRAN

# Use StockMixer/src_tc evaluators and losses
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stockmixer_dir = os.path.join(base_dir, "StockMixer")
sys.path.append(stockmixer_dir)

from src_tc.metrics import evaluate_prediction_metrics
from src_tc.backtest import backtest_topk
from src_tc.losses import netrank_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NASDAQ')
    parser.add_argument('--seed', type=int, default=123456789)
    parser.add_argument('--alpha', type=float, default=0.1, help="Base Weight for pairwise rank loss")
    parser.add_argument('--lambda_alpha', type=float, default=0.05, help="Base Weight for RA-Loss")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=25)
    
    # TC Parameters
    parser.add_argument('--gamma_net', type=float, default=1.0, help="Weight for NetRank Loss")
    parser.add_argument('--tau', type=float, default=0.10, help="Temperature for Soft-Turnover")
    parser.add_argument('--train_cost_bps', type=float, default=10.0, help="Turnover cost weight in training")
    parser.add_argument('--topk', type=int, default=5, help="Top K portfolio")
    return parser.parse_args()

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed) 
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Dataset: {args.dataset}, Seed: {args.seed}, Device: {device}, TC Gamma: {args.gamma_net}")
    
    dataset_path = os.path.join(base_dir, "StockMixer", "dataset", args.dataset)
    
    if args.dataset == 'SP500':
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
        
    stock_num = eod_data.shape[0]
    lookback_length = 16
    steps = 1
    
    if args.dataset == 'SP500':
        valid_index = 1006
        test_index = 1259
    else:
        valid_index = 756
        test_index = 1008
        
    trade_dates = mask_data.shape[1]
    fea_num = eod_data.shape[2]
    latent_dim = 32

    model = SCGRAN(
        stocks=stock_num,
        time_steps=lookback_length,
        channels=fea_num,
        latent_dim=latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    max_train_offset = valid_index - lookback_length - steps + 1
    # For Sequential TC Training, we do not shuffle offsets
    train_offsets = np.arange(start=1, stop=max_train_offset, dtype=int)
    
    def get_batch(offset):
        seq_len = lookback_length
        m_batch = mask_data[:, offset: offset + seq_len + steps]
        m_batch = np.min(m_batch, axis=1)
        
        return (
            eod_data[:, offset: offset + seq_len, :],
            np.expand_dims(m_batch, axis=1),
            np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)
        )

    def validate(start_index, end_index):
        model.eval()
        with torch.no_grad():
            cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
            
            loss_sum = 0.0
            
            # For validation we also need prev_w to calculate net_loss properly, but usually we just track prediction performance.
            # To be strictly rigorous with valid loss, we can also sequentially compute netrank_loss.
            prev_w = torch.ones(stock_num, device=device) / stock_num
            
            for cur_offset in range(
                    start_index - lookback_length - steps + 1,
                    end_index - lookback_length - steps + 1,
            ):
                data_batch, mask_batch, price_batch, gt_batch = map(
                    lambda x: torch.Tensor(x).to(device), get_batch(cur_offset)
                )
                pred, alpha_pred, lambda_alpha, lambda_beta = model(data_batch)
                
                cur_loss, cur_reg_loss, cur_rank_loss_scgran, cur_alpha_loss, cur_rr = get_scgran_loss(
                    pred, alpha_pred, gt_batch, price_batch, mask_batch, stock_num, 
                    lambda_alpha, lambda_beta, args
                )
                
                with torch.amp.autocast('cuda'):
                    net_loss, prev_w, loss_terms = netrank_loss(
                        return_ratio=cur_rr,
                        gt=gt_batch,
                        mask=mask_batch,
                        prev_w=prev_w,
                        tau=args.tau,
                        cost_bps=args.train_cost_bps,
                        rho_concentration=0.0
                    )
                
                total_loss = cur_loss + args.gamma_net * net_loss
                loss_sum += total_loss.item()
                
                idx = cur_offset - (start_index - lookback_length - steps + 1)
                cur_valid_pred[:, idx] = cur_rr[:, 0].cpu()
                cur_valid_gt[:, idx] = gt_batch[:, 0].cpu()
                cur_valid_mask[:, idx] = mask_batch[:, 0].cpu()
            
            loss_avg = loss_sum / (end_index - start_index)
            
            pred_metrics = evaluate_prediction_metrics(cur_valid_pred, cur_valid_gt, cur_valid_mask)
            bt_metrics = backtest_topk(cur_valid_pred, cur_valid_gt, cur_valid_mask, topk=5, cost_bps=10.0)
            
            cur_valid_perf = {**pred_metrics, **bt_metrics}
        return loss_avg, cur_valid_perf

    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None
    best_epoch = -1
    patience_counter = 0

    print("Starting Training TC-SCGRAN...")
    for epoch in range(args.epochs):
        model.train()
        tra_loss = 0.0
        
        # SEQUENTIAL TRAINING LOOP: Initialize Uniform Portfolio at start of epoch
        prev_w = torch.ones(stock_num, device=device) / stock_num
        
        for j in range(len(train_offsets)):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device), get_batch(train_offsets[j])
            )
            optimizer.zero_grad()
            
            pred, alpha_pred, lambda_alpha, lambda_beta = model(data_batch)
            
            cur_loss, cur_reg_loss, cur_rank_loss_scgran, cur_alpha_loss, cur_rr = get_scgran_loss(
                pred, alpha_pred, gt_batch, price_batch, mask_batch, stock_num, 
                lambda_alpha, lambda_beta, args
            )
            
            with torch.amp.autocast('cuda'):
                net_loss, prev_w, loss_terms = netrank_loss(
                    return_ratio=cur_rr,
                    gt=gt_batch,
                    mask=mask_batch,
                    prev_w=prev_w,
                    tau=args.tau,
                    cost_bps=args.train_cost_bps,
                    rho_concentration=0.0
                )
            
            total_loss = cur_loss + args.gamma_net * net_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            tra_loss += total_loss.item()
            
            # Very important: detach prev_w to prevent backward through time
            prev_w = prev_w.detach()
            
        tra_loss = tra_loss / len(train_offsets)
        
        val_loss, val_perf = validate(valid_index, test_index)
        test_loss, test_perf = validate(test_index, trade_dates)

        print(f"epoch{epoch+1}##########################################################")
        print(f"Train : loss:{tra_loss:.2e}")
        print(f"Valid : loss:{val_loss:.2e}")
        print(f"Test  : loss:{test_loss:.2e}")
        print('Valid: IC:{:.4f}, RIC:{:.4f}, Net Sharpe:{:.4f}, Turnover:{:.4f}, MDD:{:.4f}'.format(
            val_perf['IC'], val_perf['RIC'], val_perf['net_sharpe'], val_perf['avg_turnover'], val_perf['max_drawdown']))
        print('Test : IC:{:.4f}, RIC:{:.4f}, Net Sharpe:{:.4f}, Turnover:{:.4f}, MDD:{:.4f}'.format(
            test_perf['IC'], test_perf['RIC'], test_perf['net_sharpe'], test_perf['avg_turnover'], test_perf['max_drawdown']))

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_valid_perf = val_perf
            best_test_perf = test_perf
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"tc_scgran_best_{args.dataset}_{args.seed}.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    print("\n================ FINAL BEST PERFORMANCE (Min Valid Loss) ================")
    print(f"Best Epoch: {best_epoch}")
    print('Valid: IC:{:.4f}, RIC:{:.4f}, Net Sharpe:{:.4f}, Turnover:{:.4f}, MDD:{:.4f}'.format(
        best_valid_perf['IC'], best_valid_perf['RIC'], best_valid_perf['net_sharpe'], best_valid_perf['avg_turnover'], best_valid_perf['max_drawdown']))
    print('Test : IC:{:.4f}, RIC:{:.4f}, Net Sharpe:{:.4f}, Turnover:{:.4f}, MDD:{:.4f}'.format(
        best_test_perf['IC'], best_test_perf['RIC'], best_test_perf['net_sharpe'], best_test_perf['avg_turnover'], best_test_perf['max_drawdown']))
    print("=========================================================================\n")

if __name__ == '__main__':
    main()
